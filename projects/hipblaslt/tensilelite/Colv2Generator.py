# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RMSNorm col-major scale kernel generator (colv2).

colv2 applies RMSNorm in-place to a bf16 col-major matrix C using precomputed
partial sum-of-squares from partialBuf.  It is a 2D-grid kernel:
  wg_id_x (s2) selects a 256-row tile.
  wg_id_y (s3) selects a 256-column tile.

Algorithm — Phase 1 (reduction, one wave = 64 lanes per row subgroup):
  Each block of 256 threads is organised as 4 waves x 64 lanes.  The outer
  loop (s_iter = 0..63) assigns one global row per iteration.  All 64 lanes
  of the wave collectively load one element each from partialBuf for that row.
  A 6-stage ds_bpermute butterfly then sums the 64 partial values.  All lanes
  with row < M write rsqrt(inv_d * total + eps) to LDS at address
  (wave_id * 256 + s_iter * 4).

Algorithm — Phase 2 (scale, all 256 threads):
  After the barrier each thread reads rstd = LDS[Serial * 4] for its own row.
  The column loop iterates over wg_id_y * 256 .. min((wg_id_y+1)*256, N)-1,
  loading each bf16, multiplying by rstd (shift-left-16 trick), and storing
  back.

Kernarg layout (offsets fixed — host code depends on them):
  offset  0 (8B): ptr_c  — bf16, col-major M x N   value_kind=global_buffer
  offset  8 (8B): ptr_d  — f32,  row-major M x n_d  value_kind=global_buffer
  offset 16 (4B): M      (u32)                       value_kind=by_value
  offset 20 (4B): N      (u32)                       value_kind=by_value
  offset 24 (4B): n_d    (u32, columns of partialBuf) value_kind=by_value
  offset 28 (4B): inv_d  (f32, 1/N_hidden)           value_kind=by_value
  offset 32 (4B): eps    (f32)                        value_kind=by_value
  kernarg_segment_size = 40 (aligned to 8 bytes)

LDS = 1024 bytes: 4 waves x 64 iterations x 4 bytes = 1024B.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml

from rocisa.code import Module, TextBlock, SrdUpperValue
from rocisa.container import EXEC, MUBUFModifiers, vgpr, sgpr
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
import rocisa.instruction as ri

from Tensile.Common.Utilities import _global_ti
from Tensile.Common.Architectures import gfxToIsa

_KERNEL_NAME = "colv2"


# ---------------------------------------------------------------------------
# Self-contained helpers (verbatim from AuxReductionGenerator.py)
# ---------------------------------------------------------------------------


def _kernel_header(name: str, gfx_arch: str, xnack: bool = False) -> str:
    target_id = f"{gfx_arch}:xnack+" if xnack else gfx_arch
    return (
        f'.amdgcn_target "amdgcn-amd-amdhsa--{target_id}"\n'
        f".text\n"
        f".global {name}\n"
        f".p2align 8\n"
        f".type {name},@function\n"
    )


def _kernel_rodata(
    name: str,
    gfx_arch: Tuple[int, int, int],
    vgpr_count: int,
    sgpr_count: int,
    lds: int,
    use_wg_id_y: bool = False,
) -> str:
    # Round vgpr_count up to a multiple of 4 for gfx9 AGPR-capable ISAs.
    vgpr_aligned = ((vgpr_count + 3) // 4) * 4
    lines = [
        ".rodata",
        ".p2align 6",
        f".amdhsa_kernel {name}",
        ".amdhsa_user_sgpr_kernarg_segment_ptr 1",
        ".amdhsa_system_sgpr_workgroup_id_x 1",
    ]
    if use_wg_id_y:
        lines.append(".amdhsa_system_sgpr_workgroup_id_y 1")
    lines.append(".amdhsa_system_vgpr_workitem_id 0")
    # gfx950 = (9, 5, 0); also gfx90a = (9, 0, 10)
    if gfx_arch == (9, 0, 10) or ((9, 4) < gfx_arch < (10, 0, 0)):
        lines.append(f".amdhsa_accum_offset {vgpr_aligned}")
    lines += [
        f".amdhsa_next_free_vgpr {vgpr_aligned}",
        f".amdhsa_next_free_sgpr {sgpr_count}",
        f".amdhsa_group_segment_fixed_size {lds}",
        ".amdhsa_private_segment_fixed_size 0",
        ".amdhsa_float_denorm_mode_32 3",
    ]
    if _global_ti.getArchCaps().get("HasWave32", False):
        lines.append(".amdhsa_wavefront_size32 1")
    lines.append(".end_amdhsa_kernel")
    return "\n".join(lines) + "\n"


@contextmanager
def _asm_func(func_name: str, module: Module):
    module.add(TextBlock(f"{func_name}:\n"))
    try:
        yield
    finally:
        end_label = f".L{func_name}_end"
        module.add(TextBlock(f"{end_label}:\n"))
        module.add(TextBlock(f".size {func_name}, {end_label} - {func_name}\n"))


@dataclass
class KernelArgument:
    size: int
    offset: int
    value_kind: str
    address_space: Optional[str] = None

    def to_dict(self):
        d = {".size": self.size, ".offset": self.offset, ".value_kind": self.value_kind}
        if self.address_space:
            d[".address_space"] = self.address_space
        return d


@dataclass
class KernelMeta:
    name: str
    num_vgpr: int
    num_sgpr: int
    num_agpr: int
    num_lds_bytes: int
    wavefront_size: int
    max_workgroup_size: int
    args_alignment: int
    args: List[KernelArgument]

    def _get_args_size(self) -> int:
        total = sum(a.size for a in self.args)
        a = self.args_alignment
        return ((total + a - 1) // a) * a

    def to_dict(self):
        return {
            ".name": self.name,
            ".symbol": f"{self.name}.kd",
            ".kernarg_segment_size": self._get_args_size(),
            ".group_segment_fixed_size": self.num_lds_bytes,
            ".private_segment_fixed_size": 0,
            ".kernarg_segment_align": self.args_alignment,
            ".wavefront_size": self.wavefront_size,
            ".sgpr_count": self.num_sgpr,
            ".vgpr_count": self.num_vgpr,
            ".agpr_count": self.num_agpr,
            ".max_flat_workgroup_size": self.max_workgroup_size,
            ".args": [a.to_dict() for a in self.args],
        }


def _meta_str(kernels: Tuple) -> str:
    beg = ".amdgpu_metadata\n---"
    content = yaml.dump(
        {"amdhsa.version": [1, 1], "amdhsa.kernels": [k.to_dict() for k in kernels]}
    )
    end = ".end_amdgpu_metadata"
    return "\n".join([beg, content, end])


# ---------------------------------------------------------------------------
# ISA initialisation (mirrors AuxReductionGenerator)
# ---------------------------------------------------------------------------


def _ensure_isa(chip: str):
    """Initialise the rocisa global TI singleton for the given chip."""
    from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

    gfx = chip.split(":")[0]
    isa = gfxToIsa(gfx)
    toolchain = validateToolchain(ToolchainDefaults.CXX_COMPILER)
    _global_ti.init(isa, toolchain, False)
    wave_size = 32 if isa[0] in (11, 12) else 64
    _global_ti.setKernel(isa, wave_size)
    return isa


# ---------------------------------------------------------------------------
# colv2 kernel body
# ---------------------------------------------------------------------------


def _build_kernel_body(isa: Tuple[int, int, int]) -> Tuple[Module, int, int]:
    """Generate the colv2 kernel body module.

    Returns (module, vgpr_count, sgpr_count).
    """
    archCaps = _global_ti.getArchCaps()
    srdUpper = SrdUpperValue(isa).getValue()

    # ABI: s0:s1 = kernarg ptr, s2 = wg_id_x, s3 = wg_id_y, v0 = Serial.
    wg_id_x_reg = 2
    wg_id_y_reg = 3

    # SGPR pool: s0..s3 reserved by ABI.
    sgpr_pool = RegisterPool(4, RegisterType.Sgpr, True)
    sgpr_pool.addRange(4, 52)

    srd_c = sgpr_pool.checkOutAligned(4, 4)   # C SRD
    srd_d = sgpr_pool.checkOutAligned(4, 4)   # partialBuf SRD
    s_M = sgpr_pool.checkOut(1)               # M
    s_N = sgpr_pool.checkOut(1)               # N
    s_nd = sgpr_pool.checkOut(1)              # n_d (columns of partialBuf)
    s_inv_d = sgpr_pool.checkOut(1)           # inv_d (f32)
    s_eps = sgpr_pool.checkOut(1)             # eps (f32)
    s_iter = sgpr_pool.checkOut(1)            # outer loop counter (0..63)
    s_row_base = sgpr_pool.checkOut(1)        # wg_id_x * 256
    s_col_start = sgpr_pool.checkOut(1)       # wg_id_y * 256
    s_col_end = sgpr_pool.checkOut(1)         # min(col_start+256, N)
    s_exec_save = sgpr_pool.checkOutAligned(2, 2)  # phase-1 outer exec save
    s_row_mask = sgpr_pool.checkOutAligned(2, 2)   # row < M mask (used as OOB guard)
    s_tmp = sgpr_pool.checkOutAligned(2, 2)        # scratch

    # VGPR pool: v0 (Serial) reserved by ABI.
    vgpr_pool = RegisterPool(1, RegisterType.Vgpr, True)
    vgpr_pool.addRange(1, 24)

    v_lane = vgpr_pool.checkOut(1)       # lane_id = Serial & 63
    v_wave_id = vgpr_pool.checkOut(1)    # wave_id_in_block = Serial >> 6
    v_acc = vgpr_pool.checkOut(1)        # f32 accumulator (phase 1)
    v_off = vgpr_pool.checkOut(1)        # byte-offset scratch
    v_tmp = vgpr_pool.checkOut(1)        # general temp
    v_perm = vgpr_pool.checkOut(1)       # butterfly permute byte addr
    v_bfly = vgpr_pool.checkOut(1)       # butterfly fetched value
    v_rstd = vgpr_pool.checkOut(1)       # per-thread rstd from LDS
    v_lds_addr = vgpr_pool.checkOut(1)   # LDS byte address
    v_row_iter = vgpr_pool.checkOut(1)   # global row for current iteration
    v_col_d = vgpr_pool.checkOut(1)      # current col into partialBuf

    func_name = _KERNEL_NAME
    mod = Module(func_name)

    with _asm_func(func_name, mod):

        # ------------------------------------------------------------------
        # Step 1: load kernargs.
        # ------------------------------------------------------------------
        mod.add(ri.SLoadB64(sgpr(srd_c, 2), sgpr(0, 2), 0, comment="ptr_c (C bf16)"))
        mod.add(ri.SLoadB64(sgpr(srd_d, 2), sgpr(0, 2), 8, comment="ptr_d (partialBuf f32)"))
        mod.add(ri.SLoadB32(sgpr(s_M), sgpr(0, 2), 16, comment="M"))
        mod.add(ri.SLoadB32(sgpr(s_N), sgpr(0, 2), 20, comment="N"))
        mod.add(ri.SLoadB32(sgpr(s_nd), sgpr(0, 2), 24, comment="n_d"))
        mod.add(ri.SLoadB32(sgpr(s_inv_d), sgpr(0, 2), 28, comment="inv_d"))
        mod.add(ri.SLoadB32(sgpr(s_eps), sgpr(0, 2), 32, comment="eps"))
        mod.add(ri.SWaitCnt(kmcnt=0, comment="wait kernarg loads"))

        # ------------------------------------------------------------------
        # Step 2: build SRDs (unbounded buffer limit; upper word from ISA).
        # ------------------------------------------------------------------
        mod.add(ri.SMovB32(sgpr(srd_c + 2), hex(0xFFFFFFFF), comment="C limit"))
        mod.add(ri.SMovB32(sgpr(srd_c + 3), hex(srdUpper), comment="C SRD flags"))
        mod.add(ri.SMovB32(sgpr(srd_d + 2), hex(0xFFFFFFFF), comment="partialBuf limit"))
        mod.add(ri.SMovB32(sgpr(srd_d + 3), hex(srdUpper), comment="partialBuf SRD flags"))

        if archCaps.get("WorkGroupIdFromTTM", False):
            mod.add(ri.SMovB32(
                dst=sgpr(wg_id_x_reg),
                src="ttmp9",
                comment="wg_id_x from ttmp9 (WorkGroupIdFromTTM)",
            ))

        # ------------------------------------------------------------------
        # Step 3: derive scalar tile offsets and per-thread indices.
        # ------------------------------------------------------------------
        mod.add(ri.SLShiftLeftB32(
            dst=sgpr(s_row_base),
            shiftHex=hex(8),
            src=sgpr(wg_id_x_reg),
            comment="s_row_base = wg_id_x * 256",
        ))
        mod.add(ri.SLShiftLeftB32(
            dst=sgpr(s_col_start),
            shiftHex=hex(8),
            src=sgpr(wg_id_y_reg),
            comment="s_col_start = wg_id_y * 256",
        ))
        mod.add(ri.SAddU32(
            dst=sgpr(s_col_end),
            src0=sgpr(s_col_start),
            src1=256,
            comment="s_col_end = col_start + 256 (clamp below)",
        ))
        mod.add(ri.SCmpLtI32(
            src0=sgpr(s_col_end),
            src1=sgpr(s_N),
            comment="col_end < N?",
        ))
        mod.add(ri.SCSelectB32(
            dst=sgpr(s_col_end),
            src0=sgpr(s_col_end),
            src1=sgpr(s_N),
            comment="s_col_end = min(col_start+256, N)",
        ))

        mod.add(ri.VAndB32(
            dst=vgpr(v_lane),
            src0=vgpr(0),
            src1=63,
            comment="v_lane = Serial & 63",
        ))
        mod.add(ri.VLShiftRightB32(
            dst=vgpr(v_wave_id),
            shiftHex=hex(6),
            src=vgpr(0),
            comment="v_wave_id = Serial >> 6",
        ))

        # ------------------------------------------------------------------
        # Step 4: Phase 1 — outer loop (s_iter = 0..63).
        # Each iteration processes one row per wave.  All lanes share the same
        # global row: v_row_iter = s_row_base + v_wave_id*64 + s_iter.
        # Lane v_lane selects the column into partialBuf.
        # ------------------------------------------------------------------
        mod.add(ri.SMovB32(dst=sgpr(s_iter), src=0, comment="s_iter = 0"))

        loop_phase1 = ".Lcolv2_phase1"
        mod.add(TextBlock(f"{loop_phase1}:\n"))

        # Compute global row for this iteration.
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(v_row_iter),
            shiftHex=hex(6),
            src=vgpr(v_wave_id),
            comment="wave_start = v_wave_id * 64",
        ))
        mod.add(ri.VAddU32(
            vgpr(v_row_iter), vgpr(v_row_iter), sgpr(s_iter),
            comment="row_in_block = wave_start + s_iter",
        ))
        mod.add(ri.VAddU32(
            vgpr(v_row_iter), vgpr(v_row_iter), sgpr(s_row_base),
            comment="v_row_iter = row_base + row_in_block",
        ))

        # Precompute row<M mask for the OOB trick inside the inner loop.
        mod.add(ri.VCmpGtU32(
            dst=sgpr(s_row_mask, 2),
            src0=sgpr(s_M),
            src1=vgpr(v_row_iter),
            comment="s_row_mask = row < M",
        ))

        # Save EXEC before inner loop (will be restored after butterfly).
        mod.add(ri.SMovB64(dst=sgpr(s_exec_save, 2), src=EXEC(), comment="save EXEC"))

        # Init accumulator.
        mod.add(ri.VMovB32(dst=vgpr(v_acc), src=0, comment="acc = 0.0f"))

        # Init col: v_col_d = v_lane (each lane starts at its lane index).
        mod.add(ri.VMovB32(
            dst=vgpr(v_col_d), src=vgpr(v_lane), comment="v_col_d = lane_id",
        ))

        # Check initial col < n_d; if no lane has a valid col skip inner loop.
        mod.add(ri.VCmpGtU32(
            dst=sgpr(s_tmp, 2),
            src0=sgpr(s_nd),
            src1=vgpr(v_col_d),
            comment="initial col-valid mask (n_d > v_col_d)",
        ))
        mod.add(ri.SCmpEQU64(
            src0=sgpr(s_tmp, 2),
            src1=0,
            comment="all lanes OOB?",
        ))

        bfly_label = ".Lcolv2_bfly"
        mod.add(ri.SCBranchSCC1(bfly_label, comment="skip inner loop if all cols OOB"))

        # Narrow EXEC to col-valid lanes.
        mod.add(ri.SAndB64(
            dst=EXEC(),
            src0=sgpr(s_exec_save, 2),
            src1=sgpr(s_tmp, 2),
            comment="EXEC = saved & col-valid",
        ))

        # ------------------------------------------------------------------
        # Inner loop: load partialBuf[row, col], accumulate, advance col.
        # Mirrors colv2.s BB_6: EXEC is narrowed progressively (not restored
        # inside the loop); exit when no lane has col < n_d.
        # ------------------------------------------------------------------
        inner_loop = ".Lcolv2_inner"
        mod.add(TextBlock(f"{inner_loop}:\n"))

        # byte_off = (v_row_iter * n_d + v_col_d) * 4.
        mod.add(ri.VMovB32(dst=vgpr(v_tmp), src=sgpr(s_nd), comment="v_tmp = n_d"))
        mod.add(ri.VMulLOU32(
            dst=vgpr(v_off),
            src0=vgpr(v_row_iter),
            src1=vgpr(v_tmp),
            comment="row * n_d",
        ))
        mod.add(ri.VAddU32(
            vgpr(v_off), vgpr(v_off), vgpr(v_col_d),
            comment="flat_idx = row*n_d + col",
        ))
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(v_off),
            shiftHex=hex(2),
            src=vgpr(v_off),
            comment="byte_off = flat_idx * 4",
        ))

        # OOB trick: row-OOB lanes get offset -1 so buffer returns 0; zero
        # the loaded value afterward for those lanes.
        mod.add(ri.VCndMaskB32(
            vgpr(v_off), -1, vgpr(v_off), sgpr(s_row_mask, 2),
            comment="v_off = -1 for row-OOB lanes",
        ))
        mod.add(ri.BufferLoadB32(
            vgpr(v_tmp),
            vgpr(v_off),
            sgpr(srd_d, 4),
            0,
            mubuf=MUBUFModifiers(offen=True),
            comment="load partialBuf[row, col]",
        ))
        mod.add(ri.SWaitCnt(vlcnt=0, comment="wait load"))
        mod.add(ri.VCndMaskB32(
            vgpr(v_tmp), 0, vgpr(v_tmp), sgpr(s_row_mask, 2),
            comment="zero v_tmp for row-OOB lanes",
        ))
        mod.add(ri.VAddF32(
            dst=vgpr(v_acc),
            src0=vgpr(v_acc),
            src1=vgpr(v_tmp),
            comment="acc += partial",
        ))

        # Advance column and recompute col-valid mask.
        mod.add(ri.VAddU32(vgpr(v_col_d), vgpr(v_col_d), 64, comment="v_col_d += 64"))
        mod.add(ri.VCmpGtU32(
            dst=sgpr(s_tmp, 2),
            src0=sgpr(s_nd),
            src1=vgpr(v_col_d),
            comment="new col-valid: n_d > v_col_d",
        ))

        # Narrow EXEC to saved_exec & new col-valid.  Exit if no active lanes.
        mod.add(ri.SAndB64(
            dst=EXEC(),
            src0=sgpr(s_exec_save, 2),
            src1=sgpr(s_tmp, 2),
            comment="EXEC = saved & new col-valid",
        ))
        mod.add(ri.SCmpEQU64(
            src0=EXEC(),
            src1=0,
            comment="all lanes done?",
        ))
        mod.add(ri.SCBranchSCC0(inner_loop, comment="loop while active lanes remain"))

        # ------------------------------------------------------------------
        # Step 5: restore EXEC then 6-stage butterfly (strides 1..32).
        # All lanes now hold their partial sum in v_acc.
        # ------------------------------------------------------------------
        mod.add(TextBlock(f"{bfly_label}:\n"))
        mod.add(ri.SMovB64(dst=EXEC(), src=sgpr(s_exec_save, 2), comment="restore EXEC"))

        for stride in (1, 2, 4, 8, 16, 32):
            mod.add(ri.VXorB32(
                dst=vgpr(v_perm),
                src0=vgpr(v_lane),
                src1=stride,
                comment=f"partner = lane_id ^ {stride}",
            ))
            mod.add(ri.VLShiftLeftB32(
                dst=vgpr(v_perm),
                shiftHex=hex(2),
                src=vgpr(v_perm),
                comment="partner_byte = partner * 4",
            ))
            mod.add(ri.DSBPermuteB32(
                dst=vgpr(v_bfly),
                src0=vgpr(v_perm),
                src1=vgpr(v_acc),
                comment=f"bfly = partner acc (stride={stride})",
            ))
            mod.add(ri.SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
            mod.add(ri.VAddF32(
                dst=vgpr(v_acc),
                src0=vgpr(v_acc),
                src1=vgpr(v_bfly),
                comment="acc += partner",
            ))

        # ------------------------------------------------------------------
        # Step 6: rstd = rsqrt(inv_d * acc + eps).
        # ------------------------------------------------------------------
        mod.add(ri.VMovB32(dst=vgpr(v_tmp), src=sgpr(s_inv_d), comment="v_tmp = inv_d"))
        mod.add(ri.VMulF32(
            dst=vgpr(v_acc),
            src0=vgpr(v_acc),
            src1=vgpr(v_tmp),
            comment="acc *= inv_d",
        ))
        mod.add(ri.VMovB32(dst=vgpr(v_tmp), src=sgpr(s_eps), comment="v_tmp = eps"))
        mod.add(ri.VAddF32(
            dst=vgpr(v_acc),
            src0=vgpr(v_tmp),
            src1=vgpr(v_acc),
            comment="acc += eps",
        ))
        mod.add(ri.VRsqF32(dst=vgpr(v_acc), src=vgpr(v_acc), comment="rstd = rsqrt(acc)"))
        if archCaps.get("TransOpWait", False):
            mod.add(ri.SNop(waitState=0, comment="TransOpWait after VRsqF32"))

        # ------------------------------------------------------------------
        # Step 7: write rstd to LDS at wave_id*256 + s_iter*4.
        # All lanes with row < M write (they all hold the same rstd).
        # ------------------------------------------------------------------
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(v_lds_addr),
            shiftHex=hex(8),
            src=vgpr(v_wave_id),
            comment="v_lds_addr = wave_id * 256",
        ))
        mod.add(ri.VMovB32(dst=vgpr(v_tmp), src=sgpr(s_iter), comment="v_tmp = s_iter"))
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(v_tmp),
            shiftHex=hex(2),
            src=vgpr(v_tmp),
            comment="v_tmp = s_iter * 4",
        ))
        mod.add(ri.VAddU32(
            vgpr(v_lds_addr), vgpr(v_lds_addr), vgpr(v_tmp),
            comment="v_lds_addr = wave_id*256 + s_iter*4",
        ))

        # Narrow EXEC: only lanes where row < M write.
        mod.add(ri.SAndSaveExecB64(
            dst=sgpr(s_tmp, 2),
            src=sgpr(s_row_mask, 2),
            comment="apply row<M mask, save old EXEC",
        ))
        mod.add(ri.DSStoreB32(
            dstAddr=vgpr(v_lds_addr),
            src=vgpr(v_acc),
            comment="LDS[wave_id*256 + s_iter*4] = rstd",
        ))
        mod.add(ri.SWaitCnt(dscnt=0, comment="wait LDS write"))
        mod.add(ri.SMovB64(dst=EXEC(), src=sgpr(s_tmp, 2), comment="restore EXEC"))

        # ------------------------------------------------------------------
        # Step 8: advance outer loop; branch back if s_iter < 64.
        # ------------------------------------------------------------------
        mod.add(ri.SAddU32(dst=sgpr(s_iter), src0=sgpr(s_iter), src1=1, comment="++s_iter"))
        mod.add(ri.SCmpLtI32(src0=sgpr(s_iter), src1=64, comment="s_iter < 64?"))
        mod.add(ri.SCBranchSCC1(loop_phase1, comment="loop phase 1"))

        # ------------------------------------------------------------------
        # Step 9: barrier.
        # ------------------------------------------------------------------
        mod.add(ri.SBarrier(comment="all rstd values visible in LDS"))

        # ------------------------------------------------------------------
        # Step 10: Phase 2 — read rstd, scale C columns.
        # Each thread handles one fixed row: global_row = s_row_base + Serial.
        # ------------------------------------------------------------------
        # LDS read: rstd = LDS[Serial * 4].
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(v_lds_addr),
            shiftHex=hex(2),
            src=vgpr(0),
            comment="LDS addr = Serial * 4",
        ))
        mod.add(ri.DSLoadB32(
            dst=vgpr(v_rstd),
            src=vgpr(v_lds_addr),
            comment="v_rstd = LDS[Serial*4]",
        ))
        mod.add(ri.SWaitCnt(dscnt=0, comment="wait rstd load"))

        # Global row for phase 2.
        mod.add(ri.VAddU32(
            vgpr(v_row_iter), vgpr(0), sgpr(s_row_base),
            comment="global_row = Serial + s_row_base",
        ))

        # Narrow EXEC to threads where row < M.
        mod.add(ri.SMovB64(dst=sgpr(s_exec_save, 2), src=EXEC(), comment="save full EXEC"))
        mod.add(ri.VCmpGtU32(
            dst=sgpr(s_row_mask, 2),
            src0=sgpr(s_M),
            src1=vgpr(v_row_iter),
            comment="row < M mask",
        ))
        mod.add(ri.SAndB64(
            dst=EXEC(),
            src0=sgpr(s_exec_save, 2),
            src1=sgpr(s_row_mask, 2),
            comment="narrow EXEC to row-valid threads",
        ))

        # Row byte offset within column: global_row * 2 bytes.
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(v_off),
            shiftHex=hex(1),
            src=vgpr(v_row_iter),
            comment="row_byte_off = global_row * 2",
        ))

        # col_stride_bytes = M * 2 (bytes per column in col-major layout).
        mod.add(ri.SLShiftLeftB32(
            dst=sgpr(s_iter),
            shiftHex=hex(1),
            src=sgpr(s_M),
            comment="s_iter reused: col_stride = M * 2 bytes",
        ))

        # Column loop: s_col_start .. s_col_end - 1.
        # Use s_nd as the loop variable (it was already consumed above).
        mod.add(ri.SMovB32(
            dst=sgpr(s_nd),
            src=sgpr(s_col_start),
            comment="s_nd reused as col = col_start",
        ))

        col_loop_done = ".Lcolv2_col_done"
        mod.add(ri.SCmpLtI32(
            src0=sgpr(s_nd),
            src1=sgpr(s_col_end),
            comment="col < col_end?",
        ))
        mod.add(ri.SCBranchSCC0(col_loop_done, comment="skip if no columns"))

        col_loop = ".Lcolv2_col"
        mod.add(TextBlock(f"{col_loop}:\n"))

        # Byte address of C[row, col] = col * M * 2 + row * 2.
        mod.add(ri.SMulI32(
            dst=sgpr(s_inv_d),
            src0=sgpr(s_nd),
            src1=sgpr(s_iter),
            comment="col_base_bytes = col * M * 2",
        ))
        mod.add(ri.VAddU32(
            vgpr(v_tmp), sgpr(s_inv_d), vgpr(v_off),
            comment="addr = col_base + row_byte_off",
        ))

        mod.add(ri.GlobalLoadD16B16(
            dst=vgpr(v_bfly),
            vaddr=vgpr(v_tmp),
            saddr=sgpr(srd_c, 2),
            comment="load bf16 from C col-major (16-bit)",
        ))
        mod.add(ri.SWaitCnt(vlcnt=0, comment="wait load"))

        # bf16 to f32 (shift left 16), multiply by rstd, back to bf16.
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(v_bfly),
            shiftHex=hex(16),
            src=vgpr(v_bfly),
            comment="bf16 bits -> f32 bits",
        ))
        mod.add(ri.VMulF32(
            dst=vgpr(v_bfly),
            src0=vgpr(v_bfly),
            src1=vgpr(v_rstd),
            comment="val *= rstd",
        ))
        mod.add(ri.VLShiftRightB32(
            dst=vgpr(v_bfly),
            shiftHex=hex(16),
            src=vgpr(v_bfly),
            comment="f32 bits -> bf16 bits (truncate)",
        ))

        mod.add(ri.GlobalStoreB16(
            vaddr=vgpr(v_tmp),
            src=vgpr(v_bfly),
            saddr=sgpr(srd_c, 2),
            comment="store bf16 back to C",
        ))

        mod.add(ri.SAddU32(dst=sgpr(s_nd), src0=sgpr(s_nd), src1=1, comment="++col"))
        mod.add(ri.SCmpLtI32(
            src0=sgpr(s_nd),
            src1=sgpr(s_col_end),
            comment="col < col_end?",
        ))
        mod.add(ri.SCBranchSCC1(col_loop, comment="loop columns"))

        mod.add(TextBlock(f"{col_loop_done}:\n"))

        # Restore EXEC.
        mod.add(ri.SMovB64(dst=EXEC(), src=sgpr(s_exec_save, 2), comment="restore EXEC"))

        mod.add(ri.SEndpgm())

    vgpr_count = vgpr_pool.size() - vgpr_pool.availableBlockAtEnd()
    sgpr_count = sgpr_pool.size() - sgpr_pool.availableBlockAtEnd()
    return mod, vgpr_count, sgpr_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

KERNEL_NAME = _KERNEL_NAME


_MAX_M = 32767  # col * M * 2 is computed via SMulI32 (signed 32-bit); safe up to M=32767


def build_colv2(chip: str, M: int = 0) -> Tuple[str, str]:
    """Generate colv2 RMSNorm kernel assembly for the given chip.

    M is optional and used only for the size guard below.
    Returns (asm_str, kernel_name).
    Raises ValueError if M exceeds the 32-bit column-address limit.
    """
    if M > _MAX_M:
        raise ValueError(
            f"colv2 column address uses SMulI32; M={M} > {_MAX_M} would overflow. "
            "Use a kernel with 64-bit address arithmetic for larger matrices."
        )
    isa = _ensure_isa(chip)
    gfx = chip.split(":")[0]
    xnack = ":xnack+" in chip

    kernel_body, vgpr_count, sgpr_count = _build_kernel_body(isa)
    func_name = _KERNEL_NAME

    args = [
        KernelArgument(size=8, offset=0,  value_kind="global_buffer", address_space="global"),
        KernelArgument(size=8, offset=8,  value_kind="global_buffer", address_space="global"),
        KernelArgument(size=4, offset=16, value_kind="by_value"),
        KernelArgument(size=4, offset=20, value_kind="by_value"),
        KernelArgument(size=4, offset=24, value_kind="by_value"),
        KernelArgument(size=4, offset=28, value_kind="by_value"),
        KernelArgument(size=4, offset=32, value_kind="by_value"),
    ]
    meta = KernelMeta(
        name=func_name,
        num_vgpr=vgpr_count,
        num_sgpr=sgpr_count,
        num_agpr=0,
        num_lds_bytes=1024,
        wavefront_size=64,
        max_workgroup_size=256,
        args_alignment=8,
        args=args,
    )

    k_str = "\n".join([
        _kernel_header(func_name, gfx, xnack),
        str(kernel_body),
        _kernel_rodata(
            func_name,
            gfx_arch=isa,
            vgpr_count=vgpr_count,
            sgpr_count=sgpr_count,
            lds=1024,
            use_wg_id_y=True,
        ),
        _meta_str((meta,)),
    ])
    return k_str, func_name
