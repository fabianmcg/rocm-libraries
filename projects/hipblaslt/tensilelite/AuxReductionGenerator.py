# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K2 auxiliary reduction kernel generator for the GEMM+RMSNorm pipeline.

K2 is a standalone rocisa-generated kernel: one wave (64 lanes) per row reads
N_tiles_N fp32 partials from partialBuf (2D, [M_padded, N_tiles_N]), reduces
them with a 6-stage ds_bpermute butterfly, computes:
  rstd = rsqrt(total / N_hidden + eps)
and writes rstdBuf[wave_row].

Kernarg layout (offsets fixed — tests depend on them):
  offset  0: partialBuf ptr (fp32, 8B)   value_kind=global_buffer
  offset  8: rstdBuf    ptr (fp32, 8B)   value_kind=global_buffer
  offset 16: M          (u32, 4B)        value_kind=by_value
  offset 20: N_tiles_N  (u32, 4B)        value_kind=by_value  (partials per row)
  offset 24: N_hidden   (u32, 4B)        value_kind=by_value  (divisor; runtime)
  offset 28: eps        (f32, 4B)        value_kind=by_value
  kernarg_segment_size = 32

Thread geometry:
  WAVES_PER_WG = 4, WAVE_SIZE = 64 -> THREADS_PER_WG = 256, ROWS_PER_WG = 4
  grid = (ceil(M / 4), 1, 1), block = (256, 1, 1)
  wave_row = wg_id * 4 + (Serial >> 6)
  lane_id  = Serial & 63

LDS = 0: ds_bpermute uses no allocated LDS (crossbar only; no group memory consumed).
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

_WAVES_PER_WG = 4
_WAVE_SIZE    = 64
_ROWS_PER_WG  = _WAVES_PER_WG


# ---------------------------------------------------------------------------
# Self-contained helpers
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


def _kernel_rodata(name: str, gfx_arch: Tuple[int, int, int],
                   vgpr_count: int, sgpr_count: int, lds: int) -> str:
    header = ".rodata\n"
    header += ".p2align 6\n"
    header += f".amdhsa_kernel {name}\n"
    header += ".amdhsa_user_sgpr_kernarg_segment_ptr 1\n"
    header += ".amdhsa_system_sgpr_workgroup_id_x 1\n"
    header += ".amdhsa_system_vgpr_workitem_id 0\n"
    # gfx950 = (9, 5, 0); also gfx90a = (9, 0, 10)
    # Round vgpr_count up to a multiple of 4 for gfx9 AGPR-capable ISAs.
    # accum_offset must equal next_free_vgpr and be a multiple of 4.
    vgpr_aligned = ((vgpr_count + 3) // 4) * 4
    if gfx_arch == (9, 0, 10) or (gfx_arch > (9, 4) and gfx_arch < (10, 0, 0)):
        header += f".amdhsa_accum_offset {vgpr_aligned}\n"
    header += f".amdhsa_next_free_vgpr {vgpr_aligned}\n"
    header += f".amdhsa_next_free_sgpr {sgpr_count}\n"
    header += f".amdhsa_group_segment_fixed_size {lds}\n"
    header += ".amdhsa_private_segment_fixed_size 0\n"
    header += ".amdhsa_float_denorm_mode_32 3\n"
    if _global_ti.getArchCaps().get("HasWave32", False):
        header += ".amdhsa_wavefront_size32 1\n"
    header += ".end_amdhsa_kernel\n"
    return header


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
        rem = total % self.args_alignment
        if rem:
            total += self.args_alignment - rem
        return total

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
        {"amdhsa.version": [1, 1],
         "amdhsa.kernels": [k.to_dict() for k in kernels]}
    )
    end = ".end_amdgpu_metadata"
    return "\n".join([beg, content, end])


# ---------------------------------------------------------------------------
# ISA initialisation (mirrors SoftmaxGenerator __main__ lines 742-744)
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
# K2 kernel body
# ---------------------------------------------------------------------------

def _build_kernel_body(isa: Tuple[int, int, int]) -> Tuple[Module, int, int]:
    """Generate the K2 kernel body module.

    Register layout (see module docstring):
      s0:s1  kernarg ptr     (ABI)
      s2     wg_id_x         (ABI, or from ttmp9 if WorkGroupIdFromTTM)
      s3:s6  partialBuf SRD  (4 SGPRs, aligned 4)
      s7:s10 rstdBuf SRD     (4 SGPRs, aligned 4)
      s11    M               (u32)
      s12    N_tiles_N       (u32)
      s13    N_hidden        (u32)
      s14    eps             (f32)
      s15    n_iter          (loop trip count)
      s16:17 exec_save       (2 SGPRs, aligned 2)
      s18:19 mask_scratch    (2 SGPRs, aligned 2)

      v0     Serial / workitem id (ABI)
      v1     lane_id   = Serial & 63
      v2     wave_row  = wg_id*4 + (Serial>>6)
      v3     acc       (running partial sum / total / rstd)
      v4     col       (strided column index)
      v5     row_elem_base = wave_row * N_tiles_N
      v6     byte_off  (load/store offset)
      v7     tmp       (general purpose temp)
      v8     perm      (butterfly partner byte addr)
      v9     bfly_tmp  (butterfly fetched value)
      v10    n_f32 / inv_n
      v11    eps_v

    Returns (module, vgpr_count, sgpr_count).
    """
    archCaps  = _global_ti.getArchCaps()
    srdUpper  = SrdUpperValue(isa).getValue()
    wg_id_reg = 2  # s2 = workgroup-id-x (ABI given kernarg ptr in s0:s1)

    # SGPR pool: s0:s2 are reserved by ABI.
    sgpr_pool = RegisterPool(3, RegisterType.Sgpr, True)
    sgpr_pool.addRange(3, 40)

    srd_in   = sgpr_pool.checkOutAligned(4, 4)   # partialBuf SRD
    srd_out  = sgpr_pool.checkOutAligned(4, 4)   # rstdBuf SRD
    s_M      = sgpr_pool.checkOut(1)              # M
    s_Ntiles = sgpr_pool.checkOut(1)              # N_tiles_N
    s_Nhid   = sgpr_pool.checkOut(1)              # N_hidden
    s_eps    = sgpr_pool.checkOut(1)              # eps (f32)
    s_niter  = sgpr_pool.checkOut(1)              # loop trip count
    s_exec   = sgpr_pool.checkOutAligned(2, 2)   # EXEC save slot
    s_mask   = sgpr_pool.checkOutAligned(2, 2)   # scratch mask

    # VGPR pool: v0 (Serial) is reserved by ABI.
    vgpr_pool = RegisterPool(1, RegisterType.Vgpr, True)
    vgpr_pool.addRange(1, 24)

    v_lane  = vgpr_pool.checkOut(1)  # lane_id
    v_wrow  = vgpr_pool.checkOut(1)  # wave_row
    v_acc   = vgpr_pool.checkOut(1)  # accumulator
    v_col   = vgpr_pool.checkOut(1)  # strided column index
    v_rbase = vgpr_pool.checkOut(1)  # row_elem_base = wave_row * N_tiles_N
    v_off   = vgpr_pool.checkOut(1)  # byte offset
    v_tmp   = vgpr_pool.checkOut(1)  # general temp
    v_perm  = vgpr_pool.checkOut(1)  # butterfly permute byte addr
    v_bfly  = vgpr_pool.checkOut(1)  # butterfly fetched value
    v_nf32  = vgpr_pool.checkOut(1)  # float(N_hidden) / inv_n
    v_eps   = vgpr_pool.checkOut(1)  # eps as vgpr

    func_name = "aux_reduction"
    mod = Module(func_name)

    with _asm_func(func_name, mod):

        # ------------------------------------------------------------------
        # Step 1: load kernargs.
        # ------------------------------------------------------------------
        mod.add(ri.SLoadB64(sgpr(srd_in, 2),  sgpr(0, 2), 0,  comment="partialBuf ptr"))
        mod.add(ri.SLoadB64(sgpr(srd_out, 2), sgpr(0, 2), 8,  comment="rstdBuf ptr"))
        mod.add(ri.SLoadB32(sgpr(s_M),      sgpr(0, 2), 16, comment="M"))
        mod.add(ri.SLoadB32(sgpr(s_Ntiles), sgpr(0, 2), 20, comment="N_tiles_N"))
        mod.add(ri.SLoadB32(sgpr(s_Nhid),   sgpr(0, 2), 24, comment="N_hidden"))
        mod.add(ri.SLoadB32(sgpr(s_eps),    sgpr(0, 2), 28, comment="eps"))
        mod.add(ri.SWaitCnt(kmcnt=0, comment="wait for all kernarg loads"))

        # ------------------------------------------------------------------
        # Step 2: build SRDs (use unbounded byte limit; upper word from isa).
        # ------------------------------------------------------------------
        mod.add(ri.SMovB32(sgpr(srd_in + 2),  hex(0xFFFFFFFF), comment="partialBuf limit"))
        mod.add(ri.SMovB32(sgpr(srd_in + 3),  hex(srdUpper),   comment="partialBuf SRD flags"))
        mod.add(ri.SMovB32(sgpr(srd_out + 2), hex(0xFFFFFFFF), comment="rstdBuf limit"))
        mod.add(ri.SMovB32(sgpr(srd_out + 3), hex(srdUpper),   comment="rstdBuf SRD flags"))

        if archCaps.get("WorkGroupIdFromTTM", False):
            mod.add(ri.SMovB32(dst=sgpr(wg_id_reg), src="ttmp9",
                               comment="wg_id from ttmp9 (WorkGroupIdFromTTM)"))

        # ------------------------------------------------------------------
        # Step 3: derive per-thread indices.
        # Serial = v0; lane_id = v0 & 63; wave_row = wg_id*4 + (v0>>6).
        # ------------------------------------------------------------------
        mod.add(ri.VAndB32(dst=vgpr(v_lane), src0=vgpr(0), src1=63,
                           comment="lane_id = Serial & 63"))
        mod.add(ri.VMovB32(dst=vgpr(v_wrow), src=sgpr(wg_id_reg),
                           comment="wave_row = wg_id (start)"))
        mod.add(ri.VMulLOU32(dst=vgpr(v_wrow), src0=vgpr(v_wrow), src1=_WAVES_PER_WG,
                             comment="wave_row = wg_id * 4"))
        mod.add(ri.VLShiftRightB32(dst=vgpr(v_tmp), shiftHex=hex(6), src=vgpr(0),
                                   comment="wave_id_in_wg = Serial >> 6"))
        mod.add(ri.VAddU32(vgpr(v_wrow), vgpr(v_wrow), vgpr(v_tmp),
                           comment="wave_row = wg_id*4 + wave_id"))

        # ------------------------------------------------------------------
        # Step 4: row_elem_base = wave_row * N_tiles_N.
        # Move N_tiles_N to a VGPR to avoid SGPR-src0 restrictions.
        # ------------------------------------------------------------------
        mod.add(ri.VMovB32(dst=vgpr(v_tmp), src=sgpr(s_Ntiles),
                           comment="v_tmp = N_tiles_N"))
        mod.add(ri.VMulLOU32(dst=vgpr(v_rbase), src0=vgpr(v_wrow), src1=vgpr(v_tmp),
                             comment="row_elem_base = wave_row * N_tiles_N"))

        # ------------------------------------------------------------------
        # Step 5: n_iter = ceil(N_tiles_N / 64) = (N_tiles_N + 63) >> 6.
        # ------------------------------------------------------------------
        mod.add(ri.SAddU32(dst=sgpr(s_niter), src0=sgpr(s_Ntiles), src1=63,
                           comment="n_iter_tmp = N_tiles_N + 63"))
        mod.add(ri.SLShiftRightB32(dst=sgpr(s_niter), shiftHex=hex(6), src=sgpr(s_niter),
                                   comment="n_iter = ceil(N_tiles_N / 64)"))

        # ------------------------------------------------------------------
        # Step 6: Strided load loop.
        # Each iteration: predicate col < N_tiles_N → load → add to acc → col+=64.
        # Loop exit: SSubU32 n_iter; SCmpEQU32 n_iter, 0; SCBranchSCC0 → loop.
        # ------------------------------------------------------------------
        mod.add(ri.VMovB32(dst=vgpr(v_acc), src=0, comment="acc = 0.0f"))
        mod.add(ri.VMovB32(dst=vgpr(v_col), src=vgpr(v_lane),
                           comment="col = lane_id (first strided index)"))

        loop_label = ".Laux_red_loop"
        mod.add(TextBlock(f"{loop_label}:\n"))

        # Predicate: only lanes where col < N_tiles_N participate.
        mod.add(ri.SMovB64(dst=sgpr(s_exec, 2), src=EXEC(),
                           comment="save EXEC"))
        mod.add(ri.VCmpLtU32(dst=sgpr(s_mask, 2), src0=vgpr(v_col), src1=sgpr(s_Ntiles),
                              comment="mask = col < N_tiles_N"))
        mod.add(ri.SAndB64(dst=EXEC(), src0=sgpr(s_exec, 2), src1=sgpr(s_mask, 2),
                           comment="narrow EXEC to active lanes"))

        # byte_off = (row_elem_base + col) * 4.
        mod.add(ri.VAddU32(vgpr(v_off), vgpr(v_rbase), vgpr(v_col),
                           comment="flat_idx = row_elem_base + col"))
        mod.add(ri.VLShiftLeftB32(dst=vgpr(v_off), shiftHex=hex(2), src=vgpr(v_off),
                                  comment="byte_off = flat_idx * 4"))
        mod.add(ri.BufferLoadB32(vgpr(v_tmp), vgpr(v_off), sgpr(srd_in, 4), 0,
                                 mubuf=MUBUFModifiers(offen=True),
                                 comment="load partialBuf[wave_row, col]"))
        mod.add(ri.SWaitCnt(vlcnt=0, comment="wait load"))
        mod.add(ri.VAddF32(dst=vgpr(v_acc), src0=vgpr(v_acc), src1=vgpr(v_tmp),
                           comment="acc += partial (masked lanes contribute 0)"))

        # Restore EXEC before advancing col.
        mod.add(ri.SMovB64(dst=EXEC(), src=sgpr(s_exec, 2),
                           comment="restore EXEC"))

        # col += 64; decrement counter; loop if n_iter > 0.
        mod.add(ri.VAddU32(vgpr(v_col), vgpr(v_col), 64, comment="col += 64"))
        mod.add(ri.SSubU32(dst=sgpr(s_niter), src0=sgpr(s_niter), src1=1,
                           comment="--n_iter"))
        mod.add(ri.SCmpEQU32(src0=sgpr(s_niter), src1=0,
                             comment="n_iter == 0?"))
        mod.add(ri.SCBranchSCC0(loop_label, comment="loop while n_iter > 0"))

        # ------------------------------------------------------------------
        # Step 7: 6-stage butterfly (strides [32, 16, 8, 4, 2, 1]).
        # After the loop every active lane holds its strided partial in acc.
        # The butterfly accumulates from all 64 lanes within the wave.
        # ds_bpermute uses no allocated LDS (crossbar only).
        # ------------------------------------------------------------------
        for stride in (32, 16, 8, 4, 2, 1):
            mod.add(ri.VXorB32(dst=vgpr(v_perm), src0=vgpr(v_lane), src1=stride,
                               comment=f"partner = lane_id ^ {stride}"))
            mod.add(ri.VLShiftLeftB32(dst=vgpr(v_perm), shiftHex=hex(2), src=vgpr(v_perm),
                                      comment="partner_byte = partner * 4"))
            mod.add(ri.DSBPermuteB32(dst=vgpr(v_bfly), src0=vgpr(v_perm), src1=vgpr(v_acc),
                                     comment=f"bfly = partner acc (stride={stride})"))
            mod.add(ri.SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
            mod.add(ri.VAddF32(dst=vgpr(v_acc), src0=vgpr(v_acc), src1=vgpr(v_bfly),
                               comment="acc += partner acc"))

        # ------------------------------------------------------------------
        # Step 8: rsqrt: rstd = rsqrt(acc / N_hidden + eps).
        # All lanes have the same total after the butterfly.
        # ------------------------------------------------------------------
        mod.add(ri.VCvtU32toF32(dst=vgpr(v_nf32), src=sgpr(s_Nhid),
                                 comment="n_f32 = (f32)N_hidden"))
        mod.add(ri.VRcpF32(dst=vgpr(v_nf32), src=vgpr(v_nf32),
                           comment="inv_n = 1 / N_hidden"))
        if archCaps.get("TransOpWait", False):
            mod.add(ri.SNop(waitState=0, comment="TransOpWait after VRcpF32"))
        mod.add(ri.VMulF32(dst=vgpr(v_acc), src0=vgpr(v_acc), src1=vgpr(v_nf32),
                           comment="total = acc / N_hidden"))
        mod.add(ri.VMovB32(dst=vgpr(v_eps), src=sgpr(s_eps), comment="eps_v = eps"))
        mod.add(ri.VAddF32(dst=vgpr(v_acc), src0=vgpr(v_acc), src1=vgpr(v_eps),
                           comment="total += eps"))
        mod.add(ri.VRsqF32(dst=vgpr(v_acc), src=vgpr(v_acc),
                           comment="rstd = rsqrt(total)"))
        if archCaps.get("TransOpWait", False):
            mod.add(ri.SNop(waitState=0, comment="TransOpWait after VRsqF32"))

        # ------------------------------------------------------------------
        # Step 9: predicated write of rstdBuf[wave_row].
        # Only lane_id == 0 AND wave_row < M writes.
        # ------------------------------------------------------------------
        mod.add(ri.VCmpEQU32(dst=sgpr(s_exec, 2), src0=vgpr(v_lane), src1=0,
                              comment="mask_a: lane_id == 0"))
        mod.add(ri.VCmpLtU32(dst=sgpr(s_mask, 2), src0=vgpr(v_wrow), src1=sgpr(s_M),
                              comment="mask_b: wave_row < M"))
        mod.add(ri.SAndB64(dst=sgpr(s_exec, 2), src0=sgpr(s_exec, 2), src1=sgpr(s_mask, 2),
                           comment="write_mask = lane_id==0 AND wave_row<M"))
        mod.add(ri.SAndSaveExecB64(dst=sgpr(s_mask, 2), src=sgpr(s_exec, 2),
                                   comment="apply write mask, save old EXEC in s_mask"))

        mod.add(ri.VLShiftLeftB32(dst=vgpr(v_off), shiftHex=hex(2), src=vgpr(v_wrow),
                                  comment="byte_off = wave_row * 4"))
        mod.add(ri.BufferStoreB32(src=vgpr(v_acc), vaddr=vgpr(v_off),
                                   saddr=sgpr(srd_out, 4), soffset=0,
                                   mubuf=MUBUFModifiers(offen=True),
                                   comment="rstdBuf[wave_row] = rstd"))
        mod.add(ri.SWaitCnt(vlcnt=0, comment="wait store"))

        mod.add(ri.SMovB64(dst=EXEC(), src=sgpr(s_mask, 2),
                           comment="restore EXEC"))

        mod.add(ri.SEndpgm())

    vgpr_count = vgpr_pool.size()
    sgpr_count = sgpr_pool.size()
    return mod, vgpr_count, sgpr_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_aux_reduction(chip: str) -> Tuple[str, str]:
    """Generate K2 auxiliary-reduction kernel assembly for the given chip.

    Returns (asm_str, kernel_name).
    """
    isa = _ensure_isa(chip)
    gfx = chip.split(":")[0]
    xnack = ":xnack+" in chip

    kernel_body, vgpr_count, sgpr_count = _build_kernel_body(isa)
    func_name = "aux_reduction"

    args = [
        KernelArgument(size=8, offset=0,  value_kind="global_buffer", address_space="global"),
        KernelArgument(size=8, offset=8,  value_kind="global_buffer", address_space="global"),
        KernelArgument(size=4, offset=16, value_kind="by_value"),
        KernelArgument(size=4, offset=20, value_kind="by_value"),
        KernelArgument(size=4, offset=24, value_kind="by_value"),
        KernelArgument(size=4, offset=28, value_kind="by_value"),
    ]
    meta = KernelMeta(
        name=func_name,
        num_vgpr=vgpr_count,
        num_sgpr=sgpr_count,
        num_agpr=0,
        num_lds_bytes=0,
        wavefront_size=64,
        max_workgroup_size=256,
        args_alignment=8,
        args=args,
    )

    k_str = "\n".join([
        _kernel_header(func_name, gfx, xnack),
        str(kernel_body),
        _kernel_rodata(func_name, gfx_arch=isa, vgpr_count=vgpr_count,
                       sgpr_count=sgpr_count, lds=0),
        _meta_str((meta,)),
    ])
    return k_str, func_name
