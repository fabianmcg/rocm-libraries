# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RMSNorm col-major scale kernel generator (partial_rms_epilogue).

partial_rms_epilogue applies RMSNorm in-place to a bf16 col-major matrix C using
precomputed partial sum-of-squares from partialBuf.  It is a 2D-grid kernel:
  wgIdX (s2) selects a 256-row tile.
  wgIdY (s3) selects a 256-column tile.

Algorithm — Phase 1 (reduction, one wave = 64 lanes per row subgroup):
  Each block of 256 threads is organised as 4 waves x 64 lanes.  The outer
  loop (sIter = 0..63) assigns one global row per iteration.  All 64 lanes
  of the wave collectively load one element each from partialBuf for that row.
  A hybrid reduction (4× DPP row_shr + 2× ds_bpermute) sums the 64 partial values.  All lanes
  with row < M write rsqrt(invD * total + eps) to LDS at address
  (waveId * 256 + sIter * 4).

Algorithm — Phase 2 (scale, all 256 threads):
  After the barrier each thread reads rstd = LDS[Serial * 4] for its own row.
  The column loop iterates over wgIdY * 256 .. min((wgIdY+1)*256, N)-1,
  loading each bf16, multiplying by rstd (shift-left-16 trick), and storing
  back.

Kernarg layout (offsets fixed — host code depends on them):
  offset  0 (8B): ptrC  — bf16, col-major M x N   value_kind=global_buffer
  offset  8 (8B): ptrD  — f32,  row-major M x nD  value_kind=global_buffer
  offset 16 (4B): M      (u32)                       value_kind=by_value
  offset 20 (4B): N      (u32)                       value_kind=by_value
  offset 24 (4B): nD     (u32, columns of partialBuf) value_kind=by_value
  offset 28 (4B): invD   (f32, 1/N_hidden)            value_kind=by_value
  offset 32 (4B): eps    (f32)                         value_kind=by_value
  kernarg_segment_size = 40 (aligned to 8 bytes)

LDS = 1024 bytes: 4 waves x 64 iterations x 4 bytes = 1024B.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml

from rocisa.code import Module, TextBlock, SrdUpperValue
from rocisa.container import DPPModifiers, EXEC, MUBUFModifiers, vgpr, sgpr
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
import rocisa.instruction as ri

from Tensile.Common.Utilities import _global_ti
from Tensile.Common.Architectures import gfxToIsa

_KERNEL_NAME = "partial_rms_epilogue"


# ---------------------------------------------------------------------------
# Self-contained helpers (verbatim from AuxReductionGenerator.py)
# ---------------------------------------------------------------------------


def _kernel_header(name: str, gfxArch: str, xnack: bool = False) -> str:
    targetId = f"{gfxArch}:xnack+" if xnack else gfxArch
    return (
        f'.amdgcn_target "amdgcn-amd-amdhsa--{targetId}"\n'
        f".text\n"
        f".global {name}\n"
        f".p2align 8\n"
        f".type {name},@function\n"
    )


def _kernel_rodata(
    name: str,
    gfxArch: Tuple[int, int, int],
    vgprCount: int,
    sgprCount: int,
    lds: int,
    useWgIdY: bool = False,
) -> str:
    # Round vgprCount up to a multiple of 4 for gfx9 AGPR-capable ISAs.
    vgprAligned = ((vgprCount + 3) // 4) * 4
    lines = [
        ".rodata",
        ".p2align 6",
        f".amdhsa_kernel {name}",
        ".amdhsa_user_sgpr_kernarg_segment_ptr 1",
        ".amdhsa_system_sgpr_workgroup_id_x 1",
    ]
    if useWgIdY:
        lines.append(".amdhsa_system_sgpr_workgroup_id_y 1")
    lines.append(".amdhsa_system_vgpr_workitem_id 0")
    # gfx950 = (9, 5, 0); also gfx90a = (9, 0, 10).
    if gfxArch == (9, 0, 10) or ((9, 4) < gfxArch < (10, 0, 0)):
        lines.append(f".amdhsa_accum_offset {vgprAligned}")
    lines += [
        f".amdhsa_next_free_vgpr {vgprAligned}",
        f".amdhsa_next_free_sgpr {sgprCount}",
        f".amdhsa_group_segment_fixed_size {lds}",
        ".amdhsa_private_segment_fixed_size 0",
        ".amdhsa_float_denorm_mode_32 3",
    ]
    if _global_ti.getArchCaps().get("HasWave32", False):
        lines.append(".amdhsa_wavefront_size32 1")
    lines.append(".end_amdhsa_kernel")
    return "\n".join(lines) + "\n"


@contextmanager
def _asm_func(funcName: str, module: Module):
    module.add(TextBlock(f"{funcName}:\n"))
    try:
        yield
    finally:
        endLabel = f".L{funcName}_end"
        module.add(TextBlock(f"{endLabel}:\n"))
        module.add(TextBlock(f".size {funcName}, {endLabel} - {funcName}\n"))


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
    waveSize = 32 if isa[0] in (11, 12) else 64
    _global_ti.setKernel(isa, waveSize)
    return isa


# ---------------------------------------------------------------------------
# partial_rms_epilogue kernel body
# ---------------------------------------------------------------------------


def _build_kernel_body(isa: Tuple[int, int, int]) -> Tuple[Module, int, int]:
    """Generate the partial_rms_epilogue kernel body module.

    Returns (module, vgprCount, sgprCount).
    """
    archCaps = _global_ti.getArchCaps()
    srdUpper = SrdUpperValue(isa).getValue()

    # ABI: s0:s1 = kernarg ptr, s2 = wgIdX, s3 = wgIdY, v0 = Serial.
    wgIdXReg = 2
    wgIdYReg = 3

    # SGPR pool: s0..s3 reserved by ABI.
    sgprPool = RegisterPool(4, RegisterType.Sgpr, True)
    sgprPool.addRange(4, 52)

    srdC = sgprPool.checkOutAligned(4, 4)         # C SRD
    srdD = sgprPool.checkOutAligned(4, 4)         # partialBuf SRD
    sM = sgprPool.checkOut(1)                     # M
    sN = sgprPool.checkOut(1)                     # N
    sNd = sgprPool.checkOut(1)                    # nD (columns of partialBuf)
    sInvD = sgprPool.checkOut(1)                  # invD (f32)
    sEps = sgprPool.checkOut(1)                   # eps (f32)
    sIter = sgprPool.checkOut(1)                  # outer loop counter (0..63)
    sRowBase = sgprPool.checkOut(1)               # wgIdX * 256
    sColStart = sgprPool.checkOut(1)              # wgIdY * 256
    sColEnd = sgprPool.checkOut(1)                # min(colStart+256, N)
    sExecSave = sgprPool.checkOutAligned(2, 2)    # phase-1 outer exec save
    sRowMask = sgprPool.checkOutAligned(2, 2)     # row < M mask (used as OOB guard)
    sTmp = sgprPool.checkOutAligned(2, 2)         # scratch

    # VGPR pool: v0 (Serial) reserved by ABI.
    vgprPool = RegisterPool(1, RegisterType.Vgpr, True)
    vgprPool.addRange(1, 24)

    vLane = vgprPool.checkOut(1)       # laneId = Serial & 63
    vWaveId = vgprPool.checkOut(1)     # waveId in block = Serial >> 6
    vAcc = vgprPool.checkOut(1)        # f32 accumulator (phase 1)
    vOff = vgprPool.checkOut(1)        # byte-offset scratch
    vTmp = vgprPool.checkOut(1)        # general temp
    vPerm = vgprPool.checkOut(1)       # butterfly permute byte addr
    vBfly = vgprPool.checkOut(1)       # butterfly fetched value
    vRstd = vgprPool.checkOut(1)       # per-thread rstd from LDS
    vLdsAddr = vgprPool.checkOut(1)    # LDS byte address
    vRowIter = vgprPool.checkOut(1)    # global row for current iteration
    vColD = vgprPool.checkOut(1)       # current col into partialBuf

    funcName = _KERNEL_NAME
    mod = Module(funcName)

    with _asm_func(funcName, mod):

        # ------------------------------------------------------------------
        # Step 1: load kernargs.
        # ------------------------------------------------------------------
        mod.add(ri.SLoadB64(sgpr(srdC, 2), sgpr(0, 2), 0, comment="ptrC (C bf16)"))
        mod.add(ri.SLoadB64(sgpr(srdD, 2), sgpr(0, 2), 8, comment="ptrD (partialBuf f32)"))
        mod.add(ri.SLoadB32(sgpr(sM), sgpr(0, 2), 16, comment="M"))
        mod.add(ri.SLoadB32(sgpr(sN), sgpr(0, 2), 20, comment="N"))
        mod.add(ri.SLoadB32(sgpr(sNd), sgpr(0, 2), 24, comment="nD"))
        mod.add(ri.SLoadB32(sgpr(sInvD), sgpr(0, 2), 28, comment="invD"))
        mod.add(ri.SLoadB32(sgpr(sEps), sgpr(0, 2), 32, comment="eps"))
        mod.add(ri.SWaitCnt(kmcnt=0, comment="wait kernarg loads"))

        # ------------------------------------------------------------------
        # Step 2: build SRDs (unbounded buffer limit; upper word from ISA).
        # ------------------------------------------------------------------
        mod.add(ri.SMovB32(sgpr(srdC + 2), hex(0xFFFFFFFF), comment="C limit"))
        mod.add(ri.SMovB32(sgpr(srdC + 3), hex(srdUpper), comment="C SRD flags"))
        mod.add(ri.SMovB32(sgpr(srdD + 2), hex(0xFFFFFFFF), comment="partialBuf limit"))
        mod.add(ri.SMovB32(sgpr(srdD + 3), hex(srdUpper), comment="partialBuf SRD flags"))

        if archCaps.get("WorkGroupIdFromTTM", False):
            mod.add(ri.SMovB32(
                dst=sgpr(wgIdXReg),
                src="ttmp9",
                comment="wgIdX from ttmp9 (WorkGroupIdFromTTM)",
            ))

        # ------------------------------------------------------------------
        # Step 3: derive scalar tile offsets and per-thread indices.
        # ------------------------------------------------------------------
        mod.add(ri.SLShiftLeftB32(
            dst=sgpr(sRowBase),
            shiftHex=hex(8),
            src=sgpr(wgIdXReg),
            comment="sRowBase = wgIdX * 256",
        ))
        mod.add(ri.SLShiftLeftB32(
            dst=sgpr(sColStart),
            shiftHex=hex(8),
            src=sgpr(wgIdYReg),
            comment="sColStart = wgIdY * 256",
        ))
        mod.add(ri.SAddU32(
            dst=sgpr(sColEnd),
            src0=sgpr(sColStart),
            src1=256,
            comment="sColEnd = colStart + 256 (clamp below)",
        ))
        mod.add(ri.SCmpLtU32(
            src0=sgpr(sColEnd),
            src1=sgpr(sN),
            comment="colEnd < N?",
        ))
        mod.add(ri.SCSelectB32(
            dst=sgpr(sColEnd),
            src0=sgpr(sColEnd),
            src1=sgpr(sN),
            comment="sColEnd = min(colStart+256, N)",
        ))

        mod.add(ri.VAndB32(
            dst=vgpr(vLane),
            src0=vgpr(0),
            src1=63,
            comment="vLane = Serial & 63",
        ))
        mod.add(ri.VLShiftRightB32(
            dst=vgpr(vWaveId),
            shiftHex=hex(6),
            src=vgpr(0),
            comment="vWaveId = Serial >> 6",
        ))

        # ------------------------------------------------------------------
        # Step 4: Phase 1 — outer loop (sIter = 0..63).
        # Each iteration processes one row per wave.  All lanes share the same
        # global row: vRowIter = sRowBase + vWaveId*64 + sIter.
        # Lane vLane selects the column into partialBuf.
        # ------------------------------------------------------------------
        mod.add(ri.SMovB32(dst=sgpr(sIter), src=0, comment="sIter = 0"))

        loopPhase1 = ".Lpartial_rms_epilogue_phase1"
        mod.add(TextBlock(f"{loopPhase1}:\n"))

        # Compute global row for this iteration.
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(vRowIter),
            shiftHex=hex(6),
            src=vgpr(vWaveId),
            comment="waveStart = vWaveId * 64",
        ))
        mod.add(ri.VAddU32(
            vgpr(vRowIter), vgpr(vRowIter), sgpr(sIter),
            comment="rowInBlock = waveStart + sIter",
        ))
        mod.add(ri.VAddU32(
            vgpr(vRowIter), vgpr(vRowIter), sgpr(sRowBase),
            comment="vRowIter = rowBase + rowInBlock",
        ))

        # Precompute row<M mask for the OOB trick inside the inner loop.
        mod.add(ri.VCmpGtU32(
            dst=sgpr(sRowMask, 2),
            src0=sgpr(sM),
            src1=vgpr(vRowIter),
            comment="sRowMask = row < M",
        ))

        # Save EXEC before inner loop (will be restored after butterfly).
        mod.add(ri.SMovB64(dst=sgpr(sExecSave, 2), src=EXEC(), comment="save EXEC"))

        # Init accumulator.
        mod.add(ri.VMovB32(dst=vgpr(vAcc), src=0, comment="acc = 0.0f"))

        # Init col: vColD = vLane (each lane starts at its lane index).
        mod.add(ri.VMovB32(
            dst=vgpr(vColD), src=vgpr(vLane), comment="vColD = laneId",
        ))

        # Check initial col < nD; if no lane has a valid col skip inner loop.
        mod.add(ri.VCmpGtU32(
            dst=sgpr(sTmp, 2),
            src0=sgpr(sNd),
            src1=vgpr(vColD),
            comment="initial col-valid mask (nD > vColD)",
        ))
        mod.add(ri.SCmpEQU64(
            src0=sgpr(sTmp, 2),
            src1=0,
            comment="all lanes OOB?",
        ))

        bflyLabel = ".Lpartial_rms_epilogue_bfly"
        mod.add(ri.SCBranchSCC1(bflyLabel, comment="skip inner loop if all cols OOB"))

        # Narrow EXEC to col-valid lanes.
        mod.add(ri.SAndB64(
            dst=EXEC(),
            src0=sgpr(sExecSave, 2),
            src1=sgpr(sTmp, 2),
            comment="EXEC = saved & col-valid",
        ))

        # ------------------------------------------------------------------
        # Inner loop: load partialBuf[row, col], accumulate, advance col.
        # EXEC is narrowed progressively (not restored inside the loop);
        # exit when no lane has col < nD.
        # ------------------------------------------------------------------
        innerLoop = ".Lpartial_rms_epilogue_inner"
        mod.add(TextBlock(f"{innerLoop}:\n"))

        # byteOff = (vRowIter * nD + vColD) * 4.
        mod.add(ri.VMovB32(dst=vgpr(vTmp), src=sgpr(sNd), comment="vTmp = nD"))
        mod.add(ri.VMulLOU32(
            dst=vgpr(vOff),
            src0=vgpr(vRowIter),
            src1=vgpr(vTmp),
            comment="row * nD",
        ))
        mod.add(ri.VAddU32(
            vgpr(vOff), vgpr(vOff), vgpr(vColD),
            comment="flatIdx = row*nD + col",
        ))
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(vOff),
            shiftHex=hex(2),
            src=vgpr(vOff),
            comment="byteOff = flatIdx * 4",
        ))

        # OOB trick: row-OOB lanes get offset -1 so buffer returns 0; zero
        # the loaded value afterward for those lanes.
        mod.add(ri.VCndMaskB32(
            vgpr(vOff), -1, vgpr(vOff), sgpr(sRowMask, 2),
            comment="vOff = -1 for row-OOB lanes",
        ))
        mod.add(ri.BufferLoadB32(
            vgpr(vTmp),
            vgpr(vOff),
            sgpr(srdD, 4),
            0,
            mubuf=MUBUFModifiers(offen=True),
            comment="load partialBuf[row, col]",
        ))
        mod.add(ri.SWaitCnt(vlcnt=0, comment="wait load"))
        mod.add(ri.VCndMaskB32(
            vgpr(vTmp), 0, vgpr(vTmp), sgpr(sRowMask, 2),
            comment="zero vTmp for row-OOB lanes",
        ))
        mod.add(ri.VAddF32(
            dst=vgpr(vAcc),
            src0=vgpr(vAcc),
            src1=vgpr(vTmp),
            comment="acc += partial",
        ))

        # Advance column and recompute col-valid mask.
        mod.add(ri.VAddU32(vgpr(vColD), vgpr(vColD), 64, comment="vColD += 64"))
        mod.add(ri.VCmpGtU32(
            dst=sgpr(sTmp, 2),
            src0=sgpr(sNd),
            src1=vgpr(vColD),
            comment="new col-valid: nD > vColD",
        ))

        # Narrow EXEC to savedExec & new col-valid.  Exit if no active lanes.
        mod.add(ri.SAndB64(
            dst=EXEC(),
            src0=sgpr(sExecSave, 2),
            src1=sgpr(sTmp, 2),
            comment="EXEC = saved & new col-valid",
        ))
        mod.add(ri.SCmpEQU64(
            src0=EXEC(),
            src1=0,
            comment="all lanes done?",
        ))
        mod.add(ri.SCBranchSCC0(innerLoop, comment="loop while active lanes remain"))

        # ------------------------------------------------------------------
        # Step 5: restore EXEC then hybrid reduction (4× DPP + 2× ds_bpermute).
        # All lanes now hold the full 64-lane sum in vAcc.
        # ------------------------------------------------------------------
        mod.add(TextBlock(f"{bflyLabel}:\n"))
        mod.add(ri.SMovB64(dst=EXEC(), src=sgpr(sExecSave, 2), comment="restore EXEC"))

        # Hybrid reduction: 4 DPP row_shr stages (register-to-register, no LDS
        # crossbar) reduce within each aligned 16-lane row, then 2 ds_bpermute
        # stages cross the 32- and 64-lane boundaries where DPP has no operator.
        # After all 6 stages every lane holds the full 64-lane sum.
        mod.add(ri.SNop(waitState=0, comment="wait state before DPP reads vAcc"))
        mod.add(ri.SNop(waitState=0, comment="wait state before DPP reads vAcc"))
        for shr in (8, 4, 2, 1):
            mod.add(ri.VAddF32(
                dst=vgpr(vAcc),
                src0=vgpr(vAcc),
                src1=vgpr(vAcc),
                dpp=DPPModifiers(row_shr=shr, bound_ctrl=0),
                comment=f"DPP row_shr:{shr} intra-row partial sum",
            ))
        for stride in (16, 32):
            mod.add(ri.VXorB32(
                dst=vgpr(vPerm),
                src0=vgpr(vLane),
                src1=stride,
                comment=f"partner = laneId ^ {stride}",
            ))
            mod.add(ri.VLShiftLeftB32(
                dst=vgpr(vPerm),
                shiftHex=hex(2),
                src=vgpr(vPerm),
                comment="partnerByte = partner * 4",
            ))
            mod.add(ri.DSBPermuteB32(
                dst=vgpr(vBfly),
                src0=vgpr(vPerm),
                src1=vgpr(vAcc),
                comment=f"bfly = partner acc (stride={stride})",
            ))
            mod.add(ri.SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
            mod.add(ri.VAddF32(
                dst=vgpr(vAcc),
                src0=vgpr(vAcc),
                src1=vgpr(vBfly),
                comment="acc += partner",
            ))

        # ------------------------------------------------------------------
        # Step 6: rstd = rsqrt(invD * acc + eps).
        # ------------------------------------------------------------------
        mod.add(ri.VMovB32(dst=vgpr(vTmp), src=sgpr(sInvD), comment="vTmp = invD"))
        mod.add(ri.VMulF32(
            dst=vgpr(vAcc),
            src0=vgpr(vAcc),
            src1=vgpr(vTmp),
            comment="acc *= invD",
        ))
        mod.add(ri.VMovB32(dst=vgpr(vTmp), src=sgpr(sEps), comment="vTmp = eps"))
        mod.add(ri.VAddF32(
            dst=vgpr(vAcc),
            src0=vgpr(vTmp),
            src1=vgpr(vAcc),
            comment="acc += eps",
        ))
        mod.add(ri.VRsqF32(dst=vgpr(vAcc), src=vgpr(vAcc), comment="rstd = rsqrt(acc)"))
        if archCaps.get("TransOpWait", False):
            mod.add(ri.SNop(waitState=0, comment="TransOpWait after VRsqF32"))

        # ------------------------------------------------------------------
        # Step 7: write rstd to LDS at waveId*256 + sIter*4.
        # All lanes with row < M write (they all hold the same rstd).
        # ------------------------------------------------------------------
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(vLdsAddr),
            shiftHex=hex(8),
            src=vgpr(vWaveId),
            comment="vLdsAddr = waveId * 256",
        ))
        mod.add(ri.VMovB32(dst=vgpr(vTmp), src=sgpr(sIter), comment="vTmp = sIter"))
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(vTmp),
            shiftHex=hex(2),
            src=vgpr(vTmp),
            comment="vTmp = sIter * 4",
        ))
        mod.add(ri.VAddU32(
            vgpr(vLdsAddr), vgpr(vLdsAddr), vgpr(vTmp),
            comment="vLdsAddr = waveId*256 + sIter*4",
        ))

        # Narrow EXEC: only lanes where row < M write.
        mod.add(ri.SAndSaveExecB64(
            dst=sgpr(sTmp, 2),
            src=sgpr(sRowMask, 2),
            comment="apply row<M mask, save old EXEC",
        ))
        mod.add(ri.DSStoreB32(
            dstAddr=vgpr(vLdsAddr),
            src=vgpr(vAcc),
            comment="LDS[waveId*256 + sIter*4] = rstd",
        ))
        mod.add(ri.SWaitCnt(dscnt=0, comment="wait LDS write"))
        mod.add(ri.SMovB64(dst=EXEC(), src=sgpr(sTmp, 2), comment="restore EXEC"))

        # ------------------------------------------------------------------
        # Step 8: advance outer loop; branch back if sIter < 64.
        # ------------------------------------------------------------------
        mod.add(ri.SAddU32(dst=sgpr(sIter), src0=sgpr(sIter), src1=1, comment="++sIter"))
        mod.add(ri.SCmpLtU32(src0=sgpr(sIter), src1=64, comment="sIter < 64?"))
        mod.add(ri.SCBranchSCC1(loopPhase1, comment="loop phase 1"))

        # ------------------------------------------------------------------
        # Step 9: barrier.
        # ------------------------------------------------------------------
        mod.add(ri.SBarrier(comment="all rstd values visible in LDS"))

        # ------------------------------------------------------------------
        # Step 10: Phase 2 — read rstd, scale C columns.
        # Each thread handles one fixed row: globalRow = sRowBase + Serial.
        # ------------------------------------------------------------------
        # LDS read: rstd = LDS[Serial * 4].
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(vLdsAddr),
            shiftHex=hex(2),
            src=vgpr(0),
            comment="LDS addr = Serial * 4",
        ))
        mod.add(ri.DSLoadB32(
            dst=vgpr(vRstd),
            src=vgpr(vLdsAddr),
            comment="vRstd = LDS[Serial*4]",
        ))
        mod.add(ri.SWaitCnt(dscnt=0, comment="wait rstd load"))

        # Global row for phase 2.
        mod.add(ri.VAddU32(
            vgpr(vRowIter), vgpr(0), sgpr(sRowBase),
            comment="globalRow = Serial + sRowBase",
        ))

        # Narrow EXEC to threads where row < M.
        mod.add(ri.SMovB64(dst=sgpr(sExecSave, 2), src=EXEC(), comment="save full EXEC"))
        mod.add(ri.VCmpGtU32(
            dst=sgpr(sRowMask, 2),
            src0=sgpr(sM),
            src1=vgpr(vRowIter),
            comment="row < M mask",
        ))
        mod.add(ri.SAndB64(
            dst=EXEC(),
            src0=sgpr(sExecSave, 2),
            src1=sgpr(sRowMask, 2),
            comment="narrow EXEC to row-valid threads",
        ))

        # Row byte offset within column: globalRow * 2 bytes.
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(vOff),
            shiftHex=hex(1),
            src=vgpr(vRowIter),
            comment="rowByteOff = globalRow * 2",
        ))

        # colStrideBytes = M * 2 (bytes per column in col-major layout).
        mod.add(ri.SLShiftLeftB32(
            dst=sgpr(sIter),
            shiftHex=hex(1),
            src=sgpr(sM),
            comment="sIter reused: colStride = M * 2 bytes",
        ))

        # Column loop: sColStart .. sColEnd - 1.
        # Use sNd as the loop variable (it was already consumed above).
        mod.add(ri.SMovB32(
            dst=sgpr(sNd),
            src=sgpr(sColStart),
            comment="sNd reused as col = colStart",
        ))

        colLoopDone = ".Lpartial_rms_epilogue_col_done"
        mod.add(ri.SCmpLtU32(
            src0=sgpr(sNd),
            src1=sgpr(sColEnd),
            comment="col < colEnd?",
        ))
        mod.add(ri.SCBranchSCC0(colLoopDone, comment="skip if no columns"))

        colLoop = ".Lpartial_rms_epilogue_col"
        mod.add(TextBlock(f"{colLoop}:\n"))

        # Byte address of C[row, col] = col * M * 2 + row * 2.
        mod.add(ri.SMulI32(
            dst=sgpr(sInvD),
            src0=sgpr(sNd),
            src1=sgpr(sIter),
            comment="colBaseBytes = col * M * 2",
        ))
        mod.add(ri.VAddU32(
            vgpr(vTmp), sgpr(sInvD), vgpr(vOff),
            comment="addr = colBase + rowByteOff",
        ))

        mod.add(ri.GlobalLoadD16B16(
            dst=vgpr(vBfly),
            vaddr=vgpr(vTmp),
            saddr=sgpr(srdC, 2),
            comment="load bf16 from C col-major (16-bit)",
        ))
        mod.add(ri.SWaitCnt(vlcnt=0, comment="wait load"))

        # bf16 to f32 (shift left 16), multiply by rstd, back to bf16.
        mod.add(ri.VLShiftLeftB32(
            dst=vgpr(vBfly),
            shiftHex=hex(16),
            src=vgpr(vBfly),
            comment="bf16 bits -> f32 bits",
        ))
        mod.add(ri.VMulF32(
            dst=vgpr(vBfly),
            src0=vgpr(vBfly),
            src1=vgpr(vRstd),
            comment="val *= rstd",
        ))
        mod.add(ri.VLShiftRightB32(
            dst=vgpr(vBfly),
            shiftHex=hex(16),
            src=vgpr(vBfly),
            comment="f32 bits -> bf16 bits (truncate)",
        ))

        mod.add(ri.GlobalStoreB16(
            vaddr=vgpr(vTmp),
            src=vgpr(vBfly),
            saddr=sgpr(srdC, 2),
            comment="store bf16 back to C",
        ))

        mod.add(ri.SAddU32(dst=sgpr(sNd), src0=sgpr(sNd), src1=1, comment="++col"))
        mod.add(ri.SCmpLtU32(
            src0=sgpr(sNd),
            src1=sgpr(sColEnd),
            comment="col < colEnd?",
        ))
        mod.add(ri.SCBranchSCC1(colLoop, comment="loop columns"))

        mod.add(TextBlock(f"{colLoopDone}:\n"))

        # Restore EXEC.
        mod.add(ri.SMovB64(dst=EXEC(), src=sgpr(sExecSave, 2), comment="restore EXEC"))

        mod.add(ri.SEndpgm())

    vgprCount = vgprPool.size() - vgprPool.availableBlockAtEnd()
    sgprCount = sgprPool.size() - sgprPool.availableBlockAtEnd()
    return mod, vgprCount, sgprCount


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

KERNEL_NAME = _KERNEL_NAME

# colAddr uses SMulI32 (signed 32-bit); col * M * 2 is safe up to M=32767.
_MAX_M = 32767


def build_partial_rms_epilogue(chip: str, M: int = 0, N: int = 0, K: int = 0) -> Tuple[str, str]:
    """Generate partial_rms_epilogue RMSNorm kernel assembly for the given chip.

    M is optional and used only for the size guard below.
    Returns (asmStr, kernelName).
    Raises ValueError if M exceeds the 32-bit column-address limit.
    """
    if M > _MAX_M:
        raise ValueError(
            f"partial_rms_epilogue column address uses SMulI32; M={M} exceeds {_MAX_M}, "
            "which would overflow. "
            "Use a kernel with 64-bit address arithmetic for larger matrices."
        )
    isa = _ensure_isa(chip)
    waveSize = 32 if isa[0] in (11, 12) else 64
    assert waveSize == 64, f"partial_rms_epilogue requires wave64; got waveSize={waveSize} for ISA {isa}"
    gfx = chip.split(":")[0]
    xnack = ":xnack+" in chip

    kernelBody, vgprCount, sgprCount = _build_kernel_body(isa)
    funcName = _KERNEL_NAME

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
        name=funcName,
        num_vgpr=vgprCount,
        num_sgpr=sgprCount,
        num_agpr=0,
        num_lds_bytes=1024,
        wavefront_size=64,
        max_workgroup_size=256,
        args_alignment=8,
        args=args,
    )

    kStr = "\n".join([
        _kernel_header(funcName, gfx, xnack),
        str(kernelBody),
        _kernel_rodata(
            funcName,
            gfxArch=isa,
            vgprCount=vgprCount,
            sgprCount=sgprCount,
            lds=1024,
            useWgIdY=True,
        ),
        _meta_str((meta,)),
    ])
    return kStr, funcName
