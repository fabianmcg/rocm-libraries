# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MegaFusedEpilogue: single-pass fused ResidualAdd + PartialRMS + MXFP8Quant.

Ordering invariant for each (m, n, k):
  H            = acc + residual          -- residual add before gamma
  rmsPartials += H * H                   -- square pre-gamma H for RMS
  ResidualOut <- bf16(H)                 -- store pre-gamma value
  acc          = H * gamma               -- scale after squaring and store
  blkAmax[j]  = max(blkAmax, |H*gamma|) -- inline MXFP8 amax fold
"""

import math
import struct

from rocisa.code import Label, Module
from rocisa.container import (
    ContinuousRegister,
    DSModifiers,
    EXEC,
    MUBUFModifiers,
    accvgpr,
    mgpr,
    sgpr,
    vgpr,
)
from rocisa.enum import HighBitSel
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    VSubU32,
    VMed3I32,
    VCmpEQF32,
    BufferStoreB8,
    BufferLoadB32,
    BufferLoadB64,
    BufferLoadB128,
    BufferLoadD16B16,
    BufferLoadD16U8,
    BufferStoreB16,
    BufferStoreB32,
    BufferStoreB64,
    BufferStoreB128,
    DSBPermuteB32,
    DSLoadB32,
    DSLoadB64,
    DSStoreB32,
    ECvtPkBF8toF32,
    ECvtPkFP8toF32,
    SAddU32,
    SAndB64,
    SAndN2B32,
    SAndSaveExecB64,
    SBranch,
    SCBranchSCC0,
    SCmpLtU32,
    SLShiftLeftB32,
    SLShiftRightB32,
    SMovB32,
    SMovB64,
    SMulHIU32,
    SMulI32,
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAdd3U32,
    VAddF32,
    VAddPKF32,
    VAddU32,
    VAndB32,
    VCmpEQU32,
    VCmpLtU32,
    VCndMaskB32,
    VCvtBF16toFP32,
    VCvtBF8toF32,
    VCvtF16toF32,
    VCvtFP8toF32,
    VCvtPkF32toBF16,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMacF32,
    VMaxF32,
    VMovB32,
    VMulF32,
    VMulLOU32,
    VMulPKF32,
    VOrB32,
    VReadfirstlaneB32,
    VXorB32,
)
from Tensile.Common.DataType import DataType


# Maximum inline-literal integer for VOP encodings; larger immediates must be
# materialized in a VGPR before a v_add.
_INLINE_CONST_MAX = 64

# Upper sanity bound on prefetch depth across the flattened (qi, colBatch) unit
# sequence.  Prefetch depth is derived (maximized within the VGPR budget), not fixed:
# deeper prefetch hides more residual-load latency, but a single HBM round-trip is
# fully covered by only a few units of compute lookahead, so beyond this depth extra
# prefetch only trades resBank VGPRs for no additional latency coverage while raising
# spill risk on heavier tiles.  Profiling found depth 4 drains the residual-load
# `s_waitcnt` stall for the reference kernel; cap here so lightweight tiles with large
# VGPR headroom (small S) do not derive an absurdly deep, register-wasteful ring.
_PREFETCH_DEPTH_CAP = 4
# Total VGPR budget for gfx950 kernels (hardware limit).
_VGPR_BUDGET = 256
# Rough estimate of VGPRs consumed by the kernel outside the epilogue emitter's own
# residual ring and acc bank.  Reverse-engineered from MT192x256 MXFP8 (§6 of
# SUBTILE_MEGAFUSED_BATCHED_REDESIGN.md): actual fixed ≈204–205 (including the new
# vPermAddr VGPR added by the dwordx4 interior path).  Raised from 192 to 212 so
# EBC derivation keeps heavy tiles (MT384x256, MT320x320) safely below 256 VGPRs.
# Lower this constant only if the allocator reports spills.
_VGPR_FIXED_ESTIMATE = 212

# Maximum representable magnitudes of the FP8 quantization output types, used
# to scale amax so K2 can requantize. e5m2/bf8 has a far larger range than e4m3.
_fp8E4m3Max = 448.0          # OCP FP8 e4m3 (name used by the MXFP8 quant helpers).



def _sideBytes(dtype):
    """Return (bytes, log2Bytes) for a side-input element type."""
    if dtype.isSingle():
        return 4, 2
    if dtype.isAnyFloat8() or dtype.isAnyBFloat8():
        return 1, 0
    return 2, 1  # bf16 / f16.


class RMSEpilogueGeometry:
    """Compile-time MegaFusedEpilogue tile geometry derived from kernel params."""

    def __init__(self, kernel, laneSgprCount):
        self._deriveCoreGeometry(kernel, laneSgprCount)
        self._deriveSideInputGeometry(kernel)
        self._deriveQuantGeometry()
        # epilogueBatchCols (EBC) and prefetchDepth derive from the VGPR budget.
        self.epilogueBatchCols = self._deriveEpilogueBatchCols()
        self._deriveGammaLdsGeometry()

    def _deriveCoreGeometry(self, kernel, laneSgprCount):
        """Core MFMA/wave tile sizes."""
        # MXFP8 quant is derived: RMSEpilogue active and D output is F8 (OCP e4m3).
        self.useMxfp8 = (bool(kernel.get("RMSEpilogue", False))
                         and kernel["ProblemType"]["DestDataType"].isFloat8())

        self.mfmaM = kernel["MatrixInstM"]
        self.mfmaN = kernel["MatrixInstN"]
        self.waveSize = kernel["WavefrontSize"]
        assert self.waveSize == 64, "megaFused epilogue requires wavefrontSize == 64"
        self.rowsPerLane = (self.mfmaM * self.mfmaN) // self.waveSize
        wg = kernel["MIWaveGroup"]
        self.wgM = wg[0]
        self.wgN = wg[1]
        self.mmaM = (kernel["MacroTile0"] // self.mfmaM) // self.wgM
        self.mmaN = (kernel["MacroTile1"] // self.mfmaN) // self.wgN
        self.macroTile0 = kernel["MacroTile0"]
        self.macroTile1 = kernel["MacroTile1"]
        self.laneSgprCount = laneSgprCount
        self.numPartials = self.mmaN

        dt = kernel["ProblemType"]["DataType"]
        # elemBytes/log2ElemBytes encode the GEMM input element size used in colByte.
        self.elemBytes = 1 if (dt.isAnyFloat8() or dt.isAnyBFloat8()) else 2
        self.log2ElemBytes = 0 if self.elemBytes == 1 else 1

    def _deriveSideInputGeometry(self, kernel):
        """Residual and gamma side-input element geometry."""
        # Residual side input (always fused on for the MegaFused epilogue).
        self.residualType = DataType(kernel.get("RMSEpilogueResidualType") or "b")
        self.residualBytes, self.residualLog2Bytes = _sideBytes(self.residualType)
        self.useWideResidual = ((self.rowsPerLane % 4 == 0)
                                and (self.residualBytes == 1
                                     or (self.residualBytes == 2
                                         and not self.residualType.isHalf())))

        # Gamma side input is always bf16; staged to LDS and read via ds_read.
        self.gammaBytes = 2
        self.gammaLog2Bytes = 1

        # Partial-accumulation modes: this epilogue does not run on final values.
        self.isPartialAccumulation = kernel.get("_GlobalAccumulation") in (
            "MultipleBuffer", "MultipleBufferSingleKernel")

    def _deriveQuantGeometry(self):
        """MXFP8 dynamic-quant tile geometry."""
        self.streamGroup = 4
        if self.useMxfp8:
            self.tagPrefix = "mx"
            self.q0 = 32  # MXFP8 block shape is always 32x1.
            self.q1 = 1
            self.nQTilesM = (self.mmaM * self.mfmaM) // self.q0
            # subCol quant (q1 < mfmaN) is the only MXFP8 mode megaFused emits; subRow
            # (q0 < mfmaM) is excluded by the emit() assertion.
            self.subColQuant = self.q1 < self.mfmaN and not (self.q0 < self.mfmaM)
            self.tilesPerBlockM = (self.q0 // self.mfmaM) if self.subColQuant else 2
        else:
            self.tilesPerBlockM = 2 if self.mmaM % 2 == 0 else 1
            self.nQTilesM = self.mmaM // self.tilesPerBlockM

    def _deriveGammaLdsGeometry(self):
        """Gamma DTL-to-LDS broadcast geometry."""
        self.numRowGroups = self.waveSize // self.mfmaN
        self.gammaLdsWaveStride = self.tilesPerBlockM * self.mfmaM * self.gammaBytes
        self.gammaLdsBufBytes = self.wgM * self.gammaLdsWaveStride
        self.gammaBuffers = 2 if self.nQTilesM > 1 else 1

    def _deriveEpilogueBatchCols(self) -> int:
        """Maximize prefetchDepth within the VGPR budget; derive epilogueBatchCols.

        S = tilesPerBlockM * rowsPerLane is the per-column bank size in the ring.
        Grow PFD from 1 while EBC_max(PFD) >= 1 fits, capped at _PREFETCH_DEPTH_CAP;
        EBC follows as the largest column batch fitting at the chosen depth.
        """
        s = self.tilesPerBlockM * self.rowsPerLane
        budget = _VGPR_BUDGET - _VGPR_FIXED_ESTIMATE
        bestPfd = 0
        bestEbc = 0
        for pfd in range(1, _PREFETCH_DEPTH_CAP + 1):
            ebc = budget // (s * (1 + pfd))
            if ebc < 1:
                break
            bestPfd = pfd
            bestEbc = min(ebc, self.mmaN)
        if bestPfd == 0:
            raise RuntimeError(
                f"megaFused epilogue infeasible: EBC=0 at PFD=1 for this tile shape "
                f"(S={s}, mmaN={self.mmaN}, VGPRFixed est={_VGPR_FIXED_ESTIMATE})"
            )
        numUnits = self.nQTilesM * math.ceil(self.mmaN / bestEbc)
        self.prefetchDepth = max(1, min(bestPfd, numUnits - 1))
        return bestEbc


# ---------------------------------------------------------------------------
# Module-level free functions (extracted from SubtileMegaFusedEmitter).
# GROUP A: no geometry parameter needed.
# GROUP B: accept geom (RMSEpilogueGeometry) as the last positional argument.
# ---------------------------------------------------------------------------

def _isPackPair(a, b):
    """True when a,b are a consecutive even-aligned VGPR pair for packed VALU."""
    return (a % 2 == 0) and (b == a + 1)


def _issueSideLoad(module, dstVgpr: int, addrVgpr: int, srd: int,
                   comment: str, dtype) -> None:
    """Issue one side-input buffer_load without waiting (burst-friendly)."""
    module.addComment1("issue one side-input buffer_load.")
    if dtype.isSingle():
        loadCls = BufferLoadB32
    elif dtype.isAnyFloat8() or dtype.isAnyBFloat8():
        loadCls = BufferLoadD16U8
    else:
        loadCls = BufferLoadD16B16  # bf16 / f16.
    module.add(loadCls(vgpr(dstVgpr), vgpr(addrVgpr), sgpr(srd, 4), 0,
                       MUBUFModifiers(offen=True), comment=comment))


def _addImmU32(module, dst: int, src: int, imm: int, scratch: int, comment: str) -> int:
    """Compute src + imm; return src unchanged when imm == 0 (emits nothing)."""
    if imm == 0:
        return src
    module.addComment1("compute src + imm into dst.")
    # Materialize the immediate in a VGPR when it exceeds the inline-literal range.
    if imm > _INLINE_CONST_MAX:
        module.add(VMovB32(dst=vgpr(scratch), src=imm, comment=f"imm={imm}"))
        module.add(VAddU32(vgpr(dst), vgpr(src), vgpr(scratch), comment=comment))
        return dst
    module.add(VAddU32(vgpr(dst), vgpr(src), imm, comment=comment))
    return dst


def _buildBufferSrd(module, srd: int, ptrName: str, name: str) -> None:
    module.addComment1(f"build generic buffer SRD for {name}.")
    module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr(ptrName, 2), comment=f"{name} SRD base."))
    module.add(SMovB32(dst=sgpr(srd + 2), src="BufferOOB", comment=f"{name} SRD limit."))
    module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96", comment=f"{name} SRD flags."))



def _computeSwizzleStride(module, strideV: int, nTilesV: int) -> None:
    """strideV = ceil(nTiles/8) * 256 (the swizzle d0 stride)."""
    module.addComment1("compute d0 stride = ceil(nTiles/8) * 256.")
    module.add(VAddU32(vgpr(strideV), vgpr(nTilesV), 7, comment="nTiles + 7."))
    module.add(VLShiftRightB32(dst=vgpr(strideV), shiftHex=hex(3), src=vgpr(strideV),
                               comment="colBlocks = ceil(nTiles/8)."))
    module.add(VLShiftLeftB32(dst=vgpr(strideV), shiftHex=hex(8), src=vgpr(strideV),
                              comment="d0 stride = colBlocks * 256."))


def _convertGammaChunkBf16(module, base: int) -> None:
    """Unpack 4 bf16 from dwords (base, base+1) into 4 f32 at base+0..3.

    High indices first so each source dword is read before overwrite.
    """
    module.addComment1("expand 4 packed bf16 gamma dwords to 4 f32.")
    module.add(VCvtBF16toFP32(vgpr(base + 3), vgpr(base + 1), None, 1,
                               comment="gamma k=3 bf16(hi) -> f32."))
    module.add(VCvtBF16toFP32(vgpr(base + 2), vgpr(base + 1), None, 0,
                               comment="gamma k=2 bf16(lo) -> f32."))
    module.add(VCvtBF16toFP32(vgpr(base + 1), vgpr(base + 0), None, 1,
                               comment="gamma k=1 bf16(hi) -> f32."))
    module.add(VCvtBF16toFP32(vgpr(base + 0), vgpr(base + 0), None, 0,
                               comment="gamma k=0 bf16(lo) -> f32."))


def _convertResidualChunkBf16(module, base: int) -> None:
    # 4 bf16 in dwords (base, base+1) -> 4 f32; high indices first so each source
    # dword is fully read before it is overwritten.
    # sel 0 = low16 (WORD_0), sel 1 = high16 (WORD_1).
    module.addComment1("expand 4 packed bf16 dwords to 4 f32.")
    module.add(VCvtBF16toFP32(vgpr(base + 3), vgpr(base + 1), None, 1,
                               comment="residual k=3 bf16(hi) -> f32."))
    module.add(VCvtBF16toFP32(vgpr(base + 2), vgpr(base + 1), None, 0,
                               comment="residual k=2 bf16(lo) -> f32."))
    module.add(VCvtBF16toFP32(vgpr(base + 1), vgpr(base + 0), None, 1,
                               comment="residual k=1 bf16(hi) -> f32."))
    module.add(VCvtBF16toFP32(vgpr(base + 0), vgpr(base + 0), None, 0,
                               comment="residual k=0 bf16(lo) -> f32."))


def _convertSideElem(module, dstVgpr: int, comment: str, dtype) -> None:
    """Convert an already-loaded side element to fp32 in place (no-op for f32)."""
    if dtype.isSingle():
        return
    module.addComment1("in-place conversion of side element to f32.")
    if dtype.isHalf():
        module.add(VCvtF16toF32(vgpr(dstVgpr), vgpr(dstVgpr), comment=comment))
    elif dtype.isAnyFloat8():
        module.add(VCvtFP8toF32(dst=vgpr(dstVgpr), src=vgpr(dstVgpr), comment=comment))
    elif dtype.isAnyBFloat8():
        module.add(VCvtBF8toF32(dst=vgpr(dstVgpr), src=vgpr(dstVgpr), comment=comment))
    else:
        module.add(VCvtBF16toFP32(vgpr(dstVgpr), vgpr(dstVgpr), None, 0, comment=comment))


def _swizzleColBits(module, colV: int, lowV: int, tmpV: int) -> None:
    """OR d5<<6 | d4<<1 | d3<<8 into lowV using colV; clobbers tmpV."""
    module.addComment1("OR swizzle column bits d3/d4/d5 into lowV.")
    module.add(VAndB32(dst=vgpr(tmpV), src0=vgpr(colV), src1=3, comment="d5 = col & 3."))
    module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(6), src=vgpr(tmpV),
                              comment="d5 << 6."))
    module.add(VOrB32(dst=vgpr(lowV), src0=vgpr(lowV), src1=vgpr(tmpV), comment="lowV |= d5<<6."))
    module.add(VLShiftRightB32(dst=vgpr(tmpV), shiftHex=hex(2), src=vgpr(colV),
                               comment="col >> 2."))
    module.add(VAndB32(dst=vgpr(tmpV), src0=vgpr(tmpV), src1=1, comment="d4 = (col>>2)&1."))
    module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(1), src=vgpr(tmpV),
                              comment="d4 << 1."))
    module.add(VOrB32(dst=vgpr(lowV), src0=vgpr(lowV), src1=vgpr(tmpV), comment="lowV |= d4<<1."))
    module.add(VLShiftRightB32(dst=vgpr(tmpV), shiftHex=hex(3), src=vgpr(colV),
                               comment="d3 = col >> 3."))
    module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(8), src=vgpr(tmpV),
                              comment="d3 << 8."))
    module.add(VOrB32(dst=vgpr(lowV), src0=vgpr(lowV), src1=vgpr(tmpV), comment="lowV |= d3<<8."))


def _useDwordx4Interior(geom) -> bool:
    """True when the interior arm may use a dwordx4 residual load/store.

    bf16 residual only; excluded for MXFP8 (SGPR-constrained), for
    partial-accumulation modes, and when rowsPerLane != 4.
    """
    if geom.residualBytes != 2 or geom.useMxfp8 or geom.rowsPerLane != 4:
        return False
    return not geom.isPartialAccumulation


def _packResidualOutRow(module, srcRegs, packBank, geom) -> None:
    """Pack rpl bf16(H) values into packBank (rpl/2 dwords, 2-aligned) for a dwordx2 store."""
    module.addComment1("pack rpl bf16(H) values into dwordx2 store bank.")
    rpl = geom.rowsPerLane
    for p in range(rpl // 2):
        module.add(VCvtPkF32toBF16(dst=vgpr(packBank + p),
                                   src0=vgpr(srcRegs[2 * p]), src1=vgpr(srcRegs[2 * p + 1]),
                                   comment=f"pack H[{2 * p}] lo16, H[{2 * p + 1}] hi16 -> bf16x2."))



def _free0RowPos(module, dst: int, rowBase: int, rowGroupOff: int,
                 m: int, k: int, scratch: int, geom) -> None:
    # free0 row = rowBase + rowGroupOff + (m*mfmaM + k). One v_add3_u32 folds
    # the row-base, the m/k immediate, and rowGroupOff into a single VALU op.
    module.addComment1(f"compute free0 row position (m={m},k={k}).")
    mBase = m * geom.mfmaM + k
    if mBase == 0:
        module.add(VAddU32(vgpr(dst), vgpr(rowBase), vgpr(rowGroupOff),
                           comment=f"row = base + rowGroupOff (m={m},k={k})"))
        return
    if mBase <= _INLINE_CONST_MAX:
        module.add(VAdd3U32(dst=vgpr(dst), src0=vgpr(rowBase), src1=vgpr(rowGroupOff),
                            src2=mBase,
                            comment=f"row = base + {mBase} + rowGroupOff (m={m},k={k})"))
        return
    # mBase exceeds the inline-constant range: materialize it, then fold in one add3.
    module.add(VMovB32(dst=vgpr(scratch), src=mBase, comment=f"imm={mBase}"))
    module.add(VAdd3U32(dst=vgpr(dst), src0=vgpr(rowBase), src1=vgpr(rowGroupOff),
                        src2=vgpr(scratch),
                        comment=f"row = base + {mBase} + rowGroupOff (m={m},k={k})"))


class SubtileMegaFusedEmitter:
    """Emit the MegaFusedEpilogue for the Subtile gfx950 kernel."""

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        self.geom = RMSEpilogueGeometry(kernel, writer.states.laneSGPRCount)

        # ---- Shared registers allocated across the emission (init None) ----
        self.laneId = None
        self.colByte = None
        self.col = None
        self.rowGroup = None
        self.rowGroupOff = None
        self.wgRowBase = None
        self.nhBase = None
        self.partials = None
        self.waveIdV = None
        self.resSrd = None
        self.residualOutSrd = None
        self.gammaSrd = None
        self.savedExec = None
        self.laneMaskSgpr = None
        self.mxSrd = None
        self.resTokenBase = None
        self.resRowByteBase = None
        self.resAddr = None
        self.resOobV = None
        self.resOobMask = None
        self.roRowBase = None
        self.roColByteBase = None
        # dwordx4 interior path: partner-lane byte address (VGPR) and exec masks (SGPR pairs).
        self.vPermAddr = None
        self.pairLowerLaneMask = None
        self.pairUpperLaneMask = None
        # Shared per-tile nhInRange predicate (tail-wide path only); None elsewhere.
        self.nhInRangeMask = None
        # Gamma DTL broadcast: two VGPRs and two SGPRs.
        self.gammaDtlVaddr = None
        self.gammaLdsReadAddr = None
        self.gammaSoffsetSgpr = None
        self.gammaM0Base = None
        # Set when gamma LDS broadcast reads are in flight but not yet waited/converted;
        # the wait+convert is deferred to the first gammaBank consumer for latency hiding.
        self._gammaReadPending = False


    def _amaxAndWriteAcc(self, module, sk, vgprTiles, blkAmaxJ, m, n, ki) -> None:
        """Fold |H*gamma| into blkAmax (MXFP8 only) and write sk to the accumulator.

        Must be called per-k even when the multiply was packed.
        """
        module.addComment1(f"fold |H*gamma| into blkAmax and write acc (m={m},n={n},k={ki}).")
        if self.geom.useMxfp8:
            module.add(VAndB32(dst=vgpr(self._scAccTmp), src0=vgpr(sk),
                               src1=vgpr(self._scAbsMask),
                               comment=f"|H*gamma| (m={m},n={n},k={ki})."))
            module.add(VMaxF32(dst=vgpr(blkAmaxJ), src0=vgpr(blkAmaxJ),
                               src1=vgpr(self._scAccTmp),
                               comment="blkAmax = max(blkAmax, |H*gamma|)."))
        module.addComment1("write VGPR back to accumulator register file.")
        tile = vgprTiles[n * self.geom.mmaM + m]
        reg = tile.regList.indices[ki]
        if tile.regList.pool == self.writer.vgprPool:
            if sk != reg:
                module.add(VMovB32(dst=vgpr(reg), src=vgpr(sk),
                                   comment=f"write H*gamma back to acc (m={m},n={n},k={ki})."))
        else:
            module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(sk),
                                        comment=f"write H*gamma back to acc (m={m},n={n},k={ki})."))


    def _issueUnitLoads(self, resBank, mBaseV, qi, nBase, g,
                        pathInterior: bool = False) -> Module:
        """Emit residual loads for one unit into resBank (prolog or prefetch phase)."""
        module = Module(f"MegaFused issueUnitLoads qi={qi} nBase={nBase}")
        module.addComment0(f"residual loads for one unit (qi={qi},nBase={nBase}).")
        module.addComment1(f"issue all residual loads for N-group (qi={qi},nBase={nBase},g={g}).")
        rpl = self.geom.rowsPerLane
        tpb = self.geom.tilesPerBlockM
        loadsCumulative = []
        issued = 0
        for j in range(g):
            n = nBase + j
            module.addComment1(f"compute row byte base for residual (n={n}).")
            nOff = n * self.geom.mfmaN
            r = _addImmU32(module, self.resRowByteBase, self.resTokenBase, nOff, self.resAddr,
                           f"token_n = tokenBase + {nOff} (n={n}).")
            module.add(VMulLOU32(dst=vgpr(self.resRowByteBase), src0=sgpr("SizesFree+0"), src1=vgpr(r),
                                 comment="token_n * SizesFree0."))
            if self.geom.useWideResidual:
                module.add(VAddU32(vgpr(self.resRowByteBase), vgpr(self.resRowByteBase), vgpr(self.wgRowBase),
                                   comment="+ wgRowBase (fold row origin)."))
                module.add(VAddU32(vgpr(self.resRowByteBase), vgpr(self.resRowByteBase), vgpr(self.rowGroupOff),
                                   comment="+ rowGroupOff (fold row origin)."))
            module.add(VLShiftLeftB32(dst=vgpr(self.resRowByteBase), shiftHex=hex(self.geom.residualLog2Bytes),
                                      src=vgpr(self.resRowByteBase),
                                      comment="rowByteBase * residualBytes (row origin folded when wide)."))
            for mi in range(tpb):
                m = qi * tpb + mi
                burstBase = resBank + (j * tpb + mi) * rpl
                module.addComment1(f"issue residual loads for tile (m={m},n={n}).")
                rpl_irt = self.geom.rowsPerLane
                if pathInterior and _useDwordx4Interior(self.geom):
                    module.addComment1(f"exec-masked B128 load for pair (m={m},n={n}).")
                    lsc_irt = self.geom.laneSgprCount
                    rowOff_irt = m * self.geom.mfmaM * 2
                    assert rowOff_irt < 4096, f"residual dwordx4 load offset {rowOff_irt} exceeds MUBUF offset12 range"
                    loadAddr_irt = self.writer.vgprPool.checkOut(1, tag="mf_dx4LoadAddr")
                    module.add(VCndMaskB32(dst=vgpr(loadAddr_irt), src0=vgpr(self.resRowByteBase),
                                           src1=vgpr(self.resOobV), src2=sgpr(self.pairUpperLaneMask, lsc_irt),
                                           comment="upper lane -> BufferOOB (load dropped); lower lane keeps base."))
                    module.add(BufferLoadB128(vgpr(burstBase, 4), vgpr(loadAddr_irt),
                                              sgpr(self.resSrd, 4), 0,
                                              MUBUFModifiers(offen=True, offset12=rowOff_irt),
                                              comment=f"R dwordx4 (m={m},n={n}) off={rowOff_irt}: 8 bf16 for pair (full exec)."))
                    self.writer.vgprPool.checkIn(loadAddr_irt)
                    issued += 1
                elif self.geom.useWideResidual:
                    module.addComment1(f"wide residual load (m={m},n={n}).")
                    isBf16_irt = self.geom.residualBytes == 2
                    loadCls_irt = BufferLoadB64 if isBf16_irt else BufferLoadB32
                    chunkBytes_irt = 4 << self.geom.residualLog2Bytes
                    rowOff_irt = m * self.geom.mfmaM * self.geom.residualBytes
                    for c_irt in range(self.geom.rowsPerLane // 4):
                        off_irt = rowOff_irt + chunkBytes_irt * c_irt
                        assert off_irt < 4096, f"residual wide load offset {off_irt} exceeds MUBUF offset12 range"
                        dstBase_irt = burstBase + 4 * c_irt
                        dst_irt = vgpr(dstBase_irt, 2) if isBf16_irt else vgpr(dstBase_irt)
                        module.add(loadCls_irt(dst_irt, vgpr(self.resRowByteBase), sgpr(self.resSrd, 4), 0,
                                               MUBUFModifiers(offen=True, offset12=off_irt),
                                               comment=f"R wide [4 residual] (m={m},n={n},c={c_irt}) off={off_irt}."))
                    issued += rpl_irt // 4
                else:
                    for k_irt in range(rpl_irt):
                        module.addComment1(f"compute clamped residual byte address (m={m},k={k_irt}).")
                        _free0RowPos(module, self.resAddr, self.wgRowBase, self.rowGroupOff, m, k_irt, mBaseV, self.geom)
                        module.add(VCmpLtU32(dst=sgpr(self.resOobMask, self.geom.laneSgprCount), src0=vgpr(self.resAddr),
                                             src1=sgpr("SizesFree+0"), comment="inRange = nhidden_pos < N_hidden"))
                        module.add(VLShiftLeftB32(dst=vgpr(self.resAddr), shiftHex=hex(self.geom.residualLog2Bytes),
                                                  src=vgpr(self.resAddr),
                                                  comment="nhiddenByte = nhidden_pos * residualBytes."))
                        module.add(VAddU32(vgpr(self.resAddr), vgpr(self.resAddr), vgpr(self.resRowByteBase),
                                           comment="byteAddr = rowByteBase + nhiddenByte"))
                        module.add(VCndMaskB32(dst=vgpr(self.resAddr), src0=vgpr(self.resOobV), src1=vgpr(self.resAddr),
                                               src2=sgpr(self.resOobMask, self.geom.laneSgprCount),
                                               comment="clamp OOB when nhidden_pos >= N_hidden"))
                        _issueSideLoad(module, burstBase + k_irt, self.resAddr, self.resSrd,
                                           f"R[m={m},n={n},k={k_irt}].", dtype=self.geom.residualType)
                    issued += rpl_irt
                loadsCumulative.append(issued)
        return module



    def _stageGammaToLds(self, module, qi, bufIdx) -> None:
        """Stage gamma to LDS via contiguous-lane DTL; consumer reads with ds_read_b64."""
        module.addComment1(f"contiguous-lane DTL stage gamma to LDS (qi={qi},buf={bufIdx}).")
        sgprPool = self.writer.sgprPool
        lsc = self.geom.laneSgprCount
        numStageLanes = self.geom.tilesPerBlockM * self.geom.mfmaM // 2
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma DTL stage: WAR barrier before reusing LDS region."))
        savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="mf_gammaStageExec", preventOverflow=False)
        module.add(SMovB64(dst=sgpr(savedExec, lsc), src=EXEC(),
                           comment="save exec around contiguous-lane DTL gamma load."))
        # Narrow exec to laneId < numStageLanes (contiguous lanes); AND with waveN==0.
        repMask = sgprPool.checkOutAligned(lsc, lsc, tag="mf_gammaRep", preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(repMask, lsc), src0=vgpr(self.laneId), src1=numStageLanes,
                             comment=f"stageLane = laneId < {numStageLanes} (contiguous b32 DTL)."))
        if self.geom.wgN > 1:
            wtmp = sgprPool.checkOutAligned(lsc, lsc, tag="mf_gammaRepWN", preventOverflow=False)
            waveThreshold = self.geom.wgM * self.geom.waveSize
            with self.writer.allocTmpSgpr(1, tag="mf_gammaWNThr") as thr:
                module.add(SMovB32(dst=sgpr(thr.idx), src=hex(waveThreshold),
                                   comment=f"wgM*waveSize = {waveThreshold} (waveN==0 bound)."))
                module.add(VCmpLtU32(dst=sgpr(wtmp, lsc), src0=vgpr("Serial"),
                                     src1=sgpr(thr.idx),
                                     comment="waveN0 = Serial < wgM*waveSize."))
            module.add(SAndB64(dst=sgpr(repMask, lsc), src0=sgpr(repMask, lsc),
                               src1=sgpr(wtmp, lsc), comment="repLane &= (waveN == 0)."))
            sgprPool.checkIn(wtmp)
        module.add(SMovB64(dst=EXEC(), src=sgpr(repMask, lsc),
                           comment="exec = contiguous stage lanes for DTL gamma load."))
        sgprPool.checkIn(repMask)
        # M0 = waveM * ldsWaveStride + bufIdx * ldsBufBytes.
        if bufIdx == 0:
            module.add(SMovB32(dst=mgpr(0), src=sgpr(self.gammaM0Base),
                               comment="M0 = gamma LDS wave base (buf0)."))
        else:
            with self.writer.allocTmpSgpr(1, tag="mf_gammaM0") as t:
                module.add(SAddU32(dst=sgpr(t.idx), src0=sgpr(self.gammaM0Base),
                                   src1=bufIdx * self.geom.gammaLdsBufBytes,
                                   comment=f"M0 base + buf{bufIdx} offset."))
                module.add(SMovB32(dst=mgpr(0), src=sgpr(t.idx),
                                   comment=f"M0 = gamma LDS wave base (buf{bufIdx})."))
        # soffset = wgRowBase*gammaBytes + qi*tilesPerBlockM*mfmaM*gammaBytes.
        qiSoffsetAdj = qi * self.geom.tilesPerBlockM * self.geom.mfmaM * self.geom.gammaBytes
        if qiSoffsetAdj > 0:
            qiSoffSgpr = sgprPool.checkOutAligned(1, 1, tag="mf_gammaQiSoff", preventOverflow=False)
            module.add(SAddU32(dst=sgpr(qiSoffSgpr), src0=sgpr(self.gammaSoffsetSgpr),
                               src1=qiSoffsetAdj,
                               comment=f"soffset += qi*tpb*mfmaM*gammaBytes for qi={qi}."))
        else:
            qiSoffSgpr = self.gammaSoffsetSgpr
        # One b32 load covers the entire qi block: lane i reads gamma[2i] and gamma[2i+1]
        # and DTL writes them contiguously at LDS M0+i*4, matching the consumer layout.
        module.add(BufferLoadB32(
            dst=None, vaddr=vgpr(self.gammaDtlVaddr), saddr=sgpr(self.gammaSrd, 4),
            soffset=sgpr(qiSoffSgpr),
            mubuf=MUBUFModifiers(offen=True, offset12=0, lds=True),
            comment=f"gamma DTL b32 -> LDS (qi={qi},buf={bufIdx}); {numStageLanes} lanes x 2 bf16."))
        if qiSoffsetAdj > 0:
            sgprPool.checkIn(qiSoffSgpr)
        # DTL is tracked by vmcnt; wait before restoring exec and releasing the exec guard.
        module.add(SWaitCnt(vlcnt=0, comment="wait gamma DTL loads land in LDS (vmcnt tracks DTL)."))
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                           comment="restore full exec after gamma DTL stage."))
        sgprPool.checkIn(savedExec)
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma DTL stage: LDS writes visible before broadcast read."))


    def _ldsReadGammaBlockIssue(self, module, gammaBank, qi, bufIdx) -> None:
        """Issue broadcast LDS reads for staged gamma without waiting or converting.

        The wait and bf16->f32 conversion are deferred so LDS latency overlaps
        residual/RMS work.
        """
        assert not self._gammaReadPending, "gamma LDS read issued while a prior read is still pending"
        module.addComment1(f"broadcast-read gamma from LDS (qi={qi},buf={bufIdx}).")
        tpb = self.geom.tilesPerBlockM
        for mi in range(tpb):
            off = bufIdx * self.geom.gammaLdsBufBytes + mi * self.geom.mfmaM * self.geom.gammaBytes
            module.add(DSLoadB64(
                dst=vgpr(gammaBank + mi * self.geom.rowsPerLane, 2),
                src=vgpr(self.gammaLdsReadAddr),
                ds=DSModifiers(offset=off),
                comment=f"broadcast-read gamma bf16 (qi={qi},mi={mi},buf={bufIdx})."))
        self._gammaReadPending = True



    def _emitFusedBody(self, module, vgprTiles, units, resRing, accBank, gammaBank,
                       blkAmax, mBaseV, globalLoadsCum, issuedThroughUnit,
                       tileStarts, pfd, pathInterior: bool = False) -> None:
        """Issue the prolog loads then iterate all units through the PFD pipeline.

        pathInterior selects the dwordx4 bf16 load/store path (interior arm) vs
        the scalar/wide masked path (tail arm), per design §5.
        """
        module.addComment1("prolog loads then PFD pipeline iteration.")
        self._stageGammaToLds(module, 0, 0)
        # Issue the qi=0 gamma read right after the visibility barrier (no intervening
        # vmem before the DS read); the wait+convert is deferred to the first consumer
        # so the whole prolog's residual loads overlap the LDS round-trip.
        self._ldsReadGammaBlockIssue(module, gammaBank, 0, 0)
        module.addComment1("issue residual loads for the first PFD units.")
        numUnits_ep = len(units)
        for u_ep in range(min(pfd, numUnits_ep)):
            pqi_ep, pnBase_ep, pg_ep = units[u_ep]
            module.add(self._issueUnitLoads(resRing[u_ep % pfd], mBaseV, pqi_ep, pnBase_ep, pg_ep,
                                            pathInterior))
        module.addComment1("PFD-pipeline iteration over all units.")
        tpb = self.geom.tilesPerBlockM
        numUnits = len(units)
        for i, (qi, nBase, g) in enumerate(units):
            if self.geom.useMxfp8:
                module.addComment1(f"zero blkAmax[nBase..nBase+g) (qi={qi},nBase={nBase}).")
                zeroMod_eza = Module(f"MegaFused zeroBlkAmaxSlice qi={qi} nBase={nBase}")
                for j_eza in range(g):
                    zeroMod_eza.add(VMovB32(dst=vgpr(blkAmax + nBase + j_eza), src=0,
                                            comment=f"blkAmax[{nBase + j_eza}] = 0."))
                module.add(zeroMod_eza)
            # Watermark: global load count through the last in-flight prefetch unit.
            watermarkIdx = min(i + pfd - 1, numUnits - 1)
            issuedWatermark = issuedThroughUnit[watermarkIdx]
            numTilesInUnit = g * tpb
            localLoadsCum = [globalLoadsCum[tileStarts[i] + t] for t in range(numTilesInUnit)]
            _unitMod_cui = Module(f"MegaFused computeUnit qi={qi} nBase={nBase}")
            _unitMod_cui.addComment0(f"compute pass for one ring unit (qi={qi},nBase={nBase}).")
            resBank = resRing[i % pfd]
            _unitMod_cui.addComment1(f"compute pass over N-group (qi={qi},nBase={nBase},g={g}).")
            lsc = self.geom.laneSgprCount
            tpb = self.geom.tilesPerBlockM
            t = 0
            for j in range(g):
                n = nBase + j
                # Compute the token(N) OOB mask once per column n; reused across all tpb tiles.
                tokMaskSgpr = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mf_roTokMask",
                                                                   preventOverflow=False)
                self.roRowBase = self.writer.vgprPool.checkOut(1, tag="mf_roRowBase")
                self.roColByteBase = self.writer.vgprPool.checkOut(1, tag="mf_roColByteBase")
                _unitMod_cui.addComment1(f"token OOB mask and roRowBase (n={n}).")
                nOff_roBase = n * self.geom.mfmaN
                tokV_roBase = self.writer.vgprPool.checkOut(1, tag="mf_roTokV")
                scratch_roBase = self.writer.vgprPool.checkOut(1, tag="mf_roTokScratch")
                r_roBase = _addImmU32(_unitMod_cui, tokV_roBase, self.resTokenBase, nOff_roBase, scratch_roBase,
                                      f"token_n = resTokenBase + {nOff_roBase} (n={n}).")
                _unitMod_cui.add(VCmpLtU32(dst=sgpr(tokMaskSgpr, lsc), src0=vgpr(r_roBase),
                                     src1=sgpr("SizesFree+1"),
                                     comment="tokenInRange = token_n < M_tokens (ResidualOut N mask)."))
                _unitMod_cui.add(VMulLOU32(dst=vgpr(self.roRowBase), src0=sgpr("SizesFree+0"), src1=vgpr(r_roBase),
                                     comment=f"roRowBase = token_n * N_hidden (n={n}); reused across m,k."))
                _unitMod_cui.add(VAddU32(dst=vgpr(self.roColByteBase), src0=vgpr(self.roRowBase),
                                   src1=vgpr(self.wgRowBase),
                                   comment="roColBase = token_n*N_hidden + wgRowBase."))
                _unitMod_cui.add(VAddU32(dst=vgpr(self.roColByteBase), src0=vgpr(self.roColByteBase),
                                   src1=vgpr(self.rowGroupOff),
                                   comment="roColBase += rowGroupOff (per-lane row origin)."))
                _unitMod_cui.add(VLShiftLeftB32(dst=vgpr(self.roColByteBase), shiftHex=hex(1),
                                          src=vgpr(self.roColByteBase),
                                          comment="roColByteBase = roColBase * 2 (bf16)."))
                self.writer.vgprPool.checkIn(scratch_roBase)
                self.writer.vgprPool.checkIn(tokV_roBase)
                for mi in range(tpb):
                    _unitMod_cui.addComment1(f"compute pass for tile (qi={qi},j={j},mi={mi}).")
                    rpl = self.geom.rowsPerLane
                    tpb = self.geom.tilesPerBlockM
                    lsc = self.geom.laneSgprCount
                    m = qi * tpb + mi
                    n = nBase + j
                    bankBase = (j * tpb + mi) * rpl
                    burstBase = resBank + bankBase
                    coords = [(m, n, k) for k in range(rpl)]
                    dstBase = accBank + bankBase
                    accComment = f"acc m={m},n={n}."
                    _unitMod_cui.addComment1("read accumulator burst into VGPR staging buffer.")
                    srcRegs = []
                    slot = 0
                    for _i_arc, (m_, n_, k_) in enumerate(coords):
                        tile = vgprTiles[n_ * self.geom.mmaM + m_]
                        reg = tile.regList.indices[k_]
                        if tile.regList.pool == self.writer.vgprPool:
                            srcRegs.append(reg)
                            continue
                        _unitMod_cui.add(VAccvgprReadB32(vgpr(dstBase + slot), accvgpr(reg),
                                                   comment=f"{accComment} [{_i_arc}]"))
                        srcRegs.append(dstBase + slot)
                        slot += 1
                    if 0 < slot < 2:
                        _unitMod_cui.add(SNop(waitState=1, comment="fill the mandatory v_accvgpr_read->VALU wait state (gfx950)."))
                    _free0RowPos(_unitMod_cui, self.nhBase, self.wgRowBase,
                                 self.rowGroupOff, m, 0, mBaseV, self.geom)
                    # Wait only for THIS tile's residual load; later tiles' loads
                    # stay in flight (GWB decreasing-vlcnt schedule).
                    remaining = issuedWatermark - localLoadsCum[t]
                    _unitMod_cui.add(SWaitCnt(vlcnt=remaining,
                                        comment=f"wait residual tile {t}: vlcnt={issuedWatermark}-{localLoadsCum[t]}."))
                    useDx4 = pathInterior and _useDwordx4Interior(self.geom)
                    # Tail-wide path (bf16 only): check out a shared nhInRange mask block so the
                    # wide-residual OOB predicates computed per tile can be reused by the bf16
                    # address path without recomputing.  Skipped for MXFP8 kernels to avoid SGPR
                    # pressure (they recompute the predicate instead).
                    tailWide = self.geom.useWideResidual and not pathInterior and not self.geom.useMxfp8
                    if tailWide:
                        self.nhInRangeMask = self.writer.sgprPool.checkOutAligned(
                            rpl * lsc, lsc, tag="mf_nhInRange", preventOverflow=False)
                    _unitMod_cui.addComment1("convert and mask already-loaded residual tile.")
                    if pathInterior and _useDwordx4Interior(self.geom):
                        # Scatter partner rows from lower lane to upper lane, then convert.
                        _unitMod_cui.addComment1("full-exec ds_bpermute gather then branchless cndmask adoption.")
                        lsc_rdrl = self.geom.laneSgprCount
                        # Full-exec gather: every lane pulls its partner's bank[2]/bank[3].  In-place
                        # dst==src is safe because ds_bpermute snapshots all source lanes before writing.
                        _unitMod_cui.add(DSBPermuteB32(vgpr(burstBase + 2), vgpr(self.vPermAddr),
                                                 vgpr(burstBase + 2),
                                                 comment="gather partner bank[2] (rows 4,5); full exec."))
                        _unitMod_cui.add(DSBPermuteB32(vgpr(burstBase + 3), vgpr(self.vPermAddr),
                                                 vgpr(burstBase + 3),
                                                 comment="gather partner bank[3] (rows 6,7); full exec."))
                        _unitMod_cui.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute."))
                        # Adopt the gathered partner rows on the upper lane only; the lower lane keeps
                        # its own loaded bank[0,1].  Branchless v_cndmask under full exec replaces an
                        # exec-masked v_mov, dropping the save/set/restore-exec churn; the permutes
                        # above still run under full exec, preserving the 099b29d correctness fix.
                        _unitMod_cui.add(VCndMaskB32(dst=vgpr(burstBase), src0=vgpr(burstBase),
                                               src1=vgpr(burstBase + 2),
                                               src2=sgpr(self.pairUpperLaneMask, lsc_rdrl),
                                               comment="bank[0] = upperLane ? partner rows 4,5 : own rows 0,1."))
                        _unitMod_cui.add(VCndMaskB32(dst=vgpr(burstBase + 1), src0=vgpr(burstBase + 1),
                                               src1=vgpr(burstBase + 3),
                                               src2=sgpr(self.pairUpperLaneMask, lsc_rdrl),
                                               comment="bank[1] = upperLane ? partner rows 6,7 : own rows 2,3."))
                        _convertResidualChunkBf16(_unitMod_cui, burstBase)
                        # Interior lanes are guaranteed to be in-range; no OOB masking needed.
                    elif self.geom.useWideResidual:
                        if self.geom.residualBytes == 2:
                            _convertResidualChunkBf16(_unitMod_cui, burstBase)
                        else:
                            cvt_fp8 = ECvtPkFP8toF32 if self.geom.residualType.isAnyFloat8() else ECvtPkBF8toF32
                            # HIGH before LOW: HIGH reads the packed dword at base; LOW then overwrites base.
                            _unitMod_cui.addComment1("expand packed fp8 dword to 4 f32.")
                            _unitMod_cui.add(cvt_fp8(dst=vgpr(burstBase + 2, 2), src=vgpr(burstBase), sel=HighBitSel.HIGH,
                                               comment="residual pair (k=2,3) fp8 -> f32."))
                            _unitMod_cui.add(cvt_fp8(dst=vgpr(burstBase, 2), src=vgpr(burstBase), sel=HighBitSel.LOW,
                                               comment="residual pair (k=0,1) fp8 -> f32."))
                        # GlobalWriteBatch NonEdge parity: the interior arm guarantees every row is
                        # < N_hidden (wgMaxRow < N_hidden), so no element straddles the boundary and
                        # the per-element OOB mask is a no-op.  Skip it in the interior arm, matching
                        # the dwordx4 interior path; only the tail arm (Edge) needs software masking.
                        if not pathInterior:
                            _unitMod_cui.addComment1("software-mask wide residual OOB elements.")
                            lsc_mwr = self.geom.laneSgprCount
                            for k_mwr in range(self.geom.rowsPerLane):
                                maskK_mwr = (self.nhInRangeMask + k_mwr * lsc_mwr) if self.nhInRangeMask is not None \
                                            else self.resOobMask
                                nhR_mwr = _addImmU32(_unitMod_cui, self.resAddr, self.nhBase, k_mwr, self.resRowByteBase,
                                                     f"nhPos = nhBase + {k_mwr}.")
                                _unitMod_cui.add(VCmpLtU32(dst=sgpr(maskK_mwr, lsc_mwr), src0=vgpr(nhR_mwr),
                                                     src1=sgpr("SizesFree+0"),
                                                     comment=f"nhInRange[{k_mwr}] = nhPos < N_hidden (k={k_mwr})."))
                                _unitMod_cui.add(VCndMaskB32(dst=vgpr(burstBase + k_mwr), src0=0,
                                                       src1=vgpr(burstBase + k_mwr),
                                                       src2=sgpr(maskK_mwr, lsc_mwr),
                                                       comment=f"residual = nhInRange ? residual : 0 (k={k_mwr})."))
                    else:
                        for k_frt in range(rpl):
                            _convertSideElem(_unitMod_cui, burstBase + k_frt,
                                                 f"residual->fp32 (k={k_frt}).", dtype=self.geom.residualType)
                    _unitMod_cui.addComment1(f"H = acc + R, rmsSum[n] += H^2 (m={m},n={n}).")
                    for k in range(0, rpl - rpl % 2, 2):
                        sk0, sk1 = srcRegs[k], srcRegs[k + 1]
                        if _isPackPair(sk0, sk1):
                            _unitMod_cui.add(VAddPKF32(dst=vgpr(sk0, 2), src0=vgpr(sk0, 2),
                                                 src1=vgpr(burstBase + k, 2),
                                                 comment=f"H = acc + residual (packed k={k},{k+1})."))
                        else:
                            _unitMod_cui.add(VAddF32(dst=vgpr(sk0), src0=vgpr(sk0),
                                               src1=vgpr(burstBase + k),
                                               comment=f"H = acc + residual (m={m},n={n},k={k})."))
                            _unitMod_cui.add(VAddF32(dst=vgpr(sk1), src0=vgpr(sk1),
                                               src1=vgpr(burstBase + k + 1),
                                               comment=f"H = acc + residual (m={m},n={n},k={k+1})."))
                        _unitMod_cui.add(VMacF32(dst=vgpr(self.partials + n), src0=vgpr(sk0),
                                           src1=vgpr(sk0),
                                           comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k})."))
                        _unitMod_cui.add(VMacF32(dst=vgpr(self.partials + n), src0=vgpr(sk1),
                                           src1=vgpr(sk1),
                                           comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k+1})."))
                    if useDx4:
                        _unitMod_cui.addComment1(f"gather partner packed dwords, B128 store (m={m},n={n}).")
                        assert rpl % 2 == 0, "rpl must be even for bf16 packing"
                        vgprPool = self.writer.vgprPool
                        vPack = vgprPool.checkOutAligned(4, 4, tag="mf_dx4Pack")
                        _packResidualOutRow(_unitMod_cui, srcRegs, vPack, self.geom)
                        _unitMod_cui.add(DSBPermuteB32(vgpr(vPack + 2), vgpr(self.vPermAddr), vgpr(vPack),
                                                 comment="lower lane: vPack[2] <- upper lane's vPack[0]; full exec."))
                        _unitMod_cui.add(DSBPermuteB32(vgpr(vPack + 3), vgpr(self.vPermAddr), vgpr(vPack + 1),
                                                 comment="lower lane: vPack[3] <- upper lane's vPack[1]; full exec."))
                        _unitMod_cui.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute."))
                        dx4Addr = vgprPool.checkOut(1, tag="mf_dx4StoreAddr")
                        _unitMod_cui.add(VCndMaskB32(dst=vgpr(dx4Addr), src0=vgpr(self.roColByteBase),
                                               src1=vgpr(self.resOobV),
                                               src2=sgpr(self.pairUpperLaneMask, lsc),
                                               comment="upper lane -> BufferOOB (store dropped); lower lane keeps addr."))
                        rowOff = m * self.geom.mfmaM * 2
                        assert rowOff < 4096, f"residualOut row offset {rowOff} exceeds MUBUF offset12 range"
                        _unitMod_cui.add(BufferStoreB128(src=vgpr(vPack, 4), vaddr=vgpr(dx4Addr),
                                                   saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                                   mubuf=MUBUFModifiers(offen=True, offset12=rowOff),
                                                   comment=f"ResidualOut dwordx4 (m={m},n={n}) off={rowOff} (full exec, upper OOB)."))
                        vgprPool.checkIn(dx4Addr)
                        vgprPool.checkIn(vPack)
                    else:
                        _unitMod_cui.addComment1(f"pack and store rpl bf16(H) to ResidualOut (m={m},n={n}).")
                        rpl = self.geom.rowsPerLane
                        assert rpl % 2 == 0, "rpl must be even for dwordx2 bf16 packing"
                        lsc = self.geom.laneSgprCount
                        # gfx950 is wave64-only for this path; HasWave32 excludes gfx9,
                        # prereq validation rejects non-gfx950.
                        assert lsc == 2, "storeResidualOutRow hardcodes wave64 b64 exec ops"
                        vgprPool = self.writer.vgprPool
                        sgprPool = self.writer.sgprPool
                        packBank = vgprPool.checkOutAligned(rpl // 2, 2, tag="mf_roPack")
                        nhTopV   = vgprPool.checkOut(1, tag="mf_roNhTop")
                        _packResidualOutRow(_unitMod_cui, srcRegs, packBank, self.geom)
                        safe      = sgprPool.checkOutAligned(lsc, lsc, tag="mf_roSafe", preventOverflow=False)
                        wideAddrV = vgprPool.checkOut(1, tag="mf_roWideAddr")
                        _unitMod_cui.addComment1(f"full-exec dwordx2 store with BufferOOB clamp (m={m},n={n}).")
                        rpl = self.geom.rowsPerLane
                        lsc = self.geom.laneSgprCount
                        nhTop = _addImmU32(_unitMod_cui, nhTopV, self.nhBase, rpl - 1, nhTopV,
                                               f"nhTop = nhBase + {rpl - 1}.")
                        _unitMod_cui.add(VCmpLtU32(dst=sgpr(safe, lsc), src0=vgpr(nhTop),
                                             src1=sgpr("SizesFree+0"),
                                             comment="b64Safe = nhBase+rpl-1 < N_hidden (no straddle)."))
                        _unitMod_cui.add(VCndMaskB32(dst=vgpr(wideAddrV), src0=vgpr(self.resOobV),
                                               src1=vgpr(self.roColByteBase), src2=sgpr(safe, lsc),
                                               comment="wideAddr = b64Safe ? roColByteBase : BufferOOB."))
                        rowOff = m * self.geom.mfmaM * 2
                        assert rowOff < 4096, f"residualOut row offset {rowOff} exceeds MUBUF offset12 range"
                        _unitMod_cui.add(BufferStoreB64(src=vgpr(packBank, 2), vaddr=vgpr(wideAddrV),
                                                  saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                                  mubuf=MUBUFModifiers(offen=True, offset12=rowOff),
                                                  comment=f"ResidualOut dwordx2 (m={m},n={n}) off={rowOff} (full exec, straddle OOB)."))
                        _unitMod_cui.addComment1(f"full-exec per-element fallback for straddle lanes (m={m},n={n}).")
                        lsc = self.geom.laneSgprCount
                        # Straddle mask: tokMask AND NOT b64Safe.
                        # Two SAndN2B32 reuse the safe register pair; SAndB64 sets SCC for the branch.
                        _unitMod_cui.add(SAndN2B32(dst=sgpr(safe), src0=sgpr(tokMaskSgpr), src1=sgpr(safe),
                                             comment="straddle_lo = tokMask_lo & ~b64Safe_lo."))
                        _unitMod_cui.add(SAndN2B32(dst=sgpr(safe + 1), src0=sgpr(tokMaskSgpr + 1),
                                             src1=sgpr(safe + 1),
                                             comment="straddle_hi = tokMask_hi & ~b64Safe_hi."))
                        _unitMod_cui.add(SAndB64(dst=sgpr(safe, lsc), src0=sgpr(safe, lsc),
                                           src1=sgpr(safe, lsc),
                                           comment="SCC = (straddle != 0); safe still holds straddle mask."))
                        skipLabel = Label(self.writer.labels.getNameInc(f"mf_roStraddleEnd_m{m}n{n}"), "")
                        _unitMod_cui.add(SCBranchSCC0(labelName=skipLabel.getLabelName(),
                                                comment="no straddle lanes -> skip per-element fallback."))
                        # GlobalWriteBatch-style: per-element fallback runs under FULL exec; each element
                        # clamps non-straddle lanes' address to BufferOOB so only straddle lanes store.
                        for k in range(self.geom.rowsPerLane):
                            accReg_sbi = srcRegs[k]
                            _unitMod_cui.addComment1(f"inline masked bf16(H) store (m={m},n={n},k={k}).")
                            lsc_sbi = self.geom.laneSgprCount
                            addrV_sbi   = self.writer.vgprPool.checkOut(1, tag="mf_bf16Addr")
                            valV_sbi    = self.writer.vgprPool.checkOut(1, tag="mf_bf16Val")
                            nhByteV_sbi = self.writer.vgprPool.checkOut(1, tag="mf_nhByte")
                            with self.writer.allocTmpSgpr(lsc_sbi, tag="mf_nhMask") as nhMask_sbi:
                                _unitMod_cui.addComment1(f"clamped byte address for ResidualOut (n={n},k={k}).")
                                _unitMod_cui.add(VLShiftLeftB32(dst=vgpr(addrV_sbi), shiftHex=hex(1), src=vgpr(self.roRowBase),
                                                          comment="base0 = roRowBase * 2 (bf16); token_n*N_hidden reused."))
                                nh_sbi = _addImmU32(_unitMod_cui, nhByteV_sbi, self.nhBase, k, valV_sbi,
                                                    f"nhPos = nhBase + {k} (k={k}).")
                                if self.nhInRangeMask is not None:
                                    maskReg_sbi = self.nhInRangeMask + k * lsc_sbi
                                else:
                                    _unitMod_cui.add(VCmpLtU32(dst=sgpr(nhMask_sbi.idx, lsc_sbi), src0=vgpr(nh_sbi),
                                                         src1=sgpr("SizesFree+0"),
                                                         comment="nhInRange = nhPos < N_hidden."))
                                    maskReg_sbi = nhMask_sbi.idx
                                _unitMod_cui.add(VLShiftLeftB32(dst=vgpr(nhByteV_sbi), shiftHex=hex(1), src=vgpr(nh_sbi),
                                                          comment="nhByte = nhPos * 2 (bf16)."))
                                _unitMod_cui.add(VAddU32(vgpr(addrV_sbi), vgpr(addrV_sbi), vgpr(nhByteV_sbi),
                                                   comment="byteAddr = base0 + nhByte."))
                                _unitMod_cui.add(VCndMaskB32(dst=vgpr(addrV_sbi), src0=vgpr(self.resOobV), src1=vgpr(addrV_sbi),
                                                       src2=sgpr(maskReg_sbi, lsc_sbi),
                                                       comment="clamp OOB when nhPos >= N_hidden."))
                                # Store only straddle lanes; non-straddle lanes (already wide-stored, or
                                # token-OOB) -> BufferOOB so the store is a no-op under full exec.
                                _unitMod_cui.add(VCndMaskB32(dst=vgpr(addrV_sbi), src0=vgpr(self.resOobV),
                                                       src1=vgpr(addrV_sbi), src2=sgpr(safe, lsc_sbi),
                                                       comment="store only straddle lanes; others -> BufferOOB."))
                                _unitMod_cui.add(VCvtPkF32toBF16(dst=vgpr(valV_sbi), src0=vgpr(accReg_sbi),
                                                            src1=vgpr(accReg_sbi),
                                                            comment="H -> bf16 (low 16 bits)."))
                                _unitMod_cui.add(BufferStoreB16(src=vgpr(valV_sbi), vaddr=vgpr(addrV_sbi),
                                                          saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                                          mubuf=MUBUFModifiers(offen=True),
                                                          comment=f"ResidualOut bf16(H) (m={m},n={n},k={k})."))
                            self.writer.vgprPool.checkIn(nhByteV_sbi)
                            self.writer.vgprPool.checkIn(valV_sbi)
                            self.writer.vgprPool.checkIn(addrV_sbi)
                        _unitMod_cui.add(skipLabel)
                        vgprPool.checkIn(wideAddrV)
                        sgprPool.checkIn(safe)
                        vgprPool.checkIn(nhTopV)
                        vgprPool.checkIn(packBank)
                    if tailWide:
                        self.writer.sgprPool.checkIn(self.nhInRangeMask)
                        self.nhInRangeMask = None
                    # Complete any deferred gamma LDS read right before its first consumer so the
                    # LDS-read latency overlaps the residual-add/RMS/store work above.
                    if self._gammaReadPending:
                        _unitMod_cui.add(SWaitCnt(dscnt=0, comment="wait gamma LDS broadcast reads (deferred to first consumer)."))
                        for _mi_cvt in range(self.geom.tilesPerBlockM):
                            _convertGammaChunkBf16(_unitMod_cui, gammaBank + _mi_cvt * self.geom.rowsPerLane)
                        self._gammaReadPending = False
                    blkAmaxJ = (blkAmax + n) if self.geom.useMxfp8 else None
                    _unitMod_cui.addComment1(f"apply gamma, fold amax, write acc (m={m},n={n}).")
                    for k in range(0, rpl - rpl % 2, 2):
                        sk0, sk1 = srcRegs[k], srcRegs[k + 1]
                        gk = gammaBank + mi * rpl + k
                        if _isPackPair(sk0, sk1):
                            _unitMod_cui.add(VMulPKF32(dst=vgpr(sk0, 2), src0=vgpr(sk0, 2),
                                                 src1=vgpr(gk, 2),
                                                 comment=f"acc = H * gamma (packed k={k},{k+1})."))
                        else:
                            _unitMod_cui.add(VMulF32(dst=vgpr(sk0), src0=vgpr(sk0), src1=vgpr(gk),
                                               comment=f"acc = H * gamma (m={m},n={n},k={k})."))
                            _unitMod_cui.add(VMulF32(dst=vgpr(sk1), src0=vgpr(sk1), src1=vgpr(gk + 1),
                                               comment=f"acc = H * gamma (m={m},n={n},k={k+1})."))
                        self._amaxAndWriteAcc(_unitMod_cui, sk0, vgprTiles, blkAmaxJ, m, n, k)
                        self._amaxAndWriteAcc(_unitMod_cui, sk1, vgprTiles, blkAmaxJ, m, n, k + 1)
                    t = t + 1
                self.writer.vgprPool.checkIn(self.roColByteBase)
                self.roColByteBase = None
                self.writer.vgprPool.checkIn(self.roRowBase)
                self.roRowBase = None
                self.writer.sgprPool.checkIn(tokMaskSgpr)
            module.add(_unitMod_cui)
            if self.geom.useMxfp8:
                _mxDeferredTailMod = Module(f"MegaFused mxDeferredTail qi={qi} nBase={nBase}")
                _mxDeferredTailMod.addComment0(f"butterfly-reduce blkAmax, compute e8m0 scales, apply and store (qi={qi},nBase={nBase}).")
                vgprPool_mxdt = self.writer.vgprPool
                amaxSlice = blkAmax + nBase
                addrBf = vgprPool_mxdt.checkOut(1, tag="mf_addrBf")
                tmpBf = vgprPool_mxdt.checkOut(g, tag="mf_tmpBf")
                numRounds_mxdt = int(math.log2(self.geom.numRowGroups))
                for r in range(numRounds_mxdt):
                    xorVal = self.geom.mfmaN << r
                    _mxDeferredTailMod.addComment1(f"one XOR-butterfly amax reduction round (xorVal={xorVal}).")
                    _mxDeferredTailMod.add(VXorB32(dst=vgpr(addrBf), src0=vgpr(self.laneId), src1=xorVal,
                                       comment=f"partnerLane = laneId ^ {xorVal}."))
                    _mxDeferredTailMod.add(VLShiftLeftB32(dst=vgpr(addrBf), shiftHex=hex(2), src=vgpr(addrBf),
                                          comment="byteAddr = partnerLane * 4."))
                    for t in range(g):
                        _mxDeferredTailMod.add(DSBPermuteB32(vgpr(tmpBf + t), vgpr(addrBf), vgpr(amaxSlice + t),
                                             comment=f"fetch partner amax[tile={t}]."))
                    _mxDeferredTailMod.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute."))
                    for t in range(g):
                        _mxDeferredTailMod.add(VMaxF32(dst=vgpr(amaxSlice + t), src0=vgpr(amaxSlice + t),
                                           src1=vgpr(tmpBf + t),
                                           comment=f"amax[tile={t}] = max(amax, partner)."))
                vgprPool_mxdt.checkIn(tmpBf)
                vgprPool_mxdt.checkIn(addrBf)
                # Alpha fold: blkAmax[j] = |alpha * blkAmax[j]|.
                for j in range(g):
                    _mxDeferredTailMod.add(VMulF32(dst=vgpr(amaxSlice + j), src0=vgpr(amaxSlice + j),
                                       src1=sgpr("Alpha"), comment=f"blkAmax[{nBase + j}] *= alpha."))
                    _mxDeferredTailMod.add(VAndB32(dst=vgpr(amaxSlice + j), src0=vgpr(amaxSlice + j),
                                       src1=vgpr(self._scAbsMask),
                                       comment=f"blkAmax[{nBase + j}] = |alpha*blkAmax|."))
                # The scale slice is overwritten with alpha*quantMult (the apply multiplier).
                _mxDeferredTailMod.addComment1(f"compute e8m0 scales and alpha-fold for group (qi={qi},nBase={nBase},g={g}).")
                qmulBase = self.writer.vgprPool.checkOut(g, tag=f"mx_scQmul_qi{qi}_n{nBase}")
                scaleByteBank = self.writer.vgprPool.checkOut(g, tag=f"mx_scByteBank_qi{qi}_n{nBase}")
                for j in range(g):
                    # Compute e8m0 quantMult for slot j.
                    amaxVgpr_cms = amaxSlice + j
                    quantMultVgpr_cms = qmulBase + j
                    c254V_cms = self._scC254V
                    zeroMask_cms = self._scZeroMask
                    lsc_cms = self.geom.laneSgprCount
                    _mxDeferredTailMod.addComment1(f"compute e8m0 quantMult for slot {j}.")
                    # scaleF = amax * (1/448) -> into quantMultVgpr (temp for scaleByte).
                    _mxDeferredTailMod.add(VMulF32(dst=vgpr(quantMultVgpr_cms), src0=vgpr(amaxVgpr_cms),
                                       src1=vgpr(self._scInvFp8V),
                                       comment=f"scaleF[{j}] = amax * (1/448)."))
                    adjV = self.writer.vgprPool.checkOut(1, tag="mx_adj")
                    _mxDeferredTailMod.addComment1("compute ceiling adjustment from mantissa.")
                    # mantV = scaleFV << 9: discards sign and exponent, non-zero iff mantissa != 0.
                    mantV_cca = self.writer.vgprPool.checkOut(1, tag="mx_mant")
                    _mxDeferredTailMod.add(VLShiftLeftB32(dst=vgpr(mantV_cca), shiftHex=hex(9),
                                          src=vgpr(quantMultVgpr_cms),
                                          comment="mantV = scaleF << 9 (mant != 0 iff mantV != 0)."))
                    lsc_cca = self.geom.laneSgprCount
                    zmc_cca = self.writer.sgprPool.checkOutAligned(lsc_cca, lsc_cca, tag="mx_zmc", preventOverflow=False)
                    _mxDeferredTailMod.add(VCmpEQU32(dst=sgpr(zmc_cca, lsc_cca), src0=0, src1=vgpr(mantV_cca),
                                     comment="mant == 0?."))
                    # When zmc TRUE (mant==0): dst=src1=0; FALSE (mant!=0): dst=src0=1.
                    _mxDeferredTailMod.add(VCndMaskB32(dst=vgpr(adjV), src0=1, src1=0, src2=sgpr(zmc_cca, lsc_cca),
                                       comment="adj = (mant!=0) ? 1 : 0."))
                    self.writer.sgprPool.checkIn(zmc_cca)
                    self.writer.vgprPool.checkIn(mantV_cca)
                    # expByte = scaleF >> 23; & 0xFF not needed since scaleF >= 0 (sign bit = 0).
                    _mxDeferredTailMod.add(VLShiftRightB32(dst=vgpr(quantMultVgpr_cms), shiftHex=hex(23),
                                           src=vgpr(quantMultVgpr_cms),
                                           comment="expByte = scaleF >> 23."))
                    # scaleByte = expByte + adj -> into quantMultVgpr.
                    _mxDeferredTailMod.add(VAddU32(vgpr(quantMultVgpr_cms), vgpr(quantMultVgpr_cms), vgpr(adjV),
                                       comment=f"scaleByte[{j}] = expByte + ceilAdj."))
                    self.writer.vgprPool.checkIn(adjV)
                    # clamp(scaleByte, 0, 254); VMed3I32 requires src2 Container.
                    _mxDeferredTailMod.add(VMed3I32(dst=vgpr(quantMultVgpr_cms), src0=0,
                                    src1=vgpr(quantMultVgpr_cms), src2=vgpr(c254V_cms),
                                    comment=f"scaleByte[{j}] = clamp(scaleByte, 0, 254)."))
                    # the clamped scaleByte is naturally 0 when amax==0, so no zero-guard is needed.
                    _mxDeferredTailMod.add(VMovB32(dst=vgpr(scaleByteBank + j), src=vgpr(quantMultVgpr_cms),
                                       comment=f"bank scaleByte[{j}] for the store path."))
                    # qExpField = 254 - scaleByte -> into quantMultVgpr.
                    _mxDeferredTailMod.add(VSubU32(vgpr(quantMultVgpr_cms), vgpr(c254V_cms), vgpr(quantMultVgpr_cms),
                                       comment=f"qExpField[{j}] = 254 - scaleByte."))
                    # clamp(qExpField, 1, 254).
                    _mxDeferredTailMod.add(VMed3I32(dst=vgpr(quantMultVgpr_cms), src0=1,
                                    src1=vgpr(quantMultVgpr_cms), src2=vgpr(c254V_cms),
                                    comment=f"qExpField[{j}] = clamp(qExpField, 1, 254)."))
                    # quantMult = bitcast<float>(qExpField << 23).
                    _mxDeferredTailMod.add(VLShiftLeftB32(dst=vgpr(quantMultVgpr_cms), shiftHex=hex(23),
                                          src=vgpr(quantMultVgpr_cms),
                                          comment=f"quantMult[{j}] = qExpField << 23."))
                    # amax==0 override: quantMult = 0 when amax==0 (scaleByte is naturally 0).
                    _mxDeferredTailMod.add(VCmpEQF32(dst=sgpr(zeroMask_cms, lsc_cms), src0=0, src1=vgpr(amaxVgpr_cms),
                                     comment=f"amax[{j}] == 0?."))
                    _mxDeferredTailMod.add(VCndMaskB32(dst=vgpr(quantMultVgpr_cms), src0=vgpr(quantMultVgpr_cms),
                                       src1=0, src2=sgpr(zeroMask_cms, lsc_cms),
                                       comment=f"quantMult[{j}] = 0 if amax==0."))
                for j in range(g):
                    _mxDeferredTailMod.add(VMulF32(dst=vgpr(amaxSlice + j), src0=vgpr(qmulBase + j),
                                       src1=sgpr("Alpha"),
                                       comment=f"applyMult[j={j}] = alpha*quantMult."))
                self.writer.vgprPool.checkIn(qmulBase)
                applyScratch = vgprPool_mxdt.checkOut(self.geom.rowsPerLane, tag="mf_applyScratch")
                mStart = qi * self.geom.tilesPerBlockM
                mEnd = (qi + 1) * self.geom.tilesPerBlockM
                # Re-read acc registers, scale each element by applyMult, then write back.
                _mxDeferredTailMod.addComment1(f"re-read acc, scale by applyMult, write back (nBase={nBase},g={g}).")
                _mxDeferredTailMod.add(SNop(waitState=1, comment="hazard guard: accvgpr_write in element loop -> accvgpr_read here (gfx950)."))
                for j in range(g):
                    n = nBase + j
                    for m in range(mStart, mEnd):
                        coords = [(m, n, k) for k in range(self.geom.rowsPerLane)]
                        _mxDeferredTailMod.addComment1("staged read of accumulator burst into consecutive VGPRs.")
                        comment_rabs = f"reread acc[m={m},n={n}]."
                        usedAcc = False
                        for i_rabs, (m_rabs, n_rabs, k_rabs) in enumerate(coords):
                            tile = vgprTiles[n_rabs * self.geom.mmaM + m_rabs]
                            reg = tile.regList.indices[k_rabs]
                            if tile.regList.pool == self.writer.vgprPool:
                                _mxDeferredTailMod.add(VMovB32(dst=vgpr(applyScratch + i_rabs), src=vgpr(reg),
                                                   comment=f"{comment_rabs} [{i_rabs}]."))
                                continue
                            _mxDeferredTailMod.add(VAccvgprReadB32(vgpr(applyScratch + i_rabs), accvgpr(reg),
                                                       comment=f"{comment_rabs} [{i_rabs}]."))
                            usedAcc = True
                        if usedAcc and len(coords) < 2:
                            _mxDeferredTailMod.add(SNop(waitState=1, comment="s_nop after v_accvgpr_read before VALU (gfx950)."))
                        for k in range(self.geom.rowsPerLane):
                            _mxDeferredTailMod.add(VMulF32(dst=vgpr(applyScratch + k), src0=vgpr(applyScratch + k),
                                               src1=vgpr(amaxSlice + j),
                                               comment=f"acc *= alpha*quantMult[j={j}] (m={m},n={n},k={k})."))
                            src_wamx = applyScratch + k
                            _mxDeferredTailMod.addComment1("write VGPR back to MX accumulator register file.")
                            tile = vgprTiles[n * self.geom.mmaM + m]
                            reg = tile.regList.indices[k]
                            if tile.regList.pool == self.writer.vgprPool:
                                _mxDeferredTailMod.add(VMovB32(dst=vgpr(reg), src=vgpr(src_wamx),
                                                   comment=f"write acc[m={m},n={n},k={k}]."))
                            else:
                                _mxDeferredTailMod.add(VAccvgprWriteB32(accvgpr(reg), vgpr(src_wamx),
                                                            comment=f"write acc[m={m},n={n},k={k}]."))
                vgprPool_mxdt.checkIn(applyScratch)
                _mxDeferredTailMod.addComment1(f"store g e8m0 scale bytes for group (qi={qi},nBase={nBase},g={g}).")
                lsc = self.geom.laneSgprCount
                kblkV = self.writer.vgprPool.checkOut(1, tag="mx_scKblkV")
                if qi:
                    _mxDeferredTailMod.add(VAddU32(vgpr(kblkV), vgpr(self._scKblkBase), qi,
                                       comment=f"kblkV = kblkBase + qi={qi}."))
                else:
                    _mxDeferredTailMod.add(VMovB32(dst=vgpr(kblkV), src=vgpr(self._scKblkBase),
                                       comment="kblkV = kblkBase."))
                _mxDeferredTailMod.addComment1("compute rowGroup==0 AND kblk-in-range lane mask.")
                groupMask = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scGroupMask",
                                                             preventOverflow=False)
                rgCond = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scRgCond",
                                                          preventOverflow=False)
                _mxDeferredTailMod.add(VCmpEQU32(dst=sgpr(rgCond, lsc), src0=0, src1=vgpr(self.rowGroup),
                                 comment="rowGroup == 0?."))
                kblkCond = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scKblkIR",
                                                            preventOverflow=False)
                _mxDeferredTailMod.add(VCmpLtU32(dst=sgpr(kblkCond, lsc), src0=vgpr(kblkV),
                                 src1=vgpr(self._scTotalKBlocks), comment="kblkV < totalKBlocks?."))
                _mxDeferredTailMod.add(SAndB64(dst=sgpr(groupMask, lsc), src0=sgpr(rgCond, lsc),
                               src1=sgpr(kblkCond, lsc),
                               comment="group sub-mask = rowGroup==0 AND kblk in range."))
                self.writer.sgprPool.checkIn(kblkCond)
                self.writer.sgprPool.checkIn(rgCond)
                # colV=kblkV is invariant across the group's g stores; compute its swizzle
                # bits once here so they can be reused across the per-store loop.
                colLowV = self.writer.vgprPool.checkOut(1, tag="mx_scColLow")
                colLowTmp = self.writer.vgprPool.checkOut(1, tag="mx_scColLowTmp")
                _mxDeferredTailMod.add(VMovB32(dst=vgpr(colLowV), src=0, comment="init colLow=0."))
                _swizzleColBits(_mxDeferredTailMod, kblkV, colLowV, colLowTmp)
                self.writer.vgprPool.checkIn(colLowTmp)
                for j in range(g):
                    n = nBase + j
                    _mxDeferredTailMod.addComment0(f"  SubCol store qi={qi}, n={n}.")
                    _mxDeferredTailMod.addComment1(f"compute per-lane freeV = freeBase + n*mfmaN + col (n={n}).")
                    freeV = self.writer.vgprPool.checkOut(1, tag="mx_scFreeV")
                    _mxDeferredTailMod.add(VAddU32(vgpr(freeV), vgpr(self._scFreeBase), vgpr(self.col),
                                       comment="freeBase + col (per-lane)."))
                    nOff = n * self.geom.mfmaN
                    if 0 < nOff <= 64:
                        _mxDeferredTailMod.add(VAddU32(vgpr(freeV), vgpr(freeV), nOff, comment=f"+ n*mfmaN={nOff}."))
                    elif nOff > 64:
                        tmpN = self.writer.vgprPool.checkOut(1, tag="mx_scNOff")
                        _mxDeferredTailMod.add(VMovB32(dst=vgpr(tmpN), src=nOff, comment=f"n*mfmaN={nOff}."))
                        _mxDeferredTailMod.add(VAddU32(vgpr(freeV), vgpr(freeV), vgpr(tmpN), comment="+ n*mfmaN."))
                        self.writer.vgprPool.checkIn(tmpN)
                    freeCond = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scFreeIR",
                                                                preventOverflow=False)
                    _mxDeferredTailMod.add(VCmpLtU32(dst=sgpr(freeCond, lsc), src0=vgpr(freeV),
                                     src1=vgpr(self._scTotalFree), comment="freeV < totalFree?."))
                    _mxDeferredTailMod.add(SAndB64(dst=sgpr(self.laneMaskSgpr, lsc), src0=sgpr(groupMask, lsc),
                                   src1=sgpr(freeCond, lsc),
                                   comment="mask = groupMask AND freeV in range."))
                    self.writer.sgprPool.checkIn(freeCond)
                    _mxDeferredTailMod.add(SAndSaveExecB64(dst=sgpr(self.savedExec, lsc), src=sgpr(self.laneMaskSgpr, lsc),
                                           comment="save exec; set exec = write-lane mask."))
                    # Compute GFX950 pre-swizzled MXScale byte offset for this store slot.
                    strideV_stbo = self._scStrideV
                    _mxDeferredTailMod.addComment1("compute GFX950 pre-swizzled MXScale byte offset.")
                    sLow = self.writer.vgprPool.checkOut(1, tag="mx_swzLow")
                    sTmp = self.writer.vgprPool.checkOut(1, tag="mx_swzTmp")
                    ownStride = strideV_stbo is None
                    if ownStride:
                        strideV_stbo = self.writer.vgprPool.checkOut(1, tag="mx_swzStride")
                        _computeSwizzleStride(_mxDeferredTailMod, strideV_stbo, self._scTotalKBlocks)
                    # d0*stride survives the swizzle bit-mixing, so it needs a register the mixing does not touch.
                    d0Prod = strideV_stbo if ownStride else self.writer.vgprPool.checkOut(1, tag="mx_swzD0")
                    _mxDeferredTailMod.add(VLShiftRightB32(dst=vgpr(sTmp), shiftHex=hex(5), src=vgpr(freeV),
                                           comment="d0 = qTileRow >> 5."))
                    _mxDeferredTailMod.add(VMulLOU32(dst=vgpr(d0Prod), src0=vgpr(sTmp), src1=vgpr(strideV_stbo),
                                     comment="d0 * stride."))
                    _mxDeferredTailMod.addComment1("write swizzle row bits d1/d2 into lowV.")
                    _mxDeferredTailMod.add(VAndB32(dst=vgpr(sLow), src0=vgpr(freeV), src1=0xF, comment="d2 = row & 0xF."))
                    _mxDeferredTailMod.add(VLShiftLeftB32(dst=vgpr(sLow), shiftHex=hex(2), src=vgpr(sLow),
                                          comment="d2 << 2."))
                    _mxDeferredTailMod.add(VLShiftRightB32(dst=vgpr(sTmp), shiftHex=hex(4), src=vgpr(freeV),
                                           comment="row >> 4."))
                    _mxDeferredTailMod.add(VAndB32(dst=vgpr(sTmp), src0=vgpr(sTmp), src1=1, comment="d1 = (row>>4)&1."))
                    _mxDeferredTailMod.add(VOrB32(dst=vgpr(sLow), src0=vgpr(sLow), src1=vgpr(sTmp), comment="lowV |= d1."))
                    if colLowV is None:
                        _swizzleColBits(_mxDeferredTailMod, kblkV, sLow, sTmp)
                    else:
                        _mxDeferredTailMod.add(VOrB32(dst=vgpr(sLow), src0=vgpr(sLow), src1=vgpr(colLowV),
                                          comment="lowV |= precomputed col bits (hoisted)."))
                    _mxDeferredTailMod.add(VAddU32(vgpr(freeV), vgpr(d0Prod), vgpr(sLow), comment="swizzled byteOff."))
                    if ownStride:
                        self.writer.vgprPool.checkIn(strideV_stbo)
                    else:
                        self.writer.vgprPool.checkIn(d0Prod)
                    self.writer.vgprPool.checkIn(sTmp)
                    self.writer.vgprPool.checkIn(sLow)
                    _mxDeferredTailMod.add(BufferStoreB8(
                        src=vgpr(scaleByteBank + j), vaddr=vgpr(freeV),
                        saddr=sgpr(self.mxSrd, 4), soffset=0,
                        mubuf=MUBUFModifiers(offen=True),
                        comment=f"MXScale[freeV, kblkV] byte (qi={qi}, n={n})."))
                    _mxDeferredTailMod.add(SMovB64(dst=EXEC(), src=sgpr(self.savedExec, lsc),
                                   comment="restore exec mask."))
                    self.writer.vgprPool.checkIn(freeV)
                self.writer.vgprPool.checkIn(colLowV)
                self.writer.sgprPool.checkIn(groupMask)
                self.writer.vgprPool.checkIn(kblkV)
                vgprPool_mxdt.checkIn(scaleByteBank)
                module.add(_mxDeferredTailMod)
            # Prefetch the next unit AFTER compute+tail so scratch regs do not collide.
            prefetchIdx = i + pfd
            if prefetchIdx < numUnits:
                pqi, pnBase, pg = units[prefetchIdx]
                module.add(self._issueUnitLoads(resRing[prefetchIdx % pfd], mBaseV,
                                                pqi, pnBase, pg, pathInterior))
            # Ping-pong: stage next qi's gamma and issue its LDS read directly after the
            # visibility barrier (no intervening vmem before the DS read). The wait+convert
            # is deferred to qi+1's first consumer, so the next-unit prefetch loads and
            # qi+1's first-tile residual/RMS work overlap the LDS round-trip.
            if (self.geom.gammaBuffers > 1
                    and nBase + g == self.geom.mmaN and qi + 1 < self.geom.nQTilesM):
                nextBufIdx = (qi + 1) % self.geom.gammaBuffers
                self._stageGammaToLds(module, qi + 1, nextBufIdx)
                self._ldsReadGammaBlockIssue(module, gammaBank, qi + 1, nextBufIdx)






    def emit(self, vgprTiles):
        assert not self.geom.useMxfp8 or self.geom.subColQuant, \
            "megaFused MXFP8 requires subColQuant (q1 < mfmaN)"
        ebc = self.geom.epilogueBatchCols
        pfd = self.geom.prefetchDepth
        s = self.geom.tilesPerBlockM * self.geom.rowsPerLane
        module = Module("SubtileMegaFusedEpilogue")
        module.addComment0(
            f"MegaFused PFD ring: EBC={ebc} PFD={pfd} S={s} "
            f"(nQTilesM={self.geom.nQTilesM} mmaN={self.geom.mmaN}).")
        module.addComment0("top-level MegaFused epilogue emission.")
        vgprPool = self.writer.vgprPool
        self.laneId      = vgprPool.checkOut(1, tag="mf_laneId")
        self.colByte     = vgprPool.checkOut(1, tag="mf_colByte")
        self.col         = vgprPool.checkOut(1, tag="mf_col")
        self.rowGroup    = vgprPool.checkOut(1, tag="mf_rowGroup")
        self.rowGroupOff = vgprPool.checkOut(1, tag="mf_rowGroupOff")
        self.wgRowBase   = vgprPool.checkOut(1, tag="mf_wgRowBase")
        self.nhBase      = vgprPool.checkOut(1, tag="mf_nhBase")
        self.partials    = vgprPool.checkOut(self.geom.mmaN, tag="mf_rmsSum")
        if self.geom.wgM > 1:
            self.waveIdV = vgprPool.checkOut(1, tag="mf_waveIdV")
        if _useDwordx4Interior(self.geom):
            self.vPermAddr = vgprPool.checkOut(1, tag="mf_vPermAddr")
        self.gammaDtlVaddr    = vgprPool.checkOut(1, tag="mf_gammaDtlVaddr")
        self.gammaLdsReadAddr = vgprPool.checkOut(1, tag="mf_gammaLdsReadAddr")
        _setupSharedMod = Module("MegaFused shared setup")
        _setupSharedMod.addComment0("drain waits, lane arithmetic, colByte, col, rowGroup, SRDs.")
        _setupSharedMod.add(SWaitCnt(kmcnt=0, comment="drain kernarg s_loads before reading kernel args."))
        _setupSharedMod.add(SWaitCnt(vlcnt=0, comment="drain GEMM vector-memory before AGPR reuse."))
        mfmaN, waveSize = self.geom.mfmaN, self.geom.waveSize
        log2N = int(math.log2(mfmaN))
        _setupSharedMod.add(VAndB32(dst=vgpr(self.laneId), src0=vgpr("Serial"), src1=waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)."))
        if self.geom.wgM > 1:
            waveIdTmp = self.writer.vgprPool.checkOutAligned(2, 2, tag="mf_waveIdDiv")
            _setupSharedMod.add(vectorStaticDivide(self.waveIdV, "Serial", waveSize,
                                         ContinuousRegister(waveIdTmp, 2),
                                         comment="waveId = Serial / waveSize."))
            self.writer.vgprPool.checkIn(waveIdTmp)
        # col is the raw free1 column index used by the MXScale path.
        _setupSharedMod.add(VAndB32(dst=vgpr(self.col), src0=vgpr(self.laneId), src1=mfmaN - 1,
                           comment="col = laneId & (mfmaN-1)."))
        # colByte encodes the token index as col * elemBytes; wave/wg offsets added below.
        _setupSharedMod.add(VLShiftLeftB32(dst=vgpr(self.colByte), shiftHex=hex(self.geom.log2ElemBytes),
                                  src=vgpr(self.col), comment="colByte = col * elemBytes."))
        _setupSharedMod.add(VLShiftRightB32(dst=vgpr(self.rowGroup), shiftHex=hex(log2N),
                                   src=vgpr(self.laneId), comment="rowGroup = laneId >> log2(mfmaN)."))
        if self.geom.wgN > 1:
            _setupSharedMod.addComment1("add waveN column byte offset to colByte.")
            waveN = self.writer.vgprPool.checkOut(1, tag="rAdd_setupWaveN")
            tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="rAdd_setupTmp")
            tmpRes = ContinuousRegister(tmpVgpr, 2)
            _setupSharedMod.add(vectorStaticDivide(waveN, "Serial", self.geom.waveSize * self.geom.wgM, tmpRes,
                                          comment=f"waveN = Serial / {self.geom.waveSize * self.geom.wgM}"))
            colBaseBytes = self.geom.mmaN * self.geom.mfmaN * self.geom.elemBytes
            with self.writer.allocTmpSgpr(1, tag="rAdd_setupColBase") as tmpSgprInfo:
                _setupSharedMod.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(colBaseBytes),
                                   comment=f"col base bytes per wave ({colBaseBytes})"))
                _setupSharedMod.add(VMulLOU32(dst=vgpr(waveN), src0=sgpr(tmpSgprInfo.idx), src1=vgpr(waveN),
                                     comment="waveN * mmaN * mfmaN * elemBytes"))
            _setupSharedMod.add(VAddU32(vgpr(self.colByte), vgpr(self.colByte), vgpr(waveN),
                               comment="colByte += wave column base"))
            self.writer.vgprPool.checkIn(tmpVgpr)
            self.writer.vgprPool.checkIn(waveN)
        with self.writer.allocTmpSgpr(1, tag="mf_wg1ColByte") as wg1S:
            wg1Bytes = self.geom.macroTile1 * self.geom.elemBytes
            _setupSharedMod.add(SMulI32(dst=sgpr(wg1S.idx), src0=sgpr("WorkGroup1"), src1=wg1Bytes,
                               comment=f"wg1ColByte = WorkGroup1 * MT1*elemBytes ({wg1Bytes})."))
            _setupSharedMod.add(VAddU32(dst=vgpr(self.colByte), src0=vgpr(self.colByte), src1=sgpr(wg1S.idx),
                               comment="colByte += WorkGroup1 * MT1 * elemBytes."))
        _setupSharedMod.addComment1("allocate and build shared SRD SGPRs.")
        sgprPool = self.writer.sgprPool
        lsc = self.writer.states.laneSGPRCount
        self.resSrd = sgprPool.checkOutAligned(4, 4, tag="mf_resSrd", preventOverflow=False)
        _setupSharedMod.addComment1("build residual SRD with OOB bounds.")
        with self.writer.allocTmpSgpr(1, tag="rAdd_resSrdNumRec") as tmpSgpr:
            _setupSharedMod.add(SMovB64(dst=sgpr(self.resSrd, 2), src=sgpr("ResidualBuf", 2),
                               comment="residual SRD base"))
            _setupSharedMod.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("SizesFree+0"),
                               src1=sgpr("SizesFree+1"), comment="numRecords = N_hidden * M_tokens"))
            _setupSharedMod.add(SLShiftLeftB32(dst=sgpr(self.resSrd + 2), src=sgpr(tmpSgpr.idx),
                                      shiftHex=hex(self.geom.residualLog2Bytes),
                                      comment="numRecords *= residualBytes."))
        _setupSharedMod.add(SMovB32(dst=sgpr(self.resSrd + 3), src="Srd127_96", comment="residual SRD flags"))
        # ResidualOut aliases the (beta=0 unused) SrdC named SGPR, so its descriptor
        # must be built here; otherwise bf16(H) stores target the stale C buffer.
        self.residualOutSrd = self.writer.sgprs["SrdResidualOut"]
        _setupSharedMod.addComment1("build ResidualOut SRD with OOB bounds.")
        with self.writer.allocTmpSgpr(1, tag="rAdd_roSrdNumRec") as tmpSgpr:
            _setupSharedMod.add(SMovB64(dst=sgpr(self.residualOutSrd, 2), src=sgpr("AddressResidualOut", 2),
                               comment="ResidualOut SRD base."))
            _setupSharedMod.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("SizesFree+0"),
                               src1=sgpr("SizesFree+1"), comment="numRecords = N_hidden * M_tokens"))
            _setupSharedMod.add(SLShiftLeftB32(dst=sgpr(self.residualOutSrd + 2), src=sgpr(tmpSgpr.idx),
                                      shiftHex=hex(1), comment="numRecords *= 2 (bf16)."))
        _setupSharedMod.add(SMovB32(dst=sgpr(self.residualOutSrd + 3), src="Srd127_96",
                           comment="ResidualOut SRD flags."))
        self.gammaSrd  = sgprPool.checkOutAligned(4, 4, tag="mf_gammaSrd", preventOverflow=False)
        self.savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="mf_savedExec", preventOverflow=False)
        self.laneMaskSgpr  = sgprPool.checkOutAligned(lsc, lsc, tag="mf_laneMask", preventOverflow=False)
        _buildBufferSrd(_setupSharedMod, self.gammaSrd, "RMSNormGamma", "gamma")
        # MXScale SRD is only needed when MXFP8 dynamic quant is active.
        if self.geom.useMxfp8:
            self.mxSrd = sgprPool.checkOutAligned(4, 4, tag="mf_mxSrd", preventOverflow=False)
            _buildBufferSrd(_setupSharedMod, self.mxSrd, "MXScale", "mxScale")
        # dwordx4 pair masks: pairLower = lanes {0-15, 32-47}, pairUpper = the complement.
        if _useDwordx4Interior(self.geom):
            self.pairLowerLaneMask = sgprPool.checkOutAligned(
                lsc, lsc, tag="mf_pairLower", preventOverflow=False)
            self.pairUpperLaneMask = sgprPool.checkOutAligned(
                lsc, lsc, tag="mf_pairUpper", preventOverflow=False)
        # DTL gamma broadcast: per-wave soffset (wgRowBase*gammaBytes) and M0 wave base.
        self.gammaSoffsetSgpr = sgprPool.checkOutAligned(
            1, 1, tag="mf_gammaSoffset", preventOverflow=False)
        self.gammaM0Base = sgprPool.checkOutAligned(
            1, 1, tag="mf_gammaM0Base", preventOverflow=False)
        module.add(_setupSharedMod)
        module.addComment1("row geometry, rmsSum init, gamma bank, residual scratch.")
        module.addComment1("compute rowGroupOff = rowGroup * rowsPerLane.")
        log2MfmaN = int(math.log2(self.geom.mfmaN))
        module.add(VLShiftRightB32(dst=vgpr(self.rowGroupOff), shiftHex=hex(log2MfmaN),
                                   src=vgpr(self.laneId),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        module.add(VMulLOU32(dst=vgpr(self.rowGroupOff), src0=self.geom.rowsPerLane,
                             src1=vgpr(self.rowGroupOff),
                             comment=f"rowGroupOff = rowGroup * {self.geom.rowsPerLane}"))
        module.addComment1("compute free0 row base for this wave.")
        mt0Vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_rbMT0")
        module.add(VMovB32(dst=vgpr(mt0Vgpr), src=self.geom.macroTile0,
                           comment=f"MT0={self.geom.macroTile0}"))
        module.add(VMulLOU32(dst=vgpr(self.wgRowBase), src0=vgpr(mt0Vgpr),
                             src1=sgpr("WorkGroup0"), comment="rowBase = WorkGroup0 * MT0"))
        self.writer.vgprPool.checkIn(mt0Vgpr)
        if self.geom.wgM > 1:
            waveM = self.writer.vgprPool.checkOut(1, tag="pRMS_rbWaveM")
            self._computeWaveM(module, waveM)
            waveStride = self.geom.mmaM * self.geom.mfmaM
            strideV = self.writer.vgprPool.checkOut(1, tag="pRMS_rbStride")
            module.add(VMovB32(dst=vgpr(strideV), src=waveStride,
                               comment=f"waveStride = mmaM * mfmaM = {waveStride}"))
            module.add(VMulLOU32(dst=vgpr(waveM), src0=vgpr(strideV), src1=vgpr(waveM),
                                 comment="waveMOff = waveM * waveStride"))
            module.add(VAddU32(vgpr(self.wgRowBase), vgpr(self.wgRowBase), vgpr(waveM),
                               comment="rowBase += waveMOff"))
            self.writer.vgprPool.checkIn(strideV)
            self.writer.vgprPool.checkIn(waveM)
        initRmsSumMod = Module("MegaFused initRmsSum")
        initRmsSumMod.addComment0("zero-initialise rmsSum bank before the fused sweep.")
        for n in range(self.geom.mmaN):
            initRmsSumMod.add(VMovB32(dst=vgpr(self.partials + n), src=0,
                                      comment=f"rmsSum[{n}] = 0.0f."))
        module.add(initRmsSumMod)
        # Gamma stays VGPR-resident for the whole sweep; 2-aligned for dwordx2 loads.
        gammaBank = self.writer.vgprPool.checkOutAligned(
            self.geom.tilesPerBlockM * self.geom.rowsPerLane, 2, tag="mf_gamma")
        module.addComment1("allocate residual scratch and compute invariants.")
        writer = self.writer
        self.resTokenBase   = writer.vgprPool.checkOut(1, tag="mf_resTokenBase")
        self.resRowByteBase = writer.vgprPool.checkOut(1, tag="mf_resRowByteBase")
        self.resAddr        = writer.vgprPool.checkOut(1, tag="mf_resAddr")
        self.resOobV        = writer.vgprPool.checkOut(1, tag="mf_resOobV")
        self.resOobMask     = writer.sgprPool.checkOutAligned(
            self.geom.laneSgprCount, self.geom.laneSgprCount, tag="mf_resOobMask",
            preventOverflow=False)
        module.add(VLShiftRightB32(dst=vgpr(self.resTokenBase),
                                   shiftHex=hex(self.geom.log2ElemBytes),
                                   src=vgpr(self.colByte),
                                   comment="resTokenBase = colByte >> log2ElemBytes."))
        module.add(VMovB32(dst=vgpr(self.resOobV), src="BufferOOB",
                           comment="resOobV = BufferOOB (OOB loads return 0 / stores dropped)."))
        if self.geom.useMxfp8:
            module.addComment1("allocate and init shared streaming context registers.")
            lsc_bsc = self.geom.laneSgprCount
            invFp8Bits = struct.unpack('<I', struct.pack('<f', 1.0 / _fp8E4m3Max))[0]
            invFp8V = self.writer.vgprPool.checkOut(1, tag="mx_scInvFp8")
            module.add(VMovB32(dst=vgpr(invFp8V), src=hex(invFp8Bits),
                               comment=f"1/fp8_max = 1/{_fp8E4m3Max}."))
            c254V = self.writer.vgprPool.checkOut(1, tag="mx_scC254")
            module.add(VMovB32(dst=vgpr(c254V), src=254, comment="constant 254."))
            zeroMask = self.writer.sgprPool.checkOutAligned(lsc_bsc, lsc_bsc, tag="mx_scZeroMask",
                                                             preventOverflow=False)
            absMask = self.writer.vgprPool.checkOut(1, tag="mx_scAbsMask")
            module.add(VMovB32(dst=vgpr(absMask), src=hex(0x7FFFFFFF), comment="abs mask."))
            accTmp = self.writer.vgprPool.checkOut(1, tag="mx_scAccTmp")
            module.addComment1("compute waveM and waveN from waveIdx.")
            if self.geom.wgM <= 1 and self.geom.wgN <= 1:
                waveM_bsc, waveN_bsc = None, None
            else:
                log2Wave_cwi = int(math.log2(self.geom.waveSize))
                waveIdx_cwi = self.writer.vgprPool.checkOut(1, tag=f"{self.geom.tagPrefix}_waveIdx")
                module.add(VLShiftRightB32(dst=vgpr(waveIdx_cwi), shiftHex=hex(log2Wave_cwi),
                                           src=vgpr("Serial"),
                                           comment=f"waveIdx = Serial >> {log2Wave_cwi}."))
                waveM_bsc = None
                if self.geom.wgM > 1:
                    waveM_bsc = self.writer.vgprPool.checkOut(1, tag=f"{self.geom.tagPrefix}_waveM")
                    module.add(VAndB32(dst=vgpr(waveM_bsc), src0=vgpr(waveIdx_cwi),
                                       src1=self.geom.wgM - 1,
                                       comment=f"waveM = waveIdx & ({self.geom.wgM}-1)."))
                waveN_bsc = None
                if self.geom.wgN > 1:
                    log2WgM_cwi = int(math.log2(self.geom.wgM))
                    waveN_bsc = self.writer.vgprPool.checkOut(1, tag=f"{self.geom.tagPrefix}_waveN")
                    module.add(VLShiftRightB32(dst=vgpr(waveN_bsc), shiftHex=hex(log2WgM_cwi),
                                               src=vgpr(waveIdx_cwi),
                                               comment=f"waveIdx >> {log2WgM_cwi}."))
                    module.add(VAndB32(dst=vgpr(waveN_bsc), src0=vgpr(waveN_bsc), src1=self.geom.wgN - 1,
                                       comment=f"waveN = (waveIdx >> {log2WgM_cwi}) & ({self.geom.wgN}-1)."))
                self.writer.vgprPool.checkIn(waveIdx_cwi)
            totalFree = self.writer.vgprPool.checkOut(1, tag="mx_scTotalFree")
            module.addComment1("compute totalQTilesN = ceil(N/Q1).")
            with self.writer.allocTmpSgpr(1, tag=f"{self.geom.tagPrefix}_nQTNsS") as s_ctn:
                module.add(SAddU32(dst=sgpr(s_ctn.idx), src0=sgpr("SizesFree+1"),
                                   src1=self.geom.q1 - 1,
                                   comment=f"N + Q1-1 (Q1={self.geom.q1})."))
                log2q1_ctn = int(math.log2(self.geom.q1))
                module.add(SLShiftRightB32(dst=sgpr(s_ctn.idx), shiftHex=hex(log2q1_ctn),
                                           src=sgpr(s_ctn.idx),
                                           comment=f"totalQTilesN = ceil(N/Q1={self.geom.q1})."))
                module.add(VMovB32(dst=vgpr(totalFree), src=sgpr(s_ctn.idx),
                                   comment="totalQTilesN into VGPR."))
            totalKBlocks = self.writer.vgprPool.checkOut(1, tag="mx_scTotalKBlks")
            module.addComment1("compute totalQTilesM = ceil(nHidden/Q0).")
            with self.writer.allocTmpSgpr(1, tag=f"{self.geom.tagPrefix}_nQTMsS") as s_ctm:
                module.add(SAddU32(dst=sgpr(s_ctm.idx), src0=sgpr("SizesFree+0"),
                                   src1=self.geom.q0 - 1,
                                   comment=f"nHidden + Q0-1 (Q0={self.geom.q0})."))
                log2q0_ctm = int(math.log2(self.geom.q0)) if self.geom.q0 > 1 else 0
                module.add(SLShiftRightB32(dst=sgpr(s_ctm.idx), shiftHex=hex(log2q0_ctm),
                                           src=sgpr(s_ctm.idx),
                                           comment=f"totalQTilesM = ceil(nHidden/Q0={self.geom.q0})."))
                module.add(VMovB32(dst=vgpr(totalKBlocks), src=sgpr(s_ctm.idx),
                                   comment="totalQTilesM into VGPR."))
            freeBaseV = self.writer.vgprPool.checkOut(1, tag="mx_scFreeBase")
            module.addComment1("compute freeBase = WG1*MT1 + waveN*waveSpanN.")
            self._mulVgprBySgprConst(module, freeBaseV, "WorkGroup1", self.geom.macroTile1,
                                      "freeBase = WG1 * MT1.")
            if waveN_bsc is not None:
                waveSpanN_cfb = self.geom.mmaN * self.geom.mfmaN
                tmp_cfb = self.writer.vgprPool.checkOut(1, tag="mx_scFreeBaseTmp")
                self._shiftOrMulVgprConst(module, tmp_cfb, waveN_bsc, waveSpanN_cfb,
                                          f"waveN * waveSpanN={waveSpanN_cfb}.")
                module.add(VAddU32(vgpr(freeBaseV), vgpr(freeBaseV), vgpr(tmp_cfb),
                                   comment="+ waveN * waveSpanN."))
                self.writer.vgprPool.checkIn(tmp_cfb)
            kblkBaseV = self.writer.vgprPool.checkOut(1, tag="mx_scKblkBase")
            module.addComment1("compute kblkBase = WG0*(nQTilesM*wgM) + waveM*nQTilesM.")
            nQTilesMPerWG_ckb = self.geom.nQTilesM * self.geom.wgM
            self._mulVgprBySgprConst(module, kblkBaseV, "WorkGroup0", nQTilesMPerWG_ckb,
                                      f"kblkBase = WG0 * {nQTilesMPerWG_ckb}.")
            if waveM_bsc is not None:
                tmp_ckb = self.writer.vgprPool.checkOut(1, tag="mx_scKblkBaseTmp")
                self._shiftOrMulVgprConst(module, tmp_ckb, waveM_bsc, self.geom.nQTilesM,
                                          f"waveM * nQTilesM={self.geom.nQTilesM}.")
                module.add(VAddU32(vgpr(kblkBaseV), vgpr(kblkBaseV), vgpr(tmp_ckb),
                                   comment="+ waveM * nQTilesM."))
                self.writer.vgprPool.checkIn(tmp_ckb)
            strideV_bsc = self.writer.vgprPool.checkOut(1, tag="mx_scStride")
            _computeSwizzleStride(module, strideV_bsc, totalKBlocks)
            self._scInvFp8V, self._scC254V, self._scZeroMask = invFp8V, c254V, zeroMask
            self._scAbsMask, self._scAccTmp = absMask, accTmp
            self._scWaveM, self._scWaveN = waveM_bsc, waveN_bsc
            self._scTotalFree, self._scTotalKBlocks = totalFree, totalKBlocks
            self._scFreeBase, self._scKblkBase, self._scStrideV = freeBaseV, kblkBaseV, strideV_bsc
        # dwordx4 setup must happen BEFORE the interior/tail branch so that vPermAddr
        # and the pair masks are available in both arms (only interior uses them).
        if _useDwordx4Interior(self.geom):
            module.addComment1("precompute vPermAddr and pair exec masks.")
            module.add(VXorB32(dst=vgpr(self.vPermAddr), src0=vgpr(self.laneId), src1=16,
                               comment="partner = laneId XOR 16 (pair within 32-lane half)."))
            module.add(VLShiftLeftB32(dst=vgpr(self.vPermAddr), shiftHex=hex(2),
                                      src=vgpr(self.vPermAddr),
                                      comment="vPermAddr = partner * 4 (ds_bpermute byte addr)."))
            lsc = self.geom.laneSgprCount
            module.add(SMovB32(dst=sgpr(self.pairLowerLaneMask), src=hex(0x0000FFFF),
                               comment="pairLower lo: lanes 0-15."))
            module.add(SMovB32(dst=sgpr(self.pairLowerLaneMask + 1), src=hex(0x0000FFFF),
                               comment="pairLower hi: lanes 32-47."))
            module.add(SMovB32(dst=sgpr(self.pairUpperLaneMask), src=hex(0xFFFF0000),
                               comment="pairUpper lo: lanes 16-31."))
            module.add(SMovB32(dst=sgpr(self.pairUpperLaneMask + 1), src=hex(0xFFFF0000),
                               comment="pairUpper hi: lanes 48-63."))
        module.addComment1("precompute gamma DTL addresses and M0 base.")
        module.add(VLShiftLeftB32(dst=vgpr(self.gammaDtlVaddr), shiftHex=hex(2),
                                  src=vgpr(self.laneId),
                                  comment="gammaDtlVaddr = laneId * 4 (contiguous b32 DTL offset)."))
        module.add(SNop(waitState=0,
                        comment="conservative nop: wgRowBase was written well before this point, so the VALU hazard window is already closed."))
        with self.writer.allocTmpSgpr(1, tag="mf_gammaRfl") as t:
            module.add(VReadfirstlaneB32(dst=sgpr(t.idx), src=vgpr(self.wgRowBase),
                                         comment="extract uniform wgRowBase from VGPR."))
            module.add(SLShiftLeftB32(dst=sgpr(self.gammaSoffsetSgpr), src=sgpr(t.idx),
                                      shiftHex=hex(self.geom.gammaLog2Bytes),
                                      comment="gammaSoffsetSgpr = wgRowBase * gammaBytes."))
        assert (self.geom.gammaLdsWaveStride & (self.geom.gammaLdsWaveStride - 1)) == 0, \
            "gamma LDS wave stride must be a power of two for the log2 shift"
        log2LdsWaveStride = self.geom.gammaLdsWaveStride.bit_length() - 1
        if self.geom.wgM > 1:
            waveMV = self.writer.vgprPool.checkOut(1, tag="mf_gammaWaveMV")
            module.add(VAndB32(dst=vgpr(waveMV), src0=vgpr(self.waveIdV),
                               src1=self.geom.wgM - 1,
                               comment=f"waveM = waveId % {self.geom.wgM}."))
            module.add(SNop(waitState=0,
                            comment="wait for VGPR before readfirstlane (VALU write hazard)."))
            module.add(VReadfirstlaneB32(dst=sgpr(self.gammaM0Base), src=vgpr(waveMV),
                                         comment="waveM_scalar for M0 base."))
            self.writer.vgprPool.checkIn(waveMV)
            module.add(SLShiftLeftB32(dst=sgpr(self.gammaM0Base),
                                      src=sgpr(self.gammaM0Base),
                                      shiftHex=hex(log2LdsWaveStride),
                                      comment=f"gammaM0Base = waveM * ldsWaveStride({self.geom.gammaLdsWaveStride})."))
        else:
            module.add(SMovB32(dst=sgpr(self.gammaM0Base), src=0,
                               comment="gammaM0Base = 0 (wgM == 1, waveM always 0)."))
        rowGroupOffBytes = self.writer.vgprPool.checkOut(1, tag="mf_gammaRGOff")
        module.add(VLShiftLeftB32(dst=vgpr(rowGroupOffBytes),
                                  shiftHex=hex(self.geom.gammaLog2Bytes),
                                  src=vgpr(self.rowGroupOff),
                                  comment="rowGroupOff * gammaBytes for consumer LDS read base."))
        module.add(VMovB32(dst=vgpr(self.gammaLdsReadAddr), src=sgpr(self.gammaM0Base),
                           comment="gammaLdsReadAddr = waveM * ldsWaveStride (SGPR -> VGPR)."))
        module.add(VAddU32(dst=vgpr(self.gammaLdsReadAddr),
                           src0=vgpr(self.gammaLdsReadAddr),
                           src1=vgpr(rowGroupOffBytes),
                           comment="gammaLdsReadAddr += rowGroupOff * gammaBytes."))
        self.writer.vgprPool.checkIn(rowGroupOffBytes)
        tpb = self.geom.tilesPerBlockM
        units = [
            (qi, nBase, min(ebc, self.geom.mmaN - nBase))
            for qi in range(self.geom.nQTilesM)
            for nBase in range(0, self.geom.mmaN, ebc)
        ]
        loadsPerTile = self.geom.rowsPerLane // 4 if self.geom.useWideResidual else self.geom.rowsPerLane
        globalLoadsCum = []
        issuedThroughUnit = []
        running = 0
        for _qi, _nBase, g in units:
            for _j in range(g):
                for _mi in range(tpb):
                    running += loadsPerTile
                    globalLoadsCum.append(running)
            issuedThroughUnit.append(running)
        tileStarts = []
        cumTiles = 0
        for _qi, _nBase, g in units:
            tileStarts.append(cumTiles)
            cumTiles += g * tpb
        tpb = self.geom.tilesPerBlockM
        bankSize = ebc * tpb * self.geom.rowsPerLane
        resRing = [
            self.writer.vgprPool.checkOutAligned(bankSize, 2, tag=f"mf_resBank{sl}")
            for sl in range(pfd)
        ]
        accBank = self.writer.vgprPool.checkOutAligned(bankSize, 2, tag="mf_accBank")
        blkAmax = None
        if self.geom.useMxfp8:
            blkAmax = self.writer.vgprPool.checkOut(self.geom.mmaN, tag="mf_blkAmax")
        mBaseV = self.writer.vgprPool.checkOut(1, tag="mf_mBase")
        # Banks are allocated once and shared by both branch arms; only one arm
        # executes at runtime (the branch is workgroup-uniform), so VGPR usage
        # is unchanged relative to a single-path emit.
        emitBodyFn = lambda mod, pathInterior: self._emitFusedBody(
            mod, vgprTiles, units, resRing, accBank, gammaBank, blkAmax,
            mBaseV, globalLoadsCum, issuedThroughUnit, tileStarts, pfd, pathInterior)
        module.addComment1("workgroup-uniform interior/tail branch.")
        tailLabel = Label(self.writer.labels.getNameInc("mf_interiorTail_tail"),
                          "tail arm entry (wgMaxRow >= N_hidden).")
        endLabel  = Label(self.writer.labels.getNameInc("mf_interiorTail_end"),
                          "interior/tail merge point.")

        # Compute wgMaxRow and set SCC; SCC = 1 means interior-safe.
        module.addComment1("compute wgMaxRow and set SCC for interior/tail branch.")
        mt0_wmrc = self.geom.macroTile0
        with self.writer.allocTmpSgpr(1, tag="mf_wgMaxRow") as wgMaxRowS_wmrc:
            dst_wmrc = sgpr(wgMaxRowS_wmrc.idx)
            module.add(SMulI32(dst=dst_wmrc, src0=sgpr("WorkGroup0"), src1=mt0_wmrc,
                               comment=f"wgMaxRow_lo = WorkGroup0 * MT0={mt0_wmrc}."))
            module.add(SAddU32(dst=dst_wmrc, src0=dst_wmrc, src1=mt0_wmrc - 1,
                               comment=f"wgMaxRow = WorkGroup0*MT0 + (MT0-1)."))
            module.add(SCmpLtU32(src0=dst_wmrc, src1=sgpr("SizesFree+0"),
                                 comment="SCC = (wgMaxRow < N_hidden): interior path safe."))

        # Branch to tail when SCC == 0 (wgMaxRow >= N_hidden).
        module.add(SCBranchSCC0(labelName=tailLabel.getLabelName(),
                                comment="branch to tail when wgMaxRow >= N_hidden."))

        # Interior arm — falls through from the branch.
        module.addComment0("Interior arm (wgMaxRow < N_hidden): all rows fit, no boundary straddle.")
        emitBodyFn(module, pathInterior=True)
        module.add(SBranch(labelName=endLabel.getLabelName(),
                           comment="interior done; skip tail arm."))

        # Tail arm — workgroups whose last row straddles or exceeds N_hidden.
        module.add(tailLabel)
        module.addComment0("Tail arm (wgMaxRow >= N_hidden): boundary tile, scalar masked path.")
        emitBodyFn(module, pathInterior=False)

        module.add(endLabel)
        self.writer.vgprPool.checkIn(mBaseV)
        if self.geom.useMxfp8:
            self.writer.vgprPool.checkIn(blkAmax)
        self.writer.vgprPool.checkIn(accBank)
        for sl in range(pfd - 1, -1, -1):
            self.writer.vgprPool.checkIn(resRing[sl])
        # Free gamma before the butterfly to cap VGPR high-water (stack-LIFO order).
        # Gamma is fully consumed by the fused body at this point.
        self.writer.vgprPool.checkIn(gammaBank)
        # Intra-wave ds_bpermute butterfly BEFORE the store drain; overlaps in-flight stores
        # (its internal dscnt=0 wait does not drain vector stores).
        _reduceRGF0Mod = Module("PartialRMS reduceRowGroupFree0")
        _reduceRGF0Mod.addComment0("intra-wave row-group butterfly (no LDS memory, no barrier).")
        # TODO(perf): fuse the Σx² and amax row-group butterflies to share the
        # partner-address computation and dscnt wait. Deferred for simplicity.
        numRounds_rgf0 = int(math.log2(self.geom.waveSize // self.geom.mfmaN))
        rgrMod = Module("PartialRMS rowGroupReduceFree0")
        rgrMod.addComment0("XOR butterfly row-group reduction.")
        rgrMod.addComment1(
            f"PartialRMS step 2 (free0): XOR butterfly over {self.geom.waveSize // self.geom.mfmaN} row groups."
        )
        if numRounds_rgf0 == 0:
            _reduceRGF0Mod.add(rgrMod)
        else:
            addrV_rgf0 = self.writer.vgprPool.checkOut(1, tag="pRMS_rgrAddr")
            tmpV_rgf0 = self.writer.vgprPool.checkOut(self.geom.numPartials, tag="pRMS_rgrTmp")
            for i_rgf0 in range(numRounds_rgf0):
                xorVal_rgf0 = self.geom.mfmaN << i_rgf0
                rgrMod.add(
                    VXorB32(dst=vgpr(addrV_rgf0), src0=vgpr(self.laneId), src1=xorVal_rgf0,
                            comment=f"partnerLane = laneId ^ {xorVal_rgf0}")
                )
                rgrMod.add(
                    VLShiftLeftB32(dst=vgpr(addrV_rgf0), shiftHex=hex(2), src=vgpr(addrV_rgf0),
                                   comment="byteAddr = partnerLane * 4")
                )
                for n in range(self.geom.numPartials):
                    rgrMod.add(
                        DSBPermuteB32(vgpr(tmpV_rgf0 + n), vgpr(addrV_rgf0), vgpr(self.partials + n),
                                      comment=f"fetch partner partial[{n}]")
                    )
                rgrMod.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
                for n in range(self.geom.numPartials):
                    rgrMod.add(
                        VAddF32(dst=vgpr(self.partials + n), src0=vgpr(self.partials + n),
                               src1=vgpr(tmpV_rgf0 + n), comment=f"partial[{n}] + partner")
                    )
            self.writer.vgprPool.checkIn(tmpV_rgf0)
            self.writer.vgprPool.checkIn(addrV_rgf0)
            _reduceRGF0Mod.add(rgrMod)
        module.add(_reduceRGF0Mod)
        # vscnt=0 store drain; still fences the cross-wave LDS+barrier phase that follows.
        module.addComment1("drain stores, free residual scratch.")
        if self.geom.useMxfp8:
            self.writer.vgprPool.checkIn(self._scStrideV)
            self.writer.vgprPool.checkIn(self._scKblkBase)
            self.writer.vgprPool.checkIn(self._scFreeBase)
            self.writer.vgprPool.checkIn(self._scTotalKBlocks)
            self.writer.vgprPool.checkIn(self._scTotalFree)
            if self._scWaveN is not None:
                self.writer.vgprPool.checkIn(self._scWaveN)
            if self._scWaveM is not None:
                self.writer.vgprPool.checkIn(self._scWaveM)
            self.writer.vgprPool.checkIn(self._scAccTmp)
            self.writer.vgprPool.checkIn(self._scAbsMask)
            self.writer.sgprPool.checkIn(self._scZeroMask)
            self.writer.vgprPool.checkIn(self._scC254V)
            self.writer.vgprPool.checkIn(self._scInvFp8V)
        module.addComment1("drain all epilogue stores, free scratch registers.")
        writer_ers = self.writer
        module.add(SWaitCnt(vscnt=0, comment="drain ResidualOut (and MXScale for MXFP8) stores before cross-wave reduce."))
        writer_ers.sgprPool.checkIn(self.resOobMask)
        writer_ers.vgprPool.checkIn(self.resOobV)
        writer_ers.vgprPool.checkIn(self.resAddr)
        writer_ers.vgprPool.checkIn(self.resRowByteBase)
        writer_ers.vgprPool.checkIn(self.resTokenBase)
        # Cross-wave reduce (fenced by the drain above) + partialBuf write.
        _rmsModule = Module("MegaFused reduceAndWriteRms")
        _rmsModule.addComment0("reduce rmsSum across row groups and waves, write to partialBuf.")
        _rmsSgprPool = self.writer.sgprPool
        partialSrd = _rmsSgprPool.checkOutAligned(4, 4, tag="mf_partialSrd", preventOverflow=False)
        _buildBufferSrd(_rmsModule, partialSrd, "PartialBuf", "partialBuf")
        _crossWaveModule = Module("PartialRMS reduceCrossWaveFree0")
        _crossWaveModule.addComment0("cross-wave LDS reduction (fenced, wgM > 1 only).")
        if self.geom.wgM > 1:
            reduceArrays = [(self.partials, VAddF32, "+")]
            # Step 3 (free0): reduce every array in `arrays` across wgM sibling waves
            # in a single LDS pass so the three barriers are shared, not paid per array.
            # arrays: list of (baseVgpr, op, verb); array a occupies lane-slot dwords
            # [a*numPartials, (a+1)*numPartials).
            numArrays = len(reduceArrays)
            laneSlotBytes = numArrays * self.geom.numPartials * 4
            strideW = self.geom.waveSize * laneSlotBytes
            _crossWaveReduceF0Mod = Module("PartialRMS crossWaveReduceFree0")
            _crossWaveReduceF0Mod.addComment0("fused cross-wave LDS reduction pass.")
            _crossWaveReduceF0Mod.addComment1(
                f"PartialRMS step 3 (free0): cross-wave LDS reduction over wgM={self.geom.wgM}, "
                f"arrays={numArrays}."
            )
            _crossWaveReduceF0Mod.add(self.writer._syncThreads(
                self.kernel,
                "partialRMS free0 cross-wave: ensure siblings done reading LDS before scratch write."))
            writeAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WriteAddr")
            readAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadAddr")
            readTmp = self.writer.vgprPool.checkOut(numArrays * self.geom.numPartials, tag="pRMS_xwF0ReadTmp")
            _crossWaveReduceF0Mod.addComment1("compute LDS write/read addresses for cross-wave reduction.")
            # Only reached when wgM > 1, so self.waveIdV is valid. Reuse the cached
            # waveId and laneId instead of recomputing them here.
            laneSlotBytes_cwca = numArrays * self.geom.numPartials * 4
            strideW_cwca = self.geom.waveSize * laneSlotBytes_cwca
            waveM_cwca = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WaveM")
            readBaseWave_cwca = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadBase")
            laneLoc_cwca = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0Lane")
            # laneLoc is mutated in place below, so copy the cached laneId into it.
            _crossWaveReduceF0Mod.add(VMovB32(dst=vgpr(laneLoc_cwca), src=vgpr(self.laneId),
                               comment="laneId for LDS addressing (cached)"))
            _crossWaveReduceF0Mod.add(VAndB32(dst=vgpr(waveM_cwca), src0=vgpr(self.waveIdV), src1=self.geom.wgM - 1,
                               comment=f"waveM = waveId % {self.geom.wgM}"))
            # readBaseWave = waveId XOR waveM = waveN * wgM.
            _crossWaveReduceF0Mod.add(VXorB32(dst=vgpr(readBaseWave_cwca), src0=vgpr(self.waveIdV), src1=vgpr(waveM_cwca),
                               comment="readBaseWave = waveN * wgM"))
            with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0AddrSetup") as tmpSgprInfo_cwca:
                tmpSgpr_cwca = tmpSgprInfo_cwca.idx
                _crossWaveReduceF0Mod.add(SMovB32(dst=sgpr(tmpSgpr_cwca), src=hex(strideW_cwca),
                                   comment=f"strideW={strideW_cwca}"))
                _crossWaveReduceF0Mod.add(VMulLOU32(dst=vgpr(writeAddr), src0=sgpr(tmpSgpr_cwca),
                                     src1=vgpr(self.waveIdV),
                                     comment="writeAddr = waveId * strideW"))
                _crossWaveReduceF0Mod.add(VMulLOU32(dst=vgpr(readAddr), src0=sgpr(tmpSgpr_cwca),
                                     src1=vgpr(readBaseWave_cwca),
                                     comment="readAddr = readBaseWave * strideW"))
                _crossWaveReduceF0Mod.add(SMovB32(dst=sgpr(tmpSgpr_cwca), src=hex(laneSlotBytes_cwca),
                                   comment=f"laneSlotBytes={laneSlotBytes_cwca}"))
                _crossWaveReduceF0Mod.add(VMulLOU32(dst=vgpr(laneLoc_cwca), src0=sgpr(tmpSgpr_cwca),
                                     src1=vgpr(laneLoc_cwca),
                                     comment="lane * laneSlotBytes"))
                _crossWaveReduceF0Mod.add(VAddU32(vgpr(writeAddr), vgpr(writeAddr), vgpr(laneLoc_cwca),
                                   comment="writeAddr += lane*laneSlotBytes"))
                _crossWaveReduceF0Mod.add(VAddU32(vgpr(readAddr), vgpr(readAddr), vgpr(laneLoc_cwca),
                                   comment="readAddr += lane*laneSlotBytes"))
            self.writer.vgprPool.checkIn(laneLoc_cwca)
            self.writer.vgprPool.checkIn(readBaseWave_cwca)
            self.writer.vgprPool.checkIn(waveM_cwca)
            _crossWaveReduceF0Mod.addComment1("LDS store each array partial for cross-wave reduction.")
            for a_cws, (base_cws, _op_cws, _verb_cws) in enumerate(reduceArrays):
                for i_cws in range(self.geom.numPartials):
                    off_cws = (a_cws * self.geom.numPartials + i_cws) * 4
                    _crossWaveReduceF0Mod.add(DSStoreB32(dstAddr=vgpr(writeAddr), src=vgpr(base_cws + i_cws),
                                          ds=DSModifiers(offset=off_cws),
                                          comment=f"LDS store arr[{a_cws}] partial[{i_cws}]."))
            _crossWaveReduceF0Mod.add(SWaitCnt(dscnt=0, comment="wait LDS writes."))
            _crossWaveReduceF0Mod.add(self.writer._syncThreads(self.kernel, "partialRMS free0 cross-wave write."))
            # TODO(perf): prefetch wave[j+1]'s LDS loads while accumulating wave[j] to
            # overlap load and compute. Deferred: needs a second readTmp buffer.
            _crossWaveReduceF0Mod.addComment1("LDS-staged cross-wave load and reduce.")
            numArrays = len(reduceArrays)
            for j in range(self.geom.wgM):
                for a, (base, _op, _verb) in enumerate(reduceArrays):
                    for i in range(self.geom.numPartials):
                        off = (a * self.geom.numPartials + i) * 4
                        # For j==0 load directly into the accumulator base to avoid a
                        # redundant VMovB32 copy; for j>0 use the temp buffer.
                        dst = (base + i) if j == 0 else (readTmp + a * self.geom.numPartials + i)
                        _crossWaveReduceF0Mod.add(DSLoadB32(dst=vgpr(dst), src=vgpr(readAddr), ds=DSModifiers(offset=off),
                                             comment=f"LDS load wave[{j}] arr[{a}] partial[{i}]."))
                _crossWaveReduceF0Mod.add(SWaitCnt(dscnt=0, comment="wait LDS reads."))
                if j > 0:
                    # j==0 loads go directly into base+i; only j>0 reaches here.
                    _crossWaveReduceF0Mod.addComment1(f"accumulate wave[{j}] partials into base.")
                    for a_cwa, (base_cwa, op_cwa, verb_cwa) in enumerate(reduceArrays):
                        for i_cwa in range(self.geom.numPartials):
                            src_cwa = readTmp + a_cwa * self.geom.numPartials + i_cwa
                            _crossWaveReduceF0Mod.add(op_cwa(dst=vgpr(base_cwa + i_cwa), src0=vgpr(base_cwa + i_cwa),
                                              src1=vgpr(src_cwa),
                                              comment=f"arr[{a_cwa}] partial[{i_cwa}] {verb_cwa} wave[{j}]."))
                if j < self.geom.wgM - 1:
                    with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0Advance") as tmpSgprInfo:
                        _crossWaveReduceF0Mod.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(strideW),
                                           comment=f"strideW={strideW}."))
                        _crossWaveReduceF0Mod.add(VAddU32(vgpr(readAddr), vgpr(readAddr), sgpr(tmpSgprInfo.idx),
                                           comment="advance readAddr to next sibling wave."))
            # No further LDS use after this point, so the post-read WAR barrier is
            # unnecessary; the next persistent-loop tile's main loop re-barriers
            # before its own first LDS write.
            self.writer.vgprPool.checkIn(readTmp)
            self.writer.vgprPool.checkIn(readAddr)
            self.writer.vgprPool.checkIn(writeAddr)
            _crossWaveModule.add(_crossWaveReduceF0Mod)
        _rmsModule.add(_crossWaveModule)
        globalAddr = self.writer.vgprPool.checkOut(1, tag="mf_globalAddr")
        _writePartialsFree0Mod = Module("PartialRMS writePartialsFree0")
        _writePartialsFree0Mod.addComment0("predicated write of Σx² partials to partialBuf.")
        _writePartialsFree0Mod.addComment1(
            "PartialRMS step 4 (free0): predicated write of Σx² to partialBuf[token, WG0].")
        lsc = self.geom.laneSgprCount
        _writePartialsFree0Mod.addComment1("compute lane mask for rowGroup==0 and waveM==0.")
        log2MfmaN = int(math.log2(self.geom.mfmaN))
        rgV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0RowGroup")
        _writePartialsFree0Mod.add(VLShiftRightB32(dst=vgpr(rgV), shiftHex=hex(log2MfmaN), src=vgpr(self.laneId),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        if self.geom.wgM > 1:
            waveMv = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0WaveM")
            self._computeWaveM(_writePartialsFree0Mod, waveMv)
            _writePartialsFree0Mod.add(VOrB32(dst=vgpr(rgV), src0=vgpr(rgV), src1=vgpr(waveMv),
                              comment="selV = rowGroup | waveM (zero iff both zero)"))
            self.writer.vgprPool.checkIn(waveMv)
        _writePartialsFree0Mod.add(VCmpEQU32(dst=sgpr(self.laneMaskSgpr, self.geom.laneSgprCount), src0=0, src1=vgpr(rgV),
                             comment="laneMask: rowGroup==0 && waveM==0"))
        self.writer.vgprPool.checkIn(rgV)
        ntilesV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0NTiles")
        # n_d = ceil(SizesFree0 / MT0).
        _writePartialsFree0Mod.addComment1("compute n_d = ceil(SizesFree0 / MT0).")
        with self.writer.allocTmpSgpr(1, tag="pRMS_wF0NTilesS") as ntilesS:
            _writePartialsFree0Mod.add(SAddU32(dst=sgpr(ntilesS.idx), src0=sgpr("SizesFree+0"),
                               src1=self.geom.macroTile0 - 1,
                               comment=f"N_hidden + MT0-1 (MT0={self.geom.macroTile0})"))
            mt0 = self.geom.macroTile0
            if mt0 & (mt0 - 1) == 0:
                _writePartialsFree0Mod.add(SLShiftRightB32(dst=sgpr(ntilesS.idx), shiftHex=hex(mt0.bit_length() - 1),
                                           src=sgpr(ntilesS.idx),
                                           comment=f"n_d = ceil(SizesFree0 / MT0={mt0})"))
            else:
                p = (mt0 - 1).bit_length()
                magic = -(-(1 << (32 + p - 1)) // mt0)
                magic, postShift = magic & 0xFFFFFFFF, p - 1
                _writePartialsFree0Mod.add(SMulHIU32(dst=sgpr(ntilesS.idx), src0=sgpr(ntilesS.idx), src1=hex(magic),
                                     comment=f"n_d magic mul (divisor={mt0})"))
                if postShift:
                    _writePartialsFree0Mod.add(SLShiftRightB32(dst=sgpr(ntilesS.idx), shiftHex=hex(postShift),
                                               src=sgpr(ntilesS.idx),
                                               comment=f"n_d >> {postShift} (magic post-shift)"))
            _writePartialsFree0Mod.add(VMovB32(dst=vgpr(ntilesV), src=sgpr(ntilesS.idx), comment="ntilesV = n_d"))
        tokenBase = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0TokenBase")
        _writePartialsFree0Mod.add(VLShiftRightB32(dst=vgpr(tokenBase), shiftHex=hex(self.geom.log2ElemBytes),
                                   src=vgpr(self.colByte),
                                   comment="tokenBase = colByte >> log2ElemBytes."))
        _writePartialsFree0Mod.add(SAndSaveExecB64(dst=sgpr(self.savedExec, lsc), src=sgpr(self.laneMaskSgpr, lsc),
                                   comment="save exec; set exec = writing-lane mask"))
        # Strength-reduce token*n_d across the n loop: token advances by mfmaN each
        # step, so token*n_d advances by the loop-invariant stride mfmaN*n_d. This
        # replaces the per-n multiply with a single add.
        accumV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0Accum")
        # token*n_d uses 32-bit VMulLOU32; assumes token*n_d < 2^32.
        _writePartialsFree0Mod.add(VMulLOU32(dst=vgpr(accumV), src0=vgpr(ntilesV), src1=vgpr(tokenBase),
                             comment="accum = tokenBase * n_d"))
        strideV = None
        if self.geom.mmaN > 1:
            strideV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0Stride")
            _writePartialsFree0Mod.add(VMulLOU32(dst=vgpr(strideV), src0=self.geom.mfmaN, src1=vgpr(ntilesV),
                                 comment=f"stride = mfmaN({self.geom.mfmaN}) * n_d"))
        # Pre-compute byteAddr = (token*n_d + WG0) * 4 once, then stride by stride4 per n.
        _writePartialsFree0Mod.add(VAddU32(vgpr(globalAddr), vgpr(accumV), sgpr("WorkGroup0"),
                           comment="token*n_d + WorkGroup0 (n=0)"))
        _writePartialsFree0Mod.add(VLShiftLeftB32(dst=vgpr(globalAddr), shiftHex=hex(2), src=vgpr(globalAddr),
                                  comment="byteAddr = (token*n_d + WG0) * 4"))
        if strideV is not None:
            _writePartialsFree0Mod.add(VLShiftLeftB32(dst=vgpr(strideV), shiftHex=hex(2), src=vgpr(strideV),
                                      comment="stride4 = stride * 4"))
        for n in range(self.geom.mmaN):
            _writePartialsFree0Mod.add(BufferStoreB32(src=vgpr(self.partials + n), vaddr=vgpr(globalAddr),
                                      saddr=sgpr(partialSrd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"partialBuf[token+n*{self.geom.mfmaN}, WG0] = Σx² (n={n})"))
            if n < self.geom.mmaN - 1:
                _writePartialsFree0Mod.add(VAddU32(vgpr(globalAddr), vgpr(globalAddr), vgpr(strideV),
                                   comment=f"byteAddr += stride4 (advance to n={n + 1})"))
        _writePartialsFree0Mod.add(SWaitCnt(vscnt=0, comment="wait partialBuf stores"))
        _writePartialsFree0Mod.add(SMovB64(dst=EXEC(), src=sgpr(self.savedExec, lsc), comment="restore exec mask"))
        if strideV is not None:
            self.writer.vgprPool.checkIn(strideV)
        self.writer.vgprPool.checkIn(accumV)
        self.writer.vgprPool.checkIn(tokenBase)
        self.writer.vgprPool.checkIn(ntilesV)
        _rmsModule.add(_writePartialsFree0Mod)
        self.writer.vgprPool.checkIn(globalAddr)
        _rmsSgprPool.checkIn(partialSrd)
        module.add(_rmsModule)
        sgprPool = self.writer.sgprPool
        vgprPool = self.writer.vgprPool
        sgprPool.checkIn(self.gammaM0Base)
        sgprPool.checkIn(self.gammaSoffsetSgpr)
        if self.pairUpperLaneMask is not None:
            sgprPool.checkIn(self.pairUpperLaneMask)
        if self.pairLowerLaneMask is not None:
            sgprPool.checkIn(self.pairLowerLaneMask)
        if self.mxSrd is not None:
            sgprPool.checkIn(self.mxSrd)
        sgprPool.checkIn(self.laneMaskSgpr)
        sgprPool.checkIn(self.savedExec)
        sgprPool.checkIn(self.gammaSrd)
        if self.resSrd is not None:
            sgprPool.checkIn(self.resSrd)
        vgprPool.checkIn(self.gammaLdsReadAddr)
        vgprPool.checkIn(self.gammaDtlVaddr)
        if self.vPermAddr is not None:
            vgprPool.checkIn(self.vPermAddr)
        if self.waveIdV is not None:
            vgprPool.checkIn(self.waveIdV)
        vgprPool.checkIn(self.partials)
        vgprPool.checkIn(self.nhBase)
        vgprPool.checkIn(self.wgRowBase)
        vgprPool.checkIn(self.rowGroupOff)
        vgprPool.checkIn(self.rowGroup)
        vgprPool.checkIn(self.col)
        vgprPool.checkIn(self.colByte)
        vgprPool.checkIn(self.laneId)
        return module




    def _computeWaveM(self, module, dst: int) -> None:
        # Callers with wgM > 1 can use self.waveIdV, which emit() caches.
        module.addComment1("compute waveM = waveId mod wgM.")
        module.add(VAndB32(dst=vgpr(dst), src0=vgpr(self.waveIdV), src1=self.geom.wgM - 1,
                           comment=f"waveM = waveId % {self.geom.wgM}"))

















    def _mulVgprBySgprConst(self, module, dstVgpr: int, sgprName: str,
                             const: int, comment: str) -> None:
        """Emit dstVgpr = sgpr(sgprName) * const via a full-rate shift for pow2, else a literal mul."""
        module.addComment1("dstVgpr = sgpr * const (shift or mul).")
        if const > 0 and (const & (const - 1)) == 0:
            module.add(VLShiftLeftB32(dst=vgpr(dstVgpr), shiftHex=hex(int(math.log2(const))),
                                      src=sgpr(sgprName), comment=comment))
            return
        # v_mul_lo_u32 is VOP3 and rejects literal operands in any source position.
        # Materialize the constant in a temporary SGPR, then move the input SGPR to
        # the destination VGPR and multiply using the SGPR constant.
        sTmp = self.writer.sgprPool.checkOut(1, tag="mulBySgprConst")
        module.add(SMovB32(dst=sgpr(sTmp), src=const, comment=f"load {const} into SGPR."))
        module.add(VMovB32(dst=vgpr(dstVgpr), src=sgpr(sgprName), comment=comment))
        module.add(VMulLOU32(dst=vgpr(dstVgpr), src0=vgpr(dstVgpr), src1=sgpr(sTmp), comment=comment))
        self.writer.sgprPool.checkIn(sTmp)


    def _shiftOrMulVgprConst(self, module, dst: int, srcVgpr: int,
                              const: int, comment: str) -> None:
        """Emit dst = srcVgpr * const via a full-rate shift for pow2, else a literal mul."""
        module.addComment1("dst = srcVgpr * const (shift or mul).")
        if const > 0 and (const & (const - 1)) == 0:
            module.add(VLShiftLeftB32(dst=vgpr(dst), shiftHex=hex(int(math.log2(const))),
                                      src=vgpr(srcVgpr), comment=comment))
            return
        # v_mul_lo_u32 is VOP3 and rejects literal operands in any source position.
        # Materialize the constant in a temporary SGPR and multiply.
        sTmp = self.writer.sgprPool.checkOut(1, tag="mulByVgprConst")
        module.add(SMovB32(dst=sgpr(sTmp), src=const, comment=f"load {const} into SGPR."))
        module.add(VMulLOU32(dst=vgpr(dst), src0=vgpr(srcVgpr), src1=sgpr(sTmp), comment=comment))
        self.writer.sgprPool.checkIn(sTmp)










