# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MegaFusedEpilogue: single-pass fused ResidualAdd + PartialRMS + MXFP8Quant.

Ordering invariant enforced by the fused element loop for each (m, n, k):
  H            = acc + residual          -- residual add before gamma
  rmsPartials += H * H                   -- square pre-gamma H for RMS
  ResidualOut <- bf16(H)                 -- store pre-gamma value
  acc          = H * gamma               -- scale after squaring and store
  blkAmax[j]  = max(blkAmax, |H*gamma|) -- inline MXFP8 amax fold

Gamma is loaded once per quant-tile qi (wide dwordx2 or scalar) and reused
across all N-groups.  Residual elements are loaded wide (one BufferLoadB64 per
4-element chunk) when useWideResidual is set; software OOB masking corrects
elements that straddle N_hidden boundaries.

Sub-emitter roles:
  SubtileResidualAddEmitter -- address helpers and geometry constants.
  SubtilePartialRMSEmitter  -- gamma-load helpers, rmsSum reduction, partial-buf write.
  SubtileMXFP8QuantEmitter  -- streaming context and per-group quantisation helpers.
"""

import math

from rocisa.code import Label, Module
from rocisa.container import ContinuousRegister, EXEC, MUBUFModifiers, sgpr, vgpr
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadB32,
    BufferLoadB64,
    BufferStoreB16,
    BufferStoreB64,
    SAndB64,
    SAndN2B32,
    SCBranchSCC0,
    SMulI32,
    SMovB64,
    SWaitCnt,
    VAddF32,
    VAddPKF32,
    VAddU32,
    VAndB32,
    VCmpLtU32,
    VCndMaskB32,
    VCvtPkF32toBF16,
    VFmaF32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMaxF32,
    VMovB32,
    VMulF32,
    VMulLOU32,
    VMulPKF32,
)

from .SubtileResidualAddEmit import SubtileResidualAddEmitter
from .SubtilePartialRMSEmit import SubtilePartialRMSEmitter
from .SubtileDynamicQuant import SubtileMXFP8QuantEmitter


class MegaFusedCtx:
    """Holds geometry constants and shared register indices derived from the kernel.

    Centralising kernel-parameter arithmetic here keeps the free functions
    stateless with respect to kernel parsing and avoids re-deriving the same
    tiling arithmetic in each helper.
    """

    def _initGeometry(self, kernel, mx, useMxfp8):
        """Derive tile geometry constants from kernel parameters."""
        mfmaM    = kernel["MatrixInstM"]
        mfmaN    = kernel["MatrixInstN"]
        waveSize = kernel["WavefrontSize"]
        wg       = kernel["MIWaveGroup"]
        wgM, wgN = wg[0], wg[1]
        self.mfmaM       = mfmaM
        self.mfmaN       = mfmaN
        self.rowsPerLane = (mfmaM * mfmaN) // waveSize
        self.mmaN        = (kernel["MacroTile1"] // mfmaN) // wgN
        self.wgM         = wgM
        # streamGroup controls peak VGPR pressure in the N-sweep.
        self.streamGroup = 4
        if useMxfp8:
            # tilesPerBlockM is live only on the subColQuant path.
            self.tilesPerBlockM = mx.tilesPerBlockM if mx.subColQuant else 2
            self.nQTilesM       = mx.nQTilesM
            self.streamGroup    = mx.streamGroup if mx.subColQuant else 4
            return
        # No MXFP8: sweep every tile row in blocks matching the gamma-load batching.
        mmaM = (kernel["MacroTile0"] // mfmaM) // wgM
        self.tilesPerBlockM = 2 if mmaM % 2 == 0 else 1
        self.nQTilesM       = mmaM // self.tilesPerBlockM

    def __init__(self, writer, kernel, res, rms, mx, useMxfp8):
        self.writer = writer
        self.kernel = kernel
        self.res    = res
        self.rms    = rms
        self.mx     = mx
        self.useMxfp8 = useMxfp8
        self._initGeometry(kernel, mx, useMxfp8)

        # Shared VGPRs — allocated for the whole emission by _allocSharedRegs.
        self.laneId      = None
        self.colByte     = None
        self.col         = None
        self.rowGroup    = None
        self.rowGroupOff = None
        self.wgRowBase   = None
        self.nhBase      = None
        self.rmsSum      = None  # persistent ΣH² bank (one f32 per free1 MMA column).
        self.waveIdV     = None

        # Shared SGPRs — kept live until _freeSharedRegs so helpers can use them inline.
        self.resSrd    = None
        self.gammaSrd  = None
        self.savedExec = None
        self.laneMask  = None
        self.mxSrd     = None

        # Residual scratch — allocated in _beginResidualScratch, freed in _endResidualScratch.
        self.resTokenBase   = None
        self.resRowByteBase = None
        self.resAddr        = None
        self.resOobV        = None
        self.resOobMask     = None
        # Per-N-column ResidualOut row base = token_n * N_hidden (element index),
        # precomputed once per n and reused across all mi tiles and straddle-fallback k.
        self.roRowBase      = None


class SubtileMegaFusedEmitter:
    """Emit the MegaFusedEpilogue for the Subtile gfx950 kernel."""

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        # MXFP8 dynamic quant is optional; without it the fused epilogue emits bf16 D.
        self.useMxfp8 = kernel.get("DQuantType") == "MXFP8"
        self.residualEmitter = SubtileResidualAddEmitter(writer, kernel)
        # ResidualAdd drains the kernarg pointer first, so PartialRMS must not drain it again.
        self.rmsEmitter = SubtilePartialRMSEmitter(writer, kernel, kernargDrained=True)
        # The quant emitter reads _DQuantSize0/1, which are absent without MXFP8.
        self.quantEmitter = SubtileMXFP8QuantEmitter(writer, kernel) if self.useMxfp8 else None

    def _buildCtx(self) -> MegaFusedCtx:
        """Construct the shared context from the sub-emitter instances."""
        return MegaFusedCtx(self.writer, self.kernel, self.residualEmitter,
                            self.rmsEmitter, self.quantEmitter, self.useMxfp8)

    @staticmethod
    def _isPackPair(a, b):
        """True when a,b are a consecutive even-aligned VGPR pair for packed VALU."""
        return (a % 2 == 0) and (b == a + 1)

    def _allocSharedRegs(self, ctx) -> None:
        """Check out shared VGPRs that stay live for the whole emission.

        SGPRs for SRDs, exec, and lane mask are allocated in _buildAndFreeSrds and
        freed in _freeSharedRegs so they remain live across the fused element loop.
        """
        vgprPool = ctx.writer.vgprPool
        ctx.laneId      = vgprPool.checkOut(1, tag="mf_laneId")
        ctx.colByte     = vgprPool.checkOut(1, tag="mf_colByte")
        ctx.col         = vgprPool.checkOut(1, tag="mf_col")
        ctx.rowGroup    = vgprPool.checkOut(1, tag="mf_rowGroup")
        ctx.rowGroupOff = vgprPool.checkOut(1, tag="mf_rowGroupOff")
        ctx.wgRowBase   = vgprPool.checkOut(1, tag="mf_wgRowBase")
        ctx.nhBase      = vgprPool.checkOut(1, tag="mf_nhBase")
        ctx.rmsSum      = vgprPool.checkOut(ctx.mmaN, tag="mf_rmsSum")
        if ctx.wgM > 1:
            ctx.waveIdV = vgprPool.checkOut(1, tag="mf_waveIdV")

    def _freeSharedRegs(self, ctx) -> None:
        """Return shared VGPRs and SGPRs to their pools in reverse allocation order."""
        sgprPool = ctx.writer.sgprPool
        vgprPool = ctx.writer.vgprPool
        # Free SGPRs kept live across the fused loop.
        if ctx.mxSrd is not None:
            sgprPool.checkIn(ctx.mxSrd)
        sgprPool.checkIn(ctx.laneMask)
        sgprPool.checkIn(ctx.savedExec)
        sgprPool.checkIn(ctx.gammaSrd)
        if ctx.resSrd is not None:
            sgprPool.checkIn(ctx.resSrd)
        # Free VGPRs in reverse allocation order.
        if ctx.waveIdV is not None:
            vgprPool.checkIn(ctx.waveIdV)
        vgprPool.checkIn(ctx.rmsSum)
        vgprPool.checkIn(ctx.nhBase)
        vgprPool.checkIn(ctx.wgRowBase)
        vgprPool.checkIn(ctx.rowGroupOff)
        vgprPool.checkIn(ctx.rowGroup)
        vgprPool.checkIn(ctx.col)
        vgprPool.checkIn(ctx.colByte)
        vgprPool.checkIn(ctx.laneId)

    def _setupShared(self, ctx) -> Module:
        """Emit shared setup: drain waits, lane arithmetic, colByte, col, rowGroup, SRDs.

        SGPRs for SRDs, exec, and lane mask are allocated here and remain live
        until _freeSharedRegs to avoid re-building them per tile row.
        """
        module = Module("MegaFused shared setup")
        module.add(SWaitCnt(kmcnt=0, comment="drain kernarg s_loads before reading kernel args."))
        module.add(SWaitCnt(vlcnt=0, comment="drain GEMM vector-memory before AGPR reuse."))
        res = ctx.res
        mfmaN, waveSize = res.mfma_n, res.waveSize
        log2N = int(math.log2(mfmaN))
        module.add(VAndB32(dst=vgpr(ctx.laneId), src0=vgpr("Serial"), src1=waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)."))
        if ctx.wgM > 1:
            waveIdTmp = ctx.writer.vgprPool.checkOutAligned(2, 2, tag="mf_waveIdDiv")
            module.add(vectorStaticDivide(ctx.waveIdV, "Serial", waveSize,
                                         ContinuousRegister(waveIdTmp, 2),
                                         comment="waveId = Serial / waveSize."))
            ctx.writer.vgprPool.checkIn(waveIdTmp)
        # col is the raw free1 column index used by the MXScale path.
        module.add(VAndB32(dst=vgpr(ctx.col), src0=vgpr(ctx.laneId), src1=mfmaN - 1,
                           comment="col = laneId & (mfmaN-1)."))
        # colByte encodes the token index as col * elemBytes; wave/wg offsets added below.
        module.add(VLShiftLeftB32(dst=vgpr(ctx.colByte), shiftHex=hex(res.log2ElemBytes),
                                  src=vgpr(ctx.col), comment="colByte = col * elemBytes."))
        module.add(VLShiftRightB32(dst=vgpr(ctx.rowGroup), shiftHex=hex(log2N),
                                   src=vgpr(ctx.laneId), comment="rowGroup = laneId >> log2(mfmaN)."))
        res._addWaveNColByte(module, ctx.colByte)
        with ctx.writer.allocTmpSgpr(1, tag="mf_wg1ColByte") as wg1S:
            wg1Bytes = res.macro_tile1 * res.elemBytes
            module.add(SMulI32(dst=sgpr(wg1S.idx), src0=sgpr("WorkGroup1"), src1=wg1Bytes,
                               comment=f"wg1ColByte = WorkGroup1 * MT1*elemBytes ({wg1Bytes})."))
            module.add(VAddU32(dst=vgpr(ctx.colByte), src0=vgpr(ctx.colByte), src1=sgpr(wg1S.idx),
                               comment="colByte += WorkGroup1 * MT1 * elemBytes."))
        self._buildAndFreeSrds(ctx, module)
        return module

    def _buildAndFreeSrds(self, ctx, module) -> None:
        """Allocate shared SRD SGPRs and emit build instructions.

        SRDs are kept live across the entire fused element loop so helpers such as
        _subColStoreGroup can reference them without re-building per tile.
        """
        sgprPool = ctx.writer.sgprPool
        lsc = ctx.writer.states.laneSGPRCount
        if ctx.res.residualAdd:
            ctx.resSrd = sgprPool.checkOutAligned(4, 4, tag="mf_resSrd", preventOverflow=False)
            ctx.res._buildResidualSrd(module, ctx.resSrd)
        # ResidualOut aliases the (beta=0 unused) SrdC named SGPR, so its descriptor
        # must be built here; otherwise bf16(H) stores target the stale C buffer.
        if ctx.res.storeBf16D:
            ctx.res.residualOutSrd = ctx.writer.sgprs["SrdResidualOut"]
            ctx.res._buildResidualOutSrd(module, ctx.res.residualOutSrd)
        ctx.gammaSrd  = sgprPool.checkOutAligned(4, 4, tag="mf_gammaSrd", preventOverflow=False)
        ctx.savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="mf_savedExec", preventOverflow=False)
        ctx.laneMask  = sgprPool.checkOutAligned(lsc, lsc, tag="mf_laneMask", preventOverflow=False)
        ctx.rms._buildBufferSrd(module, ctx.gammaSrd, "RMSNormGamma", "gamma")
        # MXScale SRD is only needed when MXFP8 dynamic quant is active.
        if ctx.useMxfp8:
            ctx.mxSrd = sgprPool.checkOutAligned(4, 4, tag="mf_mxSrd", preventOverflow=False)
            ctx.mx._buildBufferSrd(module, ctx.mxSrd, "MXScale", "mxScale")

    def _bindSubEmitters(self, ctx) -> None:
        """Assign shared register indices to sub-emitter attributes.

        residualOutSrd is bound and populated in _buildAndFreeSrds, so it is not
        set here.
        """
        ctx.res.laneId       = ctx.laneId
        ctx.res.colByte      = ctx.colByte
        ctx.res.waveIdV      = ctx.waveIdV
        ctx.res.resSrd       = ctx.resSrd
        ctx.rms.laneId       = ctx.laneId
        ctx.rms.colByte      = ctx.colByte
        ctx.rms.waveIdV      = ctx.waveIdV
        ctx.rms.partials     = ctx.rmsSum
        ctx.rms.gammaSrd     = ctx.gammaSrd
        ctx.rms.savedExec    = ctx.savedExec
        ctx.rms.laneMaskSgpr = ctx.laneMask

    def _initRmsSum(self, ctx) -> Module:
        """Zero-initialise the rmsSum bank once before the fused sweep.

        Zeroing upfront avoids a first-element VMulF32 / is-first-element branch
        in the inner loop: VFmaF32 then works uniformly across all iterations.
        """
        module = Module("MegaFused initRmsSum")
        for n in range(ctx.mmaN):
            module.add(VMovB32(dst=vgpr(ctx.rmsSum + n), src=0,
                               comment=f"rmsSum[{n}] = 0.0f."))
        return module

    def _loadGammaBlockWide(self, module, ctx, gammaBank, qi, gammaByteV, mBaseV) -> None:
        """Issue wide (dwordx2) gamma loads for all tilesPerBlockM rows of quant-tile qi.

        gammaBank is 2-aligned so gammaBank + mi*rpl is always 2-aligned for dwordx2.
        """
        rms = ctx.rms
        rpl = ctx.rowsPerLane
        tpb = ctx.tilesPerBlockM
        for mi in range(tpb):
            m = qi * tpb + mi
            rms._free0RowPos(module, ctx.nhBase, ctx.wgRowBase, ctx.rowGroupOff, m, 0, mBaseV)
            module.add(VLShiftLeftB32(dst=vgpr(gammaByteV), shiftHex=hex(rms.gammaLog2Bytes),
                                      src=vgpr(ctx.nhBase),
                                      comment=f"gammaByte = nhBase * gammaBytes (mi={mi})."))
            module.add(BufferLoadB64(vgpr(gammaBank + mi * rpl, 2), vgpr(gammaByteV),
                                     sgpr(ctx.gammaSrd, 4), 0, MUBUFModifiers(offen=True),
                                     comment=f"gamma[nhBase..+3] dwordx2 (m={m})."))
        module.add(SWaitCnt(vlcnt=0, comment="wait wide gamma loads."))
        for mi in range(tpb):
            rms._convertGammaChunkBf16(module, gammaBank + mi * rpl)

    def _loadGammaBlockScalar(self, module, ctx, gammaBank, qi, gammaByteV, mBaseV) -> None:
        """Issue one scalar buffer_load per gamma element for quant-tile qi."""
        rms = ctx.rms
        rpl = ctx.rowsPerLane
        tpb = ctx.tilesPerBlockM
        for mi in range(tpb):
            m = qi * tpb + mi
            rms._free0RowPos(module, ctx.nhBase, ctx.wgRowBase, ctx.rowGroupOff, m, 0, mBaseV)
            for k in range(rpl):
                r = rms._addImmU32(module, gammaByteV, ctx.nhBase, k, mBaseV,
                                   f"gammaIdx = nhBase + {k} (mi={mi},k={k}).")
                module.add(VLShiftLeftB32(dst=vgpr(gammaByteV), shiftHex=hex(rms.gammaLog2Bytes),
                                          src=vgpr(r),
                                          comment="gammaByte = gammaIdx * gammaBytes."))
                rms._issueSideLoad(module, gammaBank + mi * rpl + k, gammaByteV, ctx.gammaSrd,
                                   f"gamma[m={m},k={k}].", dtype=rms.gammaType)
        module.add(SWaitCnt(vlcnt=0, comment="wait gamma loads."))
        for mi in range(tpb):
            for k in range(rpl):
                rms._convertSideElem(module, gammaBank + mi * rpl + k,
                                     f"gamma->fp32 (mi={mi},k={k}).", dtype=rms.gammaType)

    def _loadGammaBlock(self, ctx, gammaBank, qi) -> Module:
        """Load and convert gamma for the tilesPerBlockM rows of quant-tile qi.

        Gamma is per free0 row and independent of the free1 (N) sweep, so it is
        loaded once per qi and reused across all N-groups.
        """
        module = Module(f"MegaFused loadGammaBlock qi={qi}")
        gammaByteV = ctx.writer.vgprPool.checkOut(1, tag="mf_gammaByte")
        mBaseV     = ctx.writer.vgprPool.checkOut(1, tag="mf_gammaM")
        if ctx.rms.useWideGamma:
            self._loadGammaBlockWide(module, ctx, gammaBank, qi, gammaByteV, mBaseV)
        else:
            self._loadGammaBlockScalar(module, ctx, gammaBank, qi, gammaByteV, mBaseV)
        ctx.writer.vgprPool.checkIn(mBaseV)
        ctx.writer.vgprPool.checkIn(gammaByteV)
        return module

    def _beginResidualScratch(self, module, ctx) -> None:
        """Allocate per-element residual scratch VGPRs and compute invariants.

        Scratch registers are kept live across the entire N-sweep and freed in
        _endResidualScratch. resOobV and resTokenBase are needed for both the
        residual load path and the inline bf16 store path.
        """
        res = ctx.res
        writer = ctx.writer
        ctx.resTokenBase   = writer.vgprPool.checkOut(1, tag="mf_resTokenBase")
        ctx.resRowByteBase = writer.vgprPool.checkOut(1, tag="mf_resRowByteBase")
        ctx.resAddr        = writer.vgprPool.checkOut(1, tag="mf_resAddr")
        ctx.resOobV        = writer.vgprPool.checkOut(1, tag="mf_resOobV")
        ctx.resOobMask     = writer.sgprPool.checkOutAligned(
            res.lane_sgpr_count, res.lane_sgpr_count, tag="mf_resOobMask", preventOverflow=False)
        # resTokenBase = colByte >> log2ElemBytes; used by residual loads and bf16 store.
        module.add(VLShiftRightB32(dst=vgpr(ctx.resTokenBase),
                                   shiftHex=hex(res.log2ElemBytes),
                                   src=vgpr(ctx.colByte),
                                   comment="resTokenBase = colByte >> log2ElemBytes."))
        module.add(VMovB32(dst=vgpr(ctx.resOobV), src="BufferOOB",
                           comment="resOobV = BufferOOB (OOB loads return 0 / stores dropped)."))

    def _endResidualScratch(self, module, ctx) -> None:
        """Wait for pending ResidualOut stores and free residual scratch registers."""
        res = ctx.res
        writer = ctx.writer
        if res.storeBf16D:
            module.add(SWaitCnt(vscnt=0, comment="wait ResidualOut bf16 stores."))
        writer.sgprPool.checkIn(ctx.resOobMask)
        writer.vgprPool.checkIn(ctx.resOobV)
        writer.vgprPool.checkIn(ctx.resAddr)
        writer.vgprPool.checkIn(ctx.resRowByteBase)
        writer.vgprPool.checkIn(ctx.resTokenBase)

    def _computeBf16Addr(self, module, ctx, n, k, addrV, valV, nhByteV, nhMaskIdx) -> None:
        """Compute clamped byte address for ResidualOut[token_n, nhPos] at column (n, k).

        Token-OOB lanes are dropped by the ResidualOut SRD bounds, so no explicit
        token mask is applied here; only nhPos-straddle elements are clamped to
        BufferOOB to avoid aliasing the next token's row.
        """
        res = ctx.res
        lsc = res.lane_sgpr_count
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(1), src=vgpr(ctx.roRowBase),
                                  comment="base0 = roRowBase * 2 (bf16); token_n*N_hidden reused."))
        nh = res._addImmU32(module, nhByteV, ctx.nhBase, k, valV,
                            f"nhPos = nhBase + {k} (k={k}).")
        module.add(VCmpLtU32(dst=sgpr(nhMaskIdx, lsc), src0=vgpr(nh),
                             src1=sgpr("SizesFree+0"),
                             comment="nhInRange = nhPos < N_hidden."))
        module.add(VLShiftLeftB32(dst=vgpr(nhByteV), shiftHex=hex(1), src=vgpr(nh),
                                  comment="nhByte = nhPos * 2 (bf16)."))
        module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(nhByteV),
                           comment="byteAddr = base0 + nhByte."))
        module.add(VCndMaskB32(dst=vgpr(addrV), src0=vgpr(ctx.resOobV), src1=vgpr(addrV),
                               src2=sgpr(nhMaskIdx, lsc),
                               comment="clamp OOB when nhPos >= N_hidden."))

    def _computeResidualOutRowBaseAndMask(self, module, ctx, n, tokMaskSgpr) -> None:
        """Per-N-column setup: token(N) OOB mask and roRowBase = token_n * N_hidden.

        token_n is the free1 index owned by each lane and is constant across all m and k
        within the N-group, so both the mask and the row base are computed once per n and
        reused. Mirrors beta*C's GWB store, which computes its address base once rather than
        multiplying per tile. ctx.roRowBase must be checked out by the caller.
        """
        res = ctx.res
        lsc = res.lane_sgpr_count
        nOff = n * res.mfma_n
        tokV = ctx.writer.vgprPool.checkOut(1, tag="mf_roTokV")
        scratch = ctx.writer.vgprPool.checkOut(1, tag="mf_roTokScratch")
        r = res._addImmU32(module, tokV, ctx.resTokenBase, nOff, scratch,
                           f"token_n = resTokenBase + {nOff} (n={n}).")
        module.add(VCmpLtU32(dst=sgpr(tokMaskSgpr, lsc), src0=vgpr(r),
                             src1=sgpr("SizesFree+1"),
                             comment="tokenInRange = token_n < M_tokens (ResidualOut N mask)."))
        module.add(VMulLOU32(dst=vgpr(ctx.roRowBase), src0=sgpr("SizesFree+0"), src1=vgpr(r),
                             comment=f"roRowBase = token_n * N_hidden (n={n}); reused across m,k."))
        ctx.writer.vgprPool.checkIn(scratch)
        ctx.writer.vgprPool.checkIn(tokV)

    def _packResidualOutRow(self, module, ctx, srcRegs, packBank) -> None:
        """Pack rpl bf16(H) values into packBank (rpl/2 dwords, 2-aligned) for a dwordx2 store."""
        rpl = ctx.rowsPerLane
        for p in range(rpl // 2):
            module.add(VCvtPkF32toBF16(dst=vgpr(packBank + p),
                                       src0=vgpr(srcRegs[2 * p]), src1=vgpr(srcRegs[2 * p + 1]),
                                       comment=f"pack H[{2 * p}] lo16, H[{2 * p + 1}] hi16 -> bf16x2."))

    def _residualOutRowAddr(self, module, ctx, n, addrV, scratchV) -> None:
        """Compute byte address (roRowBase + nhBase) * 2 for the dwordx2 store.

        roRowBase = token_n * N_hidden is precomputed once per N-column
        (_computeResidualOutRowBaseAndMask), so no per-tile integer multiply is needed here.
        """
        module.add(VAddU32(dst=vgpr(addrV), src0=vgpr(ctx.roRowBase), src1=vgpr(ctx.nhBase),
                           comment="elemIdx = roRowBase + nhBase."))
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(1), src=vgpr(addrV),
                                  comment="byteAddr = elemIdx * 2 (bf16)."))

    def _storeResidualOutRow(self, module, ctx, srcRegs, tokMaskSgpr, m, n) -> None:
        """Store rpl bf16(H) to ResidualOut as one dwordx2 for nhidden-interior lanes.

        Interior lanes use BufferStoreB64 under the b64Safe exec mask (token-OOB lanes are dropped by the ResidualOut SRD bounds);
        straddling lanes fall back to per-element masked stores via _storeBf16ElemInline.
        A scalar SCC branch skips the fallback when no lane straddles (the common case
        for multiple-of-rpl N_hidden).  Exec is fully restored before returning.
        """
        res = ctx.res
        rpl = ctx.rowsPerLane
        assert rpl % 2 == 0, "rpl must be even for dwordx2 bf16 packing"
        lsc = res.lane_sgpr_count
        # gfx950 is wave64-only for this path; HasWave32 excludes gfx9,
        # _validateSubtileEpiloguePrereqs rejects non-gfx950.
        assert lsc == 2, "storeResidualOutRow hardcodes wave64 b64 exec ops"
        vgprPool = ctx.writer.vgprPool
        sgprPool = ctx.writer.sgprPool
        packBank = vgprPool.checkOutAligned(rpl // 2, 2, tag="mf_roPack")
        addrV    = vgprPool.checkOut(1, tag="mf_roAddr")
        scratchV = vgprPool.checkOut(1, tag="mf_roScratch")
        nhTopV   = vgprPool.checkOut(1, tag="mf_roNhTop")
        self._residualOutRowAddr(module, ctx, n, addrV, scratchV)
        self._packResidualOutRow(module, ctx, srcRegs, packBank)
        safe  = sgprPool.checkOutAligned(lsc, lsc, tag="mf_roSafe",     preventOverflow=False)
        saved = sgprPool.checkOutAligned(lsc, lsc, tag="mf_roSaveExec", preventOverflow=False)
        self._issueResidualOutWide(module, ctx, m, n, addrV, packBank,
                                   nhTopV, scratchV, safe, saved)
        self._issueResidualOutStraddle(module, ctx, tokMaskSgpr, m, n, srcRegs, safe, saved)
        sgprPool.checkIn(saved)
        sgprPool.checkIn(safe)
        vgprPool.checkIn(nhTopV)
        vgprPool.checkIn(scratchV)
        vgprPool.checkIn(addrV)
        vgprPool.checkIn(packBank)

    def _issueResidualOutWide(self, module, ctx, m, n, addrV, packBank,
                               nhTopV, scratchV, safeIdx, savedIdx) -> None:
        """Narrow exec to interior lanes and issue the dwordx2 store.

        safeIdx receives b64Safe (nhTop < N_hidden); it is consumed unchanged by
        _issueResidualOutStraddle to compute the straddle subset. Token-OOB lanes
        are silently dropped by the ResidualOut SRD bounds (no token mask folded here).
        savedIdx receives the pre-narrow full exec, restored by _issueResidualOutStraddle.
        """
        res = ctx.res
        rpl = ctx.rowsPerLane
        lsc = res.lane_sgpr_count
        nhTop = res._addImmU32(module, nhTopV, ctx.nhBase, rpl - 1, scratchV,
                               f"nhTop = nhBase + {rpl - 1}.")
        # b64Safe = interior lanes whose whole rpl-group is < N_hidden (no straddle).
        # Token-OOB lanes are dropped by the ResidualOut SRD bounds, so no token mask
        # is folded into the exec narrow here (relies on SRD OOB clamping).
        module.add(VCmpLtU32(dst=sgpr(safeIdx, lsc), src0=vgpr(nhTop),
                             src1=sgpr("SizesFree+0"),
                             comment="b64Safe = nhBase+rpl-1 < N_hidden (no straddle)."))
        # Save full exec with a plain s_mov, then set exec = b64Safe for the wide store.
        module.add(SMovB64(dst=sgpr(savedIdx, lsc), src=EXEC(),
                           comment="save full exec (plain s_mov, no exec RMW)."))
        module.add(SMovB64(dst=EXEC(), src=sgpr(safeIdx, lsc),
                           comment="exec = b64Safe (interior lanes); SRD drops token-OOB."))
        module.add(BufferStoreB64(src=vgpr(packBank, 2), vaddr=vgpr(addrV),
                                  saddr=sgpr(res.residualOutSrd, 4), soffset=0,
                                  mubuf=MUBUFModifiers(offen=True),
                                  comment=f"ResidualOut dwordx2 (m={m},n={n})."))

    def _issueResidualOutStraddle(self, module, ctx, tokMaskSgpr, m, n, srcRegs,
                                   safeIdx, savedIdx) -> None:
        """Set exec to the straddle subset and run the per-element fallback.

        safeIdx on entry holds the narrow mask from _issueResidualOutWide and is
        overwritten with the straddle mask (tokMask AND NOT b64Safe).  savedIdx holds
        the saved full exec; it is restored before returning so subsequent VALU runs
        with all lanes active.
        """
        res = ctx.res
        lsc = res.lane_sgpr_count
        # Straddle exec: tokMask AND NOT narrow (= tokMask AND NOT b64Safe).
        # Two SAndN2B32 reuse the safe register pair; SAndB64 sets SCC for the branch.
        module.add(SAndN2B32(dst=sgpr(safeIdx), src0=sgpr(tokMaskSgpr), src1=sgpr(safeIdx),
                             comment="straddle_lo = tokMask_lo & ~b64Safe_lo."))
        module.add(SAndN2B32(dst=sgpr(safeIdx + 1), src0=sgpr(tokMaskSgpr + 1),
                             src1=sgpr(safeIdx + 1),
                             comment="straddle_hi = tokMask_hi & ~b64Safe_hi."))
        module.add(SAndB64(dst=sgpr(safeIdx, lsc), src0=sgpr(safeIdx, lsc),
                           src1=sgpr(safeIdx, lsc),
                           comment="SCC = (straddle != 0); safe still holds straddle mask."))
        module.add(SMovB64(dst=EXEC(), src=sgpr(safeIdx, lsc),
                           comment="exec = straddle lanes (SCC unchanged by SMovB64)."))
        skipLabel = Label(ctx.writer.labels.getNameInc(f"mf_roStraddleEnd_m{m}n{n}"), "")
        module.add(SCBranchSCC0(labelName=skipLabel.getLabelName(),
                                comment="no straddle lanes -> skip per-element fallback."))
        for k in range(ctx.rowsPerLane):
            self._storeBf16ElemInline(module, ctx, srcRegs[k], m, n, k)
        module.add(skipLabel)
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedIdx, lsc),
                           comment="restore full exec before gamma/amax VALU."))

    def _storeBf16ElemInline(self, module, ctx, accReg, m, n, k) -> None:
        """Store bf16(accReg) to ResidualOut[token_n, nhidden_pos] with inline masking."""
        res = ctx.res
        lsc = res.lane_sgpr_count
        addrV   = ctx.writer.vgprPool.checkOut(1, tag="mf_bf16Addr")
        valV    = ctx.writer.vgprPool.checkOut(1, tag="mf_bf16Val")
        nhByteV = ctx.writer.vgprPool.checkOut(1, tag="mf_nhByte")
        with ctx.writer.allocTmpSgpr(lsc, tag="mf_nhMask") as nhMask:
            self._computeBf16Addr(module, ctx, n, k, addrV, valV, nhByteV, nhMask.idx)
            module.add(VCvtPkF32toBF16(dst=vgpr(valV), src0=vgpr(accReg),
                                        src1=vgpr(accReg),
                                        comment="H -> bf16 (low 16 bits)."))
            module.add(BufferStoreB16(src=vgpr(valV), vaddr=vgpr(addrV),
                                      saddr=sgpr(res.residualOutSrd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"ResidualOut bf16(H) (m={m},n={n},k={k})."))
        ctx.writer.vgprPool.checkIn(nhByteV)
        ctx.writer.vgprPool.checkIn(valV)
        ctx.writer.vgprPool.checkIn(addrV)

    def _issueResidualWide(self, module, ctx, m, n, burstBase) -> None:
        """Issue wide residual load(s) for tile (m, n); caller must SWaitCnt(vlcnt=0) after.

        ctx.nhBase holds wgRowBase + rowGroupOff + m*mfmaM (set by _free0RowPos).
        ctx.resRowByteBase holds token_n * N_hidden * residualBytes for this n.
        Issues one BufferLoadB64 (bf16) or BufferLoadB32 (fp8) per 4-element chunk.
        """
        res = ctx.res
        isBf16 = res.residualBytes == 2
        loadCls = BufferLoadB64 if isBf16 else BufferLoadB32
        chunkBytes = 4 << res.residualLog2Bytes
        nhByteV = ctx.writer.vgprPool.checkOut(1, tag="mf_wideNhByte")
        if isBf16:
            module.add(VLShiftLeftB32(dst=vgpr(nhByteV), shiftHex="0x1",
                                      src=vgpr(ctx.nhBase),
                                      comment=f"nhiddenByte = nhBase * 2 (m={m})."))
        else:
            module.add(VMovB32(dst=vgpr(nhByteV), src=vgpr(ctx.nhBase),
                               comment=f"nhiddenByte = nhBase (fp8, m={m})."))
        module.add(VAddU32(vgpr(ctx.resAddr), vgpr(ctx.resRowByteBase), vgpr(nhByteV),
                           comment=f"byteAddr = rowByteBase + nhiddenByte (m={m},n={n})."))
        ctx.writer.vgprPool.checkIn(nhByteV)
        for c in range(res.rows_per_lane // 4):
            if c == 0:
                addr = ctx.resAddr
            else:
                addr = ctx.writer.vgprPool.checkOut(1, tag="mf_wideChunkAddr")
                res._addImmU32(module, addr, ctx.resAddr, chunkBytes * c, ctx.resRowByteBase,
                               f"chunk byte offset {chunkBytes * c}.")
            dstBase = burstBase + 4 * c
            dst = vgpr(dstBase, 2) if isBf16 else vgpr(dstBase)
            module.add(loadCls(dst, vgpr(addr), sgpr(ctx.resSrd, 4), 0,
                               MUBUFModifiers(offen=True),
                               comment=f"R wide [4 residual] (m={m},n={n},c={c})."))
            if c > 0:
                ctx.writer.vgprPool.checkIn(addr)

    def _maskWideResidualOOB(self, module, ctx, burstBase) -> None:
        """Software-mask wide residual elements where nhBase+k >= N_hidden.

        Wide loads read rows_per_lane contiguous nhidden positions from nhBase.
        Elements straddling the N_hidden boundary alias the next token's row in
        memory instead of returning buffer-OOB zero; software masking corrects this.
        ctx.resAddr and ctx.resRowByteBase are reused as scratch (load is done).
        """
        res = ctx.res
        lsc = res.lane_sgpr_count
        for k in range(res.rows_per_lane):
            nhR = res._addImmU32(module, ctx.resAddr, ctx.nhBase, k, ctx.resRowByteBase,
                                 f"nhPos = nhBase + {k}.")
            module.add(VCmpLtU32(dst=sgpr(ctx.resOobMask, lsc), src0=vgpr(nhR),
                                 src1=sgpr("SizesFree+0"),
                                 comment=f"inRange = nhPos < N_hidden (k={k})."))
            module.add(VCndMaskB32(dst=vgpr(burstBase + k), src0=0,
                                   src1=vgpr(burstBase + k),
                                   src2=sgpr(ctx.resOobMask, lsc),
                                   comment=f"residual = inRange ? residual : 0 (k={k})."))

    def _issueResidualTile(self, module, ctx, m, n, burstBase, mBaseV) -> int:
        """Issue residual loads for tile (m, n) into burstBase; return #loads issued.

        No wait/convert here: loads are drained and converted later in the compute
        pass so the whole N-group's loads stay in flight together (GWB-style).
        Wide path: one BufferLoad per 4-element chunk. Scalar path: one per element.
        """
        res = ctx.res
        rpl = res.rows_per_lane
        if res.useWideResidual:
            ctx.rms._free0RowPos(module, ctx.nhBase, ctx.wgRowBase,
                                 ctx.rowGroupOff, m, 0, mBaseV)
            self._issueResidualWide(module, ctx, m, n, burstBase)
            return rpl // 4
        for k in range(rpl):
            res._residualElemAddr(module, ctx.resAddr, ctx.resRowByteBase,
                                  ctx.wgRowBase, ctx.rowGroupOff,
                                  ctx.resOobV, ctx.resOobMask, mBaseV, m, k)
            res._issueSideLoad(module, burstBase + k, ctx.resAddr, ctx.resSrd,
                               f"R[m={m},n={n},k={k}].", dtype=res.residualType)
        return rpl

    def _finishResidualTile(self, module, ctx, burstBase) -> None:
        """Convert (and, for wide loads, software-mask) an already-loaded residual tile.

        ctx.nhBase must hold this tile's row position (set by _free0RowPos in the
        compute pass) before calling, because the wide OOB mask reads nhBase.
        """
        res = ctx.res
        rpl = res.rows_per_lane
        if res.useWideResidual:
            if res.residualBytes == 2:
                res._convertResidualChunkBf16(module, burstBase)
            else:
                res._convertResidualChunkFp8(module, burstBase)
            self._maskWideResidualOOB(module, ctx, burstBase)
            return
        for k in range(rpl):
            res._convertSideElem(module, burstBase + k,
                                 f"residual->fp32 (k={k}).", dtype=res.residualType)

    def _pass1AccResRms(self, module, ctx, srcRegs, burstBase, m, n, rpl) -> None:
        """Fuse residual add and rmsSum accumulation: H = acc + R, rmsSum[n] += H²."""
        res = ctx.res
        if not res.residualAdd:
            for k in range(rpl):
                module.add(VFmaF32(dst=vgpr(ctx.rmsSum + n), src0=vgpr(srcRegs[k]),
                                   src1=vgpr(srcRegs[k]), src2=vgpr(ctx.rmsSum + n),
                                   comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k})."))
            return
        # Use packed add for aligned acc pairs; adds come before both squares so
        # the FMAs read the post-add values, matching the original per-element order.
        for k in range(0, rpl - rpl % 2, 2):
            sk0, sk1 = srcRegs[k], srcRegs[k + 1]
            if self._isPackPair(sk0, sk1):
                module.add(VAddPKF32(dst=vgpr(sk0, 2), src0=vgpr(sk0, 2),
                                     src1=vgpr(burstBase + k, 2),
                                     comment=f"H = acc + residual (packed k={k},{k+1})."))
            else:
                module.add(VAddF32(dst=vgpr(sk0), src0=vgpr(sk0),
                                   src1=vgpr(burstBase + k),
                                   comment=f"H = acc + residual (m={m},n={n},k={k})."))
                module.add(VAddF32(dst=vgpr(sk1), src0=vgpr(sk1),
                                   src1=vgpr(burstBase + k + 1),
                                   comment=f"H = acc + residual (m={m},n={n},k={k+1})."))
            module.add(VFmaF32(dst=vgpr(ctx.rmsSum + n), src0=vgpr(sk0),
                               src1=vgpr(sk0), src2=vgpr(ctx.rmsSum + n),
                               comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k})."))
            module.add(VFmaF32(dst=vgpr(ctx.rmsSum + n), src0=vgpr(sk1),
                               src1=vgpr(sk1), src2=vgpr(ctx.rmsSum + n),
                               comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k+1})."))
        # Defensive odd tail (rpl is even in practice).
        if rpl % 2 == 1:
            k = rpl - 1
            module.add(VAddF32(dst=vgpr(srcRegs[k]), src0=vgpr(srcRegs[k]),
                               src1=vgpr(burstBase + k),
                               comment=f"H = acc + residual (m={m},n={n},k={k})."))
            module.add(VFmaF32(dst=vgpr(ctx.rmsSum + n), src0=vgpr(srcRegs[k]),
                               src1=vgpr(srcRegs[k]), src2=vgpr(ctx.rmsSum + n),
                               comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k})."))

    def _amaxAndWriteAcc(self, module, ctx, sk, vgprTiles, blkAmaxJ, m, n, ki) -> None:
        """Fold |H*gamma| into blkAmax (MXFP8 only) and write sk back to the accumulator.

        Must be called once per k element after the multiply so the amax fold and
        accumulator writeback remain scalar (per-k) even when the multiply was packed.
        """
        if ctx.useMxfp8:
            module.add(VAndB32(dst=vgpr(ctx.mx._scAccTmp), src0=vgpr(sk),
                               src1=vgpr(ctx.mx._scAbsMask),
                               comment=f"|H*gamma| (m={m},n={n},k={ki})."))
            module.add(VMaxF32(dst=vgpr(blkAmaxJ), src0=vgpr(blkAmaxJ),
                               src1=vgpr(ctx.mx._scAccTmp),
                               comment="blkAmax = max(blkAmax, |H*gamma|)."))
        ctx.rms._writeAccFrom(module, sk, vgprTiles, m, n, ki,
                              f"write H*gamma back to acc (m={m},n={n},k={ki}).")

    def _pass3GammaAmax(self, module, ctx, srcRegs, vgprTiles, gammaBank, blkAmaxJ,
                        mi, m, n, rpl) -> None:
        """Apply gamma, fold |H*gamma| into blkAmax for MXFP8, and write the result back to acc.

        The gamma-scaled value is written to the accumulator so the deferred MXFP8
        tail can re-read it. The amax fold and writeAccFrom MUST stay scalar per-k;
        only the multiply is packed.
        """
        for k in range(0, rpl - rpl % 2, 2):
            sk0, sk1 = srcRegs[k], srcRegs[k + 1]
            # gammaBank is 2-aligned; when rpl % 2 == 0 (production gfx950 config),
            # mi*rpl+k is even so gk is 2-aligned by construction.
            gk = gammaBank + mi * rpl + k
            if self._isPackPair(sk0, sk1):
                module.add(VMulPKF32(dst=vgpr(sk0, 2), src0=vgpr(sk0, 2),
                                     src1=vgpr(gk, 2),
                                     comment=f"acc = H * gamma (packed k={k},{k+1})."))
            else:
                module.add(VMulF32(dst=vgpr(sk0), src0=vgpr(sk0), src1=vgpr(gk),
                                   comment=f"acc = H * gamma (m={m},n={n},k={k})."))
                module.add(VMulF32(dst=vgpr(sk1), src0=vgpr(sk1), src1=vgpr(gk + 1),
                                   comment=f"acc = H * gamma (m={m},n={n},k={k+1})."))
            self._amaxAndWriteAcc(module, ctx, sk0, vgprTiles, blkAmaxJ, m, n, k)
            self._amaxAndWriteAcc(module, ctx, sk1, vgprTiles, blkAmaxJ, m, n, k + 1)
        # Defensive odd tail (rpl is even in practice).
        if rpl % 2 == 1:
            k = rpl - 1
            acc = srcRegs[k]
            gammaReg = gammaBank + mi * rpl + k
            module.add(VMulF32(dst=vgpr(acc), src0=vgpr(acc), src1=vgpr(gammaReg),
                               comment=f"acc = H * gamma (m={m},n={n},k={k})."))
            self._amaxAndWriteAcc(module, ctx, acc, vgprTiles, blkAmaxJ, m, n, k)

    def _prologResidualLoads(self, module, ctx, resBank, mBaseV, qi, nBase, g):
        """Issue every residual load for the N-group into resBank so loads overlap.

        Returns (loadsCumulative, totalIssued): loadsCumulative[t] is the number
        of residual loads issued up to and including tile t (issue order equals
        compute order), which drives the per-tile decreasing vlcnt in the compute
        pass.
        """
        res = ctx.res
        rpl = ctx.rowsPerLane
        tpb = ctx.tilesPerBlockM
        loadsCumulative = []
        issued = 0
        if not res.residualAdd:
            return loadsCumulative, issued
        for j in range(g):
            n = nBase + j
            res._residualRowByteBase(module, ctx.resRowByteBase,
                                     ctx.resTokenBase, n, ctx.resAddr)
            for mi in range(tpb):
                m = qi * tpb + mi
                burstBase = resBank + (j * tpb + mi) * rpl
                issued += self._issueResidualTile(module, ctx, m, n, burstBase, mBaseV)
                loadsCumulative.append(issued)
        return loadsCumulative, issued

    def _computePassTile(self, module, ctx, vgprTiles, accBank, resBank, gammaBank,
                         blkAmax, loadsCumulative, totalIssued, mBaseV, tokMaskSgpr,
                         qi, nBase, j, mi, t) -> int:
        """Emit instructions for one (mi, j) tile in the compute pass; returns updated t."""
        res = ctx.res
        rms = ctx.rms
        rpl = ctx.rowsPerLane
        tpb = ctx.tilesPerBlockM
        m = qi * tpb + mi
        n = nBase + j
        bankBase = (j * tpb + mi) * rpl
        burstBase = (resBank + bankBase) if res.residualAdd else None
        coords = [(m, n, k) for k in range(rpl)]
        srcRegs = rms._readAccBurst(module, accBank + bankBase, vgprTiles,
                                    coords, f"acc m={m},n={n}.")
        rms._free0RowPos(module, ctx.nhBase, ctx.wgRowBase,
                         ctx.rowGroupOff, m, 0, mBaseV)
        if res.residualAdd:
            # Wait only for THIS tile's residual load; later tiles' loads
            # stay in flight (GWB decreasing-vlcnt schedule).
            remaining = totalIssued - loadsCumulative[t]
            module.add(SWaitCnt(vlcnt=remaining,
                                comment=f"wait residual tile {t}: vlcnt={totalIssued}-{loadsCumulative[t]}."))
            self._finishResidualTile(module, ctx, burstBase)
        self._pass1AccResRms(module, ctx, srcRegs, burstBase, m, n, rpl)
        if res.storeBf16D:
            self._storeResidualOutRow(module, ctx, srcRegs, tokMaskSgpr, m, n)
        blkAmaxJ = (blkAmax + n) if ctx.useMxfp8 else None
        self._pass3GammaAmax(module, ctx, srcRegs, vgprTiles, gammaBank, blkAmaxJ,
                             mi, m, n, rpl)
        return t + 1

    def _computePass(self, module, ctx, vgprTiles, accBank, resBank, gammaBank,
                     blkAmax, loadsCumulative, totalIssued, mBaseV, qi, nBase, g) -> None:
        """Drain residual loads per tile, then run residual add, bf16 store, rmsSum, gamma/amax.

        Uses a GWB decreasing-vlcnt schedule: each tile waits only for its own
        residual load so later tiles' loads stay in flight.
        """
        res = ctx.res
        lsc = res.lane_sgpr_count
        tpb = ctx.tilesPerBlockM
        t = 0
        for j in range(g):
            n = nBase + j
            # Compute the token(N) OOB mask once per column n; reused across all tpb tiles.
            tokMaskSgpr = None
            if res.storeBf16D:
                tokMaskSgpr = ctx.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mf_roTokMask",
                                                                   preventOverflow=False)
                ctx.roRowBase = ctx.writer.vgprPool.checkOut(1, tag="mf_roRowBase")
                self._computeResidualOutRowBaseAndMask(module, ctx, n, tokMaskSgpr)
            for mi in range(tpb):
                t = self._computePassTile(module, ctx, vgprTiles, accBank, resBank, gammaBank,
                                          blkAmax, loadsCumulative, totalIssued, mBaseV,
                                          tokMaskSgpr, qi, nBase, j, mi, t)
            if res.storeBf16D:
                ctx.writer.vgprPool.checkIn(ctx.roRowBase)
                ctx.roRowBase = None
                ctx.writer.sgprPool.checkIn(tokMaskSgpr)

    def _fusedElementLoop(self, module, ctx, vgprTiles, accBank, resBank, gammaBank,
                          blkAmax, qi, nBase, g) -> None:
        """Emit the fused loop as a GWB-style split: a load prolog then a compute pass.

        The prolog issues every residual load for the N-group into resBank so the
        loads overlap; the compute pass drains them per tile and runs residual add,
        bf16 store, rmsSum, and gamma/amax.
        """
        mBaseV = ctx.writer.vgprPool.checkOut(1, tag="mf_mBase")
        loadsCumulative, totalIssued = self._prologResidualLoads(
            module, ctx, resBank, mBaseV, qi, nBase, g)
        self._computePass(module, ctx, vgprTiles, accBank, resBank, gammaBank,
                          blkAmax, loadsCumulative, totalIssued, mBaseV, qi, nBase, g)
        ctx.writer.vgprPool.checkIn(mBaseV)

    def _initBlkAmax(self, ctx, blkAmax) -> Module:
        """Zero the per-qi persistent blkAmax bank (one f32 per absolute N column)."""
        module = Module("MegaFused initBlkAmax")
        for n in range(ctx.mmaN):
            module.add(VMovB32(dst=vgpr(blkAmax + n), src=0, comment=f"blkAmax[{n}] = 0."))
        return module

    def _fusedFrontHalf(self, ctx, vgprTiles, gammaBank, blkAmax, qi, nBase, g) -> Module:
        """Emit one N-group's element loop (residual add, bf16 store, rmsSum, gamma).

        For MXFP8 the gamma-scaled result is written back to the accumulator and
        |H*gamma| is folded into the persistent blkAmax bank; the group's MXFP8 tail
        is deferred and emitted later by _mxDeferredTail.
        """
        module = Module(f"MegaFused frontHalf qi={qi} nBase={nBase}")
        vgprPool = ctx.writer.vgprPool
        bankSize = g * ctx.tilesPerBlockM * ctx.rowsPerLane
        # 2-aligned so AGPR-staged acc pairs are packed-VALU eligible.
        accBank = vgprPool.checkOutAligned(bankSize, 2, tag="mf_accBank")
        # Whole-N-group residual bank so all residual loads overlap (2-aligned for
        # the wide BufferLoadB64 path).
        resBank = vgprPool.checkOutAligned(bankSize, 2, tag="mf_resBank") \
            if ctx.res.residualAdd else None
        self._fusedElementLoop(module, ctx, vgprTiles, accBank, resBank, gammaBank,
                               blkAmax, qi, nBase, g)
        if resBank is not None:
            vgprPool.checkIn(resBank)
        vgprPool.checkIn(accBank)
        return module

    def _mxDeferredTail(self, ctx, vgprTiles, blkAmax, qi, nBase, g) -> Module:
        """Deferred MXFP8 tail for one N-group: butterfly-reduce blkAmax, compute e8m0
        scales, re-read the accumulator to apply alpha*quantMult, and store MXScale bytes.

        blkAmax is the persistent mmaN bank; this group owns the slice [nBase, nBase+g).
        """
        module = Module(f"MegaFused mxDeferredTail qi={qi} nBase={nBase}")
        vgprPool = ctx.writer.vgprPool
        amaxSlice = blkAmax + nBase
        addrBf = vgprPool.checkOut(1, tag="mf_addrBf")
        tmpBf = vgprPool.checkOut(g, tag="mf_tmpBf")
        for r in range(2):
            ctx.mx._butterflyRound(module, addrBf, tmpBf, amaxSlice, g, ctx.laneId,
                                   ctx.mfmaN << r)
        vgprPool.checkIn(tmpBf)
        vgprPool.checkIn(addrBf)
        # Alpha fold: blkAmax[j] = |alpha * blkAmax[j]|, matching _streamSubColGroup.
        for j in range(g):
            module.add(VMulF32(dst=vgpr(amaxSlice + j), src0=vgpr(amaxSlice + j),
                               src1=sgpr("Alpha"), comment=f"blkAmax[{nBase + j}] *= alpha."))
            module.add(VAndB32(dst=vgpr(amaxSlice + j), src0=vgpr(amaxSlice + j),
                               src1=vgpr(ctx.mx._scAbsMask),
                               comment=f"blkAmax[{nBase + j}] = |alpha*blkAmax|."))
        # _computeSubColScales overwrites the slice with alpha*quantMult (the apply multiplier).
        scaleByteBank = ctx.mx._computeSubColScales(module, amaxSlice, qi, nBase, g)
        applyScratch = vgprPool.checkOut(ctx.rowsPerLane, tag="mf_applyScratch")
        mStart = qi * ctx.tilesPerBlockM
        mEnd = (qi + 1) * ctx.tilesPerBlockM
        ctx.mx._subColApplyFromAcc(module, vgprTiles, amaxSlice, applyScratch,
                                   mStart, mEnd, nBase, g)
        vgprPool.checkIn(applyScratch)
        ctx.mx._subColStoreGroup(module, ctx.mxSrd, scaleByteBank, ctx.col, ctx.rowGroup,
                                 ctx.savedExec, ctx.laneMask, qi, nBase, g)
        vgprPool.checkIn(scaleByteBank)
        return module

    def _reduceAndWriteRms(self, ctx) -> Module:
        """Finalise rmsSum: reduce across row groups and waves, then write to partialBuf.

        PartialBuf SRD is built here (deferred from setup to reduce SGPR pressure
        during the fused element loop) and freed before returning.
        """
        module = Module("MegaFused reduceAndWriteRms")
        sgprPool = ctx.writer.sgprPool
        partialSrd = sgprPool.checkOutAligned(4, 4, tag="mf_partialSrd", preventOverflow=False)
        ctx.rms._buildBufferSrd(module, partialSrd, "PartialBuf", "partialBuf")
        module.add(ctx.rms._reduceFree0())
        globalAddr = ctx.writer.vgprPool.checkOut(1, tag="mf_globalAddr")
        module.add(ctx.rms._writePartialsFree0(
            ctx.rmsSum, partialSrd, ctx.laneId, ctx.savedExec, ctx.laneMask,
            globalAddr, ctx.colByte))
        ctx.writer.vgprPool.checkIn(globalAddr)
        sgprPool.checkIn(partialSrd)
        return module

    def emit(self, vgprTiles):
        ctx = self._buildCtx()
        assert not ctx.useMxfp8 or ctx.mx.subColQuant, \
            "megaFused MXFP8 requires subColQuant (q1 < mfmaN)"
        module = Module("SubtileMegaFusedEpilogue")
        self._allocSharedRegs(ctx)
        module.add(self._setupShared(ctx))
        self._bindSubEmitters(ctx)

        # Compute per-wave row geometry used by gamma loads and residual addressing.
        ctx.rms._computeRowGroupOff(module, ctx.rowGroupOff)
        ctx.rms._computeFree0RowBase(module, ctx.wgRowBase)

        module.add(self._initRmsSum(ctx))

        # 2-aligned so wide BufferLoadB64 (gammaBank + mi*rpl, 2) is valid.
        gammaBank = ctx.writer.vgprPool.checkOutAligned(
            ctx.tilesPerBlockM * ctx.rowsPerLane, 2, tag="mf_gamma")
        self._beginResidualScratch(module, ctx)
        # The MXFP8 stream context must be live for the whole sweep; skip it without quant.
        if ctx.useMxfp8:
            ctx.mx._beginStreamContext(module)

        for qi in range(ctx.nQTilesM):
            module.add(self._loadGammaBlock(ctx, gammaBank, qi))
            blkAmax = None
            if ctx.useMxfp8:
                # Persist blkAmax across the qi's N-groups so the MXFP8 tails can be
                # deferred until every group's residual/RMS/gamma work is done.
                blkAmax = ctx.writer.vgprPool.checkOut(ctx.mmaN, tag="mf_blkAmax")
                module.add(self._initBlkAmax(ctx, blkAmax))
            for nBase in range(0, ctx.mmaN, ctx.streamGroup):
                g = min(ctx.streamGroup, ctx.mmaN - nBase)
                module.add(self._fusedFrontHalf(ctx, vgprTiles, gammaBank, blkAmax, qi, nBase, g))
            if ctx.useMxfp8:
                for nBase in range(0, ctx.mmaN, ctx.streamGroup):
                    g = min(ctx.streamGroup, ctx.mmaN - nBase)
                    module.add(self._mxDeferredTail(ctx, vgprTiles, blkAmax, qi, nBase, g))
                ctx.writer.vgprPool.checkIn(blkAmax)

        if ctx.useMxfp8:
            ctx.mx._endStreamContext()
        # One vscnt=0 drains both MXScale stores (from _subColStoreGroup) and ResidualOut stores.
        module.add(SWaitCnt(vscnt=0, comment="drain MXScale and ResidualOut stores."))
        self._endResidualScratch(module, ctx)
        ctx.writer.vgprPool.checkIn(gammaBank)

        module.add(self._reduceAndWriteRms(ctx))
        self._freeSharedRegs(ctx)
        return module
