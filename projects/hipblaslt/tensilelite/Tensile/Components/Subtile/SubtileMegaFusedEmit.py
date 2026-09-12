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

from rocisa.code import Module
from rocisa.container import ContinuousRegister, MUBUFModifiers, sgpr, vgpr
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadB32,
    BufferLoadB64,
    BufferStoreB16,
    BufferStoreB32,
    SMulI32,
    SWaitCnt,
    VAddF32,
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

    def _initGeometry(self, kernel, mx):
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
        # tilesPerBlockM is live only on the subColQuant path.
        self.tilesPerBlockM = mx.tilesPerBlockM if mx.subColQuant else 2
        self.nQTilesM    = mx.nQTilesM
        # streamGroup controls peak VGPR pressure in the N-sweep.
        self.streamGroup = mx.streamGroup if mx.subColQuant else 4

    def __init__(self, writer, kernel, res, rms, mx):
        self.writer = writer
        self.kernel = kernel
        self.res    = res
        self.rms    = rms
        self.mx     = mx
        self._initGeometry(kernel, mx)

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
        self.resBurst       = None
        self.resOobMask     = None


class SubtileMegaFusedEmitter:
    """Emit the MegaFusedEpilogue for the Subtile gfx950 kernel."""

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        self.residualEmitter = SubtileResidualAddEmitter(writer, kernel)
        # ResidualAdd drains the kernarg pointer first, so PartialRMS must not drain it again.
        self.rmsEmitter = SubtilePartialRMSEmitter(writer, kernel, kernargDrained=True)
        self.quantEmitter = SubtileMXFP8QuantEmitter(writer, kernel)

    def _buildCtx(self) -> MegaFusedCtx:
        """Construct the shared context from the three sub-emitter instances."""
        return MegaFusedCtx(self.writer, self.kernel,
                            self.residualEmitter, self.rmsEmitter, self.quantEmitter)

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
        ctx.mxSrd     = sgprPool.checkOutAligned(4, 4, tag="mf_mxSrd", preventOverflow=False)
        ctx.rms._buildBufferSrd(module, ctx.gammaSrd, "RMSNormGamma", "gamma")
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
        # BufferLoadB64 requires a 2-aligned VGPR destination.
        ctx.resBurst       = writer.vgprPool.checkOutAligned(ctx.rowsPerLane, 2, tag="mf_resBurst")
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
        writer.vgprPool.checkIn(ctx.resBurst)
        writer.vgprPool.checkIn(ctx.resOobV)
        writer.vgprPool.checkIn(ctx.resAddr)
        writer.vgprPool.checkIn(ctx.resRowByteBase)
        writer.vgprPool.checkIn(ctx.resTokenBase)

    def _computeBf16Addr(self, module, ctx, n, k, addrV, valV, nhByteV, tokMaskIdx, nhMaskIdx) -> None:
        """Compute clamped byte address for ResidualOut[token_n, nhPos] at column (n, k)."""
        res = ctx.res
        lsc = res.lane_sgpr_count
        nOff = n * res.mfma_n
        r = res._addImmU32(module, addrV, ctx.resTokenBase, nOff, valV,
                           f"token_n = resTokenBase + {nOff} (n={n}).")
        module.add(VCmpLtU32(dst=sgpr(tokMaskIdx, lsc), src0=vgpr(r),
                             src1=sgpr("SizesFree+1"),
                             comment="tokenInRange = token_n < M_tokens."))
        module.add(VMulLOU32(dst=vgpr(addrV), src0=sgpr("SizesFree+0"), src1=vgpr(r),
                             comment="base0 = token_n * N_hidden."))
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(1), src=vgpr(addrV),
                                  comment="base0 *= 2 (bf16)."))
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
        module.add(VCndMaskB32(dst=vgpr(addrV), src0=vgpr(ctx.resOobV), src1=vgpr(addrV),
                               src2=sgpr(tokMaskIdx, lsc),
                               comment="clamp OOB when token_n >= M_tokens."))

    def _storeBf16ElemInline(self, module, ctx, accReg, m, n, k) -> None:
        """Store bf16(accReg) to ResidualOut[token_n, nhidden_pos] with inline masking."""
        res = ctx.res
        lsc = res.lane_sgpr_count
        addrV   = ctx.writer.vgprPool.checkOut(1, tag="mf_bf16Addr")
        valV    = ctx.writer.vgprPool.checkOut(1, tag="mf_bf16Val")
        nhByteV = ctx.writer.vgprPool.checkOut(1, tag="mf_nhByte")
        with ctx.writer.allocTmpSgpr(lsc, tag="mf_tokMask") as tokMask:
            with ctx.writer.allocTmpSgpr(lsc, tag="mf_nhMask") as nhMask:
                self._computeBf16Addr(module, ctx, n, k, addrV, valV, nhByteV,
                                      tokMask.idx, nhMask.idx)
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

    def _computeBf16PairAddr(self, module, ctx, n, k, addrV, valV, nhByteV, tokMaskIdx, nhMaskIdx) -> None:
        """Compute clamped byte address for a wide bf16 pair store at (n, k) and k+1.

        OOB check uses nhPos at k+1 conservatively: if k+1 is in range, k is too,
        matching the wide-store convention in the residual emitter.
        """
        res = ctx.res
        lsc = res.lane_sgpr_count
        nOff = n * res.mfma_n
        r = res._addImmU32(module, addrV, ctx.resTokenBase, nOff, valV,
                           f"token_n = resTokenBase + {nOff} (n={n}).")
        module.add(VCmpLtU32(dst=sgpr(tokMaskIdx, lsc), src0=vgpr(r),
                             src1=sgpr("SizesFree+1"),
                             comment="tokenInRange = token_n < M_tokens."))
        module.add(VMulLOU32(dst=vgpr(addrV), src0=sgpr("SizesFree+0"), src1=vgpr(r),
                             comment="base0 = token_n * N_hidden."))
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(1), src=vgpr(addrV),
                                  comment="base0 *= 2 (bf16)."))
        nh1 = res._addImmU32(module, nhByteV, ctx.nhBase, k + 1, valV,
                             f"nhPos1 = nhBase + {k + 1} (k={k}).")
        module.add(VCmpLtU32(dst=sgpr(nhMaskIdx, lsc), src0=vgpr(nh1),
                             src1=sgpr("SizesFree+0"),
                             comment="pairInRange = nhPos1 < N_hidden."))
        # nhByte for k is the lower address of the pair.
        nh0 = res._addImmU32(module, nhByteV, ctx.nhBase, k, valV,
                             f"nhPos0 = nhBase + {k} (k={k}).")
        module.add(VLShiftLeftB32(dst=vgpr(nhByteV), shiftHex=hex(1), src=vgpr(nh0),
                                  comment="nhByte = nhPos0 * 2 (bf16)."))
        module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(nhByteV),
                           comment="byteAddr = base0 + nhByte(k)."))
        module.add(VCndMaskB32(dst=vgpr(addrV), src0=vgpr(ctx.resOobV), src1=vgpr(addrV),
                               src2=sgpr(nhMaskIdx, lsc),
                               comment="clamp OOB when nhPos1 >= N_hidden."))
        module.add(VCndMaskB32(dst=vgpr(addrV), src0=vgpr(ctx.resOobV), src1=vgpr(addrV),
                               src2=sgpr(tokMaskIdx, lsc),
                               comment="clamp OOB when token_n >= M_tokens."))

    def _storeBf16PairInline(self, module, ctx, acc0, acc1, m, n, k) -> None:
        """Pack and store bf16(acc0), bf16(acc1) with one BufferStoreB32."""
        res = ctx.res
        lsc = res.lane_sgpr_count
        addrV   = ctx.writer.vgprPool.checkOut(1, tag="mf_bf16Addr")
        valV    = ctx.writer.vgprPool.checkOut(1, tag="mf_bf16Val")
        nhByteV = ctx.writer.vgprPool.checkOut(1, tag="mf_nhByte")
        with ctx.writer.allocTmpSgpr(lsc, tag="mf_tokMask") as tokMask:
            with ctx.writer.allocTmpSgpr(lsc, tag="mf_nhMask") as nhMask:
                self._computeBf16PairAddr(module, ctx, n, k, addrV, valV, nhByteV,
                                          tokMask.idx, nhMask.idx)
                # src0=acc0 -> lo16 = bf16(H[k]); src1=acc1 -> hi16 = bf16(H[k+1]).
                module.add(VCvtPkF32toBF16(dst=vgpr(valV), src0=vgpr(acc0), src1=vgpr(acc1),
                                            comment="pack H[k] lo16, H[k+1] hi16 -> bf16x2."))
                module.add(BufferStoreB32(src=vgpr(valV), vaddr=vgpr(addrV),
                                          saddr=sgpr(res.residualOutSrd, 4), soffset=0,
                                          mubuf=MUBUFModifiers(offen=True),
                                          comment=f"ResidualOut bf16x2 (m={m},n={n},k={k})."))
        ctx.writer.vgprPool.checkIn(nhByteV)
        ctx.writer.vgprPool.checkIn(valV)
        ctx.writer.vgprPool.checkIn(addrV)

    def _issueResidualWide(self, module, ctx, m, n) -> None:
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
            dstBase = ctx.resBurst + 4 * c
            dst = vgpr(dstBase, 2) if isBf16 else vgpr(dstBase)
            module.add(loadCls(dst, vgpr(addr), sgpr(ctx.resSrd, 4), 0,
                               MUBUFModifiers(offen=True),
                               comment=f"R wide [4 residual] (m={m},n={n},c={c})."))
            if c > 0:
                ctx.writer.vgprPool.checkIn(addr)

    def _maskWideResidualOOB(self, module, ctx) -> None:
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
            module.add(VCndMaskB32(dst=vgpr(ctx.resBurst + k), src0=0,
                                   src1=vgpr(ctx.resBurst + k),
                                   src2=sgpr(ctx.resOobMask, lsc),
                                   comment=f"residual = inRange ? residual : 0 (k={k})."))

    def _loadResidualRow(self, module, ctx, m, n, mBaseV) -> None:
        """Load, wait, convert, and mask residual for tile (m, n).

        ctx.nhBase holds wgRowBase + rowGroupOff + m*mfmaM (set before this call).
        ctx.resRowByteBase holds token_n * N_hidden * residualBytes (set per n column).
        Wide path: one BufferLoad per chunk of 4 elements, software OOB masking.
        Scalar path: one BufferLoad per element with per-element address clamping.
        """
        res = ctx.res
        rpl = res.rows_per_lane
        if res.useWideResidual:
            self._issueResidualWide(module, ctx, m, n)
            module.add(SWaitCnt(vlcnt=0, comment="wait wide residual load."))
            if res.residualBytes == 2:
                res._convertResidualChunkBf16(module, ctx.resBurst)
            else:
                res._convertResidualChunkFp8(module, ctx.resBurst)
            self._maskWideResidualOOB(module, ctx)
            return
        for k in range(rpl):
            res._residualElemAddr(module, ctx.resAddr, ctx.resRowByteBase,
                                  ctx.wgRowBase, ctx.rowGroupOff,
                                  ctx.resOobV, ctx.resOobMask, mBaseV, m, k)
            res._issueSideLoad(module, ctx.resBurst + k, ctx.resAddr, ctx.resSrd,
                               f"R[m={m},n={n},k={k}].", dtype=res.residualType)
        module.add(SWaitCnt(vlcnt=0, comment="wait residual loads."))
        for k in range(rpl):
            res._convertSideElem(module, ctx.resBurst + k,
                                 f"residual->fp32 (k={k}).", dtype=res.residualType)

    def _pass1AccResRms(self, module, ctx, srcRegs, m, n, rpl) -> None:
        """Fuse residual add and rmsSum accumulation: H = acc + R, rmsSum[n] += H²."""
        res = ctx.res
        for k in range(rpl):
            acc = srcRegs[k]
            if res.residualAdd:
                module.add(VAddF32(dst=vgpr(acc), src0=vgpr(acc),
                                   src1=vgpr(ctx.resBurst + k),
                                   comment=f"H = acc + residual (m={m},n={n},k={k})."))
            module.add(VFmaF32(dst=vgpr(ctx.rmsSum + n), src0=vgpr(acc),
                               src1=vgpr(acc), src2=vgpr(ctx.rmsSum + n),
                               comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k})."))

    def _pass2StoreBf16(self, module, ctx, srcRegs, m, n, rpl, useWidePair) -> None:
        """Store bf16(H) to ResidualOut before gamma is applied."""
        if not ctx.res.storeBf16D:
            return
        if useWidePair:
            for k in range(0, rpl, 2):
                self._storeBf16PairInline(module, ctx, srcRegs[k], srcRegs[k + 1], m, n, k)
        else:
            for k in range(rpl):
                self._storeBf16ElemInline(module, ctx, srcRegs[k], m, n, k)

    def _pass3GammaAmax(self, module, ctx, srcRegs, gammaBank, blkAmaxJ,
                        accBank, bankBase, mi, m, n, rpl) -> None:
        """Apply gamma, fold |H*gamma| into blkAmax[j], and stage result to accBank."""
        for k in range(rpl):
            acc = srcRegs[k]
            gammaReg = gammaBank + mi * rpl + k
            module.add(VMulF32(dst=vgpr(acc), src0=vgpr(acc), src1=vgpr(gammaReg),
                               comment=f"acc = H * gamma (m={m},n={n},k={k})."))
            module.add(VAndB32(dst=vgpr(ctx.mx._scAccTmp), src0=vgpr(acc),
                               src1=vgpr(ctx.mx._scAbsMask),
                               comment=f"|H*gamma| (m={m},n={n},k={k})."))
            module.add(VMaxF32(dst=vgpr(blkAmaxJ), src0=vgpr(blkAmaxJ),
                               src1=vgpr(ctx.mx._scAccTmp),
                               comment=f"blkAmax = max(blkAmax, |H*gamma|)."))
            bankReg = accBank + bankBase + k
            if acc != bankReg:
                module.add(VMovB32(dst=vgpr(bankReg), src=vgpr(acc),
                                   comment=f"stage H*gamma to accBank (m={m},n={n},k={k})."))

    def _fusedElementLoop(self, module, ctx, vgprTiles, accBank, gammaBank,
                          blkAmax, qi, nBase, g) -> None:
        """Emit the three-pass fused loop (residual add, bf16 store, gamma/amax) per tile.

        Row byte base is hoisted before the tile-row mi loop because it depends
        only on n, not on m.
        """
        res = ctx.res
        rms = ctx.rms
        rpl = ctx.rowsPerLane
        tpb = ctx.tilesPerBlockM
        useWidePair = res.storeBf16D and rpl % 2 == 0
        mBaseV = ctx.writer.vgprPool.checkOut(1, tag="mf_mBase")
        for j in range(g):
            n = nBase + j
            if res.residualAdd:
                res._residualRowByteBase(module, ctx.resRowByteBase,
                                         ctx.resTokenBase, n, ctx.resAddr)
            for mi in range(tpb):
                m = qi * tpb + mi
                bankBase = (j * tpb + mi) * rpl
                coords = [(m, n, k) for k in range(rpl)]
                srcRegs = rms._readAccBurst(module, accBank + bankBase, vgprTiles,
                                            coords, f"acc m={m},n={n}.")
                rms._free0RowPos(module, ctx.nhBase, ctx.wgRowBase,
                                 ctx.rowGroupOff, m, 0, mBaseV)
                if res.residualAdd:
                    self._loadResidualRow(module, ctx, m, n, mBaseV)
                self._pass1AccResRms(module, ctx, srcRegs, m, n, rpl)
                self._pass2StoreBf16(module, ctx, srcRegs, m, n, rpl, useWidePair)
                self._pass3GammaAmax(module, ctx, srcRegs, gammaBank, blkAmax + j,
                                     accBank, bankBase, mi, m, n, rpl)
        ctx.writer.vgprPool.checkIn(mBaseV)

    def _fusedGroup(self, ctx, vgprTiles, gammaBank, qi, nBase, g) -> Module:
        """Emit the fused element loop and inline MXFP8 tail for one (qi, nBase) N-group.

        Checks out an accumulator staging bank and a per-N-column amax bank, runs
        the element loop (which folds |H*gamma| into blkAmax), then calls _mxFusedTail
        to butterfly-reduce, compute e8m0 scales, apply, and store MXScale bytes.
        """
        module = Module(f"MegaFused fusedGroup qi={qi} nBase={nBase}")
        vgprPool = ctx.writer.vgprPool
        bankSize = g * ctx.tilesPerBlockM * ctx.rowsPerLane
        accBank = vgprPool.checkOut(bankSize, tag="mf_accBank")
        blkAmax = vgprPool.checkOut(g, tag="mf_blkAmax")
        for j in range(g):
            module.add(VMovB32(dst=vgpr(blkAmax + j), src=0, comment=f"blkAmax[{j}] = 0."))
        self._fusedElementLoop(module, ctx, vgprTiles, accBank, gammaBank,
                               blkAmax, qi, nBase, g)
        self._mxFusedTail(module, ctx, vgprTiles, accBank, blkAmax, qi, nBase, g)
        vgprPool.checkIn(blkAmax)
        vgprPool.checkIn(accBank)
        return module

    def _mxFusedTail(self, module, ctx, vgprTiles, accBank, blkAmax,
                     qi, nBase, g) -> None:
        """Butterfly reduce blkAmax, compute e8m0 scales, apply to accBank, store MXScale bytes.

        After this call, the AGPR holds the final alpha*quantMult-scaled fp8 value and
        the MXScale buffer holds the e8m0 scale byte for each N-group column j.
        """
        vgprPool = ctx.writer.vgprPool
        addrBf = vgprPool.checkOut(1, tag="mf_addrBf")
        tmpBf = vgprPool.checkOut(g, tag="mf_tmpBf")
        for r in range(2):
            ctx.mx._butterflyRound(module, addrBf, tmpBf, blkAmax, g, ctx.laneId,
                                   ctx.mfmaN << r)
        vgprPool.checkIn(tmpBf)
        vgprPool.checkIn(addrBf)
        # Alpha fold: blkAmax[j] = |alpha * blkAmax[j]|, matching _streamSubColGroup.
        for j in range(g):
            module.add(VMulF32(dst=vgpr(blkAmax + j), src0=vgpr(blkAmax + j),
                               src1=sgpr("Alpha"), comment=f"blkAmax[{j}] *= alpha."))
            module.add(VAndB32(dst=vgpr(blkAmax + j), src0=vgpr(blkAmax + j),
                               src1=vgpr(ctx.mx._scAbsMask),
                               comment=f"blkAmax[{j}] = |alpha*blkAmax|."))
        # _computeSubColScales overwrites blkAmax with alpha*quantMult (the apply multiplier).
        scaleByteBank = ctx.mx._computeSubColScales(module, blkAmax, qi, nBase, g)
        mStart = qi * ctx.tilesPerBlockM
        mEnd = (qi + 1) * ctx.tilesPerBlockM
        ctx.mx._subColApply(module, vgprTiles, accBank, blkAmax, mStart, mEnd, nBase, g)
        ctx.mx._subColStoreGroup(module, ctx.mxSrd, scaleByteBank, ctx.col, ctx.rowGroup,
                                 ctx.savedExec, ctx.laneMask, qi, nBase, g)
        vgprPool.checkIn(scaleByteBank)

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
        assert ctx.mx.subColQuant, "megaFused requires subColQuant (q1 < mfmaN)"
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
        # Stream context must be live for the entire fused sweep so _fusedGroup helpers work.
        ctx.mx._beginStreamContext(module)

        for qi in range(ctx.nQTilesM):
            module.add(self._loadGammaBlock(ctx, gammaBank, qi))
            for nBase in range(0, ctx.mmaN, ctx.streamGroup):
                g = min(ctx.streamGroup, ctx.mmaN - nBase)
                module.add(self._fusedGroup(ctx, vgprTiles, gammaBank, qi, nBase, g))

        ctx.mx._endStreamContext()
        # One vscnt=0 drains both MXScale stores (from _subColStoreGroup) and ResidualOut stores.
        module.add(SWaitCnt(vscnt=0, comment="drain MXScale and ResidualOut stores."))
        self._endResidualScratch(module, ctx)
        ctx.writer.vgprPool.checkIn(gammaBank)

        module.add(self._reduceAndWriteRms(ctx))
        self._freeSharedRegs(ctx)
        return module
