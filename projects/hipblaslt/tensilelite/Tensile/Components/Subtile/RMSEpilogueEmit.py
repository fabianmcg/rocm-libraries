# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RMSEpilogueEmitter: faithful from-scratch RMS epilogue emitter (Subtile gfx950).

Mirrors the reference kernel mega_fused_epilogue.mlir (@mega_fused_epilogue). The
native path stores D = H*gamma directly and is exclusive: when it is active it is the
only epilogue (GlobalWriteBatch is skipped) and it does not apply alpha, beta*C, bias,
activation, scales, E, or amaxD. Every Subtile-layout, non-MXFP8 bf16 config takes the
native path; MXFP8 (float8 D) configs delegate to SubtileMegaFusedEmitter instead.
"""

import math

from rocisa.code import Label, Module
from rocisa.container import EXEC, DSModifiers, MUBUFModifiers, accvgpr, mgpr, sgpr, vgpr
from rocisa.instruction import (
    BufferLoadB32,
    BufferLoadB128,
    BufferLoadD16B16,
    BufferStoreB16,
    BufferStoreB32,
    BufferStoreB128,
    DSBPermuteB32,
    DSLoadB32,
    DSLoadB64,
    DSStoreB16,
    DSStoreB32,
    SAddU32,
    SAndB32,
    SAndSaveExecB64,
    SBranch,
    SCBranchSCC0,
    SCmpEQU32,
    SLShiftLeftB32,
    SLShiftRightB32,
    SMovB32,
    SMovB64,
    SMulHIU32,
    SMulI32,
    SNop,
    SSubU32,
    SWaitCnt,
    VAccvgprReadB32,
    VAddF32,
    VAddPKF32,
    VAddU32,
    VAndB32,
    VCmpEQU32,
    VCmpLtU32,
    VCndMaskB32,
    VCvtBF16toFP32,
    VCvtPkF32toBF16,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMovB32,
    VMulF32,
    VMulLOU32,
    VMulPKF32,
    VNop,
    VOrB32,
    VPermlane16SwapB32,
    VReadfirstlaneB32,
    VXorB32,
    _SWaitCnt,
)


class RMSEpilogueEmitter:
    """Emit the fused RMS/Residual epilogue for the Subtile gfx950 kernel."""

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel

        # ---- Core config (gfx950 wave64 MI16x16x32 Subtile) ----
        assert kernel["WavefrontSize"] == 64, "RMS epilogue requires wavefrontSize == 64"
        self.mfmaM = 16
        self.mfmaN = 16
        self.waveSize = 64
        # R: contiguous M-rows a lane owns per tile (Subtile vw0 = vw1 = 1).
        self.rowsPerLane = 4

        self.wgM = kernel["MIWaveGroup"][0]
        self.wgN = kernel["MIWaveGroup"][1]
        # Subtile: vw0 = vw1 = 1, so T_M/T_N are plain per-wave tile counts.
        self.T_M = (kernel["MacroTile0"] // 16) // self.wgM
        self.T_N = (kernel["MacroTile1"] // 16) // self.wgN
        self.G_M = self.wgM
        self.G_N = self.wgN
        self.MT0 = kernel["MacroTile0"]
        self.MT1 = kernel["MacroTile1"]
        self.numPairs = (self.T_M // 2) * self.T_N
        # PREFETCH must divide numPairs for the two-phase constexpr group loop;
        # deepen to a depth-2 ring only when numPairs is even and >= 2.
        self.PREFETCH = 2 if (self.numPairs % 2 == 0 and self.numPairs >= 2) else 1

        # MXFP8 dynamic quant is derived: RMSEpilogue active and D output is F8.
        self.useMxfp8 = (bool(kernel.get("RMSEpilogue", False))
                         and kernel["ProblemType"]["DestDataType"].isFloat8())
        # Subtile uses a wave-contiguous layout; CMS kernels are wave-interleaved.
        self.interleaved = not bool(kernel.get("UseSubtileImpl"))
        # Residual stream is bf16 in/out.
        self.residualBytes = 2

        self.laneSGPRCount = writer.states.laneSGPRCount

        # ---- Setup registers (allocated in _emitSetup; init None) ----
        self.laneV = None
        self.cV = None
        self.gV = None
        self.waveIdV = None
        self.waveMV = None
        self.waveNV = None
        self.rowOriginV = None
        self.colOriginV = None
        # Residual-load address constants (allocated in _computeAddrConstants).
        self.colBaseV = None
        # Tile-relative D column base (no WG1*MT1; SrdD already folds it).
        self.colBaseDV = None
        self.pairRowOffV = None
        self.g4V = None
        self.oobV = None
        self.resSrd = None
        self.residualOutSrd = None
        self.dOutSrd = None
        self.dStrideSgpr = None
        self.gammaSrd = None
        # Byte offset of the flat gamma LDS buffer (set in _emitGammaPrefetch).
        self.gammaLdsBase = None
        # Static byte footprint of the flat gamma LDS buffer (set in prefetch).
        self.gammaLdsBytes = None
        # Per-lane gamma LDS read base byte offset (set in _computeGammaReadBase).
        self.gammaReadBaseV = None

    def _nativeEligible(self):
        # The native RMS epilogue is exclusive and self-contained: when active it is
        # the only epilogue, it stores D = H*gamma directly, and GlobalWriteBatch is
        # skipped. By design it does NOT apply alpha, beta*C, bias, activation, scales,
        # E, or amaxD -- those features are simply not applied for a config routed here.
        # Delegate to the writer's authoritative gate so the store-skip predicate and
        # this emitter never drift (rowsPerLane is the only emitter-local extra check).
        return self.rowsPerLane == 4 and self.writer._nativeRmsEpilogueActive(self.kernel)

    def emit(self, vgprTiles):
        if self._nativeEligible():
            return self._emitNative(vgprTiles)
        from .SubtileMegaFusedEmit import SubtileMegaFusedEmitter
        return SubtileMegaFusedEmitter(self.writer, self.kernel).emit(vgprTiles)

    def _emitNative(self, vgprTiles):
        module = Module("RMSEpilogue native (faithful MLIR paired-permlane).")
        assert self.kernel["ProblemType"]["DestDataType"].isBFloat16(), \
            "native RMS epilogue emits bf16 D only; non-bf16 dest must route to the old path"
        # Drain GEMM vector memory before reusing AGPRs/LDS (epilogue entry).
        module.add(SWaitCnt(vlcnt=0, comment="drain GEMM loads before epilogue."))
        self._emitSetup(module)
        self._computeAddrConstants(module)
        self._computeGammaReadBase(module)
        # Top-level is_x4 = (rowExtent % 8 == 0) branch (MLIR:552-554).
        narrowLabel = Label(self.writer.labels.getNameInc("rms_narrow"),
                            "narrow (rowExtent%8 != 0) arm.")
        endLabel = Label(self.writer.labels.getNameInc("rms_end"), "epilogue merge point.")
        with self.writer.allocTmpSgpr(1, tag="rms_isx4") as t:
            module.add(SAndB32(dst=sgpr(t.idx), src0=sgpr("SizesFree+0"), src1=7,
                               comment="rowExtent & 7."))
            module.add(SCmpEQU32(src0=sgpr(t.idx), src1=0,
                                 comment="SCC = (rowExtent%8 == 0): wide path."))
        module.add(SCBranchSCC0(labelName=narrowLabel.getLabelName(),
                                comment="branch to narrow when rowExtent%8 != 0."))
        # Wide arm: DTL gamma + barrier + dwordx4 body.
        module.addComment0("Wide arm (rowExtent%8==0): dwordx4 coalesced.")
        self._emitGammaPrefetchDTL(module)
        self._emitBody(module, vgprTiles, isX4=True)
        module.add(SBranch(labelName=endLabel.getLabelName(), comment="wide done; skip narrow."))
        # Narrow arm: software gamma + barrier + scalar body.
        module.add(narrowLabel)
        module.addComment0("Narrow arm (rowExtent%8!=0): scalar bf16.")
        self._emitGammaPrefetch(module)
        self._emitBody(module, vgprTiles, isX4=False)
        module.add(endLabel)
        self._emitTeardown(module)
        return module

    def _emitTeardown(self, module):
        """Free every persistent register checked out by setup/address/gamma-base.

        Balances the checkouts in _emitSetup, _computeAddrConstants, and
        _computeGammaReadBase. residualOutSrd aliases SrdResidualOut (not
        pool-allocated) and is intentionally not freed.
        """
        module.addComment1("RMS epilogue teardown: free persistent setup registers.")
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        for attr in ("laneV", "cV", "gV", "waveIdV", "waveMV", "waveNV",
                     "rowOriginV", "colOriginV", "colBaseV", "colBaseDV", "pairRowOffV",
                     "g4V", "oobV", "gammaReadBaseV"):
            vgprPool.checkIn(getattr(self, attr))
            setattr(self, attr, None)
        for attr in ("resSrd", "gammaSrd"):
            sgprPool.checkIn(getattr(self, attr))
            setattr(self, attr, None)

    def _emitSetup(self, module):
        """Emit the kernel prologue: lane decode, row/col origins, buffer SRDs.

        Faithful port of @mega_fused_epilogue lines 470-546. The register
        checkouts persist for the whole emission and are freed in the final
        teardown milestone.
        """
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool

        # ---- Lane decode (MLIR 499-514) ----
        self.laneV   = vgprPool.checkOut(1, tag="rms_lane")
        self.cV      = vgprPool.checkOut(1, tag="rms_c")
        self.gV      = vgprPool.checkOut(1, tag="rms_g")
        self.waveIdV = vgprPool.checkOut(1, tag="rms_waveId")
        self.waveMV  = vgprPool.checkOut(1, tag="rms_waveM")
        self.waveNV  = vgprPool.checkOut(1, tag="rms_waveN")
        module.addComment1("RMS epilogue setup: lane decode (lane, c, g, waveId).")
        module.add(VAndB32(dst=vgpr(self.laneV), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="lane = Serial & 63."))
        module.add(VAndB32(dst=vgpr(self.cV), src0=vgpr(self.laneV), src1=self.mfmaN - 1,
                           comment="c = lane & 15 (column within MI tile)."))
        module.add(VLShiftRightB32(dst=vgpr(self.gV), shiftHex=hex(int(math.log2(self.mfmaN))),
                                   src=vgpr(self.laneV), comment="g = lane >> 4 (row-group)."))
        module.add(VLShiftRightB32(dst=vgpr(self.waveIdV),
                                   shiftHex=hex(int(math.log2(self.waveSize))),
                                   src=vgpr("Serial"), comment="waveId = Serial >> 6."))
        # M-fastest wave linearization: waveId = waveN*G_M + waveM, so
        # waveM = waveId % G_M and waveN = waveId // G_M (G_M is a power of two).
        if self.G_M == 1:
            module.add(VMovB32(dst=vgpr(self.waveMV), src=0, comment="waveM = 0 (G_M == 1)."))
            module.add(VMovB32(dst=vgpr(self.waveNV), src=vgpr(self.waveIdV),
                               comment="waveN = waveId (G_M == 1)."))
        else:
            assert self.G_M & (self.G_M - 1) == 0, "G_M must be a power of two"
            module.add(VAndB32(dst=vgpr(self.waveMV), src0=vgpr(self.waveIdV), src1=self.G_M - 1,
                               comment="waveM = waveId & (G_M - 1)."))
            module.add(VLShiftRightB32(dst=vgpr(self.waveNV),
                                       shiftHex=hex(int(math.log2(self.G_M))),
                                       src=vgpr(self.waveIdV), comment="waveN = waveId >> log2(G_M)."))

        # ---- Row/col origins (MLIR 515-522) ----
        self.rowOriginV = vgprPool.checkOut(1, tag="rms_rowOrigin")
        self.colOriginV = vgprPool.checkOut(1, tag="rms_colOrigin")
        tmpV = vgprPool.checkOut(1, tag="rms_originTmp")
        module.addComment1("row origin = WorkGroup0*MT0 + waveM*(T_M*16).")
        module.add(VMovB32(dst=vgpr(tmpV), src=self.MT0, comment=f"MT0={self.MT0}."))
        module.add(VMulLOU32(dst=vgpr(self.rowOriginV), src0=vgpr(tmpV), src1=sgpr("WorkGroup0"),
                             comment="rowBase = WorkGroup0 * MT0."))
        rwSpanM = self.T_M * 16
        module.add(VMovB32(dst=vgpr(tmpV), src=rwSpanM, comment=f"T_M*16={rwSpanM}."))
        module.add(VMulLOU32(dst=vgpr(tmpV), src0=vgpr(tmpV), src1=vgpr(self.waveMV),
                             comment="waveMOff = waveM * (T_M*16)."))
        module.add(VAddU32(vgpr(self.rowOriginV), vgpr(self.rowOriginV), vgpr(tmpV),
                           comment="rowOrigin += waveMOff."))
        module.addComment1("col origin = WorkGroup1*MT1 + waveN*(T_N*16).")
        module.add(VMovB32(dst=vgpr(tmpV), src=self.MT1, comment=f"MT1={self.MT1}."))
        module.add(VMulLOU32(dst=vgpr(self.colOriginV), src0=vgpr(tmpV), src1=sgpr("WorkGroup1"),
                             comment="colBase = WorkGroup1 * MT1."))
        cwSpanN = self.T_N * 16
        module.add(VMovB32(dst=vgpr(tmpV), src=cwSpanN, comment=f"T_N*16={cwSpanN}."))
        module.add(VMulLOU32(dst=vgpr(tmpV), src0=vgpr(tmpV), src1=vgpr(self.waveNV),
                             comment="waveNOff = waveN * (T_N*16)."))
        module.add(VAddU32(vgpr(self.colOriginV), vgpr(self.colOriginV), vgpr(tmpV),
                           comment="colOrigin += waveNOff."))
        vgprPool.checkIn(tmpV)

        # ---- Buffer SRDs (MLIR 523-546) ----
        # TODO(body/address milestone): fold the WorkGroup2 batch offset into these
        # base addresses when StridedBatched; F1 builds the base SRDs only.
        self.resSrd   = sgprPool.checkOutAligned(4, 4, tag="rms_resSrd", preventOverflow=False)
        self.gammaSrd = sgprPool.checkOutAligned(4, 4, tag="rms_gammaSrd", preventOverflow=False)
        # ResidualOut aliases the (beta=0 unused) SrdC named SGPR.
        self.residualOutSrd = self.writer.sgprs["SrdResidualOut"]
        # D output SRD is already built (batch offset folded) and live here; reuse it.
        self.dOutSrd = self.writer.sgprs["SrdD"]
        self.dStrideSgpr = "StrideD" + self.writer.states.indexChars[self.kernel["PackedC1IndicesX"][0]]
        self._boundDOutSrdRecords(module)
        self._buildResidualSrd(module, self.resSrd, "ResidualBuf", "residual")
        self._buildResidualSrd(module, self.residualOutSrd, "AddressResidualOut", "residualOut")
        self._buildGammaSrd(module, self.gammaSrd)

    def _buildResidualSrd(self, module, srd, base, name):
        """Build a residual SRD: numRecords = SizesFree0 * SizesFree1 * 2 (bf16)."""
        module.addComment1(f"build {name} SRD (numRecords = N*M * 2 bytes).")
        module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr(base, 2), comment=f"{name} SRD base."))
        module.add(SMulI32(dst=sgpr(srd + 2), src0=sgpr("SizesFree+0"), src1=sgpr("SizesFree+1"),
                           comment="numRecords = SizesFree0 * SizesFree1."))
        module.add(SLShiftLeftB32(dst=sgpr(srd + 2), src=sgpr(srd + 2), shiftHex=hex(1),
                                  comment="numRecords *= 2 (bf16)."))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96", comment=f"{name} SRD flags."))

    def _boundDOutSrdRecords(self, module):
        """Bound the reused D SRD to this workgroup's valid column span.

        SrdD's base folds WG1*MT1 columns but its numRecords is left unbounded
        (BufferOOB). The native D store only clamps rows, so a partial-N tile would
        write columns past N into a neighbour row. Mirror the residual SRD's bound
        (column-major: validCols * rowExtent * bpe) so overflow stores are dropped.
        """
        module.addComment1("bound D SRD numRecords to valid columns (SrdD base folds WG1*MT1).")
        with self.writer.allocTmpSgpr(1, tag="rms_dRecords") as t:
            module.add(SMulI32(dst=sgpr(t.idx), src0=self.MT1, src1=sgpr("WorkGroup1"),
                               comment="colFolded = WorkGroup1 * MT1 (already in SrdD base)."))
            module.add(SSubU32(dst=sgpr(t.idx), src0=sgpr("SizesFree+1"), src1=sgpr(t.idx),
                               comment="validCols = SizesFree1 - colFolded."))
            module.add(SMulI32(dst=sgpr(t.idx), src0=sgpr(t.idx), src1=sgpr("SizesFree+0"),
                               comment="records = validCols * rowExtent."))
            module.add(SLShiftLeftB32(dst=sgpr(self.dOutSrd + 2), src=sgpr(t.idx), shiftHex=hex(1),
                                      comment="SrdD numRecords = records * 2 (bf16)."))

    def _buildGammaSrd(self, module, srd):
        """Build the gamma SRD: numRecords = SizesFree0 * 2 (bf16, one per row)."""
        module.addComment1("build gamma SRD (numRecords = SizesFree0 * 2 bytes).")
        module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr("RMSNormGamma", 2), comment="gamma SRD base."))
        module.add(SLShiftLeftB32(dst=sgpr(srd + 2), src=sgpr("SizesFree+0"), shiftHex=hex(1),
                                  comment="numRecords = SizesFree0 * 2 (bf16)."))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96", comment="gamma SRD flags."))

    def _emitGammaPrefetch(self, module):
        """Prefetch this block's MT0 gamma rows into a flat workgroup LDS buffer.

        Faithful port of the SOFTWARE gamma prefetch (@mega_fused_epilogue lines
        600-620, the narrow path). It fills LDS so that
        ``LDS[idx] = gamma[rowWaveBase + idx]`` for idx in [0, MT0), with
        ``rowWaveBase = WorkGroup0 * MT0``. The body (F4) reads each lane's four
        gamma rows straight from this flat buffer.

        The software prefetch (plain buffer_load + ds_write_b16) is correct for
        every MT0; the hardware direct-to-LDS variant (two cooperating waves, one
        dword per lane) is a perf optimization deferred to a later milestone.
        """
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        blockDim = self.wgM * self.wgN * self.waveSize
        numPasses = (self.MT0 + blockDim - 1) // blockDim

        # LDS region: the gamma buffer occupies MT0*2 bytes at the start of the
        # epilogue LDS (the main-loop LDS, free post-loop). gammaLdsBase is the
        # byte offset handed to ds_write/ds_read; the body (F4) reads from it.
        self.gammaLdsBase = 0
        self.gammaLdsBytes = self.MT0 * 2

        module.addComment1(
            f"gamma LDS prefetch: flat LDS[idx]=gamma[rowWaveBase+idx], idx in [0,{self.MT0}).")
        rowWaveBaseS = sgprPool.checkOut(1, tag="rms_rowWaveBase")
        module.add(SMulI32(dst=sgpr(rowWaveBaseS), src0=sgpr("WorkGroup0"), src1=hex(self.MT0),
                           comment=f"rowWaveBase = WorkGroup0 * MT0({self.MT0})."))
        # WAR: sibling waves may still read the main-loop LDS; fence before reuse.
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma prefetch: WAR barrier before reusing LDS."))

        idxV    = vgprPool.checkOut(1, tag="rms_gammaIdx")
        growV   = vgprPool.checkOut(1, tag="rms_gammaRow")
        gvalV   = vgprPool.checkOut(1, tag="rms_gammaVal")
        ldsOffV = vgprPool.checkOut(1, tag="rms_gammaLdsOff")
        for p in range(numPasses):
            needGuard = (p + 1) * blockDim > self.MT0
            self._emitGammaPrefetchPass(module, p, blockDim, needGuard,
                                        rowWaveBaseS, idxV, growV, gvalV, ldsOffV)
        vgprPool.checkIn(idxV)
        vgprPool.checkIn(growV)
        vgprPool.checkIn(gvalV)
        vgprPool.checkIn(ldsOffV)
        sgprPool.checkIn(rowWaveBaseS)

        # Publish: wait for every ds_write, then barrier so the body reads a fully
        # populated LDS (MLIR 620: vmcnt0 lgkmcnt0 + s_barrier).
        module.add(SWaitCnt(dscnt=0, comment="wait all gamma ds_writes (lgkmcnt 0)."))
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma prefetch: LDS fully populated before body reads."))

    def _emitGammaPrefetchPass(self, module, p, blockDim, needGuard,
                               rowWaveBaseS, idxV, growV, gvalV, ldsOffV):
        """Emit one prefetch pass: load gamma[rowWaveBase+idx] -> LDS[idx]."""
        lsc = self.laneSGPRCount
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        if p == 0:
            module.add(VMovB32(dst=vgpr(idxV), src=vgpr("Serial"), comment="idx = Serial."))
        else:
            self._vaddImm(module, idxV, "Serial", p * blockDim, f"idx = Serial + {p}*blockDim.")
        savedExec = None
        if needGuard:
            mask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_gammaMask", preventOverflow=False)
            savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="rms_gammaExec", preventOverflow=False)
            # MT0 exceeds the inline cap, so materialize it before the compare.
            mt0V = vgprPool.checkOut(1, tag="rms_gammaMt0")
            module.add(VMovB32(dst=vgpr(mt0V), src=self.MT0, comment=f"MT0 = {self.MT0}."))
            module.add(VCmpLtU32(dst=sgpr(mask, lsc), src0=vgpr(idxV), src1=vgpr(mt0V),
                                 comment=f"active = idx < MT0({self.MT0})."))
            vgprPool.checkIn(mt0V)
            module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(mask, lsc),
                                       comment="narrow exec to active prefetch lanes."))
            sgprPool.checkIn(mask)
        # grow = rowWaveBase + idx; gamma byte offset = grow*2.
        module.add(VAddU32(vgpr(growV), sgpr(rowWaveBaseS), vgpr(idxV),
                           comment="grow = rowWaveBase + idx."))
        module.add(VLShiftLeftB32(dst=vgpr(growV), shiftHex=hex(1), src=vgpr(growV),
                                  comment="gamma byte offset = grow * 2 (bf16)."))
        module.add(BufferLoadD16B16(vgpr(gvalV), vgpr(growV), sgpr(self.gammaSrd, 4), 0,
                                    MUBUFModifiers(offen=True),
                                    comment="gval = gamma[grow] (buffer OOB -> 0)."))
        module.add(SWaitCnt(vlcnt=0, comment="wait gamma load (vmcnt 0)."))
        # LDS byte offset = gammaLdsBase + idx*2.
        module.add(VLShiftLeftB32(dst=vgpr(ldsOffV), shiftHex=hex(1), src=vgpr(idxV),
                                  comment="lds byte offset = idx * 2 (bf16)."))
        if self.gammaLdsBase:
            module.add(VAddU32(vgpr(ldsOffV), vgpr(ldsOffV), self.gammaLdsBase,
                               comment=f"lds offset += gammaLdsBase({self.gammaLdsBase})."))
        module.add(DSStoreB16(dstAddr=vgpr(ldsOffV), src=vgpr(gvalV), comment="LDS[idx] = gval."))
        if needGuard:
            module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                               comment="restore full exec after prefetch pass."))
            sgprPool.checkIn(savedExec)

    def _emitGammaPrefetchDTL(self, module):
        """Prefetch this block's MT0 gamma rows into flat LDS via direct-to-LDS.

        Faithful port of the WIDE gamma prefetch (@mega_fused_epilogue lines
        555-591, the dwordx4-coalesced path). It fills the SAME flat layout as the
        software prefetch -- ``LDS[idx] = gamma[rowWaveBase + idx]`` for idx in
        [0, MT0), with ``rowWaveBase = WorkGroup0 * MT0`` -- but cooperatively: the
        first ``loaderWaves = ceil(MT0 / 128)`` waves each issue one b32
        direct-to-LDS load whose 64 lanes read two bf16 gamma rows apiece
        (lane l -> gamma[chunkBase + 2l], gamma[chunkBase + 2l + 1]).

        Each physical wave handles exactly its own chunk ``w = waveId``:
        ``chunkBase = rowWaveBase + w*128`` rows, global byte soffset
        ``soffB = chunkBase*2``, per-lane vaddr ``voff = lane*4``, and per-wave LDS
        base ``M0 = gammaLdsBase + w*256`` bytes. Waves >= loaderWaves skip the load
        (narrowed exec) but still reach the publish barrier.

        Dead code pending wiring (F2b); emit() still delegates.
        """
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        lsc = self.laneSGPRCount
        loaderWaves = (self.MT0 + 127) // 128
        numWaves = self.wgM * self.wgN
        assert loaderWaves <= numWaves, "gamma DTL prefetch needs loaderWaves <= wgM*wgN waves"

        # LDS region: the flat gamma buffer starts the epilogue LDS. The DTL path
        # rounds its footprint up to whole 128-row (256-byte) chunks; partial
        # last-wave OOB rows land in unread slots >= MT0.
        self.gammaLdsBase = 0
        self.gammaLdsBytes = max(self.MT0 * 2, loaderWaves * 256)

        module.addComment1(
            f"gamma DTL prefetch: flat LDS[idx]=gamma[rowWaveBase+idx], "
            f"loaderWaves={loaderWaves} (ceil(MT0({self.MT0})/128)).")
        rowWaveBaseS = sgprPool.checkOut(1, tag="rms_dtlRowWaveBase")
        module.add(SMulI32(dst=sgpr(rowWaveBaseS), src0=sgpr("WorkGroup0"), src1=hex(self.MT0),
                           comment=f"rowWaveBase = WorkGroup0 * MT0({self.MT0})."))
        # WAR: sibling waves may still read the main-loop LDS; fence before reuse.
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma DTL prefetch: WAR barrier before reusing LDS."))

        # waveId = Serial >> 6 (uniform per wave); scalar copy drives M0/soffB.
        waveIdV = vgprPool.checkOut(1, tag="rms_dtlWaveId")
        module.add(VLShiftRightB32(dst=vgpr(waveIdV),
                                   shiftHex=hex(int(math.log2(self.waveSize))),
                                   src=vgpr("Serial"), comment="waveId = Serial >> 6."))
        module.add(SNop(waitState=0, comment="VALU write -> readfirstlane hazard."))
        wS = sgprPool.checkOut(1, tag="rms_dtlWave")
        module.add(VReadfirstlaneB32(dst=sgpr(wS), src=vgpr(waveIdV),
                                     comment="w = waveId (uniform scalar)."))
        # w256 = w * 256 feeds both M0 (= gammaLdsBase + w*256) and soffB.
        w256S = sgprPool.checkOut(1, tag="rms_dtlW256")
        module.add(SLShiftLeftB32(dst=sgpr(w256S), src=sgpr(wS), shiftHex=hex(8),
                                  comment="w256 = w * 256 (128 rows * 2 bytes)."))
        # soffB = chunkBase*2 = rowWaveBase*2 + w*256 (byte soffset).
        soffBS = sgprPool.checkOut(1, tag="rms_dtlSoff")
        module.add(SLShiftLeftB32(dst=sgpr(soffBS), src=sgpr(rowWaveBaseS), shiftHex=hex(1),
                                  comment="rowWaveBase * 2 (bf16 byte offset)."))
        module.add(SAddU32(dst=sgpr(soffBS), src0=sgpr(soffBS), src1=sgpr(w256S),
                           comment="soffB = (rowWaveBase + w*128) * 2."))
        # Per-lane vaddr voff = lane*4 = (Serial & 63) << 2.
        voffV = vgprPool.checkOut(1, tag="rms_dtlVoff")
        module.add(VAndB32(dst=vgpr(voffV), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="lane = Serial & 63."))
        module.add(VLShiftLeftB32(dst=vgpr(voffV), shiftHex=hex(2), src=vgpr(voffV),
                                  comment="voff = lane * 4 (b32 DTL byte offset)."))

        # Guard: narrow exec to loader waves (waveId < loaderWaves); the whole wave
        # is uniform, so a loader wave keeps exec full and others drop to zero.
        mask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_dtlMask", preventOverflow=False)
        savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="rms_dtlExec", preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(mask, lsc), src0=vgpr(waveIdV), src1=loaderWaves,
                             comment=f"isLoader = waveId < loaderWaves({loaderWaves})."))
        module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(mask, lsc),
                                   comment="narrow exec to loader waves."))
        sgprPool.checkIn(mask)

        # M0 = gammaLdsBase + w*256 (per-wave LDS base), then one b32 DTL load.
        if self.gammaLdsBase:
            m0S = sgprPool.checkOut(1, tag="rms_dtlM0")
            module.add(SAddU32(dst=sgpr(m0S), src0=sgpr(w256S), src1=self.gammaLdsBase,
                               comment=f"M0 = gammaLdsBase({self.gammaLdsBase}) + w*256."))
            module.add(SMovB32(dst=mgpr(0), src=sgpr(m0S), comment="M0 = per-wave LDS base."))
            sgprPool.checkIn(m0S)
        else:
            module.add(SMovB32(dst=mgpr(0), src=sgpr(w256S),
                               comment="M0 = gammaLdsBase(0) + w*256 = per-wave LDS base."))
        module.add(BufferLoadB32(
            dst=None, vaddr=vgpr(voffV), saddr=sgpr(self.gammaSrd, 4), soffset=sgpr(soffBS),
            mubuf=MUBUFModifiers(offen=True, offset12=0, lds=True),
            comment="gamma b32 DTL -> LDS[M0 + lane*4] = gamma[chunkBase + 2*lane]."))

        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                           comment="restore full exec after DTL issue."))
        sgprPool.checkIn(savedExec)

        vgprPool.checkIn(waveIdV)
        vgprPool.checkIn(voffV)
        sgprPool.checkIn(wS)
        sgprPool.checkIn(w256S)
        sgprPool.checkIn(soffBS)
        sgprPool.checkIn(rowWaveBaseS)

        # Publish: drain the DTL writes, then barrier so the body reads full LDS.
        module.add(SWaitCnt(vlcnt=0, dscnt=0, comment="wait gamma DTL loads (vmcnt0 lgkmcnt0)."))
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma DTL prefetch: LDS fully populated before body reads."))

    def _computeAddrConstants(self, module):
        """Per-lane residual-address constants (precompute once per emission).

        Faithful to the #eoff column-major model of @mega_fused_epilogue: the
        residual byte address is (column*rowExtent + row) * 2. colBase, pairRowOff
        and g4 are the index-independent lane terms; the per-load code adds the
        pair/tile/column terms on top. These VGPRs are freed in the final teardown
        milestone.
        """
        vgprPool = self.writer.vgprPool
        self.colBaseV    = vgprPool.checkOut(1, tag="rms_colBase")
        self.colBaseDV   = vgprPool.checkOut(1, tag="rms_colBaseD")
        self.pairRowOffV = vgprPool.checkOut(1, tag="rms_pairRowOff")
        self.g4V         = vgprPool.checkOut(1, tag="rms_g4")
        self.oobV        = vgprPool.checkOut(1, tag="rms_oob")
        module.addComment1("residual load constants: colBase, colBaseD, pairRowOff, g4, oob.")
        # colBase = colOrigin + c (the lane's column within the wave tile).
        module.add(VAddU32(vgpr(self.colBaseV), vgpr(self.colOriginV), vgpr(self.cV),
                           comment="colBase = colOrigin + c."))
        # colBaseD = waveN*(T_N*16) + c: tile-relative (SrdD folds WG1*MT1, so the
        # D store must NOT re-add it, unlike the residual store's global colBase).
        cwSpanN = self.T_N * 16
        tmpColD = vgprPool.checkOut(1, tag="rms_colBaseDTmp")
        module.add(VMovB32(dst=vgpr(tmpColD), src=cwSpanN, comment=f"T_N*16={cwSpanN}."))
        module.add(VMulLOU32(dst=vgpr(self.colBaseDV), src0=vgpr(tmpColD), src1=vgpr(self.waveNV),
                             comment="colBaseD = waveN * (T_N*16)."))
        vgprPool.checkIn(tmpColD)
        module.add(VAddU32(vgpr(self.colBaseDV), vgpr(self.colBaseDV), vgpr(self.cV),
                           comment="colBaseD = waveN*(T_N*16) + c (tile-relative; SrdD folds WG1*MT1)."))
        # pairRowOff = (g&1)*16 + (g>>1)*8: tileSel*16 + half*8.
        tmpV = vgprPool.checkOut(1, tag="rms_pairRowTmp")
        module.add(VAndB32(dst=vgpr(tmpV), src0=vgpr(self.gV), src1=1, comment="tileSel = g & 1."))
        module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(4), src=vgpr(tmpV),
                                  comment="tileSel * 16."))
        module.add(VLShiftRightB32(dst=vgpr(self.pairRowOffV), shiftHex=hex(1), src=vgpr(self.gV),
                                   comment="half = g >> 1."))
        module.add(VLShiftLeftB32(dst=vgpr(self.pairRowOffV), shiftHex=hex(3),
                                  src=vgpr(self.pairRowOffV), comment="half * 8."))
        module.add(VAddU32(vgpr(self.pairRowOffV), vgpr(self.pairRowOffV), vgpr(tmpV),
                           comment="pairRowOff = tileSel*16 + half*8."))
        vgprPool.checkIn(tmpV)
        # g4 = g * 4 (native-tile row-group offset).
        module.add(VLShiftLeftB32(dst=vgpr(self.g4V), shiftHex=hex(2), src=vgpr(self.gV),
                                  comment="g4 = g * 4."))
        # oob = BufferOOB sentinel address that clamps an OOB row to a no-op load.
        module.add(VMovB32(dst=vgpr(self.oobV), src="BufferOOB", comment="oob = BufferOOB sentinel."))

    def _loadRaw(self, module, bank, mp, n, isX4):
        """Load the raw residual for pair ``mp``, column-tile ``n`` into ``bank``.

        Faithful port of @load_raw (MLIR 84-137). WIDE issues one dwordx4 with a
        per-pair all-or-nothing row clamp; NARROW issues eight scalar bf16 with a
        per-row clamp. ``bank`` is an 8-wide, 4-aligned VGPR block checked out by
        the caller. No @pair_shuffle here -- the caller applies it (WIDE only).
        """
        if isX4:
            self._loadRawWide(module, bank, mp, n)
        else:
            self._loadRawNarrow(module, bank, mp, n)

    def _loadRawWide(self, module, bank, mp, n):
        """One dwordx4 per pair, per-pair row clamp (MLIR 91-96)."""
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        lsc = self.laneSGPRCount
        m = 2 * mp
        rowBaseP = vgprPool.checkOut(1, tag="rms_rowBaseP")
        col      = vgprPool.checkOut(1, tag="rms_wideCol")
        addr     = vgprPool.checkOut(1, tag="rms_wideAddr")
        # rowBaseP = rowOrigin + pairRowOff + m*16.
        module.add(VAddU32(vgpr(rowBaseP), vgpr(self.rowOriginV), vgpr(self.pairRowOffV),
                           comment="rowBaseP = rowOrigin + pairRowOff."))
        self._vaddImm(module, rowBaseP, rowBaseP, m * 16, f"rowBaseP += m*16 (m={m}).")
        # col = colBase + n*16.
        self._vaddImm(module, col, self.colBaseV, n * 16, f"col = colBase + n*16 (n={n}).")
        # addr = (column*rowExtent + rowBaseP) << 1 (bf16 column-major byte offset).
        module.add(VMulLOU32(dst=vgpr(addr), src0=sgpr("SizesFree+0"), src1=vgpr(col),
                             comment="addr = column * rowExtent."))
        module.add(VAddU32(vgpr(addr), vgpr(addr), vgpr(rowBaseP), comment="addr += rowBaseP."))
        module.add(VLShiftLeftB32(dst=vgpr(addr), shiftHex=hex(1), src=vgpr(addr),
                                  comment="addr *= 2 (bf16 bytes)."))
        # inRange = rowBaseP < rowExtent: one clamp covers the pair's 8 rows
        # (rowExtent % 8 == 0 and rowBaseP is 8-aligned).
        mask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_wideMask", preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(mask, lsc), src0=vgpr(rowBaseP), src1=sgpr("SizesFree+0"),
                             comment="inRange = rowBaseP < rowExtent."))
        module.add(VCndMaskB32(dst=vgpr(addr), src0=vgpr(self.oobV), src1=vgpr(addr),
                               src2=sgpr(mask, lsc), comment="OOB addr when rowBaseP >= rowExtent."))
        module.add(BufferLoadB128(vgpr(bank, 4), vgpr(addr), sgpr(self.resSrd, 4), 0,
                                  MUBUFModifiers(offen=True),
                                  comment="raw dwordx4 = residual[pair 8 rows]."))
        sgprPool.checkIn(mask)
        vgprPool.checkIn(rowBaseP)
        vgprPool.checkIn(col)
        vgprPool.checkIn(addr)

    def _loadRawNarrow(self, module, bank, mp, n):
        """Eight scalar bf16, per-row clamp (MLIR 97-134).

        Native order: bank[0..3] = tile m rows g*4+0..3, bank[4..7] = tile m1.
        """
        self._loadRawNarrowTile(module, bank, 0, 2 * mp, n)
        self._loadRawNarrowTile(module, bank, 4, 2 * mp + 1, n)

    def _loadRawNarrowTile(self, module, bank, slotBase, ti, n):
        """Load one native tile's four bf16 rows into bank[slotBase .. slotBase+3]."""
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        lsc = self.laneSGPRCount
        rowBaseT = vgprPool.checkOut(1, tag="rms_rowBaseT")
        col      = vgprPool.checkOut(1, tag="rms_narrowCol")
        row      = vgprPool.checkOut(1, tag="rms_narrowRow")
        addr     = vgprPool.checkOut(1, tag="rms_narrowAddr")
        # rowBaseT = rowOrigin + g4 + ti*16 (native-tile base row for this lane).
        module.add(VAddU32(vgpr(rowBaseT), vgpr(self.rowOriginV), vgpr(self.g4V),
                           comment="rowBaseT = rowOrigin + g4."))
        self._vaddImm(module, rowBaseT, rowBaseT, ti * 16, f"rowBaseT += ti*16 (ti={ti}).")
        # col = colBase + n*16.
        self._vaddImm(module, col, self.colBaseV, n * 16, f"col = colBase + n*16 (n={n}).")
        for k in range(4):
            mask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_narrowMask", preventOverflow=False)
            module.add(VAddU32(vgpr(row), vgpr(rowBaseT), k, comment=f"row = rowBaseT + {k}."))
            module.add(VCmpLtU32(dst=sgpr(mask, lsc), src0=vgpr(row), src1=sgpr("SizesFree+0"),
                                 comment="inRange = row < rowExtent."))
            module.add(VMulLOU32(dst=vgpr(addr), src0=sgpr("SizesFree+0"), src1=vgpr(col),
                                 comment="addr = column * rowExtent."))
            module.add(VAddU32(vgpr(addr), vgpr(addr), vgpr(row), comment="addr += row."))
            module.add(VLShiftLeftB32(dst=vgpr(addr), shiftHex=hex(1), src=vgpr(addr),
                                      comment="addr *= 2 (bf16 bytes)."))
            module.add(VCndMaskB32(dst=vgpr(addr), src0=vgpr(self.oobV), src1=vgpr(addr),
                                   src2=sgpr(mask, lsc), comment="OOB addr when row >= rowExtent."))
            module.add(BufferLoadD16B16(vgpr(bank + slotBase + k), vgpr(addr), sgpr(self.resSrd, 4), 0,
                                        MUBUFModifiers(offen=True), comment=f"raw bf16 = residual[row {k}]."))
            sgprPool.checkIn(mask)
        vgprPool.checkIn(rowBaseT)
        vgprPool.checkIn(col)
        vgprPool.checkIn(row)
        vgprPool.checkIn(addr)

    def _pairShuffle(self, module, bank):
        """Cross-lane pair shuffle (WIDE only): tile-contiguous raw -> native.

        Faithful port of @pair_shuffle (MLIR 144-158). Two involutive
        permlane16_swap ops over the (g, g^1) lane pair. After: bank[0..3] =
        native dwords [a0,a1,b0,b1] = [tileM r0,r1 | tileM r2,r3 |
        tileM1 r0,r1 | tileM1 r2,r3].
        """
        module.add(VPermlane16SwapB32(dst=vgpr(bank + 0), src=vgpr(bank + 2),
                                      comment="pair swap dword 0 <-> 2."))
        module.add(VPermlane16SwapB32(dst=vgpr(bank + 1), src=vgpr(bank + 3),
                                      comment="pair swap dword 1 <-> 3."))

    def _residualToF32(self, module, bank, isX4):
        """Expand the raw bf16 residual in ``bank`` to 8 native-order f32.

        Faithful port of the ``%H8 = extf %native`` step (MLIR 352). WIDE unpacks
        4 native packed dwords into 8 f32 high-index-first (so each packed source
        dword is fully read before its low half is overwritten); NARROW converts
        8 lo16 bf16 in place. Returns [bank+0 .. bank+7] in native f32 order
        [tileM r0..3, tileM1 r0..3].
        """
        if isX4:
            for i in range(7, -1, -1):
                module.add(VCvtBF16toFP32(vgpr(bank + i), vgpr(bank + i // 2), None, i % 2,
                                          comment=f"f32[{i}] = extf(dword {i // 2}, half {i % 2})."))
        else:
            for j in range(8):
                module.add(VCvtBF16toFP32(vgpr(bank + j), vgpr(bank + j), None, 0,
                                          comment=f"f32[{j}] = extf(lo16 bf16[{j}])."))
        return [bank + j for j in range(8)]

    def _computeGammaReadBase(self, module):
        """Precompute this lane's gamma LDS read base byte offset.

        Faithful to the #rowg/#goff gamma LDS index of @mega_fused_body (MLIR
        280-284, 354-355): LDS[idx] = gamma[WorkGroup0*MT0 + idx] with
        idx = waveM*(T_M*16) + m*16 + g*4 + k. The index-independent per-lane
        term is waveM*(T_M*16) + g*4; its byte offset is that * 2 (bf16). The
        body (F4) adds the m*32 tile term per pair. Freed in the final teardown.
        """
        vgprPool = self.writer.vgprPool
        self.gammaReadBaseV = vgprPool.checkOut(1, tag="rms_gammaReadBase")
        module.addComment1("gamma read base = (waveM*(T_M*16) + g*4) * 2 (bf16 LDS byte offset).")
        rwSpanM = self.T_M * 16
        # waveMTerm = waveM * (T_M*16): shift when power of two, else materialized mul.
        if rwSpanM & (rwSpanM - 1) == 0:
            module.add(VLShiftLeftB32(dst=vgpr(self.gammaReadBaseV),
                                      shiftHex=hex(int(math.log2(rwSpanM))),
                                      src=vgpr(self.waveMV),
                                      comment=f"waveMTerm = waveM * {rwSpanM} (T_M*16)."))
        else:
            tmpV = vgprPool.checkOut(1, tag="rms_gammaReadTmp")
            module.add(VMovB32(dst=vgpr(tmpV), src=rwSpanM, comment=f"T_M*16={rwSpanM}."))
            module.add(VMulLOU32(dst=vgpr(self.gammaReadBaseV), src0=vgpr(tmpV), src1=vgpr(self.waveMV),
                                 comment=f"waveMTerm = waveM * {rwSpanM} (T_M*16)."))
            vgprPool.checkIn(tmpV)
        module.add(VAddU32(vgpr(self.gammaReadBaseV), vgpr(self.gammaReadBaseV), vgpr(self.g4V),
                           comment="+= g4 (native row-group offset)."))
        module.add(VLShiftLeftB32(dst=vgpr(self.gammaReadBaseV), shiftHex=hex(1),
                                  src=vgpr(self.gammaReadBaseV),
                                  comment="gamma read base byte offset = idx * 2 (bf16)."))
        if self.gammaLdsBase:
            module.add(VAddU32(vgpr(self.gammaReadBaseV), vgpr(self.gammaReadBaseV), self.gammaLdsBase,
                               comment=f"+= gammaLdsBase({self.gammaLdsBase})."))

    def _readGammaLds(self, module, g8Bank, mp):
        """Read this pair's two native gamma tiles from flat gamma LDS into g8Bank.

        Faithful port of the gamma LDS read in @mega_fused_body (MLIR 353-371):
        tile m -> g8Bank[0..3], tile m+1 -> g8Bank[4..7], native f32 order. The
        two tiles are 16 rows (32 bf16 bytes) apart in LDS. ``g8Bank`` is an
        8-wide VGPR block owned by the caller.
        """
        vgprPool = self.writer.vgprPool
        m = 2 * mp
        gldsAddr = vgprPool.checkOut(1, tag="rms_gldsAddr")
        self._vaddImm(module, gldsAddr, self.gammaReadBaseV, m * 16 * 2,
                      f"gldsAddr = gammaReadBase + m*32 (m={m}).")
        module.add(DSLoadB64(dst=vgpr(g8Bank + 0, 2), src=vgpr(gldsAddr),
                             ds=DSModifiers(offset=0), comment="tile m gamma: 4 bf16 (2 dwords)."))
        module.add(DSLoadB64(dst=vgpr(g8Bank + 4, 2), src=vgpr(gldsAddr),
                             ds=DSModifiers(offset=32), comment="tile m+1 gamma: 4 bf16 (+16 rows)."))
        module.add(SWaitCnt(dscnt=0, comment="wait gamma ds_reads (lgkmcnt 0)."))
        # Expand each 2-dword half to 4 f32, high-index first (in place).
        for i in range(3, -1, -1):
            module.add(VCvtBF16toFP32(vgpr(g8Bank + 0 + i), vgpr(g8Bank + 0 + i // 2), None, i % 2,
                                      comment=f"tile m gamma f32[{i}] = extf(dword {i // 2}, half {i % 2})."))
        for i in range(3, -1, -1):
            module.add(VCvtBF16toFP32(vgpr(g8Bank + 4 + i), vgpr(g8Bank + 4 + i // 2), None, i % 2,
                                      comment=f"tile m+1 gamma f32[{i}] = extf(dword {i // 2}, half {i % 2})."))
        vgprPool.checkIn(gldsAddr)

    def _storeResidualOut(self, module, hRegs, mp, n, isX4, srd, colStride, colBase):
        """Store bf16(H) for pair ``mp``, column-tile ``n`` to the given SRD.

        Faithful port of @store_data (MLIR 166-218). ``hRegs`` lists the 8
        native-order f32 H VGPRs. WIDE packs, pair-shuffles, and issues one
        dwordx4; NARROW issues eight per-row clamped bf16 stores. Addressing
        mirrors the residual load (@load_raw). ``srd`` is the SGPR name for the
        4-sgpr buffer descriptor; ``colStride`` is the SGPR name for the column
        leading-dimension stride (use "SizesFree+0" for the residual stream,
        self.dStrideSgpr for the D stream). ``colBase`` is the column-base VGPR:
        the GLOBAL self.colBaseV for the residual stream, the tile-relative
        self.colBaseDV for the D stream (SrdD already folds WG1*MT1).
        """
        if isX4:
            self._storeResidualOutWide(module, hRegs, mp, n, srd, colStride, colBase)
        else:
            self._storeResidualOutNarrow(module, hRegs, mp, n, srd, colStride, colBase)

    def _storeResidualOutWide(self, module, hRegs, mp, n, srd, colStride, colBase):
        """Pack 8 f32 -> 4 bf16 dwords, pair-shuffle, one dwordx4 store (MLIR 173-178)."""
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        lsc = self.laneSGPRCount
        m = 2 * mp
        vPack = vgprPool.checkOutAligned(4, 4, tag="rms_resStorePack")
        for p in range(4):
            module.add(VCvtPkF32toBF16(dst=vgpr(vPack + p), src0=vgpr(hRegs[2 * p]),
                                       src1=vgpr(hRegs[2 * p + 1]),
                                       comment=f"pack f32 pair ({2 * p},{2 * p + 1}) -> bf16 dword {p}."))
        # native -> tile-contiguous (involutive) before the coalesced store.
        self._pairShuffle(module, vPack)
        rowBaseP = vgprPool.checkOut(1, tag="rms_resStoreRow")
        col      = vgprPool.checkOut(1, tag="rms_resStoreCol")
        addr     = vgprPool.checkOut(1, tag="rms_resStoreAddr")
        module.add(VAddU32(vgpr(rowBaseP), vgpr(self.rowOriginV), vgpr(self.pairRowOffV),
                           comment="rowBaseP = rowOrigin + pairRowOff."))
        self._vaddImm(module, rowBaseP, rowBaseP, m * 16, f"rowBaseP += m*16 (m={m}).")
        self._vaddImm(module, col, colBase, n * 16, f"col = colBase + n*16 (n={n}).")
        module.add(VMulLOU32(dst=vgpr(addr), src0=sgpr(colStride), src1=vgpr(col),
                             comment="addr = column * rowExtent."))
        module.add(VAddU32(vgpr(addr), vgpr(addr), vgpr(rowBaseP), comment="addr += rowBaseP."))
        module.add(VLShiftLeftB32(dst=vgpr(addr), shiftHex=hex(1), src=vgpr(addr),
                                  comment="addr *= 2 (bf16 bytes)."))
        mask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_resStoreMask", preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(mask, lsc), src0=vgpr(rowBaseP), src1=sgpr("SizesFree+0"),
                             comment="inRange = rowBaseP < rowExtent."))
        module.add(VCndMaskB32(dst=vgpr(addr), src0=vgpr(self.oobV), src1=vgpr(addr),
                               src2=sgpr(mask, lsc), comment="OOB addr when rowBaseP >= rowExtent."))
        module.add(BufferStoreB128(src=vgpr(vPack, 4), vaddr=vgpr(addr),
                                   saddr=sgpr(srd, 4), soffset=0,
                                   mubuf=MUBUFModifiers(offen=True),
                                   comment="store dwordx4 bf16 -> output buffer."))
        sgprPool.checkIn(mask)
        vgprPool.checkIn(rowBaseP)
        vgprPool.checkIn(col)
        vgprPool.checkIn(addr)
        vgprPool.checkIn(vPack)

    def _storeResidualOutNarrow(self, module, hRegs, mp, n, srd, colStride, colBase):
        """Eight per-row clamped bf16 stores, one per native tile (MLIR 179-216)."""
        self._storeResidualOutNarrowTile(module, hRegs, 0, 2 * mp, n, srd, colStride, colBase)
        self._storeResidualOutNarrowTile(module, hRegs, 4, 2 * mp + 1, n, srd, colStride, colBase)

    def _storeResidualOutNarrowTile(self, module, hRegs, slotBase, ti, n, srd, colStride, colBase):
        """Store one native tile's four bf16 H rows, per-row clamped."""
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        lsc = self.laneSGPRCount
        rowBaseT = vgprPool.checkOut(1, tag="rms_resStoreRowT")
        col      = vgprPool.checkOut(1, tag="rms_resStoreColN")
        row      = vgprPool.checkOut(1, tag="rms_resStoreRowN")
        addr     = vgprPool.checkOut(1, tag="rms_resStoreAddrN")
        val      = vgprPool.checkOut(1, tag="rms_resStoreVal")
        module.add(VAddU32(vgpr(rowBaseT), vgpr(self.rowOriginV), vgpr(self.g4V),
                           comment="rowBaseT = rowOrigin + g4."))
        self._vaddImm(module, rowBaseT, rowBaseT, ti * 16, f"rowBaseT += ti*16 (ti={ti}).")
        self._vaddImm(module, col, colBase, n * 16, f"col = colBase + n*16 (n={n}).")
        for k in range(4):
            mask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_resStoreMaskN", preventOverflow=False)
            module.add(VAddU32(vgpr(row), vgpr(rowBaseT), k, comment=f"row = rowBaseT + {k}."))
            module.add(VCmpLtU32(dst=sgpr(mask, lsc), src0=vgpr(row), src1=sgpr("SizesFree+0"),
                                 comment="inRange = row < rowExtent."))
            module.add(VMulLOU32(dst=vgpr(addr), src0=sgpr(colStride), src1=vgpr(col),
                                 comment="addr = column * colStride."))
            module.add(VAddU32(vgpr(addr), vgpr(addr), vgpr(row), comment="addr += row."))
            module.add(VLShiftLeftB32(dst=vgpr(addr), shiftHex=hex(1), src=vgpr(addr),
                                      comment="addr *= 2 (bf16 bytes)."))
            module.add(VCndMaskB32(dst=vgpr(addr), src0=vgpr(self.oobV), src1=vgpr(addr),
                                   src2=sgpr(mask, lsc), comment="OOB addr when row >= rowExtent."))
            module.add(VCvtPkF32toBF16(dst=vgpr(val), src0=vgpr(hRegs[slotBase + k]),
                                       src1=vgpr(hRegs[slotBase + k]),
                                       comment=f"val = bf16(H[{slotBase + k}])."))
            module.add(BufferStoreB16(src=vgpr(val), vaddr=vgpr(addr),
                                      saddr=sgpr(srd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"store bf16 = residualOut[row {k}]."))
            sgprPool.checkIn(mask)
        vgprPool.checkIn(rowBaseT)
        vgprPool.checkIn(col)
        vgprPool.checkIn(row)
        vgprPool.checkIn(addr)
        vgprPool.checkIn(val)

    def _emitBody(self, module, vgprTiles, isX4):
        """Per-arm two-phase pair loop with a depth-``PREFETCH`` residual ring.

        Faithful port of the @mega_fused_body two-phase loop (MLIR 290-397) over
        all pairs in COLUMN-FASTEST order (n fastest): t -> n = t % T_N,
        mp = t // T_N. Each group of ``PREFETCH`` pairs first issues all P loads
        (phase 1), then consumes each in turn (phase 2), so the next group's loads
        cannot overwrite a bank until every consume of this group has finished in
        program order (no WAR hazard on the ring).

        Per-pair vmem = 1 load + 2 stores (residualOut store, then D store). gfx950
        has a single combined vmcnt (loads+stores, FIFO). At consume d, load_d must
        be complete; the VMEM ops issued more recently than load_d that may still be
        in flight are the (P-1-d) not-yet-consumed prefetch loads plus the 2*d stores
        from the d earlier consumes in this group, so the wait is vmcnt((P-1-d)+2d)
        = vmcnt(P-1+d). For P=2 this gives consume0 vmcnt(1) and consume1 vmcnt(2),
        matching the two-store reference schedule.
        """
        P = self.PREFETCH
        module.addComment0(f"MegaFused body (PREFETCH={P}, isX4={isX4}).")
        vgprPool = self.writer.vgprPool
        resBanks = [vgprPool.checkOutAligned(8, 4, tag=f"rms_resBank{d}") for d in range(P)]
        accStage = vgprPool.checkOutAligned(8, 2, tag="rms_accStage")
        g8Bank   = vgprPool.checkOutAligned(8, 2, tag="rms_g8Bank")
        ssqAcc   = vgprPool.checkOut(self.T_N, tag="rms_ssqAcc")
        self.ssqAccBase = ssqAcc
        module.addComment1("zero-init ssqAcc[0..T_N-1].")
        for n in range(self.T_N):
            module.add(VMovB32(dst=vgpr(ssqAcc + n), src=0, comment=f"ssqAcc[{n}] = 0.0"))
        for grp in range(0, self.numPairs, P):
            # Phase 1: issue all P prefetch loads for this group.
            for d in range(P):
                t = grp + d
                n = t % self.T_N
                mp = t // self.T_N
                module.addComment1(f"prefetch load pair t={t} (mp={mp}, n={n}) -> bank {d}.")
                self._loadRaw(module, resBanks[d], mp, n, isX4)
            # Phase 2: consume each. At consume d the in-flight VMEM ops newer than
            # load_d are (P-1-d) remaining prefetch loads + 2*d earlier-consume
            # stores = P-1+d, so wait vmcnt(P-1+d) (two stores/pair; P=2 -> 1,2).
            for d in range(P):
                t = grp + d
                n = t % self.T_N
                mp = t // self.T_N
                module.addComment1(f"consume pair t={t} (mp={mp}, n={n}) from bank {d}.")
                module.add(_SWaitCnt(lgkmcnt=-1, vmcnt=P - 1 + d,
                                     comment=f"combined vmcnt({P - 1 + d}): (P-1-d) loads + 2d stores in flight."))
                if isX4:
                    self._pairShuffle(module, resBanks[d])
                residualF32 = self._residualToF32(module, resBanks[d], isX4)
                self._computePair(module, vgprTiles, residualF32, accStage, g8Bank, ssqAcc, mp, n, isX4)
        vgprPool.checkIn(g8Bank)
        vgprPool.checkIn(accStage)
        for b in reversed(resBanks):
            vgprPool.checkIn(b)
        # Reduction: combine row-groups, cross-wave combine, and partial write.
        self._combineRowGroups(module, ssqAcc)
        self._writePartials(module, ssqAcc)
        vgprPool.checkIn(ssqAcc)

    def _combineRowGroups(self, module, ssqAccBase):
        """Intra-wave row-group XOR butterfly over ssqAcc (no LDS).

        Faithful port of @combine_rowgroups (@mega_fused_epilogue 226-239) and
        its per-column use (402-406). Each ssqAcc[n] holds this lane's per-row-
        group partial sum of H^2; a 2-round XOR butterfly over the
        waveSize//mfmaN = 4 row-groups (g = lane>>4) folds them so every lane
        ends with the wavefront-complete partial for its column. Rounds XOR the
        lane with bit 4 (16) then bit 5 (32); ds_bpermute gathers the partner
        lane's running sum, which is then added.
        """
        module.addComment0("intra-wave row-group XOR butterfly over ssqAcc (no LDS).")
        vgprPool = self.writer.vgprPool
        numRounds = int(math.log2(self.waveSize // self.mfmaN))
        # Precompute the two partner byte-addresses (independent of n).
        addrs = []
        for r in range(numRounds):
            xorVal = self.mfmaN << r
            a = vgprPool.checkOut(1, tag=f"rms_bpAddr{r}")
            module.add(VXorB32(dst=vgpr(a), src0=vgpr(self.laneV), src1=xorVal,
                               comment=f"partner = lane ^ {xorVal}."))
            module.add(VLShiftLeftB32(dst=vgpr(a), shiftHex=hex(2), src=vgpr(a),
                                      comment="byteAddr = partner * 4."))
            addrs.append(a)
        tmp = vgprPool.checkOut(1, tag="rms_bpTmp")
        for n in range(self.T_N):
            for r in range(numRounds):
                module.add(DSBPermuteB32(vgpr(tmp), vgpr(addrs[r]), vgpr(ssqAccBase + n),
                                         comment=f"fetch partner ssqAcc[{n}] (round {r})."))
                module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute."))
                module.add(VAddF32(dst=vgpr(ssqAccBase + n), src0=vgpr(ssqAccBase + n),
                                   src1=vgpr(tmp), comment=f"ssqAcc[{n}] += partner."))
        vgprPool.checkIn(tmp)
        for a in reversed(addrs):
            vgprPool.checkIn(a)

    def _vaddImm(self, module, dst, src, imm, comment):
        """dst = src + imm, materializing imm in a temp VGPR above the inline cap.

        Inline VALU literals cap at 64, and a VOP2 32-bit literal is only encodable
        in the src0 slot, so a larger immediate second operand is first loaded into
        a temp VGPR. The temp is independent of src, so this is safe in place
        (dst == src).
        """
        if imm <= 64:
            module.add(VAddU32(vgpr(dst), vgpr(src), imm, comment=comment))
            return
        vgprPool = self.writer.vgprPool
        tmp = vgprPool.checkOut(1, tag="rms_immTmp")
        module.add(VMovB32(dst=vgpr(tmp), src=imm, comment=f"imm = {imm}."))
        module.add(VAddU32(vgpr(dst), vgpr(src), vgpr(tmp), comment=comment))
        vgprPool.checkIn(tmp)

    def _addColOffset(self, module, dst, base, off):
        """dst = base + off (column = colBase + n*16); see _vaddImm for the cap handling."""
        self._vaddImm(module, dst, base, off, f"column = colBase + {off}.")

    def _storePartialColumn(self, module, valReg, n, ntV, partialSrd, colV, addrV, colMask):
        """Predicated store of one column's partial to partialBuf (shared by both G_M paths).

        Faithful to the per-column store in @mega_fused_body (MLIR 445-448,
        459-463): column = colBase + n*16, the partial byte offset is
        (column*numRowTiles + WorkGroup0)*4, OOB-clamped when column>=colExtent.
        """
        lsc = self.laneSGPRCount
        self._addColOffset(module, colV, self.colBaseV, n * 16)
        module.add(VCmpLtU32(dst=sgpr(colMask, lsc), src0=vgpr(colV), src1=sgpr("SizesFree+1"),
                             comment="column < colExtent."))
        module.add(VMulLOU32(dst=vgpr(addrV), src0=vgpr(ntV), src1=vgpr(colV),
                             comment="column*numRowTiles."))
        module.add(VAddU32(vgpr(addrV), vgpr(addrV), sgpr("WorkGroup0"),
                           comment="+ WorkGroup0 (rowTileIdx)."))
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(2), src=vgpr(addrV),
                                  comment="byte = idx*4."))
        module.add(VCndMaskB32(dst=vgpr(addrV), src0=vgpr(self.oobV), src1=vgpr(addrV),
                               src2=sgpr(colMask, lsc), comment="OOB when column>=colExtent."))
        module.add(BufferStoreB32(src=vgpr(valReg), vaddr=vgpr(addrV),
                                  saddr=sgpr(partialSrd, 4), soffset=0,
                                  mubuf=MUBUFModifiers(offen=True),
                                  comment=f"partialBuf[column,WG0] = partial[n={n}]."))

    def _writePartials(self, module, ssqAccBase):
        """Predicated write of each column's wavefront-complete Sigma(H^2) to partialBuf.

        Faithful port of @mega_fused_body's partial write (MLIR 399-466). Only the
        WRITER lanes (waveM==0 && g==0) store; the partial element byte offset is
        (column*numRowTiles + WorkGroup0)*4 with numRowTiles = ceil(SizesFree0/MT0)
        and column = colBase + n*16. This emits the shared SRD/mask setup plus the
        G_M==1 direct-store path; the G_M>1 cross-wave combine is a later milestone.
        """
        module.addComment0("predicated write of per-column Sigma(H^2) to partialBuf.")
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        lsc = self.laneSGPRCount
        # ---- partialBuf SRD (base PartialBuf, numRecords = SizesFree1*numRowTiles*4) ----
        partialSrd = sgprPool.checkOutAligned(4, 4, tag="rms_partialSrd", preventOverflow=False)
        module.add(SMovB64(dst=sgpr(partialSrd, 2), src=sgpr("PartialBuf", 2),
                           comment="partialBuf SRD base."))
        # numRowTiles = ceil(SizesFree0 / MT0).
        ntV = vgprPool.checkOut(1, tag="rms_numRowTiles")
        mt0 = self.MT0
        with self.writer.allocTmpSgpr(1, tag="rms_ntS") as nt:
            module.add(SAddU32(dst=sgpr(nt.idx), src0=sgpr("SizesFree+0"), src1=mt0 - 1,
                               comment="N + MT0-1."))
            if mt0 & (mt0 - 1) == 0:
                module.add(SLShiftRightB32(dst=sgpr(nt.idx), shiftHex=hex(mt0.bit_length() - 1),
                                           src=sgpr(nt.idx),
                                           comment=f"numRowTiles = ceil(N/MT0={mt0})."))
            else:
                p = (mt0 - 1).bit_length()
                magic = (-(-(1 << (32 + p - 1)) // mt0)) & 0xFFFFFFFF
                postShift = p - 1
                module.add(SMulHIU32(dst=sgpr(nt.idx), src0=sgpr(nt.idx), src1=hex(magic),
                                     comment=f"numRowTiles magic mul (MT0={mt0})."))
                if postShift:
                    module.add(SLShiftRightB32(dst=sgpr(nt.idx), shiftHex=hex(postShift),
                                               src=sgpr(nt.idx),
                                               comment=f"numRowTiles >> {postShift}."))
            module.add(VMovB32(dst=vgpr(ntV), src=sgpr(nt.idx), comment="numRowTiles -> VGPR."))
            # numRecords (bytes) = SizesFree1 * numRowTiles * 4.
            with self.writer.allocTmpSgpr(1, tag="rms_nrec") as nr:
                module.add(SMulI32(dst=sgpr(nr.idx), src0=sgpr("SizesFree+1"), src1=sgpr(nt.idx),
                                   comment="SizesFree1 * numRowTiles."))
                module.add(SLShiftLeftB32(dst=sgpr(partialSrd + 2), shiftHex=hex(2),
                                          src=sgpr(nr.idx), comment="numRecords *= 4 (f32)."))
        module.add(SMovB32(dst=sgpr(partialSrd + 3), src="Srd127_96", comment="partialBuf SRD flags."))
        # ---- writer mask: waveM==0 && g==0 ----
        selV = vgprPool.checkOut(1, tag="rms_writerSel")
        module.add(VOrB32(dst=vgpr(selV), src0=vgpr(self.gV), src1=vgpr(self.waveMV),
                          comment="sel = g | waveM (0 iff writer)."))
        writerMask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_writerMask", preventOverflow=False)
        module.add(VCmpEQU32(dst=sgpr(writerMask, lsc), src0=0, src1=vgpr(selV),
                             comment="writerMask = (g==0 && waveM==0)."))
        vgprPool.checkIn(selV)
        savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="rms_pwExec", preventOverflow=False)
        if self.G_M > 1:
            # ---- G_M>1: cross-wave LDS combine, then the writer stores the sum ----
            module.addComment1("cross-wave LDS combine (G_M>1).")
            tidX4 = vgprPool.checkOut(1, tag="rms_pwTidX4")
            module.add(VLShiftLeftB32(dst=vgpr(tidX4), shiftHex=hex(2), src=vgpr("Serial"),
                                      comment="tid*4 (LDS byte base)."))
            # WAR barrier: gamma LDS reads done before reusing LDS base 0.
            module.add(self.writer._syncThreads(
                self.kernel, "cross-wave: ensure gamma LDS free before ssq write."))
            module.addComment1("full-exec: each lane writes ssqAcc[n] to lds[n*256 + tid].")
            for n in range(self.T_N):
                module.add(DSStoreB32(dstAddr=vgpr(tidX4), src=vgpr(ssqAccBase + n),
                                      ds=DSModifiers(offset=n * 256 * 4),
                                      comment=f"lds[n={n}, tid] = ssqAcc[{n}]."))
            module.add(SWaitCnt(dscnt=0, comment="wait ssq LDS writes."))
            module.add(self.writer._syncThreads(self.kernel, "cross-wave: publish ssq LDS."))
            # writer-exec: sum the G_M sibling waves' slots, then store.
            module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(writerMask, lsc),
                                       comment="exec = writer lanes."))
            strideBytes = 64 * 4   # waveLdsStride = 64 elements (M-fastest: consecutive waveIds are the G_M siblings).
            rowSum = vgprPool.checkOut(1, tag="rms_pwRowSum")
            tmp = vgprPool.checkOut(1, tag="rms_pwTmp")
            colV = vgprPool.checkOut(1, tag="rms_pwCol")
            addrV = vgprPool.checkOut(1, tag="rms_pwAddr")
            colMask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_pwColMask", preventOverflow=False)
            for n in range(self.T_N):
                for wm in range(self.G_M):
                    off = n * 256 * 4 + wm * strideBytes
                    assert off < 65536, "lds ds offset exceeds 16-bit; fold into the address vgpr"
                    module.add(DSLoadB32(dst=vgpr(tmp), src=vgpr(tidX4),
                                         ds=DSModifiers(offset=off),
                                         comment=f"lds sibling wave {wm}, col {n}."))
                    module.add(SWaitCnt(dscnt=0, comment="wait lds read."))
                    if wm == 0:
                        module.add(VMovB32(dst=vgpr(rowSum), src=vgpr(tmp),
                                           comment="rowSum = wave0 partial."))
                    else:
                        module.add(VAddF32(dst=vgpr(rowSum), src0=vgpr(rowSum), src1=vgpr(tmp),
                                           comment=f"rowSum += wave{wm} partial."))
                self._storePartialColumn(module, rowSum, n, ntV, partialSrd, colV, addrV, colMask)
            module.add(SWaitCnt(vscnt=0, comment="drain partialBuf stores."))
            module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc), comment="restore exec."))
            vgprPool.checkIn(addrV)
            vgprPool.checkIn(colV)
            vgprPool.checkIn(tmp)
            vgprPool.checkIn(rowSum)
            vgprPool.checkIn(tidX4)
            sgprPool.checkIn(colMask)
        else:
            # ---- G_M==1: writer stores ssqAcc[n] directly, OOB-clamped by column<colExtent ----
            module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(writerMask, lsc),
                                       comment="exec = writer lanes."))
            colV = vgprPool.checkOut(1, tag="rms_pwCol")
            addrV = vgprPool.checkOut(1, tag="rms_pwAddr")
            colMask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_pwColMask", preventOverflow=False)
            for n in range(self.T_N):
                self._storePartialColumn(module, ssqAccBase + n, n, ntV, partialSrd,
                                         colV, addrV, colMask)
            module.add(SWaitCnt(vscnt=0, comment="drain partialBuf stores."))
            module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc), comment="restore exec."))
            vgprPool.checkIn(addrV)
            vgprPool.checkIn(colV)
            sgprPool.checkIn(colMask)
        # ---- shared teardown (both branches) ----
        vgprPool.checkIn(ntV)
        sgprPool.checkIn(savedExec)
        sgprPool.checkIn(writerMask)
        sgprPool.checkIn(partialSrd)

    def _isPackPair(self, a, b):
        """True iff (a, b) is an even-aligned consecutive VGPR pair (packable)."""
        return (a % 2 == 0) and (b == a + 1)

    def _pkMulPairs(self, module, dstBase, aBase, bBase, comment):
        """Emit dst = a * b over 8 contiguous f32, packing even-aligned pairs.

        dstBase/aBase/bBase are the base VGPRs of three 8-wide contiguous banks;
        each packable pair becomes one v_pk_mul_f32, else two scalar v_mul_f32.
        """
        for p in range(4):
            d0 = dstBase + 2 * p
            a0 = aBase + 2 * p
            b0 = bBase + 2 * p
            if (self._isPackPair(d0, d0 + 1) and self._isPackPair(a0, a0 + 1)
                    and self._isPackPair(b0, b0 + 1)):
                module.add(VMulPKF32(dst=vgpr(d0, 2), src0=vgpr(a0, 2), src1=vgpr(b0, 2),
                                     comment=f"{comment} (packed {2 * p},{2 * p + 1})."))
                continue
            module.add(VMulF32(dst=vgpr(d0), src0=vgpr(a0), src1=vgpr(b0),
                               comment=f"{comment} ({2 * p})."))
            module.add(VMulF32(dst=vgpr(d0 + 1), src0=vgpr(a0 + 1), src1=vgpr(b0 + 1),
                               comment=f"{comment} ({2 * p + 1})."))

    def _pkAddPairs(self, module, dstBase, aRegs, bRegs, comment):
        """Emit dst = a + b over 8 f32, packing per-pair when all three align.

        dstBase is the base of an 8-wide contiguous bank; aRegs/bRegs are LISTS
        of 8 VGPRs (acc may be aliased tile regs, not a contiguous bank). A pair
        packs only when its dst, a, and b sub-pairs are each even-aligned
        consecutive; otherwise it falls back to two scalar v_add_f32.
        """
        for p in range(4):
            d0 = dstBase + 2 * p
            a0, a1 = aRegs[2 * p], aRegs[2 * p + 1]
            b0, b1 = bRegs[2 * p], bRegs[2 * p + 1]
            if (self._isPackPair(d0, d0 + 1) and self._isPackPair(a0, a1)
                    and self._isPackPair(b0, b1)):
                module.add(VAddPKF32(dst=vgpr(d0, 2), src0=vgpr(a0, 2), src1=vgpr(b0, 2),
                                     comment=f"{comment} (packed {2 * p},{2 * p + 1})."))
                continue
            module.add(VAddF32(dst=vgpr(d0), src0=vgpr(a0), src1=vgpr(b0),
                               comment=f"{comment} ({2 * p})."))
            module.add(VAddF32(dst=vgpr(d0 + 1), src0=vgpr(a1), src1=vgpr(b1),
                               comment=f"{comment} ({2 * p + 1})."))

    def _computePair(self, module, vgprTiles, residualF32, accStage, g8Bank,
                     ssqAccBase, mp, n, isX4):
        """Per-pair compute: H = residual + acc, ssq partial, residualOut, D = H*gamma.

        Faithful port of @mega_fused_body (MLIR 352-396) for pair ``mp``,
        column-tile ``n``. ``residualF32`` is an 8-wide caller-owned bank holding
        the native f32 residual on entry; it holds H8 after the add and Dout after
        the gamma multiply. ``accStage`` stages AGPR accumulator reads, ``g8Bank``
        receives the pair's native gamma, and ``ssqAccBase`` is the T_N-wide f32
        column partial array (caller zero-inits before the pair loop).

        The one sanctioned deviation from MLIR is H = extf(native) + acc; the
        reference uses H = residual directly.
        """
        vgprPool = self.writer.vgprPool
        # ---- Step 1: read the accumulator into native-order accRegs ----
        accRegs = []
        slot = 0
        for j in range(8):
            tileIdx = 2 * mp if j < 4 else 2 * mp + 1
            k = j % 4
            tile = vgprTiles[n * self.T_M + tileIdx]
            reg = tile.regList.indices[k]
            if tile.regList.pool == self.writer.vgprPool:
                accRegs.append(reg)
                continue
            module.add(VAccvgprReadB32(dst=vgpr(accStage + slot), src=accvgpr(reg),
                                       comment=f"stage acc (mp={mp},n={n},j={j})."))
            accRegs.append(accStage + slot)
            slot += 1
        if slot > 0:
            module.add(SNop(waitState=1, comment="accvgpr_read->VALU hazard (gfx950)"))

        # ---- Step 2: H8 = residual + acc (in place, MLIR 352 + acc deviation) ----
        # residualF32 is a contiguous even-aligned bank (packable); accRegs packs
        # when AGPR-staged into the even-aligned accStage, else falls back scalar.
        self._pkAddPairs(module, residualF32[0], residualF32, accRegs,
                         "H8 = residual + acc")

        # ---- Step 3: ssq pairwise tree (MLIR 374-386), scalar faithful tree ----
        sq = vgprPool.checkOutAligned(8, 2, tag="rms_ssq")
        self._pkMulPairs(module, sq, residualF32[0], residualF32[0], "sq = H8^2")
        module.add(VAddF32(dst=vgpr(sq + 0), src0=vgpr(sq + 0), src1=vgpr(sq + 1), comment="s01."))
        module.add(VAddF32(dst=vgpr(sq + 2), src0=vgpr(sq + 2), src1=vgpr(sq + 3), comment="s23."))
        module.add(VAddF32(dst=vgpr(sq + 4), src0=vgpr(sq + 4), src1=vgpr(sq + 5), comment="s45."))
        module.add(VAddF32(dst=vgpr(sq + 6), src0=vgpr(sq + 6), src1=vgpr(sq + 7), comment="s67."))
        module.add(VAddF32(dst=vgpr(sq + 0), src0=vgpr(sq + 0), src1=vgpr(sq + 2), comment="s0123."))
        module.add(VAddF32(dst=vgpr(sq + 4), src0=vgpr(sq + 4), src1=vgpr(sq + 6), comment="s4567."))
        module.add(VAddF32(dst=vgpr(sq + 0), src0=vgpr(sq + 0), src1=vgpr(sq + 4), comment="s = s0123 + s4567."))
        module.add(VAddF32(dst=vgpr(ssqAccBase + n), src0=vgpr(ssqAccBase + n), src1=vgpr(sq + 0),
                           comment=f"ssqAcc[{n}] += s."))
        vgprPool.checkIn(sq)

        # ---- Step 5: gamma read (MLIR 353-371), before the stores ----
        self._readGammaLds(module, g8Bank, mp)

        # ---- Step 6: D = H8 * gamma into a SEPARATE bank (MLIR 372). Computing out
        # of place keeps residualF32 = H8 live so the residual store below still
        # writes H (the reference keeps %native and %Dout both live).
        dBank = vgprPool.checkOutAligned(8, 2, tag="rms_dBank")
        self._pkMulPairs(module, dBank, residualF32[0], g8Bank, "Dout = H8 * gamma")

        # ---- Steps 4 + 6b: both stores issued back-to-back after H*gamma (residual
        # H8 first, then D), matching the @mega_fused_body schedule (MLIR 389-392). ----
        self._storeResidualOut(module, residualF32, mp, n, isX4,
                               self.residualOutSrd, "SizesFree+0", self.colBaseV)
        # WAR hazard: the residual store's dwordx4 reads its bf16 pack registers, and
        # the D store reuses the same pack bank. Let the store's source read drain
        # before the D pack overwrites it (the reference spaces these with two nops).
        module.add(VNop(2, comment="WAR: drain residual store pack read before D pack reuses it."))
        self._storeResidualOut(module, [dBank + j for j in range(8)], mp, n, isX4,
                               self.dOutSrd, self.dStrideSgpr, self.colBaseDV)
        vgprPool.checkIn(dBank)

        # D is stored directly to SrdD in Step 6b and GlobalWriteBatch is skipped on
        # the native path, so there is no downstream accumulator reader: the former
        # Step 7 writeback of D into the acc/C-tile registers is dead and removed.
