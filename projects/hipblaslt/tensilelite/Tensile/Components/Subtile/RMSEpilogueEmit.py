# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RMSEpilogueEmitter: faithful from-scratch RMS epilogue emitter (Subtile gfx950).

This is a ground-up rewrite that mirrors the reference kernel
mega_fused_epilogue.mlir (@mega_fused_epilogue). Milestone F1 provides only the
scaffold: config/geometry, emit() routing, and _emitSetup (the MLIR prologue,
lines 470-546). The native compute path (gamma prefetch/load/compute/reduction
and the body) arrives in later milestones (F2-F7); until then every config
DELEGATES to SubtileMegaFusedEmitter so the correctness harness stays green.
"""

import math

from rocisa.code import Module
from rocisa.container import EXEC, MUBUFModifiers, mgpr, sgpr, vgpr
from rocisa.instruction import (
    BufferLoadB32,
    BufferLoadB128,
    BufferLoadD16B16,
    DSStoreB16,
    SAddU32,
    SAndSaveExecB64,
    SLShiftLeftB32,
    SMovB32,
    SMovB64,
    SMulI32,
    SNop,
    SWaitCnt,
    VAddU32,
    VAndB32,
    VCmpLtU32,
    VCndMaskB32,
    VCvtBF16toFP32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMovB32,
    VMulLOU32,
    VPermlane16SwapB32,
    VReadfirstlaneB32,
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
        self.PREFETCH = 1

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
        self.pairRowOffV = None
        self.g4V = None
        self.oobV = None
        self.resSrd = None
        self.residualOutSrd = None
        self.gammaSrd = None
        # Byte offset of the flat gamma LDS buffer (set in _emitGammaPrefetch).
        self.gammaLdsBase = None
        # Static byte footprint of the flat gamma LDS buffer (set in prefetch).
        self.gammaLdsBytes = None

    def _nativeEligible(self):
        return (not self.useMxfp8 and not self.interleaved
                and self.rowsPerLane == 4 and self.T_M % 2 == 0)

    def emit(self, vgprTiles):
        if self._nativeEligible():
            return self._emitNative(vgprTiles)
        from .SubtileMegaFusedEmit import SubtileMegaFusedEmitter
        return SubtileMegaFusedEmitter(self.writer, self.kernel).emit(vgprTiles)

    def _emitNative(self, vgprTiles):
        # TODO(F2-F7): replace with the faithful native implementation.
        from .SubtileMegaFusedEmit import SubtileMegaFusedEmitter
        return SubtileMegaFusedEmitter(self.writer, self.kernel).emit(vgprTiles)

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
        # waveM = waveId // G_N; waveN = waveId % G_N (G_N is a power of two).
        if self.G_N == 1:
            module.add(VMovB32(dst=vgpr(self.waveMV), src=vgpr(self.waveIdV),
                               comment="waveM = waveId (G_N == 1)."))
            module.add(VMovB32(dst=vgpr(self.waveNV), src=0, comment="waveN = 0 (G_N == 1)."))
        else:
            assert self.G_N & (self.G_N - 1) == 0, "G_N must be a power of two"
            module.add(VLShiftRightB32(dst=vgpr(self.waveMV),
                                       shiftHex=hex(int(math.log2(self.G_N))),
                                       src=vgpr(self.waveIdV), comment="waveM = waveId >> log2(G_N)."))
            module.add(VAndB32(dst=vgpr(self.waveNV), src0=vgpr(self.waveIdV), src1=self.G_N - 1,
                               comment="waveN = waveId & (G_N - 1)."))

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
        sgprPool = self.writer.sgprPool
        if p == 0:
            module.add(VMovB32(dst=vgpr(idxV), src=vgpr("Serial"), comment="idx = Serial."))
        else:
            module.add(VAddU32(vgpr(idxV), vgpr("Serial"), p * blockDim,
                               comment=f"idx = Serial + {p}*blockDim."))
        savedExec = None
        if needGuard:
            mask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_gammaMask", preventOverflow=False)
            savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="rms_gammaExec", preventOverflow=False)
            module.add(VCmpLtU32(dst=sgpr(mask, lsc), src0=vgpr(idxV), src1=self.MT0,
                                 comment=f"active = idx < MT0({self.MT0})."))
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
        self.pairRowOffV = vgprPool.checkOut(1, tag="rms_pairRowOff")
        self.g4V         = vgprPool.checkOut(1, tag="rms_g4")
        self.oobV        = vgprPool.checkOut(1, tag="rms_oob")
        module.addComment1("residual load constants: colBase, pairRowOff, g4, oob.")
        # colBase = colOrigin + c (the lane's column within the wave tile).
        module.add(VAddU32(vgpr(self.colBaseV), vgpr(self.colOriginV), vgpr(self.cV),
                           comment="colBase = colOrigin + c."))
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
        module.add(VAddU32(vgpr(rowBaseP), vgpr(rowBaseP), m * 16, comment=f"rowBaseP += m*16 (m={m})."))
        # col = colBase + n*16.
        module.add(VAddU32(vgpr(col), vgpr(self.colBaseV), n * 16, comment=f"col = colBase + n*16 (n={n})."))
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
        module.add(VAddU32(vgpr(rowBaseT), vgpr(rowBaseT), ti * 16, comment=f"rowBaseT += ti*16 (ti={ti})."))
        # col = colBase + n*16.
        module.add(VAddU32(vgpr(col), vgpr(self.colBaseV), n * 16, comment=f"col = colBase + n*16 (n={n})."))
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
