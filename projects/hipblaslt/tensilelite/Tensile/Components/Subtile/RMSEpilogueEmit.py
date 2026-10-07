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
from rocisa.container import EXEC, MUBUFModifiers, sgpr, vgpr
from rocisa.instruction import (
    BufferLoadD16B16,
    DSStoreB16,
    SAndSaveExecB64,
    SLShiftLeftB32,
    SMovB32,
    SMovB64,
    SMulI32,
    SWaitCnt,
    VAddU32,
    VAndB32,
    VCmpLtU32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMovB32,
    VMulLOU32,
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
        self.resSrd = None
        self.residualOutSrd = None
        self.gammaSrd = None
        # Byte offset of the flat gamma LDS buffer (set in _emitGammaPrefetch).
        self.gammaLdsBase = None

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
