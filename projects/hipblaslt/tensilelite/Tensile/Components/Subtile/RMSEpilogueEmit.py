# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RMSEpilogueEmitter: fused ResidualAdd + PartialRMS (+ MXFP8Quant) epilogue.

Replacement emitter for the Subtile/CMS MegaFused epilogue. Milestone 1 is a
thin delegating shim over the existing SubtileMegaFusedEmitter; subsequent
milestones move the emission logic here, restructured to mirror the reference
kernel mega_fused_epilogue.mlir (load_raw / pair_shuffle / store_data /
combine_rowgroups), and integrate the AGPR-held accumulator (H) path that the
reference kernel does not model.
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

from .SubtileMegaFusedEmit import (
    SubtileMegaFusedEmitter,
    RMSEpilogueGeometry,
    _buildBufferSrd,
    _useDwordx4Interior,
)


class RMSEpilogueEmitter:
    """Emit the fused RMS/Residual epilogue for the Subtile/CMS gfx950 kernel."""

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        # Milestone 1: delegate verbatim. Later milestones replace this with a
        # native implementation and drop the delegate.
        self._delegate = SubtileMegaFusedEmitter(writer, kernel)
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
        # Gamma VGPR bank (allocated during _emitSharedSetup).
        self.gammaBank = None

    def _computeWaveM(self, module, dst: int) -> None:
        # Callers with wgM > 1 can use self.waveIdV, which emit() caches.
        module.addComment1("compute waveM = waveId mod wgM.")
        module.add(VAndB32(dst=vgpr(dst), src0=vgpr(self.waveIdV), src1=self.geom.wgM - 1,
                           comment=f"waveM = waveId % {self.geom.wgM}"))

    def _emitSharedSetup(self, module) -> None:
        """Emit the non-MXFP8 shared setup: register checkouts, SRDs, row geometry.

        Faithful port of SubtileMegaFusedEmitter.emit() lines 2021-2183. The
        resulting gammaBank is stored as self.gammaBank for later milestones.
        """
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
        # colByte encodes the per-lane token index as (vw1*laneN) * elemBytes; VW1 sub-tiling
        # places vw1 consecutive physical columns under one lane, so laneN scales by vw1.
        # wave/wg offsets added below. For vw1=1 log2Vw1=0, so this is unchanged.
        log2Vw1 = int(math.log2(self.geom.vw1))
        _setupSharedMod.add(VLShiftLeftB32(dst=vgpr(self.colByte), shiftHex=hex(self.geom.log2ElemBytes + log2Vw1),
                                  src=vgpr(self.col), comment="colByte = (vw1*col) * elemBytes."))
        _setupSharedMod.add(VLShiftRightB32(dst=vgpr(self.rowGroup), shiftHex=hex(log2N),
                                   src=vgpr(self.laneId), comment="rowGroup = laneId >> log2(mfmaN)."))
        if self.geom.wgN > 1:
            _setupSharedMod.addComment1("add waveN column byte offset to colByte.")
            waveN = self.writer.vgprPool.checkOut(1, tag="rAdd_setupWaveN")
            tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="rAdd_setupTmp")
            tmpRes = ContinuousRegister(tmpVgpr, 2)
            _setupSharedMod.add(vectorStaticDivide(waveN, "Serial", self.geom.waveSize * self.geom.wgM, tmpRes,
                                          comment=f"waveN = Serial / {self.geom.waveSize * self.geom.wgM}"))
            colBaseBytes = self.geom.waveStrideN * self.geom.elemBytes
            with self.writer.allocTmpSgpr(1, tag="rAdd_setupColBase") as tmpSgprInfo:
                _setupSharedMod.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(colBaseBytes),
                                   comment=f"col base bytes per wave ({colBaseBytes})"))
                _setupSharedMod.add(VMulLOU32(dst=vgpr(waveN), src0=sgpr(tmpSgprInfo.idx), src1=vgpr(waveN),
                                     comment="waveN * waveStrideN * elemBytes"))
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
            waveStride = self.geom.waveStrideM
            strideV = self.writer.vgprPool.checkOut(1, tag="pRMS_rbStride")
            module.add(VMovB32(dst=vgpr(strideV), src=waveStride,
                               comment=f"waveStride = waveStrideM = {waveStride}"))
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
        self.gammaBank = self.writer.vgprPool.checkOutAligned(
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

    def emit(self, vgprTiles):
        return self._delegate.emit(vgprTiles)
