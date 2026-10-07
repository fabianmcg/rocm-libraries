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
    VPermlane16SwapB32,
    VReadfirstlaneB32,
    VXorB32,
)
from Tensile.Common.DataType import DataType

from .SubtileMegaFusedEmit import (
    SubtileMegaFusedEmitter,
    RMSEpilogueGeometry,
    _buildBufferSrd,
    _convertGammaChunk,
    _convertGammaChunkBf16,
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

    def _emitGammaDtlLoadContiguous(self, module, qi) -> None:
        """Issue one qi-block's DTL gamma load for the contiguous (Subtile) layout.

        One b32 DTL load covers all tilesPerBlockM tiles because they occupy contiguous
        global gamma rows; lane i reads gamma[2i],gamma[2i+1] and DTL writes them
        contiguously at LDS M0+i*4, matching the consumer layout.
        """
        sgprPool = self.writer.sgprPool
        # M0 = waveM*ldsWaveStride + qi*gammaLdsBufBytes (slot qi's per-wave write base).
        if qi == 0:
            module.add(SMovB32(dst=mgpr(0), src=sgpr(self.gammaM0Base),
                               comment="M0 = gamma LDS wave base (slot 0)."))
        else:
            with self.writer.allocTmpSgpr(1, tag="mf_gammaM0") as t:
                module.add(SAddU32(dst=sgpr(t.idx), src0=sgpr(self.gammaM0Base),
                                   src1=qi * self.geom.gammaLdsBufBytes,
                                   comment=f"M0 base + slot{qi} offset."))
                module.add(SMovB32(dst=mgpr(0), src=sgpr(t.idx),
                                   comment=f"M0 = gamma LDS wave base (slot {qi})."))
        # soffset = wgRowBase*gammaBytes + qi*gammaLdsWaveStride (next qi-block of gamma).
        qiSoffsetAdj = qi * self.geom.gammaLdsWaveStride
        if qiSoffsetAdj > 0:
            qiSoffSgpr = sgprPool.checkOutAligned(1, 1, tag="mf_gammaQiSoff", preventOverflow=False)
            module.add(SAddU32(dst=sgpr(qiSoffSgpr), src0=sgpr(self.gammaSoffsetSgpr),
                               src1=qiSoffsetAdj,
                               comment=f"soffset += qi*gammaLdsWaveStride for qi={qi}."))
        else:
            qiSoffSgpr = self.gammaSoffsetSgpr
        module.add(BufferLoadB32(
            dst=None, vaddr=vgpr(self.gammaDtlVaddr), saddr=sgpr(self.gammaSrd, 4),
            soffset=sgpr(qiSoffSgpr),
            mubuf=MUBUFModifiers(offen=True, offset12=0, lds=True),
            comment=f"gamma DTL b32 -> LDS slot {qi}."))
        if qiSoffsetAdj > 0:
            sgprPool.checkIn(qiSoffSgpr)

    def _emitGammaDtlLoad(self, module, qi) -> None:
        """Issue one qi-block's Direct-To-LDS gamma load(s) into the resident LDS slot(s).

        Contiguous layout (Subtile): delegates to _emitGammaDtlLoadContiguous.

        Interleaved layout (CMS): the tilesPerBlockM tiles map to non-contiguous global
        gamma rows (stride tileStrideM between tiles), so issue one b32 DTL load per
        tile, each at its own global soffset, into the same per-mi LDS slot the consumer
        already reads.  The LDS layout and the consumer are unchanged.

        The caller owns exec narrowing, the vmcnt wait, and the barriers.
        """
        if not self.geom.interleavedWaves:
            self._emitGammaDtlLoadContiguous(module, qi)
            return
        # Interleaved layout (CMS): tiles in a qi-block map to non-contiguous global
        # gamma rows (stride tileStrideM = wgM*mfmaM between consecutive tiles).
        # Issue one b32 DTL load per tile at its interleaved global soffset, writing
        # into the same per-mi LDS slot the consumer reads.
        tpb = self.geom.tilesPerBlockM
        for mi in range(tpb):
            m = qi * tpb + mi
            # Global byte soffset for this tile:
            #   gammaSoffsetSgpr already = wgRowBase * gammaBytes (wgRowBase includes
            #   waveM * waveStrideM from the wgRowBase setup), so adding
            #   m * tileStrideM * gammaBytes gives the correct interleaved row.
            tileSoffsetAdj = m * self.geom.tileStrideM * self.geom.gammaBytes
            if tileSoffsetAdj > 0:
                with self.writer.allocTmpSgpr(1, tag="mf_gammaTileSoff") as ts:
                    module.add(SAddU32(dst=sgpr(ts.idx), src0=sgpr(self.gammaSoffsetSgpr),
                                       src1=tileSoffsetAdj,
                                       comment=f"soffset = wgRowBase*gammaBytes + m*tileStrideM*gammaBytes (m={m})."))
                    self._emitGammaDtlInterleavedTileLoad(module, qi, mi, ts.idx)
            else:
                self._emitGammaDtlInterleavedTileLoad(module, qi, mi, self.gammaSoffsetSgpr)

    def _emitGammaDtlInterleavedTileLoad(self, module, qi, mi, soffSgpr) -> None:
        """Set M0 for one interleaved tile's LDS slot and issue its DTL load(s).

        For VW0=1 a single b32 DTL (2 bf16 per lane) fills a 32-byte slot per tile.
        For VW0=2 two b32 DTL loads fill a 64-byte slot per tile: the first covers the
        lower mfmaM gamma rows, the second (at M0+mfmaM*gammaBytes, soffset+mfmaM*gammaBytes)
        covers the upper mfmaM rows. The consumer reads two DSLoadB64 at off and off+8.
        """
        halfBytes = self.geom.mfmaM * self.geom.gammaBytes   # bytes for one b32-DTL half (VW0=1 full slot)
        m0Adj = qi * self.geom.gammaLdsBufBytes + mi * self.geom.mfmaM * self.geom.vw0 * self.geom.gammaBytes
        if m0Adj == 0:
            module.add(SMovB32(dst=mgpr(0), src=sgpr(self.gammaM0Base),
                               comment=f"M0 = gammaM0Base (qi={qi},mi={mi})."))
        else:
            with self.writer.allocTmpSgpr(1, tag="mf_gammaTileM0") as tm:
                module.add(SAddU32(dst=sgpr(tm.idx), src0=sgpr(self.gammaM0Base),
                                   src1=m0Adj,
                                   comment=f"M0 = gammaM0Base + qi*ldsStride + mi*mfmaM*vw0*gammaBytes (qi={qi},mi={mi})."))
                module.add(SMovB32(dst=mgpr(0), src=sgpr(tm.idx),
                                   comment=f"M0 = per-tile LDS slot base (qi={qi},mi={mi})."))
        module.add(BufferLoadB32(
            dst=None, vaddr=vgpr(self.gammaDtlVaddr), saddr=sgpr(self.gammaSrd, 4),
            soffset=sgpr(soffSgpr),
            mubuf=MUBUFModifiers(offen=True, offset12=0, lds=True),
            comment=f"gamma DTL b32 half 0 -> LDS slot (qi={qi},mi={mi})."))
        # For vw0>1: issue one b32 DTL load per additional half; each advances M0 and
        # soffset by halfBytes so successive mfmaM-row blocks map to contiguous LDS.
        for h in range(1, self.geom.vw0):
            m0AdjH = m0Adj + h * halfBytes
            with self.writer.allocTmpSgpr(1, tag="mf_gammaTileM0h") as tmh:
                module.add(SAddU32(dst=sgpr(tmh.idx), src0=sgpr(self.gammaM0Base),
                                   src1=m0AdjH,
                                   comment=f"M0 = slot base + {h}*halfBytes (qi={qi},mi={mi},h={h})."))
                module.add(SMovB32(dst=mgpr(0), src=sgpr(tmh.idx),
                                   comment=f"M0 = LDS slot for half {h} (qi={qi},mi={mi})."))
            with self.writer.allocTmpSgpr(1, tag="mf_gammaSoffH") as tsh:
                module.add(SAddU32(dst=sgpr(tsh.idx), src0=sgpr(soffSgpr),
                                   src1=h * halfBytes,
                                   comment=f"soffset += {h}*halfBytes (half {h}, qi={qi},mi={mi})."))
                module.add(BufferLoadB32(
                    dst=None, vaddr=vgpr(self.gammaDtlVaddr), saddr=sgpr(self.gammaSrd, 4),
                    soffset=sgpr(tsh.idx),
                    mubuf=MUBUFModifiers(offen=True, offset12=0, lds=True),
                    comment=f"gamma DTL b32 half {h} -> LDS slot (qi={qi},mi={mi})."))

    def _issueAllGammaToLds(self, module) -> None:
        """Issue every nQTilesM gamma qi-block's Direct-To-LDS load (no drain, no barrier).

        One WAR barrier guards the prior (mainloop) LDS users; one exec narrowing covers
        every DTL load.  Exec is restored (and its saved-exec SGPR checked in) immediately
        after issuing the loads -- exec only gates which lanes *issue* the op, so restoring
        it does not require waiting for completion (vlcnt tracks that).  The caller issues
        independent work next, then calls _drainGammaToLds to wait + publish.  Distinct qi
        slots never alias, so no inter-stage WAR barrier is needed.
        """
        module.addComment1(f"issue all gamma qi-block DTL loads up front (nQTilesM={self.geom.nQTilesM}).")
        sgprPool = self.writer.sgprPool
        lsc = self.geom.laneSgprCount
        # Interleaved layout: each per-tile DTL load needs mfmaM//2 lanes (lane i reads
        # gamma[2i,2i+1]); all per-tile loads share the same low-lane mask, so we narrow
        # exec once to mfmaM//2.  Contiguous layout: one load per qi covers tilesPerBlockM
        # tiles, requiring tilesPerBlockM*mfmaM//2 lanes.
        numStageLanes = (self.geom.mfmaM // 2) if self.geom.interleavedWaves \
            else (self.geom.tilesPerBlockM * self.geom.mfmaM // 2)
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma DTL stage: WAR barrier before reusing LDS region."))
        savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="mf_gammaStageExec", preventOverflow=False)
        module.add(SMovB64(dst=sgpr(savedExec, lsc), src=EXEC(),
                           comment="save exec around contiguous-lane DTL gamma loads."))
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
                           comment="exec = contiguous stage lanes for DTL gamma loads."))
        sgprPool.checkIn(repMask)
        for qi in range(self.geom.nQTilesM):
            self._emitGammaDtlLoad(module, qi)
        # Restore exec right after issuing; completion is tracked by vlcnt regardless of exec.
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                           comment="restore full exec after issuing gamma DTL loads."))
        sgprPool.checkIn(savedExec)

    def _ldsReadGammaBlockIssue(self, module, gammaBank, qi) -> None:
        """Issue broadcast LDS reads for staged gamma without waiting or converting.

        The wait and bf16->f32 conversion are deferred so LDS latency overlaps
        residual/RMS work.

        For VW0=1 (rowsPerLane=4) a single DSLoadB64 reads 8 bytes = 4 packed bf16
        into 2 VGPRs (base+0,1). _convertGammaChunk then expands those 4 values.

        For VW0=2 (rowsPerLane=8) two DSLoadB64 reads fill 4 contiguous VGPRs
        (base+0..3): the first read covers the lower mfmaM gamma rows (off1), the
        second covers the upper mfmaM rows (off1+mfmaM*gammaBytes). Together they
        provide the 4 packed dwords that _convertGammaChunk expands to 8 f32.
        """
        assert not self._gammaReadPending, "gamma LDS read issued while a prior read is still pending"
        module.addComment1(f"broadcast-read gamma from LDS (qi={qi}).")
        tpb = self.geom.tilesPerBlockM
        rpl = self.geom.rowsPerLane
        # Each rowGroup region in LDS spans rowsPerLane*gammaBytes bytes total.
        # The first DSLoadB64 reads k=0..3 (8 bytes); the second k=4..7 (next 8 bytes).
        # This inner gap is always 8 bytes (4 bf16), independent of VW0 or tileStrideM.
        # The second DSLoadB64 half sits at off + outputsPerMFMA gamma values =
        # 4*gammaBytes = 8 bytes = one DSLoadB64 width within the rowGroup's slice.
        innerGapBytes = 4 * self.geom.gammaBytes  # 8 bytes for 4 bf16 per DSLoadB64
        for mi in range(tpb):
            off = qi * self.geom.gammaLdsBufBytes + mi * self.geom.mfmaM * self.geom.vw0 * self.geom.gammaBytes
            # Lower half: 4 packed bf16 in 2 dwords -> VGPRs base+0, base+1.
            module.add(DSLoadB64(
                dst=vgpr(gammaBank + mi * rpl, 2),
                src=vgpr(self.gammaLdsReadAddr),
                ds=DSModifiers(offset=off),
                comment=f"broadcast-read gamma lower 4 bf16 (qi={qi},mi={mi})."))
            # Each additional half (h=1..vw0-1) reads the next block of 4 packed bf16
            # into contiguous VGPRs so _convertGammaChunk sees vw0 packed dwords at base+0..2*vw0-1.
            for h in range(1, self.geom.vw0):
                module.add(DSLoadB64(
                    dst=vgpr(gammaBank + mi * rpl + h * 2, 2),
                    src=vgpr(self.gammaLdsReadAddr),
                    ds=DSModifiers(offset=off + h * innerGapBytes),
                    comment=f"broadcast-read gamma half {h} (qi={qi},mi={mi}) off+{h * innerGapBytes}."))
        self._gammaReadPending = True

    def _pairShuffle(self, module, bank) -> None:
        """Emit the involutive pair_shuffle: 2x VPermlane16SwapB32, raw<->native.

        Mirrors @pair_shuffle in mega_fused_epilogue.mlir (lines 144-158): swaps
        d0<->d2 and d1<->d3 across lane-16 halves so the layout converts between
        8 contiguous bf16 rows per lane (raw dwordx4 load) and per-lane native
        [tileA 4 rows, tileB 4 rows] (accumulator layout). Involutive: applying
        it twice is identity.
        """
        module.addComment1("pair_shuffle: 2x permlane16_swap, raw<->native (involutive).")
        module.add(VPermlane16SwapB32(dst=vgpr(bank + 0), src=vgpr(bank + 2),
                                      comment="swap d0<->d2 across lane-16 halves (a0,b0)."))
        module.add(VPermlane16SwapB32(dst=vgpr(bank + 1), src=vgpr(bank + 3),
                                      comment="swap d1<->d3 across lane-16 halves (a1,b1)."))

    def _pairedDwordx4Eligible(self) -> bool:
        # Faithful MLIR paired dwordx4 wide path: bf16, non-MXFP8, rowsPerLane==4,
        # not partial-accum, and an even mmaM so (m, m+1) pairs are well-formed.
        return _useDwordx4Interior(self.geom) and self.geom.mmaM % 2 == 0

    def emit(self, vgprTiles):
        return self._delegate.emit(vgprTiles)
