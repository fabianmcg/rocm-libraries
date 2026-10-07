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
    _addImmU32,
    _buildBufferSrd,
    _convertGammaChunk,
    _convertGammaChunkBf16,
    _free0RowPos,
    _INLINE_CONST_MAX,
    _issueSideLoad,
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
        # Paired dwordx4 per-lane row origin; allocated by _computePairRowOff.
        self.pairRowOff = None
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

    def _computeResidualRowByteBase(self, module, n) -> None:
        """Compute self.resRowByteBase for column n (sets the attribute in place).

        The result encodes token_n*N_hidden (plus wgRowBase+rowGroupOff when wide)
        multiplied by residualBytes; subsequent tile loads add only the row offset.
        """
        module.addComment1(f"compute row byte base for residual (n={n}).")
        nOff = self.geom.colOffset(n)
        r = _addImmU32(module, self.resRowByteBase, self.resTokenBase, nOff, self.resAddr,
                       f"token_n = tokenBase + {nOff} (n={n}).")
        module.add(VMulLOU32(dst=vgpr(self.resRowByteBase), src0=sgpr("SizesFree+0"),
                             src1=vgpr(r), comment="token_n * SizesFree0."))
        if self.geom.useWideResidual:
            module.add(VAddU32(vgpr(self.resRowByteBase), vgpr(self.resRowByteBase),
                               vgpr(self.wgRowBase), comment="+ wgRowBase (fold row origin)."))
            module.add(VAddU32(vgpr(self.resRowByteBase), vgpr(self.resRowByteBase),
                               vgpr(self.rowGroupOff), comment="+ rowGroupOff (fold row origin)."))
        module.add(VLShiftLeftB32(dst=vgpr(self.resRowByteBase),
                                  shiftHex=hex(self.geom.residualLog2Bytes),
                                  src=vgpr(self.resRowByteBase),
                                  comment="rowByteBase * residualBytes (row origin folded when wide)."))

    def _loadNativeTile(self, module, bank, bankSlot, m, n, mBaseV) -> None:
        """Issue rpl clamped bf16 loads for one native tile into bank at bankSlot*rpl.

        Assumes self.resRowByteBase is set for column n by _computeResidualRowByteBase.
        """
        rpl = self.geom.rowsPerLane
        burstBase = bank + bankSlot * rpl
        module.addComment1(f"issue native per-element residual loads for tile (m={m},n={n}).")
        for k_irt in range(rpl):
            module.addComment1(f"compute clamped residual byte address (m={m},k={k_irt}).")
            _free0RowPos(module, self.resAddr, self.wgRowBase, self.rowGroupOff,
                         m, k_irt, mBaseV, self.geom)
            module.add(VCmpLtU32(dst=sgpr(self.resOobMask, self.geom.laneSgprCount),
                                 src0=vgpr(self.resAddr), src1=sgpr("SizesFree+0"),
                                 comment="inRange = nhidden_pos < N_hidden"))
            module.add(VLShiftLeftB32(dst=vgpr(self.resAddr),
                                      shiftHex=hex(self.geom.residualLog2Bytes),
                                      src=vgpr(self.resAddr),
                                      comment="nhiddenByte = nhidden_pos * residualBytes."))
            module.add(VAddU32(vgpr(self.resAddr), vgpr(self.resAddr),
                               vgpr(self.resRowByteBase),
                               comment="byteAddr = rowByteBase + nhiddenByte"))
            module.add(VCndMaskB32(dst=vgpr(self.resAddr), src0=vgpr(self.resOobV),
                                   src1=vgpr(self.resAddr),
                                   src2=sgpr(self.resOobMask, self.geom.laneSgprCount),
                                   comment="clamp OOB when nhidden_pos >= N_hidden"))
            _issueSideLoad(module, burstBase + k_irt, self.resAddr, self.resSrd,
                           f"R[m={m},n={n},k={k_irt}].", dtype=self.geom.residualType)

    def _computePairRowOff(self, module) -> None:
        """Allocate self.pairRowOff and compute (g&1)*16 + (g>>1)*8 per lane.

        pairRowOff is the per-lane paired row origin for the interior dwordx4 path;
        it converts the 8-contiguous-rows raw layout produced by the wide load to
        the native per-lane accumulator layout.
        """
        module.addComment1("compute pairRowOff = (g&1)*16 + (g>>1)*8 (paired dwordx4 row origin).")
        self.pairRowOff = self.writer.vgprPool.checkOut(1, tag="mf_pairRowOff")
        tmp = self.writer.vgprPool.checkOut(1, tag="mf_pairRowOffTmp")
        module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(self.rowGroup), src1=1, comment="g & 1."))
        module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(4), src=vgpr(tmp),
                                  comment="(g&1) << 4 = tileSel*16."))
        module.add(VLShiftRightB32(dst=vgpr(self.pairRowOff), shiftHex=hex(1),
                                   src=vgpr(self.rowGroup), comment="g >> 1."))
        module.add(VLShiftLeftB32(dst=vgpr(self.pairRowOff), shiftHex=hex(3),
                                  src=vgpr(self.pairRowOff), comment="(g>>1) << 3 = half*8."))
        module.add(VAddU32(vgpr(self.pairRowOff), vgpr(self.pairRowOff), vgpr(tmp),
                           comment="pairRowOff = tileSel*16 + half*8."))
        self.writer.vgprPool.checkIn(tmp)

    def _computeResidualRowByteBaseP(self, module, n) -> None:
        """Compute self.resRowByteBase for column n using pairRowOff instead of rowGroupOff.

        Identical to _computeResidualRowByteBase except that the wide-residual fold
        adds pairRowOff (paired row origin) rather than rowGroupOff; used by the
        interior paired dwordx4 path.
        """
        module.addComment1(f"compute paired row byte base for residual (n={n}).")
        nOff = self.geom.colOffset(n)
        r = _addImmU32(module, self.resRowByteBase, self.resTokenBase, nOff, self.resAddr,
                       f"token_n = tokenBase + {nOff} (n={n}).")
        module.add(VMulLOU32(dst=vgpr(self.resRowByteBase), src0=sgpr("SizesFree+0"),
                             src1=vgpr(r), comment="token_n * SizesFree0."))
        if self.geom.useWideResidual:
            module.add(VAddU32(vgpr(self.resRowByteBase), vgpr(self.resRowByteBase),
                               vgpr(self.wgRowBase), comment="+ wgRowBase (fold row origin)."))
            module.add(VAddU32(vgpr(self.resRowByteBase), vgpr(self.resRowByteBase),
                               vgpr(self.pairRowOff), comment="+ pairRowOff (fold paired row origin)."))
        module.add(VLShiftLeftB32(dst=vgpr(self.resRowByteBase),
                                  shiftHex=hex(self.geom.residualLog2Bytes),
                                  src=vgpr(self.resRowByteBase),
                                  comment="rowByteBase * residualBytes (paired row origin folded when wide)."))

    def _loadRaw(self, module, bank, mp, n, pathInterior) -> None:
        """Issue residual loads for pair mp (tiles m=2*mp and m1=2*mp+1) into bank.

        pathInterior=True: one 64-lane paired dwordx4 load, no per-row clamp.
        pathInterior=False: two native per-row-clamped load series (tail path).
        The caller applies _pairShuffle after this call for the interior path only;
        this method issues loads only and performs no computation.
        """
        m = 2 * mp
        m1 = 2 * mp + 1
        if pathInterior:
            self._computeResidualRowByteBaseP(module, n)
            rowOff = m * self.geom.tileStrideM * self.geom.residualBytes
            assert rowOff < 4096, \
                f"paired dwordx4 load offset {rowOff} exceeds MUBUF offset12 range"
            module.add(BufferLoadB128(
                vgpr(bank, 4), vgpr(self.resRowByteBase), sgpr(self.resSrd, 4), 0,
                MUBUFModifiers(offen=True, offset12=rowOff),
                comment=f"R paired dwordx4 (mp={mp},n={n}) off={rowOff}: 8 bf16, 64 lanes, raw."))
            return
        self._computeResidualRowByteBase(module, n)
        mBaseV = self.writer.vgprPool.checkOut(1, tag="mf_loadRawMBase")
        self._loadNativeTile(module, bank, 0, m, n, mBaseV)
        self._loadNativeTile(module, bank, 1, m1, n, mBaseV)
        self.writer.vgprPool.checkIn(mBaseV)

    def _computeRoBaseP(self, module, n):
        """Compute roRowBase and roColByteBase for column n using pairRowOff.

        Used by the interior paired dwordx4 store path. Returns the allocated
        tokMaskSgpr so the caller can check it in after use.
        """
        lsc = self.geom.laneSgprCount
        tokMaskSgpr = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mf_roTokMask", preventOverflow=False)
        self.roRowBase = self.writer.vgprPool.checkOut(1, tag="mf_roRowBase")
        self.roColByteBase = self.writer.vgprPool.checkOut(1, tag="mf_roColByteBase")
        nOff = self.geom.colOffset(n)
        tokV = self.writer.vgprPool.checkOut(1, tag="mf_roTokV")
        scratch = self.writer.vgprPool.checkOut(1, tag="mf_roTokScratch")
        r = _addImmU32(module, tokV, self.resTokenBase, nOff, scratch, f"token_n = resTokenBase + {nOff} (n={n}).")
        module.add(VCmpLtU32(dst=sgpr(tokMaskSgpr, lsc), src0=vgpr(r), src1=sgpr("SizesFree+1"),
                             comment="tokenInRange = token_n < M_tokens."))
        module.add(VMulLOU32(dst=vgpr(self.roRowBase), src0=sgpr("SizesFree+0"), src1=vgpr(r),
                             comment=f"roRowBase = token_n * N_hidden (n={n})."))
        module.add(VAddU32(dst=vgpr(self.roColByteBase), src0=vgpr(self.roRowBase), src1=vgpr(self.wgRowBase),
                           comment="roColBase = token_n*N_hidden + wgRowBase."))
        module.add(VAddU32(dst=vgpr(self.roColByteBase), src0=vgpr(self.roColByteBase), src1=vgpr(self.pairRowOff),
                           comment="roColBase += pairRowOff (paired per-lane row origin)."))
        module.add(VLShiftLeftB32(dst=vgpr(self.roColByteBase), shiftHex=hex(1), src=vgpr(self.roColByteBase),
                                  comment="roColByteBase = roColBase * 2 (bf16)."))
        self.writer.vgprPool.checkIn(scratch)
        self.writer.vgprPool.checkIn(tokV)
        return tokMaskSgpr

    def _computeRoBaseNative(self, module, n):
        """Compute roRowBase and roColByteBase for column n using rowGroupOff.

        Used by the tail per-row-clamped store path. Returns the allocated
        tokMaskSgpr so the caller can check it in after use.
        """
        lsc = self.geom.laneSgprCount
        tokMaskSgpr = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mf_roTokMask", preventOverflow=False)
        self.roRowBase = self.writer.vgprPool.checkOut(1, tag="mf_roRowBase")
        self.roColByteBase = self.writer.vgprPool.checkOut(1, tag="mf_roColByteBase")
        nOff = self.geom.colOffset(n)
        tokV = self.writer.vgprPool.checkOut(1, tag="mf_roTokV")
        scratch = self.writer.vgprPool.checkOut(1, tag="mf_roTokScratch")
        r = _addImmU32(module, tokV, self.resTokenBase, nOff, scratch, f"token_n = resTokenBase + {nOff} (n={n}).")
        module.add(VCmpLtU32(dst=sgpr(tokMaskSgpr, lsc), src0=vgpr(r), src1=sgpr("SizesFree+1"),
                             comment="tokenInRange = token_n < M_tokens."))
        module.add(VMulLOU32(dst=vgpr(self.roRowBase), src0=sgpr("SizesFree+0"), src1=vgpr(r),
                             comment=f"roRowBase = token_n * N_hidden (n={n})."))
        module.add(VAddU32(dst=vgpr(self.roColByteBase), src0=vgpr(self.roRowBase), src1=vgpr(self.wgRowBase),
                           comment="roColBase = token_n*N_hidden + wgRowBase."))
        module.add(VAddU32(dst=vgpr(self.roColByteBase), src0=vgpr(self.roColByteBase), src1=vgpr(self.rowGroupOff),
                           comment="roColBase += rowGroupOff (per-lane row origin)."))
        module.add(VLShiftLeftB32(dst=vgpr(self.roColByteBase), shiftHex=hex(1), src=vgpr(self.roColByteBase),
                                  comment="roColByteBase = roColBase * 2 (bf16)."))
        self.writer.vgprPool.checkIn(scratch)
        self.writer.vgprPool.checkIn(tokV)
        return tokMaskSgpr

    def _storeResidualOut(self, module, nativeRegs, mp, n, pathInterior):
        """Emit bf16(H) stores to residualOut for pair mp (tiles m=2*mp and m+1).

        Interior path: one paired dwordx4 store per pair (no masking, 64 lanes).
        Tail path: per-row-clamped B16 stores for each of the two tiles.
        """
        if pathInterior:
            tokMask = self._computeRoBaseP(module, n)
            vPack = self.writer.vgprPool.checkOutAligned(4, 4, tag="mf_roPackP")
            module.addComment1("pack 8 native f32 H -> 4 bf16 dwords (dword p = [n2p, n2p+1]).")
            for p in range(4):
                module.add(VCvtPkF32toBF16(dst=vgpr(vPack + p), src0=vgpr(nativeRegs[2*p]),
                                           src1=vgpr(nativeRegs[2*p+1]),
                                           comment=f"pack native[{2*p}] lo16, native[{2*p+1}] hi16 -> bf16x2."))
            self._pairShuffle(module, vPack)
            rowOff = (2*mp) * self.geom.tileStrideM * self.geom.residualBytes
            assert rowOff < 4096, f"residualOut paired dwordx4 offset {rowOff} exceeds MUBUF offset12 range"
            module.add(BufferStoreB128(src=vgpr(vPack, 4), vaddr=vgpr(self.roColByteBase),
                                       saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                       mubuf=MUBUFModifiers(offen=True, offset12=rowOff),
                                       comment=f"ResidualOut paired dwordx4 (mp={mp},n={n}) off={rowOff}: 64 lanes."))
            self.writer.vgprPool.checkIn(vPack)
            self.writer.sgprPool.checkIn(tokMask)
            self.writer.vgprPool.checkIn(self.roColByteBase); self.roColByteBase = None
            self.writer.vgprPool.checkIn(self.roRowBase); self.roRowBase = None
            return
        tokMask = self._computeRoBaseNative(module, n)
        rpl = self.geom.rowsPerLane
        lsc = self.geom.laneSgprCount
        for t in range(2):
            mTile = 2*mp + t
            nhScratch = self.writer.vgprPool.checkOut(1, tag="mf_roTailScratch")
            # self.nhBase is the setup-allocated scratch (tag "mf_nhBase"); reuse it directly here.
            _free0RowPos(module, self.nhBase, self.wgRowBase, self.rowGroupOff, mTile, 0, nhScratch, self.geom)
            for k in range(rpl):
                addrV   = self.writer.vgprPool.checkOut(1, tag="mf_roTailAddr")
                valV    = self.writer.vgprPool.checkOut(1, tag="mf_roTailVal")
                nhByteV = self.writer.vgprPool.checkOut(1, tag="mf_roTailNhByte")
                with self.writer.allocTmpSgpr(lsc, tag="mf_roTailMask") as nhMask:
                    module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(1), src=vgpr(self.roRowBase),
                                              comment="base0 = roRowBase * 2 (bf16)."))
                    nh = _addImmU32(module, nhByteV, self.nhBase, k, valV, f"nhPos = nhBase + {k} (t={t},k={k}).")
                    module.add(VCmpLtU32(dst=sgpr(nhMask.idx, lsc), src0=vgpr(nh), src1=sgpr("SizesFree+0"),
                                         comment="nhInRange = nhPos < N_hidden."))
                    module.add(VLShiftLeftB32(dst=vgpr(nhByteV), shiftHex=hex(1), src=vgpr(nh),
                                              comment="nhByte = nhPos * 2 (bf16)."))
                    module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(nhByteV),
                                       comment="byteAddr = base0 + nhByte."))
                    module.add(VCndMaskB32(dst=vgpr(addrV), src0=vgpr(self.resOobV), src1=vgpr(addrV),
                                           src2=sgpr(nhMask.idx, lsc),
                                           comment="clamp OOB when nhPos >= N_hidden."))
                    module.add(VCvtPkF32toBF16(dst=vgpr(valV), src0=vgpr(nativeRegs[t*4 + k]),
                                               src1=vgpr(nativeRegs[t*4 + k]),
                                               comment="H -> bf16 (low 16)."))
                    module.add(BufferStoreB16(src=vgpr(valV), vaddr=vgpr(addrV),
                                             saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                             mubuf=MUBUFModifiers(offen=True),
                                             comment=f"ResidualOut bf16(H) (mp={mp},t={t},k={k})."))
                self.writer.vgprPool.checkIn(nhByteV)
                self.writer.vgprPool.checkIn(valV)
                self.writer.vgprPool.checkIn(addrV)
            self.writer.vgprPool.checkIn(nhScratch)
        self.writer.sgprPool.checkIn(tokMask)
        self.writer.vgprPool.checkIn(self.roColByteBase); self.roColByteBase = None
        self.writer.vgprPool.checkIn(self.roRowBase); self.roRowBase = None

    def _computePair(self, module, vgprTiles, residualF32Regs, accStageBank, mp, n, pathInterior):
        """Compute one pair mp for column n: H=acc+residual, RMS, residualOut, gamma, writeback.

        Scalar-only; packed promotion is a later milestone.
        residualF32Regs: list of 8 VGPR ints holding the residual as f32, native order.
        accStageBank: base VGPR of an 8-slot staging area for AGPR reads.
        """
        m = 2 * mp
        m1 = 2 * mp + 1

        # Step 1: read accumulator for both tiles into VGPRs (mirror lines 967-981).
        module.addComment1(f"read accumulator into VGPR staging area (mp={mp},n={n}).")
        accRegs = []
        slot = 0
        for j in range(8):
            tileIdx = m if j < 4 else m1
            k = j % 4
            tileInfo = vgprTiles[n * self.geom.mmaM + tileIdx]
            reg = tileInfo.regList.indices[k]
            if tileInfo.regList.pool == self.writer.vgprPool:
                accRegs.append(reg)
            else:
                module.add(VAccvgprReadB32(vgpr(accStageBank + slot), accvgpr(reg),
                                           comment=f"acc[mp={mp},n={n},j={j}] agpr -> vgpr."))
                accRegs.append(accStageBank + slot)
                slot += 1
        # Mandatory hazard nop when fewer than 2 AGPRs were read (gfx950 v_accvgpr_read->VALU).
        if 0 < slot < 2:
            module.add(SNop(waitState=1,
                            comment="fill the mandatory v_accvgpr_read->VALU wait state (gfx950)."))

        # Step 2: H = acc + residual (mirror lines 1065-1078, scalar only).
        module.addComment1(f"H = acc + residual (mp={mp},n={n}).")
        for j in range(8):
            module.add(VAddF32(dst=vgpr(accRegs[j]), src0=vgpr(accRegs[j]),
                               src1=vgpr(residualF32Regs[j]),
                               comment=f"H = acc + residual (j={j}, n={n})."))

        # Step 3: RMS accumulate (mirror lines 1079-1084).
        module.addComment1(f"rmsSum[{n}] += H*H (mp={mp},n={n}).")
        for j in range(8):
            module.add(VMacF32(dst=vgpr(self.partials + n), src0=vgpr(accRegs[j]),
                               src1=vgpr(accRegs[j]),
                               comment=f"rmsSum[{n}] += H*H (j={j})."))

        # Step 4: store bf16(H) to residualOut (H values are pre-gamma).
        self._storeResidualOut(module, accRegs, mp, n, pathInterior)

        # Step 5: D = H * gamma (mirror lines 1225-1238, scalar only).
        module.addComment1(f"D = H * gamma (mp={mp},n={n}).")
        for j in range(8):
            module.add(VMulF32(dst=vgpr(accRegs[j]), src0=vgpr(accRegs[j]),
                               src1=vgpr(self.gammaBank + j),
                               comment=f"D = H * gamma (j={j}, n={n})."))

        # Step 6: write D back to the accumulator register file (mirror _amaxAndWriteAcc
        # lines 533-542, non-MXFP8 branch only — no amax, no MXFP8).
        module.addComment1(f"write D back to accumulator register file (mp={mp},n={n}).")
        for j in range(8):
            tileIdx = m if j < 4 else m1
            k = j % 4
            tileInfo = vgprTiles[n * self.geom.mmaM + tileIdx]
            reg = tileInfo.regList.indices[k]
            sk = accRegs[j]
            if tileInfo.regList.pool == self.writer.vgprPool:
                if sk != reg:
                    module.add(VMovB32(dst=vgpr(reg), src=vgpr(sk),
                                       comment=f"write D back to acc (tile={tileIdx},n={n},k={k})."))
            else:
                module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(sk),
                                            comment=f"write D back to acc (tile={tileIdx},n={n},k={k})."))

    def _residualToF32(self, module, bank, pathInterior):
        """Convert 8 bf16 residual VGPRs to f32 in place, native order.

        Interior path: bank[0..3] hold 4 packed bf16 dwords (2 bf16 per dword)
        after pair_shuffle; expand high-index first so each source dword is read
        before its low half is overwritten.

        Tail path: bank[0..7] hold 8 independent lo16 bf16 values; convert in
        forward order.

        Returns a list of 8 VGPR ints [bank+0 .. bank+7] holding f32.
        """
        if pathInterior:
            module.addComment1("expand 8 native bf16 residual -> 8 f32 (high index first, in place).")
            for i in range(7, -1, -1):
                module.add(VCvtBF16toFP32(vgpr(bank + i), vgpr(bank + i // 2), None, i % 2,
                                          comment=f"residual native[{i}] bf16({'hi' if i%2 else 'lo'}) -> f32."))
        else:
            module.addComment1("convert 8 native lo16 bf16 residual -> 8 f32 (in place).")
            for j in range(8):
                module.add(VCvtBF16toFP32(vgpr(bank + j), vgpr(bank + j), None, 0,
                                          comment=f"residual native[{j}] lo16 -> f32."))
        return [bank + j for j in range(8)]

    def _emitPairSweep(self, module, vgprTiles, pathInterior):
        """Emit the per-pair body with PREFETCH=1 (simplest correct schedule).

        No prefetch ring yet: load one pair-column, wait, convert, compute, repeat.
        """
        module.addComment0(f"MegaFused pair sweep (PREFETCH=1, pathInterior={pathInterior}).")
        rpl = self.geom.rowsPerLane
        tpb = self.geom.tilesPerBlockM
        vgprPool = self.writer.vgprPool
        # resBank: 8 VGPRs, 4-aligned (B128 needs 4-aligned base; also holds 8 expanded f32).
        resBank = vgprPool.checkOutAligned(8, 4, tag="mf_resBank")
        accStageBank = vgprPool.checkOut(8, tag="mf_accStage")
        for qi in range(self.geom.nQTilesM):
            module.addComment1(f"gamma block load/convert for qi={qi}.")
            self._ldsReadGammaBlockIssue(module, self.gammaBank, qi)
            module.add(SWaitCnt(dscnt=0, comment=f"wait gamma LDS read (qi={qi})."))
            for mi in range(tpb):
                _convertGammaChunk(module, self.gammaBank + mi * rpl, rpl)
            self._gammaReadPending = False
            for n in range(self.geom.mmaN):
                module.addComment1(f"pair compute (qi={qi}, n={n}).")
                self._loadRaw(module, resBank, qi, n, pathInterior)
                module.add(SWaitCnt(vlcnt=0, comment="wait residual load (PREFETCH=1)."))
                if pathInterior:
                    self._pairShuffle(module, resBank)
                residualF32 = self._residualToF32(module, resBank, pathInterior)
                self._computePair(module, vgprTiles, residualF32, accStageBank, qi, n, pathInterior)
        vgprPool.checkIn(accStageBank)
        vgprPool.checkIn(resBank)

    def _reduceAndWritePartials(self, module) -> None:
        """Reduce rmsSum intra-wave, fence stores, and write partials to partialBuf.

        Faithful port of SubtileMegaFusedEmitter.emit() lines 2450-2489 (intra-wave
        butterfly), line 2509 (store drain), and lines 2516-2737 (cross-wave reduce
        and partialBuf write). Dead code: not called yet.
        """
        # Block 1: intra-wave row-group XOR butterfly reduction (lines 2450-2489).
        _reduceRGF0Mod = Module("PartialRMS reduceRowGroupFree0")
        _reduceRGF0Mod.addComment0("intra-wave row-group butterfly (no LDS memory, no barrier).")
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
        # Block 2: store drain fencing the cross-wave phase (line 2509 only).
        module.add(SWaitCnt(vscnt=0, comment="drain ResidualOut (and MXScale for MXFP8) stores before cross-wave reduce."))
        # Block 3: cross-wave reduce + partialBuf write (lines 2516-2737).
        _rmsModule = Module("MegaFused reduceAndWriteRms")
        _rmsModule.addComment0("reduce rmsSum across row groups and waves, write to partialBuf.")
        _rmsSgprPool = self.writer.sgprPool
        partialSrd = _rmsSgprPool.checkOutAligned(4, 4, tag="mf_partialSrd", preventOverflow=False)
        _buildBufferSrd(_rmsModule, partialSrd, "PartialBuf", "partialBuf")
        _crossWaveModule = Module("PartialRMS reduceCrossWaveFree0")
        _crossWaveModule.addComment0("cross-wave LDS reduction (fenced, wgM > 1 only).")
        if self.geom.wgM > 1:
            reduceArrays = [(self.partials, VAddF32, "+")]
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
            laneSlotBytes_cwca = numArrays * self.geom.numPartials * 4
            strideW_cwca = self.geom.waveSize * laneSlotBytes_cwca
            waveM_cwca = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WaveM")
            readBaseWave_cwca = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadBase")
            laneLoc_cwca = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0Lane")
            _crossWaveReduceF0Mod.add(VMovB32(dst=vgpr(laneLoc_cwca), src=vgpr(self.laneId),
                               comment="laneId for LDS addressing (cached)"))
            _crossWaveReduceF0Mod.add(VAndB32(dst=vgpr(waveM_cwca), src0=vgpr(self.waveIdV), src1=self.geom.wgM - 1,
                               comment=f"waveM = waveId % {self.geom.wgM}"))
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
            _crossWaveReduceF0Mod.addComment1("LDS-staged cross-wave load and reduce.")
            numArrays = len(reduceArrays)
            for j in range(self.geom.wgM):
                for a, (base, _op, _verb) in enumerate(reduceArrays):
                    for i in range(self.geom.numPartials):
                        off = (a * self.geom.numPartials + i) * 4
                        dst = (base + i) if j == 0 else (readTmp + a * self.geom.numPartials + i)
                        _crossWaveReduceF0Mod.add(DSLoadB32(dst=vgpr(dst), src=vgpr(readAddr), ds=DSModifiers(offset=off),
                                             comment=f"LDS load wave[{j}] arr[{a}] partial[{i}]."))
                _crossWaveReduceF0Mod.add(SWaitCnt(dscnt=0, comment="wait LDS reads."))
                if j > 0:
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
        # Strength-reduce token*n_d across the n loop; see SubtileMegaFusedEmitter comment.
        accumV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0Accum")
        _writePartialsFree0Mod.add(VMulLOU32(dst=vgpr(accumV), src0=vgpr(ntilesV), src1=vgpr(tokenBase),
                             comment="accum = tokenBase * n_d"))
        strideV = None
        subStrideV = None
        if self.geom.mmaN > 1:
            groupDelta = self.geom.tileStrideN - (self.geom.vw1 - 1)
            strideV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0Stride")
            if groupDelta > _INLINE_CONST_MAX:
                _writePartialsFree0Mod.add(VMovB32(dst=vgpr(strideV), src=groupDelta,
                                                   comment=f"groupDelta={groupDelta} (too large for inline)."))
                _writePartialsFree0Mod.add(VMulLOU32(dst=vgpr(strideV), src0=vgpr(strideV), src1=vgpr(ntilesV),
                                     comment=f"groupStride = (tileStrideN-(vw1-1)={groupDelta}) * n_d"))
            else:
                _writePartialsFree0Mod.add(VMulLOU32(dst=vgpr(strideV), src0=groupDelta, src1=vgpr(ntilesV),
                                     comment=f"groupStride = (tileStrideN-(vw1-1)={groupDelta}) * n_d"))
            if self.geom.vw1 > 1:
                subStrideV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0SubStride")
        _writePartialsFree0Mod.add(VAddU32(vgpr(globalAddr), vgpr(accumV), sgpr("WorkGroup0"),
                           comment="token*n_d + WorkGroup0 (n=0)"))
        _writePartialsFree0Mod.add(VLShiftLeftB32(dst=vgpr(globalAddr), shiftHex=hex(2), src=vgpr(globalAddr),
                                  comment="byteAddr = (token*n_d + WG0) * 4"))
        if strideV is not None:
            _writePartialsFree0Mod.add(VLShiftLeftB32(dst=vgpr(strideV), shiftHex=hex(2), src=vgpr(strideV),
                                      comment="groupStride4 = groupStride * 4"))
        if subStrideV is not None:
            _writePartialsFree0Mod.add(VLShiftLeftB32(dst=vgpr(subStrideV), shiftHex=hex(2), src=vgpr(ntilesV),
                                      comment="subStride4 = n_d * 4 (within-group delta=1)"))
        for n in range(self.geom.mmaN):
            _writePartialsFree0Mod.add(BufferStoreB32(src=vgpr(self.partials + n), vaddr=vgpr(globalAddr),
                                      saddr=sgpr(partialSrd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"partialBuf[token+colOffset({n})={self.geom.colOffset(n)}, WG0] = Σx²"))
            if n < self.geom.mmaN - 1:
                boundary = ((n + 1) % self.geom.vw1) == 0
                adv = strideV if boundary else subStrideV
                _writePartialsFree0Mod.add(VAddU32(vgpr(globalAddr), vgpr(globalAddr), vgpr(adv),
                                   comment=f"byteAddr += {'groupStride4' if boundary else 'subStride4'} (advance to n={n + 1})"))
        _writePartialsFree0Mod.add(SWaitCnt(vscnt=0, comment="wait partialBuf stores"))
        _writePartialsFree0Mod.add(SMovB64(dst=EXEC(), src=sgpr(self.savedExec, lsc), comment="restore exec mask"))
        if subStrideV is not None:
            self.writer.vgprPool.checkIn(subStrideV)
        if strideV is not None:
            self.writer.vgprPool.checkIn(strideV)
        self.writer.vgprPool.checkIn(accumV)
        self.writer.vgprPool.checkIn(tokenBase)
        self.writer.vgprPool.checkIn(ntilesV)
        _rmsModule.add(_writePartialsFree0Mod)
        self.writer.vgprPool.checkIn(globalAddr)
        _rmsSgprPool.checkIn(partialSrd)
        module.add(_rmsModule)

    def emit(self, vgprTiles):
        return self._delegate.emit(vgprTiles)
