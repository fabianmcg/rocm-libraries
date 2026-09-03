# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ResidualAdd epilogue emitter for the Subtile gfx950 kernel.

Loads a row-major residual tensor, computes H = GEMM_acc + residual, writes H
back into the accumulator registers (AGPRs/VGPRs) so the following PartialRMS
epilogue can read H from there, and optionally stores H as bf16 to ResidualOut.

This emitter runs before PartialRMS; because it holds only the residual burst
in scratch VGPRs (released before PartialRMS begins), the VGPR peak for each
epilogue is lower than the old fused single-pass version.
"""

import math
import types

from rocisa.code import Module
from rocisa.container import ContinuousRegister, MUBUFModifiers, accvgpr, sgpr, vgpr
from rocisa.enum import HighBitSel
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadB32,
    BufferLoadB64,
    BufferLoadD16B16,
    BufferLoadD16U8,
    ECvtPkBF8toF32,
    ECvtPkFP8toF32,
    SLShiftLeftB32,
    SMovB32,
    SMovB64,
    SMulI32,
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddF32,
    VAddU32,
    VAndB32,
    VCmpLtU32,
    VCndMaskB32,
    VCvtBF16toFP32,
    VCvtBF8toF32,
    VCvtF16toF32,
    VCvtFP8toF32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMovB32,
    VMulLOU32,
)
from Tensile.Common.DataType import DataType

# Maximum inline-literal integer for VOP encodings; larger immediates must be
# materialized in a VGPR before a v_add.
_INLINE_CONST_MAX = 64


class SubtileResidualAddEmitter:
    """Emit the ResidualAdd epilogue for the Subtile gfx950 kernel.

    Runs before PartialRMS. Holds only the residual burst in scratch VGPRs,
    which are released before PartialRMS begins, keeping the peak lower than
    a fused single-pass approach.
    """

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        self.archCaps = writer.states.archCaps

        self.mfma_m = kernel["MatrixInstM"]
        self.mfma_n = kernel["MatrixInstN"]
        self.waveSize = kernel["WavefrontSize"]
        assert self.waveSize == 64, "residualAdd epilogue requires wavefrontSize == 64"
        self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.waveSize

        wg = kernel["MIWaveGroup"]
        self.wg_m = wg[0]
        self.wg_n = wg[1]

        self.mma_m = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
        self.mma_n = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
        self.macro_tile0 = kernel["MacroTile0"]
        self.macro_tile1 = kernel["MacroTile1"]

        self.lane_sgpr_count = writer.states.laneSGPRCount
        self.residualAdd = bool(kernel.get("PartialRMSResidualAdd", False))
        self.storeBf16D = bool(kernel.get("PartialRMSStoreBf16D", False))

        dt = kernel["ProblemType"]["DataType"]
        # elemBytes/log2ElemBytes encode the GEMM input element size used in the
        # token-index encoding in colByte.
        self.elemBytes = 1 if (dt.isAnyFloat8() or dt.isAnyBFloat8()) else 2
        self.log2ElemBytes = 0 if self.elemBytes == 1 else 1

        self.residualType = DataType(kernel.get("PartialRMSResidualType") or "b")
        self.residualBytes, self.residualLog2Bytes = self._sideBytes(self.residualType)
        # Wide residual load: fp8/bf8 packs 4 per dword, bf16 packs 4 per dwordx2.
        self.useWideResidual = (self.residualAdd and (self.rows_per_lane % 4 == 0)
                                and (self.residualBytes == 1
                                     or (self.residualBytes == 2
                                         and not self.residualType.isHalf())))

    @staticmethod
    def _sideBytes(dtype):
        """Return (bytes, log2Bytes) for a side-input element type."""
        if dtype.isSingle():
            return 4, 2
        if dtype.isAnyFloat8() or dtype.isAnyBFloat8():
            return 1, 0
        return 2, 1  # bf16 / f16.

    def _sideLoadClass(self, dtype):
        if dtype.isSingle():
            return BufferLoadB32
        if dtype.isAnyFloat8() or dtype.isAnyBFloat8():
            return BufferLoadD16U8
        return BufferLoadD16B16  # bf16 / f16.

    def _issueSideLoad(self, module, dstVgpr: int, addrVgpr: int, srd: int,
                       comment: str, dtype) -> None:
        """Issue one residual buffer_load without waiting (burst-friendly)."""
        loadCls = self._sideLoadClass(dtype)
        module.add(loadCls(vgpr(dstVgpr), vgpr(addrVgpr), sgpr(srd, 4), 0,
                           MUBUFModifiers(offen=True), comment=comment))

    def _convertSideElem(self, module, dstVgpr: int, comment: str, dtype) -> None:
        """Convert an already-loaded side element to fp32 in place (no-op for f32)."""
        if dtype.isSingle():
            return
        if dtype.isHalf():
            module.add(VCvtF16toF32(vgpr(dstVgpr), vgpr(dstVgpr), comment=comment))
        elif dtype.isAnyFloat8():
            module.add(VCvtFP8toF32(dst=vgpr(dstVgpr), src=vgpr(dstVgpr), comment=comment))
        elif dtype.isAnyBFloat8():
            module.add(VCvtBF8toF32(dst=vgpr(dstVgpr), src=vgpr(dstVgpr), comment=comment))
        else:
            module.add(VCvtBF16toFP32(vgpr(dstVgpr), vgpr(dstVgpr), None, 0, comment=comment))

    def _readAccBurst(self, module, dstBase: int, vgprTiles, coords, comment: str) -> None:
        """Read a burst of accumulator elements into consecutive VGPRs [dstBase, dstBase+len).

        coords is a list of (m, n, k) tile coordinates. The reads are issued
        back-to-back so the mandatory post-v_accvgpr_read wait state is filled by
        the following reads and the compute that consumes earlier entries, hiding
        the hazard.
        """
        usedAcc = False
        for i, (m, n, k) in enumerate(coords):
            tile = vgprTiles[n * self.mma_m + m]
            reg = tile.regList.indices[k]
            if tile.regList.pool == self.writer.vgprPool:
                module.add(VMovB32(dst=vgpr(dstBase + i), src=vgpr(reg),
                                   comment=f"{comment} [{i}]"))
                continue
            module.add(VAccvgprReadB32(vgpr(dstBase + i), accvgpr(reg),
                                       comment=f"{comment} [{i}]"))
            usedAcc = True
        # gfx950 requires one wait state between v_accvgpr_read_b32 and a dependent
        # VALU consumer. A burst of >= 2 reads already fills that gap; only a lone
        # read (mma_n == 1) needs an explicit s_nop.
        if usedAcc and len(coords) < 2:
            module.add(SNop(waitState=1, comment="fill the mandatory v_accvgpr_read->VALU wait state (gfx950)."))

    def _writeAccFrom(self, module, src: int, vgprTiles, m: int, n: int, k: int, comment: str) -> None:
        """Write VGPR src back into accumulator element (m, n, k), selecting the right register file."""
        tile = vgprTiles[n * self.mma_m + m]
        reg = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            module.add(VMovB32(dst=vgpr(reg), src=vgpr(src), comment=comment))
            return
        module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(src), comment=comment))

    def _addImmU32(self, module, dst: int, src: int, imm: int, scratch: int, comment: str) -> None:
        # imm == 0 is a no-op add; emit a copy only when the value must move registers.
        if imm == 0:
            if dst != src:
                module.add(VMovB32(dst=vgpr(dst), src=vgpr(src), comment=comment))
            return
        # Materialize the immediate in a VGPR when it exceeds the inline-literal range.
        if imm > _INLINE_CONST_MAX:
            module.add(VMovB32(dst=vgpr(scratch), src=imm, comment=f"imm={imm}"))
            module.add(VAddU32(vgpr(dst), vgpr(src), vgpr(scratch), comment=comment))
            return
        module.add(VAddU32(vgpr(dst), vgpr(src), imm, comment=comment))

    def _computeWaveM(self, module, dst: int) -> None:
        # waveId is cached once in _setup (self.waveIdV); only callers with wg_m > 1
        # reach here, so self.waveIdV is always valid.
        module.add(VAndB32(dst=vgpr(dst), src0=vgpr(self.waveIdV), src1=self.wg_m - 1,
                           comment=f"waveM = waveId % {self.wg_m}"))

    def _computeRowGroupOff(self, module, dst: int) -> None:
        # Reuse the cached self.laneId instead of recomputing Serial & (waveSize-1).
        log2MfmaN = int(math.log2(self.mfma_n))
        module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(log2MfmaN), src=vgpr(self.laneId),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        module.add(VMulLOU32(dst=vgpr(dst), src0=self.rows_per_lane, src1=vgpr(dst),
                             comment=f"rowGroupOff = rowGroup * {self.rows_per_lane}"))

    def _computeFree0RowBase(self, module, dst: int) -> None:
        # rowBase = WorkGroup0 * MT0 (+ waveM * mma_m*mfma_m when wg_m > 1).
        mt0Vgpr = self.writer.vgprPool.checkOut(1, tag="rAdd_rbMT0")
        module.add(VMovB32(dst=vgpr(mt0Vgpr), src=self.macro_tile0, comment=f"MT0={self.macro_tile0}"))
        module.add(VMulLOU32(dst=vgpr(dst), src0=vgpr(mt0Vgpr), src1=sgpr("WorkGroup0"),
                             comment="rowBase = WorkGroup0 * MT0"))
        self.writer.vgprPool.checkIn(mt0Vgpr)
        if self.wg_m <= 1:
            return
        waveM = self.writer.vgprPool.checkOut(1, tag="rAdd_rbWaveM")
        self._computeWaveM(module, waveM)
        waveStride = self.mma_m * self.mfma_m
        strideV = self.writer.vgprPool.checkOut(1, tag="rAdd_rbStride")
        module.add(VMovB32(dst=vgpr(strideV), src=waveStride,
                           comment=f"waveStride = mma_m * mfma_m = {waveStride}"))
        module.add(VMulLOU32(dst=vgpr(waveM), src0=vgpr(strideV), src1=vgpr(waveM),
                             comment="waveMOff = waveM * waveStride"))
        module.add(VAddU32(vgpr(dst), vgpr(dst), vgpr(waveM), comment="rowBase += waveMOff"))
        self.writer.vgprPool.checkIn(strideV)
        self.writer.vgprPool.checkIn(waveM)

    def _free0RowPos(self, module, dst: int, rowBase: int, rowGroupOff: int,
                     m: int, k: int, scratch: int) -> None:
        # free0 row = rowBase + rowGroupOff + (m*mfma_m + k).
        mBase = m * self.mfma_m + k
        self._addImmU32(module, dst, rowBase, mBase, scratch, f"row = base + {mBase} (m={m},k={k})")
        module.add(VAddU32(vgpr(dst), vgpr(dst), vgpr(rowGroupOff), comment="row += rowGroupOff"))

    def _addWaveNColByte(self, module, colByte: int) -> None:
        if self.wg_n <= 1:
            return
        waveN = self.writer.vgprPool.checkOut(1, tag="rAdd_setupWaveN")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="rAdd_setupTmp")
        tmpRes = ContinuousRegister(tmpVgpr, 2)
        module.add(vectorStaticDivide(waveN, "Serial", self.waveSize * self.wg_m, tmpRes,
                                      comment=f"waveN = Serial / {self.waveSize * self.wg_m}"))
        colBaseBytes = self.mma_n * self.mfma_n * self.elemBytes
        with self.writer.allocTmpSgpr(1, tag="rAdd_setupColBase") as tmpSgprInfo:
            module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(colBaseBytes),
                               comment=f"col base bytes per wave ({colBaseBytes})"))
            module.add(VMulLOU32(dst=vgpr(waveN), src0=sgpr(tmpSgprInfo.idx), src1=vgpr(waveN),
                                 comment="waveN * mma_n * mfma_n * elemBytes"))
        module.add(VAddU32(vgpr(colByte), vgpr(colByte), vgpr(waveN),
                           comment="colByte += wave column base"))
        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(waveN)

    def _buildResidualSrd(self, module, resSrd: int) -> None:
        # Residual SRD bounds = M_tokens*N_hidden*elemBytes so tail-WG OOB lanes read 0.
        with self.writer.allocTmpSgpr(1, tag="rAdd_resSrdNumRec") as tmpSgpr:
            module.add(SMovB64(dst=sgpr(resSrd, 2), src=sgpr("ResidualBuf", 2),
                               comment="residual SRD base"))
            module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("SizesFree+0"),
                               src1=sgpr("SizesFree+1"), comment="numRecords = N_hidden * M_tokens"))
            module.add(SLShiftLeftB32(dst=sgpr(resSrd + 2), src=sgpr(tmpSgpr.idx),
                                      shiftHex=hex(self.residualLog2Bytes),
                                      comment="numRecords *= residualBytes."))
        module.add(SMovB32(dst=sgpr(resSrd + 3), src="Srd127_96", comment="residual SRD flags"))

    def _buildResidualOutSrd(self, module, srd: int) -> None:
        # ResidualOut SRD bounds = M_tokens*N_hidden*2 (bf16) so OOB lanes store to /dev/null.
        with self.writer.allocTmpSgpr(1, tag="rAdd_roSrdNumRec") as tmpSgpr:
            module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr("AddressResidualOut", 2),
                               comment="ResidualOut SRD base."))
            module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("SizesFree+0"),
                               src1=sgpr("SizesFree+1"), comment="numRecords = N_hidden * M_tokens"))
            module.add(SLShiftLeftB32(dst=sgpr(srd + 2), src=sgpr(tmpSgpr.idx),
                                      shiftHex=hex(1), comment="numRecords *= 2 (bf16)."))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96",
                           comment="ResidualOut SRD flags."))

    def _residualRowByteBase(self, module, dst: int, tokenBase: int, n: int, scratch: int) -> None:
        nOff = n * self.mfma_n
        self._addImmU32(module, dst, tokenBase, nOff, scratch, f"token_n = tokenBase + {nOff} (n={n})")
        # token_n * SizesFree0 uses 32-bit VMulLOU32; valid while the element index
        # token_n * N_hidden stays below 2^32 (all currently supported tensor sizes).
        module.add(VMulLOU32(dst=vgpr(dst), src0=sgpr("SizesFree+0"), src1=vgpr(dst),
                             comment="token_n * SizesFree0"))
        module.add(VLShiftLeftB32(dst=vgpr(dst), shiftHex=hex(self.residualLog2Bytes), src=vgpr(dst),
                                  comment="rowByteBase = token_n * SizesFree0 * residualBytes."))

    def _residualElemAddr(self, module, resAddr: int, rowByteBase: int, nhiddenBase: int,
                          rowGroupOff: int, oobV: int, oobMask: int, scratch: int,
                          m: int, k: int) -> None:
        self._free0RowPos(module, resAddr, nhiddenBase, rowGroupOff, m, k, scratch)
        module.add(VCmpLtU32(dst=sgpr(oobMask, self.lane_sgpr_count), src0=vgpr(resAddr),
                             src1=sgpr("SizesFree+0"), comment="inRange = nhidden_pos < N_hidden"))
        module.add(VLShiftLeftB32(dst=vgpr(resAddr), shiftHex=hex(self.residualLog2Bytes),
                                  src=vgpr(resAddr),
                                  comment="nhiddenByte = nhidden_pos * residualBytes."))
        module.add(VAddU32(vgpr(resAddr), vgpr(resAddr), vgpr(rowByteBase),
                           comment="byteAddr = rowByteBase + nhiddenByte"))
        module.add(VCndMaskB32(dst=vgpr(resAddr), src0=vgpr(oobV), src1=vgpr(resAddr),
                               src2=sgpr(oobMask, self.lane_sgpr_count),
                               comment="clamp OOB when nhidden_pos >= N_hidden"))

    def _residualOutAcquireScratch(self, cvtVgprStruct):
        """Return (effectiveCvt, selfVgprs) for the ResidualOut permute/pack scratch.

        Reuses cvtVgprStruct when it already has valid permute fields (bf16/fp16
        D dest); otherwise self-allocates 7 scratch VGPRs so fp8/other D dest
        can still emit the store. selfVgprs lists the allocated indices in
        allocation order; the caller frees them after the store completes.
        """
        if (cvtVgprStruct is not None
                and hasattr(cvtVgprStruct, 'vgprPermAddr')
                and cvtVgprStruct.vgprPermAddr >= 0):
            return cvtVgprStruct, []
        vgprPool = self.writer.vgprPool
        # 4 VGPRs, 2-aligned: satisfies buffer_store_dwordx4 src alignment.
        packBase = vgprPool.checkOutAligned(4, 2, tag="rAdd_roPack")
        permAddr = vgprPool.checkOut(1, tag="rAdd_roPermAddr")
        lgDelta = vgprPool.checkOut(1, tag="rAdd_roLGDelta")
        addrScratch = vgprPool.checkOut(1, tag="rAdd_roAddrScratch")
        scratch = types.SimpleNamespace(
            vgprBf16Temp=packBase,
            vgprPermAddr=permAddr,
            vgprLaneGroupDelta=lgDelta,
            vgprAddrScratch=addrScratch,
        )
        return scratch, [packBase, permAddr, lgDelta, addrScratch]

    def _residualOutSetupAddresses(self, module, gwbw, wgRowBase: int):
        """Build ResidualOut SRD, permute state, guards, and per-n row-byte VGPRs.

        Returns (roRowBase, lgByte, wgRowBase_bytes) checked out from the VGPR pool;
        the caller frees them after the store loop.
        """
        self._buildResidualOutSrd(module, self.residualOutSrd)
        # Permute/LGDelta setup: fills vgprPermAddr and vgprLaneGroupDelta in cvtVgprStruct.
        gwbw._emitSubtilePermuteAndLGDelta(module)
        # Edge guards: populates SubtileMGuard, SubtileNGuard, and leaks a temp SGPR
        # into self.writer.states.subtileTotalMOffsetSgpr (freed in the outer function).
        self.writer._emitSubtileGuards(self.kernel, module)

        vgprPool = self.writer.vgprPool
        # Pre-compute per-n token row byte bases.
        #   roRowBase[n] = (tokenBase + n*mfma_n) * SizesFree0 * 2.
        tokenBase = vgprPool.checkOut(1, tag="rAdd_roMT_tokenBase")
        module.add(VLShiftRightB32(dst=vgpr(tokenBase), shiftHex=hex(self.log2ElemBytes),
                                    src=vgpr(self.colByte),
                                    comment="tokenBase = colByte >> log2ElemBytes."))
        roRowBase = vgprPool.checkOut(self.mma_n, tag="rAdd_roMT_roRow")
        roScratch = vgprPool.checkOut(1, tag="rAdd_roMT_roScratch")
        for n in range(self.mma_n):
            self._addImmU32(module, roRowBase + n, tokenBase, n * self.mfma_n, roScratch,
                            f"token_n = tokenBase + {n * self.mfma_n} (n={n}).")
            module.add(VMulLOU32(dst=vgpr(roRowBase + n), src0=sgpr("SizesFree+0"),
                                 src1=vgpr(roRowBase + n),
                                 comment=f"token_n * SizesFree0 (n={n})."))
            module.add(VLShiftLeftB32(dst=vgpr(roRowBase + n), shiftHex=hex(1),
                                       src=vgpr(roRowBase + n),
                                       comment=f"roRowBase[{n}] = token_n * SizesFree0 * 2."))
        vgprPool.checkIn(roScratch)
        vgprPool.checkIn(tokenBase)

        # lane_group*16 bytes; for wave64 with mfma_n=16: laneId & 0x30 = lane_group<<4.
        lgByte = vgprPool.checkOut(1, tag="rAdd_roMT_lgByte")
        module.add(VAndB32(dst=vgpr(lgByte), src0=vgpr(self.laneId), src1=0x30,
                           comment="lgByte = lane_group*16 bytes = laneId & 0x30."))
        # wgRowBase_bytes = wgRowBase * 2 (bf16, 2 bytes/element).
        wgRowBase_bytes = vgprPool.checkOut(1, tag="rAdd_roMT_wgRowBaseB")
        module.add(VLShiftLeftB32(dst=vgpr(wgRowBase_bytes), shiftHex=hex(1),
                                   src=vgpr(wgRowBase),
                                   comment="wgRowBase_bytes = wgRowBase * 2."))
        return roRowBase, lgByte, wgRowBase_bytes

    def _residualOutStoreLoop(self, module, gwbw, vgprTiles,
                              roRowBase: int, lgByte: int, wgRowBase_bytes: int) -> None:
        """Iterate over paired (m, m+1) / n tile positions and emit paired dwordx4 stores."""
        useAlign8 = self.writer.states.storeAlign8
        vgprPool = self.writer.vgprPool
        vaddrVgpr = vgprPool.checkOut(1, tag="rAdd_roMT_vaddr")
        nhBase_bytes = vgprPool.checkOut(1, tag="rAdd_roMT_nhBase")
        # 2-aligned burst for 8 f32 values; satisfies buffer_store_dwordx4 src alignment.
        burstBase = vgprPool.checkOutAligned(8, 2, tag="rAdd_roMT_burst")

        def srcVc(pair, vi):
            return vgpr(burstBase + pair * self.rows_per_lane + vi)

        for m in range(0, self.mma_m, 2):
            # nhBase_pair_bytes = (wgRowBase + m*mfma_m)*2 = wgRowBase_bytes + m*mfma_m*2.
            mOff_bytes = m * self.mfma_m * 2
            self._addImmU32(module, nhBase_bytes, wgRowBase_bytes, mOff_bytes, vaddrVgpr,
                            f"nhBase_bytes = wgRowBase_bytes + {mOff_bytes} (m={m}).")
            # Add lane_group*16 correction: after permute each LG has 8 contiguous nhidden rows.
            module.add(VAddU32(vgpr(nhBase_bytes), vgpr(nhBase_bytes), vgpr(lgByte),
                               comment=f"nhBase_bytes += lgByte (lane_group*16) (m={m})."))
            for n in range(self.mma_n):
                # Full per-lane byte address: token row base + nhidden byte base.
                module.add(VAddU32(vgpr(vaddrVgpr), vgpr(roRowBase + n), vgpr(nhBase_bytes),
                                   comment=f"vaddrVgpr = roRowBase[{n}] + nhBase_bytes (m={m},n={n})."))
                # Read 8 H f32 values from acc: 4 from tile m then 4 from tile m+1.
                coords = [(m, n, k) for k in range(self.rows_per_lane)] + \
                         [(m + 1, n, k) for k in range(self.rows_per_lane)]
                self._readAccBurst(module, burstBase, vgprTiles, coords,
                                   f"read H burst for ResidualOut (m={m},{m+1},n={n}).")
                module.add(gwbw._emit16bitPairedDwordx4Core(
                    srcVc=srcVc, vaddrVgpr=vaddrVgpr, offset12=0,
                    srd="SrdResidualOut", applyLGDeltaToAddr=False,
                    useAlign8=useAlign8, blockIdxM=m, blockIdxN=n,
                    isGlc=False, isSlc=False, isNT=False, tt0=m))

        module.add(SWaitCnt(vscnt=0, comment="wait ResidualOut dwordx4 stores."))
        vgprPool.checkIn(burstBase)
        vgprPool.checkIn(nhBase_bytes)
        vgprPool.checkIn(vaddrVgpr)

    def _emitResidualOutMacroTileStore(self, module, vgprTiles, wgRowBase: int, cvtVgprStruct) -> None:
        """Store H (in accumulator tiles) as bf16 to ResidualOut using paired buffer_store_dwordx4.

        Reuses _emit16bitPairedDwordx4Core from GlobalWriteBatchWriter.
        Requires rows_per_lane == 4 (mfma_m=16, mfma_n=16, waveSize=64) and even mma_m.

        Address formula per pair (m, m+1) and token column n:
          vaddrVgpr = roRowBase[n] + (wgRowBase + m*mfma_m)*2 + lane_group*16
        where roRowBase[n] = (tokenBase + n*mfma_n)*SizesFree0*2
        and lane_group*16 = laneId & 0x30 (the 8-row-per-LG byte correction after permute).
        """
        assert self.storeBf16D, "residualOut store requires storeBf16D"
        assert self.mma_m % 2 == 0, "mma_m must be even for paired dwordx4 ResidualOut store"
        assert self.rows_per_lane == 4, "ResidualOut macro-tile dwordx4 store requires rows_per_lane == 4"

        from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter

        effectiveCvt, selfVgprs = self._residualOutAcquireScratch(cvtVgprStruct)

        lsc = self.writer.states.laneSGPRCount
        # Bare GlobalWriteBatchWriter shell giving access to the core store primitives.
        # Attributes accessed by the called methods:
        #   kernel, cvtVgprStruct, parentWriter, tmpSgpr.
        gwbw = GlobalWriteBatchWriter.__new__(GlobalWriteBatchWriter)
        gwbw.kernel = self.kernel
        gwbw.cvtVgprStruct = effectiveCvt
        gwbw.parentWriter = self.writer
        tmpSgpr = self.writer.sgprPool.checkOutAligned(
            2 * lsc, 2, tag="rAdd_roMT_tmpS", preventOverflow=False)
        gwbw.tmpSgpr = tmpSgpr

        roRowBase, lgByte, wgRowBase_bytes = self._residualOutSetupAddresses(module, gwbw, wgRowBase)
        self._residualOutStoreLoop(module, gwbw, vgprTiles, roRowBase, lgByte, wgRowBase_bytes)

        self.writer.vgprPool.checkIn(wgRowBase_bytes)
        self.writer.vgprPool.checkIn(lgByte)
        self.writer.vgprPool.checkIn(roRowBase)
        for v in reversed(selfVgprs):
            self.writer.vgprPool.checkIn(v)

        # Return the leaked subtileTotalMOffsetSgpr so the later D-store batch can
        # recompute its own guards byte-identically.
        if self.writer.states.subtileTotalMOffsetSgpr is not None:
            self.writer.sgprPool.checkIn(self.writer.states.subtileTotalMOffsetSgpr)
            self.writer.states.subtileTotalMOffsetSgpr = None

        self.writer.sgprPool.checkIn(tmpSgpr)

    def _beginResidual(self, module) -> int:
        """Build the residual SRD and check out per-m-row load scratch; returns the residual burst base."""
        self._buildResidualSrd(module, self.resSrd)
        # 2-aligned so packed-convert dst pairs (base, base+2) are even.
        self._resBurst = self.writer.vgprPool.checkOutAligned(
            self.mma_n * self.rows_per_lane, 2, tag="rAdd_fusResBurst")
        self._resRowByteBase = self.writer.vgprPool.checkOut(1, tag="rAdd_fusResRowByte")
        self._resTokenBase = self.writer.vgprPool.checkOut(1, tag="rAdd_fusResToken")
        self._resOobMask = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="rAdd_fusResOobMask",
            preventOverflow=False)
        module.add(VLShiftRightB32(dst=vgpr(self._resTokenBase), shiftHex=hex(self.log2ElemBytes),
                                   src=vgpr(self.colByte),
                                   comment="tokenBase = colByte >> log2ElemBytes."))
        if self.useWideResidual:
            self._beginResidualWideBase(module)
        if not self.useWideResidual:
            self._resOobV = self.writer.vgprPool.checkOut(1, tag="rAdd_fusResOob")
            module.add(VMovB32(dst=vgpr(self._resOobV), src="BufferOOB",
                               comment="OOB byte offset -> residual load returns 0."))
        return self._resBurst

    def _beginResidualWideBase(self, module) -> None:
        """Precompute the m-invariant residual row byte base and per-token stride.

        base0 = tokenBase * SizesFree0 * residualBytes replaces the per-(m,n)
        multiply; rowStride advances byteAddr by one MMA-tile of tokens per n.
        """
        self._resRowStrideS = self.writer.sgprPool.checkOut(1, tag="rAdd_fusResRowStride")
        module.add(VMulLOU32(dst=vgpr(self._resTokenBase), src0=sgpr("SizesFree+0"),
                             src1=vgpr(self._resTokenBase), comment="base0 = tokenBase * SizesFree0."))
        if self.residualLog2Bytes:
            module.add(VLShiftLeftB32(dst=vgpr(self._resTokenBase),
                                      shiftHex=hex(self.residualLog2Bytes),
                                      src=vgpr(self._resTokenBase), comment="base0 *= residualBytes."))
        module.add(SMulI32(dst=sgpr(self._resRowStrideS), src0=sgpr("SizesFree+0"),
                           src1=self.mfma_n, comment="rowStride = SizesFree0 * mfma_n."))
        if self.residualLog2Bytes:
            module.add(SLShiftLeftB32(dst=sgpr(self._resRowStrideS), src=sgpr(self._resRowStrideS),
                                      shiftHex=hex(self.residualLog2Bytes),
                                      comment="rowStride *= residualBytes."))

    def _endResidual(self) -> None:
        if self.useWideResidual:
            self.writer.sgprPool.checkIn(self._resRowStrideS)
        if not self.useWideResidual:
            self.writer.vgprPool.checkIn(self._resOobV)
        self.writer.sgprPool.checkIn(self._resOobMask)
        self.writer.vgprPool.checkIn(self._resTokenBase)
        self.writer.vgprPool.checkIn(self._resRowByteBase)
        self.writer.vgprPool.checkIn(self._resBurst)

    def _issueResidualLoads(self, module, m: int, resBurst: int, wgRowBase: int,
                            rowGroupOff: int, scratch: int) -> None:
        """Issue all residual loads for tile row m (no wait)."""
        if self.useWideResidual:
            self._issueResidualLoadsWide(module, m, resBurst, wgRowBase, rowGroupOff, scratch)
            return
        for n in range(self.mma_n):
            self._residualRowByteBase(module, self._resRowByteBase, self._resTokenBase, n, scratch)
            for k in range(self.rows_per_lane):
                self._residualElemAddr(module, self.resAddr, self._resRowByteBase, wgRowBase,
                                       rowGroupOff, self._resOobV, self._resOobMask, scratch, m, k)
                self._issueSideLoad(module, resBurst + n * self.rows_per_lane + k, self.resAddr,
                                    self.resSrd, f"R[token_n, nhidden_pos] (m={m},n={n},k={k}).",
                                    dtype=self.residualType)

    def _issueResidualLoadsWide(self, module, m: int, resBurst: int, wgRowBase: int,
                                rowGroupOff: int, scratch: int) -> None:
        rpl = self.rows_per_lane
        isBf16 = self.residualBytes == 2
        loadCls = BufferLoadB64 if isBf16 else BufferLoadB32
        chunkBytes = 4 << self.residualLog2Bytes
        # byteAddr for n=0 this m: base0 (token row) + nhidden byte offset.
        if isBf16:
            module.add(VLShiftLeftB32(dst=vgpr(scratch), shiftHex=hex(self.residualLog2Bytes),
                                      src=vgpr(self._nhBaseV),
                                      comment="nhiddenByte = nhBase * residualBytes."))
            module.add(VAddU32(vgpr(self.resAddr), vgpr(self._resTokenBase), vgpr(scratch),
                               comment=f"byteAddr = base0 + nhiddenByte (m={m},n=0)."))
        else:
            module.add(VAddU32(vgpr(self.resAddr), vgpr(self._resTokenBase), vgpr(self._nhBaseV),
                               comment=f"byteAddr = base0 + nhBase (m={m},n=0)."))
        for n in range(self.mma_n):
            if n > 0:
                module.add(VAddU32(vgpr(self.resAddr), vgpr(self.resAddr),
                                   sgpr(self._resRowStrideS),
                                   comment=f"byteAddr += rowStride (advance to n={n})."))
            for c in range(rpl // 4):
                addr = self.resAddr
                # c > 0 is dead for rows_per_lane == 4; exists for rows_per_lane >= 8.
                if c > 0:
                    self._addImmU32(module, scratch, self.resAddr, chunkBytes * c, scratch,
                                    f"chunk byte offset {chunkBytes * c}.")
                    addr = scratch
                dstBase = resBurst + n * rpl + 4 * c
                dst = vgpr(dstBase, 2) if isBf16 else vgpr(dstBase)
                module.add(loadCls(dst, vgpr(addr), sgpr(self.resSrd, 4), 0,
                                   MUBUFModifiers(offen=True),
                                   comment=f"R wide [4 residual] (m={m},n={n},c={c})."))

    def _convertResidualRow(self, module, resBurst: int) -> None:
        if self.useWideResidual:
            self._convertResidualRowWide(module, resBurst)
            return
        for i in range(self.mma_n * self.rows_per_lane):
            self._convertSideElem(module, resBurst + i, f"residual -> fp32 ({i}).", dtype=self.residualType)

    def _convertResidualRowWide(self, module, resBurst: int) -> None:
        rpl = self.rows_per_lane
        for n in range(self.mma_n):
            for c in range(rpl // 4):
                base = resBurst + n * rpl + 4 * c
                if self.residualBytes == 1:
                    self._convertResidualChunkFp8(module, base)
                else:
                    self._convertResidualChunkBf16(module, base)
        self._maskResidualOOB(module, resBurst)

    def _convertResidualChunkFp8(self, module, base: int) -> None:
        cvt = ECvtPkFP8toF32 if self.residualType.isAnyFloat8() else ECvtPkBF8toF32
        # HIGH before LOW: HIGH reads the packed dword at base; LOW then overwrites base.
        module.add(cvt(dst=vgpr(base + 2, 2), src=vgpr(base), sel=HighBitSel.HIGH,
                       comment="residual pair (k=2,3) fp8 -> f32."))
        module.add(cvt(dst=vgpr(base, 2), src=vgpr(base), sel=HighBitSel.LOW,
                       comment="residual pair (k=0,1) fp8 -> f32."))

    def _convertResidualChunkBf16(self, module, base: int) -> None:
        # 4 bf16 in dwords (base, base+1) -> 4 f32; high indices first so each source
        # dword is fully read before it is overwritten.
        # sel 0 = low16 (WORD_0), sel 1 = high16 (WORD_1).
        module.add(VCvtBF16toFP32(vgpr(base + 3), vgpr(base + 1), None, 1,
                                  comment="residual k=3 bf16(hi) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 2), vgpr(base + 1), None, 0,
                                  comment="residual k=2 bf16(lo) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 1), vgpr(base + 0), None, 1,
                                  comment="residual k=1 bf16(hi) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 0), vgpr(base + 0), None, 0,
                                  comment="residual k=0 bf16(lo) -> f32."))

    def _maskResidualOOB(self, module, resBurst: int) -> None:
        # Zero residual elements whose nhidden_pos >= N_hidden; mask depends only on (m,k).
        lsc = self.lane_sgpr_count
        for k in range(self.rows_per_lane):
            nhpos = self._nhBaseV
            if k > 0:
                self._addImmU32(module, self.resAddr, self._nhBaseV, k, self._resRowByteBase,
                                f"nhidden_pos = nhBase + {k}.")
                nhpos = self.resAddr
            module.add(VCmpLtU32(dst=sgpr(self._resOobMask, lsc), src0=vgpr(nhpos),
                                 src1=sgpr("SizesFree+0"),
                                 comment=f"inRange = nhidden_pos < N_hidden (k={k})."))
            for n in range(self.mma_n):
                r = resBurst + n * self.rows_per_lane + k
                module.add(VCndMaskB32(dst=vgpr(r), src0=0, src1=vgpr(r),
                                       src2=sgpr(self._resOobMask, lsc),
                                       comment="residual = inRange ? residual : 0."))

    def _allocEpilogueRegs(self) -> None:
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        self.laneId = vgprPool.checkOut(1, tag="rAdd_laneId")
        self.waveIdV = vgprPool.checkOut(1, tag="rAdd_waveId") if self.wg_m > 1 else None
        self.colByte = vgprPool.checkOut(1, tag="rAdd_colByte")
        if self.residualAdd:
            self.resAddr = vgprPool.checkOut(1, tag="rAdd_resAddr")
        if self.residualAdd:
            self.resSrd = sgprPool.checkOutAligned(4, 4, tag="rAdd_resSrd", preventOverflow=False)
        if self.storeBf16D:
            self.residualOutSrd = self.writer.sgprs["SrdResidualOut"]

    def _freeEpilogueRegs(self) -> None:
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        if self.residualAdd:
            sgprPool.checkIn(self.resSrd)
        if self.residualAdd:
            vgprPool.checkIn(self.resAddr)
        if self.wg_m > 1:
            vgprPool.checkIn(self.waveIdV)
        vgprPool.checkIn(self.colByte)
        vgprPool.checkIn(self.laneId)

    def _setup(self, laneId: int, colByte: int) -> Module:
        module = Module("ResidualAdd setup")
        # Both passes must drain kernarg s_loads before reading kernel arguments.
        module.add(SWaitCnt(kmcnt=0, comment="wait for ResidualAdd kernarg s_load"))
        module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)"))
        if self.wg_m > 1:
            waveIdTmp = self.writer.vgprPool.checkOutAligned(2, 2, tag="rAdd_setupWaveIdDiv")
            waveIdRes = ContinuousRegister(waveIdTmp, 2)
            module.add(vectorStaticDivide(self.waveIdV, "Serial", self.waveSize, waveIdRes,
                                          comment="waveId = Serial / WavefrontSize (cached once)"))
            self.writer.vgprPool.checkIn(waveIdTmp)
        module.add(VAndB32(dst=vgpr(colByte), src0=vgpr(laneId), src1=self.mfma_n - 1,
                           comment=f"colInMma = laneId % {self.mfma_n}"))
        module.add(VLShiftLeftB32(dst=vgpr(colByte), shiftHex=hex(self.log2ElemBytes), src=vgpr(colByte),
                                  comment="colByte = colInMma * elemBytes."))
        self._addWaveNColByte(module, colByte)
        with self.writer.allocTmpSgpr(1, tag="rAdd_setupWG1") as wg1S:
            module.add(SMulI32(dst=sgpr(wg1S.idx), src0=sgpr("WorkGroup1"),
                               src1=self.macro_tile1 * self.elemBytes,
                               comment=f"wg1ColByte = WorkGroup1 * MT1*elemBytes (MT1={self.macro_tile1})"))
            module.add(VAddU32(vgpr(colByte), vgpr(colByte), sgpr(wg1S.idx),
                               comment="colByte += WorkGroup1 * MT1 * elemBytes"))
        return module

    def _residualAccElement(self, module, vgprTiles, accReg: int, resReg, m: int, n: int, k: int) -> None:
        if resReg is not None:
            module.add(VAddF32(dst=vgpr(accReg), src0=vgpr(accReg), src1=vgpr(resReg),
                               comment="H = GEMM + residual."))
            self._writeAccFrom(module, accReg, vgprTiles, m, n, k, f"write H back to acc (m={m},n={n},k={k}).")

    def _residualAccRow(self, module, vgprTiles, accBurst: int, resBurst, m: int) -> None:
        for k in range(self.rows_per_lane):
            coords = [(m, n, k) for n in range(self.mma_n)]
            self._readAccBurst(module, accBurst, vgprTiles, coords, f"read acc m={m},k={k}.")
            for n in range(self.mma_n):
                resReg = None if resBurst is None else resBurst + n * self.rows_per_lane + k
                self._residualAccElement(module, vgprTiles, accBurst + n, resReg, m, n, k)

    def _residualPassFree0(self, vgprTiles, cvtVgprStruct=None) -> Module:
        module = Module("ResidualAdd residualPassFree0")
        module.addComment1("ResidualAdd pass (free0): load residual, H = GEMM + residual, write H back, bf16 store")
        vgprPool = self.writer.vgprPool
        rowGroupOff = vgprPool.checkOut(1, tag="rAdd_rgOff")
        wgRowBase   = vgprPool.checkOut(1, tag="rAdd_wgRowBase")
        self._computeRowGroupOff(module, rowGroupOff)
        self._computeFree0RowBase(module, wgRowBase)
        self._nhBaseV = vgprPool.checkOut(1, tag="rAdd_nhBase")
        mBaseVgpr = vgprPool.checkOut(1, tag="rAdd_mBase")
        accBurst  = vgprPool.checkOut(self.mma_n, tag="rAdd_accBurst")
        resBurst  = self._beginResidual(module) if self.residualAdd else None
        for m in range(self.mma_m):
            if m == 0:
                self._free0RowPos(module, self._nhBaseV, wgRowBase, rowGroupOff, 0, 0, mBaseVgpr)
            else:
                self._addImmU32(module, self._nhBaseV, self._nhBaseV, self.mfma_m, mBaseVgpr,
                                f"nhBase += mfma_m (advance to m={m}).")
            if resBurst is not None:
                self._issueResidualLoads(module, m, resBurst, wgRowBase, rowGroupOff, mBaseVgpr)
                module.add(SWaitCnt(vlcnt=0, comment="wait residual row burst."))
                self._convertResidualRow(module, resBurst)
            self._residualAccRow(module, vgprTiles, accBurst, resBurst, m)
        if resBurst is not None:
            self._endResidual()
        if self.storeBf16D:
            # H is now in the accumulator tiles; emit the single macro-tile write-back.
            self._emitResidualOutMacroTileStore(module, vgprTiles, wgRowBase, cvtVgprStruct)
        vgprPool.checkIn(accBurst)
        vgprPool.checkIn(mBaseVgpr)
        vgprPool.checkIn(self._nhBaseV)
        vgprPool.checkIn(wgRowBase)
        vgprPool.checkIn(rowGroupOff)
        return module

    def emit(self, vgprTiles, cvtVgprStruct=None) -> Module:
        module = Module("ResidualAdd epilogue")
        module.addComment1("ResidualAdd: load residual, H = GEMM + residual, store ResidualOut bf16")
        self._allocEpilogueRegs()
        module.add(self._setup(self.laneId, self.colByte))
        module.add(self._residualPassFree0(vgprTiles, cvtVgprStruct))
        self._freeEpilogueRegs()
        return module
