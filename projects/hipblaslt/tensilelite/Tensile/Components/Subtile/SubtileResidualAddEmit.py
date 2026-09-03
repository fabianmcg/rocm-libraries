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

from rocisa.code import Label, Module
from rocisa.container import ContinuousRegister, MUBUFModifiers, accvgpr, sgpr, vgpr
from rocisa.enum import HighBitSel
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadB32,
    BufferLoadB64,
    BufferLoadD16B16,
    BufferLoadD16U8,
    BufferStoreB16,
    BufferStoreB64,
    ECvtPkBF8toF32,
    ECvtPkFP8toF32,
    SAndB32,
    SBranch,
    SCBranchSCC1,
    SCmpEQU32,
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
    VCvtPkF32toBF16,
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
        # Wide bf16 store: pack rows_per_lane bf16 per (m,n) into one dwordx2;
        # requires residualAdd (uses the residual burst as a packing buffer) and
        # 4-aligned rows_per_lane.
        self.useWideBf16Store = (self.storeBf16D and self.residualAdd
                                 and self.rows_per_lane % 4 == 0)

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

    def _residualOutRowByteBase(self, module, dst: int, tokenBase: int, n: int, scratch: int) -> None:
        nOff = n * self.mfma_n
        self._addImmU32(module, dst, tokenBase, nOff, scratch, f"token_n = tokenBase + {nOff} (n={n}).")
        # token_n * SizesFree0 uses 32-bit VMulLOU32; valid while token_n * N_hidden
        # stays below 2^32 (all currently supported tensor sizes).
        module.add(VMulLOU32(dst=vgpr(dst), src0=sgpr("SizesFree+0"), src1=vgpr(dst),
                             comment="token_n * SizesFree0."))
        module.add(VLShiftLeftB32(dst=vgpr(dst), shiftHex=hex(1), src=vgpr(dst),
                                  comment="roRowByteBase = token_n * SizesFree0 * 2 (bf16)."))

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

    def _beginBf16Store(self, module) -> None:
        """Build the ResidualOut SRD and precompute per-n row-byte bases for bf16 stores."""
        self._buildResidualOutSrd(module, self.residualOutSrd)
        self._roTokenBase = self.writer.vgprPool.checkOut(1, tag="rAdd_roToken")
        self._roRowBase   = self.writer.vgprPool.checkOut(self.mma_n, tag="rAdd_roRowBase")
        self._roAddr      = self.writer.vgprPool.checkOut(1, tag="rAdd_roAddr")
        self._roVal       = self.writer.vgprPool.checkOut(1, tag="rAdd_roVal")
        self._roOobV      = self.writer.vgprPool.checkOut(1, tag="rAdd_roOob")
        self._roNhByte    = self.writer.vgprPool.checkOut(1, tag="rAdd_roNhByte")
        self._roOobMask   = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="rAdd_roOobMask", preventOverflow=False)
        # Single token-in-range mask slot; recomputed per-n in _storeBf16Elem to save
        # (mma_n - 1) * lane_sgpr_count SGPRs vs the old per-n precomputed layout.
        self._roTokenMask = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="rAdd_roTokMask",
            preventOverflow=False)
        # Alignment remainder is computed once and kept for the whole store pass so
        # _storeBf16RowWide can branch at runtime between wide and scalar paths.
        self._roNAlignRem = self.writer.sgprPool.checkOut(1, tag="rAdd_roNAlignRem")
        module.add(VLShiftRightB32(dst=vgpr(self._roTokenBase), shiftHex=hex(self.log2ElemBytes),
                                   src=vgpr(self.colByte), comment="tokenBase = colByte >> log2ElemBytes."))
        module.add(VMovB32(dst=vgpr(self._roOobV), src="BufferOOB",
                           comment="OOB byte offset -> ResidualOut store dropped."))
        module.add(SAndB32(dst=sgpr(self._roNAlignRem), src0=sgpr("SizesFree+0"),
                           src1=self.rows_per_lane - 1,
                           comment="N_hidden % rows_per_lane (rows_per_lane is pow2)."))
        for n in range(self.mma_n):
            self._residualOutRowByteBase(module, self._roRowBase + n, self._roTokenBase, n,
                                         self._roNhByte)

    def _endBf16Store(self, module) -> None:
        """Wait for all pending ResidualOut bf16 stores and free scratch registers."""
        module.add(SWaitCnt(vscnt=0, comment="wait ResidualOut bf16 stores."))
        self.writer.sgprPool.checkIn(self._roNAlignRem)
        self.writer.sgprPool.checkIn(self._roTokenMask)
        self.writer.sgprPool.checkIn(self._roOobMask)
        self.writer.vgprPool.checkIn(self._roNhByte)
        self.writer.vgprPool.checkIn(self._roOobV)
        self.writer.vgprPool.checkIn(self._roVal)
        self.writer.vgprPool.checkIn(self._roAddr)
        self.writer.vgprPool.checkIn(self._roRowBase)
        self.writer.vgprPool.checkIn(self._roTokenBase)

    def _computeBf16NhByte(self, module, m: int, k: int) -> None:
        """Compute the nhidden byte offset and OOB mask for element (m, k) from the shared nhBase."""
        self._addImmU32(module, self._roNhByte, self._nhBaseV, k, self._roAddr,
                        f"nhidden_pos = nhBase + {k} (m={m},k={k}).")
        module.add(VCmpLtU32(dst=sgpr(self._roOobMask, self.lane_sgpr_count),
                             src0=vgpr(self._roNhByte), src1=sgpr("SizesFree+0"),
                             comment="inRange = nhidden_pos < N_hidden."))
        module.add(VLShiftLeftB32(dst=vgpr(self._roNhByte), shiftHex=hex(1), src=vgpr(self._roNhByte),
                                  comment="nhByte = nhidden_pos * 2 (bf16)."))

    def _storeBf16Elem(self, module, accReg: int, n: int) -> None:
        """Store bf16(accReg) to ResidualOut[token, nhidden], dropping OOB lanes."""
        # Recompute token_n for this n into _roAddr (safe: _roAddr is overwritten next anyway).
        nOff = n * self.mfma_n
        self._addImmU32(module, self._roAddr, self._roTokenBase, nOff, self._roAddr,
                        f"token_n = tokenBase + {nOff} (n={n}).")
        module.add(VCmpLtU32(dst=sgpr(self._roTokenMask, self.lane_sgpr_count),
                             src0=vgpr(self._roAddr), src1=sgpr("SizesFree+1"),
                             comment="tokenInRange = token_n < M_tokens."))
        module.add(VAddU32(dst=vgpr(self._roAddr), src0=vgpr(self._roRowBase + n),
                           src1=vgpr(self._roNhByte),
                           comment="byteAddr = roRowByteBase[n] + nhByte."))
        # Clamp the full address so an OOB lane lands on exactly BufferOOB and is dropped.
        module.add(VCndMaskB32(dst=vgpr(self._roAddr), src0=vgpr(self._roOobV),
                               src1=vgpr(self._roAddr),
                               src2=sgpr(self._roTokenMask, self.lane_sgpr_count),
                               comment="clamp OOB when token_n >= M_tokens."))
        module.add(VCndMaskB32(dst=vgpr(self._roAddr), src0=vgpr(self._roOobV),
                               src1=vgpr(self._roAddr),
                               src2=sgpr(self._roOobMask, self.lane_sgpr_count),
                               comment="clamp OOB when nhidden_pos >= N_hidden."))
        module.add(VCvtPkF32toBF16(dst=vgpr(self._roVal), src0=vgpr(accReg), src1=vgpr(accReg),
                                   comment="H+residual f32 -> bf16 (low16)."))
        module.add(BufferStoreB16(src=vgpr(self._roVal), vaddr=vgpr(self._roAddr),
                                  saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                  mubuf=MUBUFModifiers(offen=True),
                                  comment="ResidualOut[token, nhidden] = bf16(H+residual)."))

    def _storeBf16RowWide(self, module, resBurst: int, m: int) -> None:
        """Dispatch bf16 H stores for tile row m: wide (aligned) or scalar (unaligned).

        The wide dwordx2 path is taken only when N_hidden is a multiple of
        rows_per_lane (checked at runtime); otherwise the per-element scalar path
        preserves correctness for straddling groups.
        """
        alignedLabel = Label(self.writer.labels.getNameInc(f"rAdd_roWide_m{m}"), "")
        endLabel     = Label(self.writer.labels.getNameInc(f"rAdd_roWideEnd_m{m}"), "")
        module.add(SCmpEQU32(src0=sgpr(self._roNAlignRem), src1=0,
                             comment="N_hidden aligned to rows_per_lane."))
        module.add(SCBranchSCC1(labelName=alignedLabel.getLabelName(),
                                comment="aligned -> wide dwordx2 store."))
        self._storeBf16RowScalar(module, resBurst, m)
        module.add(SBranch(labelName=endLabel.getLabelName(), comment="skip wide store."))
        module.add(alignedLabel)
        self._storeBf16RowWideAligned(module, resBurst, m)
        module.add(endLabel)

    def _storeBf16RowScalar(self, module, resBurst: int, m: int) -> None:
        """Per-element bf16 H store fallback for N_hidden not aligned to rows_per_lane."""
        for k in range(self.rows_per_lane):
            self._computeBf16NhByte(module, m, k)
            for n in range(self.mma_n):
                self._storeBf16Elem(module, resBurst + n * self.rows_per_lane + k, n)

    def _setupBf16WideNhMask(self, module, m: int) -> None:
        """Compute whole-group OOB mask and nhByte for wide bf16 row m from the shared nhBase."""
        rpl = self.rows_per_lane
        lsc = self.lane_sgpr_count
        # Whole-group nhidden mask: in range iff the last element (k=rpl-1) < N_hidden.
        self._addImmU32(module, self._roAddr, self._nhBaseV, rpl - 1, self._roAddr,
                        f"nhiddenLast = nhBase + {rpl - 1}.")
        module.add(VCmpLtU32(dst=sgpr(self._roOobMask, lsc), src0=vgpr(self._roAddr),
                             src1=sgpr("SizesFree+0"),
                             comment="group in range = nhiddenLast < N_hidden."))
        module.add(VLShiftLeftB32(dst=vgpr(self._roNhByte), shiftHex=hex(1),
                                  src=vgpr(self._nhBaseV),
                                  comment="nhByte = nhBase * 2 (bf16)."))

    def _storeBf16WideElem(self, module, resBurst: int, m: int, n: int) -> None:
        """Pack and store rows_per_lane bf16 values for element (m, n) via buffer_store_dwordx2."""
        rpl = self.rows_per_lane
        lsc = self.lane_sgpr_count
        nOff = n * self.mfma_n
        self._addImmU32(module, self._roAddr, self._roTokenBase, nOff, self._roAddr,
                        f"token_n = tokenBase + {nOff} (n={n}).")
        module.add(VCmpLtU32(dst=sgpr(self._roTokenMask, lsc), src0=vgpr(self._roAddr),
                             src1=sgpr("SizesFree+1"),
                             comment="tokenInRange = token_n < M_tokens."))
        module.add(VAddU32(dst=vgpr(self._roAddr), src0=vgpr(self._roRowBase + n),
                           src1=vgpr(self._roNhByte),
                           comment="byteAddr = roRowByteBase[n] + nhByte(k=0)."))
        module.add(VCndMaskB32(dst=vgpr(self._roAddr), src0=vgpr(self._roOobV),
                               src1=vgpr(self._roAddr),
                               src2=sgpr(self._roTokenMask, lsc),
                               comment="clamp OOB when token_n >= M_tokens."))
        module.add(VCndMaskB32(dst=vgpr(self._roAddr), src0=vgpr(self._roOobV),
                               src1=vgpr(self._roAddr),
                               src2=sgpr(self._roOobMask, lsc),
                               comment="clamp OOB when nhidden group >= N_hidden."))
        for c in range(rpl // 4):
            # c > 0 is dead for rows_per_lane == 4; live only for rows_per_lane >= 8.
            base = resBurst + n * rpl + 4 * c
            # Pack in order: dword0 from k0,k1 into base+0, then dword1 from k2,k3
            # into base+1. The first pack reads base+0,base+1 before clobbering base+0;
            # the second pack reads base+2,base+3 before clobbering base+1.
            module.add(VCvtPkF32toBF16(dst=vgpr(base + 0), src0=vgpr(base + 0),
                                       src1=vgpr(base + 1),
                                       comment="pack H k0,k1 -> bf16 dword."))
            module.add(VCvtPkF32toBF16(dst=vgpr(base + 1), src0=vgpr(base + 2),
                                       src1=vgpr(base + 3),
                                       comment="pack H k2,k3 -> bf16 dword."))
            stAddr = self._roAddr
            if c > 0:
                self._addImmU32(module, self._roVal, self._roAddr, 8 * c, self._roVal,
                                f"chunk byte offset {8 * c}.")
                stAddr = self._roVal
            module.add(BufferStoreB64(src=vgpr(base + 0, 2), vaddr=vgpr(stAddr),
                                      saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"ResidualOut b64 [4 bf16] (m={m},n={n},c={c})."))

    def _storeBf16RowWideAligned(self, module, resBurst: int, m: int) -> None:
        """Store rows_per_lane bf16 H values per n with buffer_store_dwordx2.

        Only called when N_hidden is a multiple of rows_per_lane, so a group of
        rows_per_lane elements is either entirely in range or entirely OOB.
        """
        self._setupBf16WideNhMask(module, m)
        for n in range(self.mma_n):
            self._storeBf16WideElem(module, resBurst, m, n)

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
        # When useWideBf16Store: H is left in resReg (the residual burst) for wide packing,
        # then written back to the accumulator. When residualAdd is False but storeBf16D is
        # True, the accumulator already holds H (= GEMM acc), so no writeback is needed.
        if self.useWideBf16Store:
            module.add(VAddF32(dst=vgpr(resReg), src0=vgpr(accReg), src1=vgpr(resReg),
                               comment="H = GEMM + residual (kept in burst for wide store)."))
            self._writeAccFrom(module, resReg, vgprTiles, m, n, k, f"write H back to acc (m={m},n={n},k={k}).")
            return
        if resReg is not None:
            module.add(VAddF32(dst=vgpr(accReg), src0=vgpr(accReg), src1=vgpr(resReg),
                               comment="H = GEMM + residual."))
        if self.storeBf16D:
            self._storeBf16Elem(module, accReg, n)
        if resReg is not None:
            self._writeAccFrom(module, accReg, vgprTiles, m, n, k, f"write H back to acc (m={m},n={n},k={k}).")

    def _residualAccRow(self, module, vgprTiles, accBurst: int, resBurst, m: int) -> None:
        for k in range(self.rows_per_lane):
            coords = [(m, n, k) for n in range(self.mma_n)]
            self._readAccBurst(module, accBurst, vgprTiles, coords, f"read acc m={m},k={k}.")
            if self.storeBf16D and not self.useWideBf16Store:
                self._computeBf16NhByte(module, m, k)
            for n in range(self.mma_n):
                resReg = None if resBurst is None else resBurst + n * self.rows_per_lane + k
                self._residualAccElement(module, vgprTiles, accBurst + n, resReg, m, n, k)
        # Wide store packs (and clobbers) the residual burst in place, so the element
        # loop writeback must complete before _storeBf16RowWide runs.
        if self.useWideBf16Store:
            self._storeBf16RowWide(module, resBurst, m)

    def _residualPassFree0(self, vgprTiles) -> Module:
        module = Module("ResidualAdd residualPassFree0")
        module.addComment1("ResidualAdd pass (free0): load residual, H = GEMM + residual, write H back, bf16 store")
        vgprPool = self.writer.vgprPool
        rowGroupOff = vgprPool.checkOut(1, tag="rAdd_rgOff")
        wgRowBase   = vgprPool.checkOut(1, tag="rAdd_wgRowBase")
        self._computeRowGroupOff(module, rowGroupOff)
        self._computeFree0RowBase(module, wgRowBase)
        self._nhBaseV = vgprPool.checkOut(1, tag="rAdd_nhBase")
        if self.storeBf16D:
            self._beginBf16Store(module)
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
        if self.storeBf16D:
            self._endBf16Store(module)
        if resBurst is not None:
            self._endResidual()
        vgprPool.checkIn(accBurst)
        vgprPool.checkIn(mBaseVgpr)
        vgprPool.checkIn(self._nhBaseV)
        vgprPool.checkIn(wgRowBase)
        vgprPool.checkIn(rowGroupOff)
        return module

    def emit(self, vgprTiles) -> Module:
        module = Module("ResidualAdd epilogue")
        module.addComment1("ResidualAdd: load residual, H = GEMM + residual, store ResidualOut bf16")
        self._allocEpilogueRegs()
        module.add(self._setup(self.laneId, self.colByte))
        module.add(self._residualPassFree0(vgprTiles))
        self._freeEpilogueRegs()
        return module
