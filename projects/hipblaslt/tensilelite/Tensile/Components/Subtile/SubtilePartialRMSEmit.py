# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PartialRMS fused epilogue emitter for the Subtile kernel (gfx950, bf16).

Reduces the fp32 GEMM accumulator (AGPRs) over free0 (the N_hidden dimension),
producing one fp32 Σx² per output token (free1 row) per free0 tile (WorkGroup0).
Also applies the gamma weight (bf16) to the accumulator in-place.

This is Phase 1 (K1) of a two-kernel RMSNorm pipeline operating on row-major output:
  - K1 (this kernel): free0=N_hidden, free1=M_tokens. Each WG writes
    partialBuf[token, WorkGroup0] = Σ_{i in tile} h1[token, i]².
  - K2 (row_div): reads all partialBuf tiles per token, reduces, computes
    rstd = rsqrt(Σx²/N_hidden + eps), and divides D in-place.

partialBuf layout contract (2D, row-major):
  - Logical shape [M_padded, n_d], n_d = ceil(SizesFree0 / MT0).
  - partialBuf[token, t] = Σ_{i in WG t's free0 columns} h1[token, i]².
  - Byte offset for (token, t) = (token * n_d + t) * 4.
  - Token index = WorkGroup1*MT1 + intra-tile token offset.
  - Tile column t = WorkGroup0 (the WG's index along free0).
  - n_d is computed on device from SizesFree0; it is not a kernarg.

Reduction stages (free0 axis):
  1. Sum acc²  over all free0 MMA-tiles (mma_m) and k-offsets (rows_per_lane)
     within each wave, yielding one partial[n] per free1 lane-column (mfma_n lanes).
  2. XOR butterfly via ds_bpermute over waveSize/mfma_n row groups so every
     lane holds the full free0 sum for its n column.
  3. (When wg_m > 1) LDS cross-wave reduction over wg_m sibling waves.
  4. Lanes with rowGroup==0 and waveM==0 write partialBuf[token, WG0].

Gamma application:
  Loads gamma (bf16) for each free0 position via BufferLoadD16B16, converts to
  fp32, and multiplies each accumulator element in-place before the store path
  writes D as bf16.

MFMA layout (gfx950, waveSize=64, 16x16 MFMA):
  - lane % mfma_n = free1 column within MMA tile (token lane)
  - rows_per_lane = (mfma_m * mfma_n) // waveSize

Acc VGPR ordering (N-outer, M-inner):
  acc_idx(base, m, n, k) = base + (n*mma_m + m)*rows_per_lane + k

Alpha=1, beta=0 must be passed by the host.
"""

import math
import struct

from rocisa.code import Module
from rocisa.container import (
    ContinuousRegister,
    DPPModifiers,
    DSModifiers,
    EXEC,
    MUBUFModifiers,
    accvgpr,
    sgpr,
    vgpr,
)
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadD16B16,
    BufferStoreB32,
    DSBPermuteB32,
    DSLoadB32,
    DSStoreB32,
    SAndSaveExecB64,
    SMovB32,
    SMovB64,
    SMulHIU32,
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddF32,
    VAddU32,
    VAndB32,
    VCmpEQU32,
    VCmpLtU32,
    VCndMaskB32,
    VCvtBF16toFP32,
    VFmaF32,
    VLShiftLeftB32,
    SAddU32,
    SLShiftLeftB32,
    SLShiftRightB32,
    VMaxF32,
    VMulF32,
    VMulLOU32,
    VMovB32,
    VLShiftRightB32,
    VOrB32,
    VXorB32,
    SMulI32,
)


# OCP FP8 e4m3 maximum representable magnitude.
_FP8_E4M3_MAX = 448.0


# Returns (magic, postShift) for unsigned floor-division by constant d via SMulHIU32.
# floor(x/d) = mulhi(x, magic) >> postShift; for ceil-div pre-add (d-1). Valid for d >= 2.
def _ceilDivMagic(d: int):
    p = (d - 1).bit_length()          # smallest p such that 2^p >= d
    magic = -(-( 1 << (32 + p - 1)) // d)  # ceil(2^(32+p-1) / d), using integer ceiling
    return magic & 0xFFFFFFFF, p - 1  # postShift = p-1 (mulhi already shifts by 32)


class SubtilePartialRMSEmitter:
    """Emit the PartialRMS epilogue for the Subtile gfx950 bf16 kernel.

    Computes per-row Σx² from fp32 AGPRs, writes to partialBuf (fp32),
    and applies gamma (bf16) in-place to the accumulator.
    """

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        self.archCaps = writer.states.archCaps

        # Derive all geometry from kernel params; no module-level constants.
        self.mfma_m = kernel["MatrixInstM"]
        self.mfma_n = kernel["MatrixInstN"]
        self.waveSize = kernel["WavefrontSize"]
        self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.waveSize

        wg = kernel["MIWaveGroup"]
        self.wg_m = wg[0]
        self.wg_n = wg[1]

        self.mma_m = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
        self.mma_n = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
        self.macro_tile0 = kernel["MacroTile0"]
        self.macro_tile1 = kernel["MacroTile1"]
        self.numRows = self.mma_m * self.rows_per_lane
        self.numPartials = self.mma_n

        # laneSGPRCount: 1 for wave32, 2 for wave64.
        self.lane_sgpr_count = writer.states.laneSGPRCount
        self.residualAdd = bool(kernel.get("PartialRMSResidualAdd", False))
        self.quant = bool(kernel.get("PartialRMSQuant", False))

    def _readAccInto(self, module, dst: int, vgprTiles, m: int, n: int, k: int, comment: str) -> None:
        """Copy accumulator element (m, n, k) into VGPR dst, selecting the right register file."""
        tile = vgprTiles[n * self.mma_m + m]
        reg = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            module.add(VMovB32(dst=vgpr(dst), src=vgpr(reg), comment=comment))
            return
        module.add(VAccvgprReadB32(vgpr(dst), accvgpr(reg), comment=comment))
        module.add(SNop(waitState=1, comment="s_nop after v_accvgpr_read before VALU (gfx950)."))

    def _writeAccFrom(self, module, src: int, vgprTiles, m: int, n: int, k: int, comment: str) -> None:
        """Write VGPR src back into accumulator element (m, n, k), selecting the right register file."""
        tile = vgprTiles[n * self.mma_m + m]
        reg = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            module.add(VMovB32(dst=vgpr(reg), src=vgpr(src), comment=comment))
            return
        module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(src), comment=comment))

    def _buildBufferSrd(self, module, srd: int, ptrName: str, name: str) -> None:
        module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr(ptrName, 2), comment=f"{name} SRD base"))
        module.add(SMovB32(dst=sgpr(srd + 2), src="BufferOOB", comment=f"{name} SRD limit"))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96", comment=f"{name} SRD flags"))

    def _addImmU32(self, module, dst: int, src: int, imm: int, scratch: int, comment: str) -> None:
        # Materialize the immediate in a VGPR when it exceeds the inline-literal range.
        if imm > 64:
            module.add(VMovB32(dst=vgpr(scratch), src=imm, comment=f"imm={imm}"))
            module.add(VAddU32(vgpr(dst), vgpr(src), vgpr(scratch), comment=comment))
            return
        module.add(VAddU32(vgpr(dst), vgpr(src), imm, comment=comment))

    def _computeWaveM(self, module, dst: int) -> None:
        waveId = self.writer.vgprPool.checkOut(1, tag="pRMS_waveId")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_waveMDiv")
        tmpRes = ContinuousRegister(tmpVgpr, 2)
        module.add(vectorStaticDivide(waveId, "Serial", self.waveSize, tmpRes,
                                      comment="waveId = Serial / WavefrontSize"))
        module.add(VAndB32(dst=vgpr(dst), src0=vgpr(waveId), src1=self.wg_m - 1,
                           comment=f"waveM = waveId % {self.wg_m}"))
        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(waveId)

    def _computeRowGroupOff(self, module, dst: int) -> None:
        log2MfmaN = int(math.log2(self.mfma_n))
        laneIdV = self.writer.vgprPool.checkOut(1, tag="pRMS_rgoLaneId")
        module.add(VAndB32(dst=vgpr(laneIdV), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)"))
        module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(log2MfmaN), src=vgpr(laneIdV),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        module.add(VMulLOU32(dst=vgpr(dst), src0=self.rows_per_lane, src1=vgpr(dst),
                             comment=f"rowGroupOff = rowGroup * {self.rows_per_lane}"))
        self.writer.vgprPool.checkIn(laneIdV)

    def _computeFree0RowBase(self, module, dst: int) -> None:
        # rowBase = WorkGroup0 * MT0 (+ waveM * mma_m*mfma_m when wg_m > 1).
        mt0Vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_rbMT0")
        module.add(VMovB32(dst=vgpr(mt0Vgpr), src=self.macro_tile0, comment=f"MT0={self.macro_tile0}"))
        module.add(VMulLOU32(dst=vgpr(dst), src0=vgpr(mt0Vgpr), src1=sgpr("WorkGroup0"),
                             comment="rowBase = WorkGroup0 * MT0"))
        self.writer.vgprPool.checkIn(mt0Vgpr)
        if self.wg_m <= 1:
            return
        waveM = self.writer.vgprPool.checkOut(1, tag="pRMS_rbWaveM")
        self._computeWaveM(module, waveM)
        waveStride = self.mma_m * self.mfma_m
        strideV = self.writer.vgprPool.checkOut(1, tag="pRMS_rbStride")
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

    def emit(self, vgprTiles) -> Module:
        # vgprTiles is dtileInfo.vgprTiles, the per-tile allocator records for the D accumulator.
        numAccVgpr = self.mma_m * self.mma_n * self.rows_per_lane
        accVgprBase = vgprTiles[0].regList.indices[0] if vgprTiles else 0
        module = Module("PartialRMS epilogue")
        module.addComment1("PartialRMS: fused partial sum-of-squares + gamma epilogue")
        module.addComment0(
            f"  Acc AGPRs [{accVgprBase}, {accVgprBase + numAccVgpr}), "
            f"mma_m={self.mma_m}, mma_n={self.mma_n}, "
            f"MT0={self.macro_tile0}, MT1={self.macro_tile1}"
        )
        module.addComment0(
            "  partialBuf: raw Σx² per row (K2 divides by N_hidden, not this kernel)"
        )

        # Allocate all VGPRs for temporaries.
        partials = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_partials")
        accTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_accTmp")
        gammaTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_gammaTmp")
        laneId = self.writer.vgprPool.checkOut(1, tag="pRMS_laneId")
        # colByte is computed and consumed outside the EXEC-narrowed window in _writePartials.
        colByte = self.writer.vgprPool.checkOut(1, tag="pRMS_colByte")
        globalAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_globalAddr")
        if self.quant:
            amaxPartials = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_amaxPartials")
            mPaddedV     = self.writer.vgprPool.checkOut(1, tag="pRMS_mPaddedV")
            amaxScaleV   = self.writer.vgprPool.checkOut(1, tag="pRMS_amaxScale")
        if self.residualAdd:
            resTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_resTmp")
            resAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_resAddr")

        # Allocate SGPRs: gamma SRD, partialBuf SRD, saved exec.
        # savedExec and laneMaskSgpr must be 2-aligned for 64-bit EXEC operations.
        # rowBase = WorkGroup0 * MT0 is computed into globalAddr on demand (no SGPR).
        # tileCol = sgpr("WorkGroup1"), live named SGPR, no allocation needed.
        # preventOverflow=False: the epilogue SGPRs are temporary (live only during
        # the epilogue) and hardware SGPR budget has been verified to accommodate them.
        gammaSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_gammaSrd",
                                                        preventOverflow=False)
        partialSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_partialSrd",
                                                          preventOverflow=False)
        savedExec = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_savedExec",
            preventOverflow=False,
        )
        laneMaskSgpr = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_laneMask",
            preventOverflow=False,
        )
        if self.residualAdd:
            resSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_resSrd",
                                                          preventOverflow=False)

        # Flush MFMA pipeline before reading AGPRs.
        module.add(
            SWaitCnt(waitAll=True, comment="flush MFMA pipeline before PartialRMS")
        )

        module.add(self._setup(gammaSrd, partialSrd, laneId, colByte))
        if self.residualAdd:
            module.add(
                self._addResidualFree0(vgprTiles, accTmp, resTmp, resAddr, colByte, resSrd)
            )
        module.add(self._squareAndLaneSumFree0(vgprTiles, partials, accTmp))
        module.add(self._rowGroupReduceFree0(partials))
        if self.wg_m > 1:
            module.add(self._crossWaveReduceFree0(partials))
        module.add(
            self._writePartialsFree0(
                partials, partialSrd, laneId, savedExec, laneMaskSgpr,
                globalAddr, colByte
            )
        )
        module.add(
            self._applyGammaFree0(vgprTiles, gammaSrd, gammaTmp, accTmp, globalAddr)
        )
        if self.quant:
            module.add(self._amaxEpilogueFree0(
                vgprTiles, amaxPartials, accTmp, partialSrd, laneId,
                savedExec, laneMaskSgpr, globalAddr, colByte, mPaddedV, amaxScaleV))

        if self.residualAdd:
            self.writer.sgprPool.checkIn(resSrd)
        self.writer.sgprPool.checkIn(laneMaskSgpr)
        self.writer.sgprPool.checkIn(savedExec)
        self.writer.sgprPool.checkIn(partialSrd)
        self.writer.sgprPool.checkIn(gammaSrd)
        self.writer.vgprPool.checkIn(globalAddr)
        self.writer.vgprPool.checkIn(colByte)
        self.writer.vgprPool.checkIn(laneId)
        self.writer.vgprPool.checkIn(gammaTmp)
        self.writer.vgprPool.checkIn(accTmp)
        self.writer.vgprPool.checkIn(partials)
        if self.residualAdd:
            self.writer.vgprPool.checkIn(resAddr)
            self.writer.vgprPool.checkIn(resTmp)
        if self.quant:
            self.writer.vgprPool.checkIn(amaxScaleV)
            self.writer.vgprPool.checkIn(mPaddedV)
            self.writer.vgprPool.checkIn(amaxPartials)

        return module

    def _setup(self, gammaSrd: int, partialSrd: int, laneId: int, colByte: int) -> Module:
        module = Module("PartialRMS setup")
        module.add(SWaitCnt(kmcnt=0, comment="wait for PartialRMS kernarg s_load"))
        self._buildBufferSrd(module, gammaSrd, "RMSNormGamma", "gamma")
        self._buildBufferSrd(module, partialSrd, "PartialBuf", "partialBuf")
        module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)"))
        module.add(VAndB32(dst=vgpr(colByte), src0=vgpr(laneId), src1=self.mfma_n - 1,
                           comment=f"colInMma = laneId % {self.mfma_n}"))
        module.add(VLShiftLeftB32(dst=vgpr(colByte), shiftHex=hex(1), src=vgpr(colByte),
                                  comment="colByte = colInMma * 2 (bf16 size)"))
        self._addWaveNColByte(module, colByte)
        with self.writer.allocTmpSgpr(1, tag="pRMS_setupWG1") as wg1S:
            module.add(SMulI32(dst=sgpr(wg1S.idx), src0=sgpr("WorkGroup1"), src1=self.macro_tile1 * 2,
                               comment=f"wg1ColByte = WorkGroup1 * MT1*2 (MT1={self.macro_tile1})"))
            module.add(VAddU32(vgpr(colByte), vgpr(colByte), sgpr(wg1S.idx),
                               comment="colByte += WorkGroup1 * MT1 * 2"))
        return module

    def _addWaveNColByte(self, module, colByte: int) -> None:
        if self.wg_n <= 1:
            return
        waveN = self.writer.vgprPool.checkOut(1, tag="pRMS_setupWaveN")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_setupTmp")
        tmpRes = ContinuousRegister(tmpVgpr, 2)
        module.add(vectorStaticDivide(waveN, "Serial", self.waveSize * self.wg_m, tmpRes,
                                      comment=f"waveN = Serial / {self.waveSize * self.wg_m}"))
        colBaseBytes = self.mma_n * self.mfma_n * 2
        with self.writer.allocTmpSgpr(1, tag="pRMS_setupColBase") as tmpSgprInfo:
            module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(colBaseBytes),
                               comment=f"col base bytes per wave ({colBaseBytes})"))
            module.add(VMulLOU32(dst=vgpr(waveN), src0=sgpr(tmpSgprInfo.idx), src1=vgpr(waveN),
                                 comment="waveN * mma_n * mfma_n * 2"))
        module.add(VAddU32(vgpr(colByte), vgpr(colByte), vgpr(waveN),
                           comment="colByte += wave column base"))
        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(waveN)

    def _addResidualFree0(self, vgprTiles, accTmp: int, resTmp: int, resAddr: int,
                          colByte: int, resSrd: int) -> Module:
        module = Module("PartialRMS addResidualFree0")
        module.addComment1("PartialRMS residual add (free0): acc[m,n,k] += R[token, nhidden_pos]")
        # Residual SRD bounds = M_tokens*N_hidden*2 so tail-WG OOB lanes read 0.
        with self.writer.allocTmpSgpr(1, tag="pRMS_rF0SrdNumRec") as tmpSgpr:
            module.add(SMovB64(dst=sgpr(resSrd, 2), src=sgpr("ResidualBuf", 2),
                               comment="residual SRD base"))
            module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("SizesFree+0"),
                               src1=sgpr("SizesFree+1"), comment="numRecords = N_hidden * M_tokens"))
            module.add(SLShiftLeftB32(dst=sgpr(resSrd + 2), src=sgpr(tmpSgpr.idx), shiftHex=hex(1),
                                      comment="numRecords *= 2 (bf16 element size)"))
        module.add(SMovB32(dst=sgpr(resSrd + 3), src="Srd127_96", comment="residual SRD flags"))
        rowGroupOff = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0RGOff")
        nhiddenBase = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0NHiddenBase")
        tokenBase = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0TokenBase")
        self._computeRowGroupOff(module, rowGroupOff)
        self._computeFree0RowBase(module, nhiddenBase)
        module.add(VLShiftRightB32(dst=vgpr(tokenBase), shiftHex=hex(1), src=vgpr(colByte),
                                   comment="tokenBase = colByte >> 1"))
        module.add(self._loopResidualFree0(vgprTiles, accTmp, resTmp, resAddr,
                                           nhiddenBase, tokenBase, rowGroupOff, resSrd))
        self.writer.vgprPool.checkIn(tokenBase)
        self.writer.vgprPool.checkIn(nhiddenBase)
        self.writer.vgprPool.checkIn(rowGroupOff)
        return module

    def _residualRowByteBase(self, module, dst: int, tokenBase: int, n: int, scratch: int) -> None:
        nOff = n * self.mfma_n
        self._addImmU32(module, dst, tokenBase, nOff, scratch, f"token_n = tokenBase + {nOff} (n={n})")
        module.add(VMulLOU32(dst=vgpr(dst), src0=sgpr("SizesFree+0"), src1=vgpr(dst),
                             comment="token_n * SizesFree0"))
        module.add(VLShiftLeftB32(dst=vgpr(dst), shiftHex=hex(1), src=vgpr(dst),
                                  comment="rowByteBase = token_n * SizesFree0 * 2"))

    def _addResidualElement(self, module, vgprTiles, accTmp: int, resTmp: int, resAddr: int,
                            rowByteBase: int, nhiddenBase: int, rowGroupOff: int, resSrd: int,
                            oobV: int, oobMask: int, scratch: int, m: int, n: int, k: int) -> None:
        self._free0RowPos(module, resAddr, nhiddenBase, rowGroupOff, m, k, scratch)
        module.add(VCmpLtU32(dst=sgpr(oobMask, self.lane_sgpr_count), src0=vgpr(resAddr),
                             src1=sgpr("SizesFree+0"), comment="inRange = nhidden_pos < N_hidden"))
        module.add(VLShiftLeftB32(dst=vgpr(resAddr), shiftHex=hex(1), src=vgpr(resAddr),
                                  comment="nhiddenByte = nhidden_pos * 2"))
        module.add(VAddU32(vgpr(resAddr), vgpr(resAddr), vgpr(rowByteBase),
                           comment="byteAddr = rowByteBase + nhiddenByte"))
        module.add(VCndMaskB32(dst=vgpr(resAddr), src0=vgpr(oobV), src1=vgpr(resAddr),
                               src2=sgpr(oobMask, self.lane_sgpr_count),
                               comment="clamp OOB when nhidden_pos >= N_hidden"))
        module.add(BufferLoadD16B16(vgpr(resTmp), vgpr(resAddr), sgpr(resSrd, 4), 0,
                                    MUBUFModifiers(offen=True),
                                    comment=f"R[token_n, nhidden_pos] bf16 (m={m},n={n},k={k})"))
        module.add(SWaitCnt(vlcnt=0, comment="wait residual load"))
        module.add(VCvtBF16toFP32(vgpr(resTmp), vgpr(resTmp), None, 0, comment="residual bf16 -> fp32"))
        self._readAccInto(module, accTmp, vgprTiles, m, n, k, f"read acc[m={m},n={n},k={k}]")
        module.add(VAddF32(dst=vgpr(accTmp), src0=vgpr(accTmp), src1=vgpr(resTmp),
                           comment="H = GEMM + residual"))
        self._writeAccFrom(module, accTmp, vgprTiles, m, n, k, f"write acc[m={m},n={n},k={k}] += residual")

    def _loopResidualFree0(self, vgprTiles, accTmp: int, resTmp: int, resAddr: int,
                           nhiddenBase: int, tokenBase: int, rowGroupOff: int, resSrd: int) -> Module:
        module = Module("PartialRMS loopResidualFree0")
        scratch = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0NOff")
        rowByteBase = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0RowByteBase")
        oobV = self.writer.vgprPool.checkOut(1, tag="pRMS_rF0Oob")
        module.add(VMovB32(dst=vgpr(oobV), src="BufferOOB",
                           comment="OOB byte offset -> residual load returns 0"))
        oobMask = self.writer.sgprPool.checkOutAligned(self.lane_sgpr_count, self.lane_sgpr_count,
                                                       tag="pRMS_rF0OobMask", preventOverflow=False)
        for n in range(self.mma_n):
            self._residualRowByteBase(module, rowByteBase, tokenBase, n, scratch)
            for m in range(self.mma_m):
                for k in range(self.rows_per_lane):
                    self._addResidualElement(module, vgprTiles, accTmp, resTmp, resAddr,
                                             rowByteBase, nhiddenBase, rowGroupOff, resSrd,
                                             oobV, oobMask, scratch, m, n, k)
        self.writer.sgprPool.checkIn(oobMask)
        self.writer.vgprPool.checkIn(oobV)
        self.writer.vgprPool.checkIn(rowByteBase)
        self.writer.vgprPool.checkIn(scratch)
        return module

    def _squareAndLaneSumFree0(self, vgprTiles, partials: int, accTmp: int) -> Module:
        # Step 1 (free0): per-column Σx² over M-rows from fp32 AGPRs.
        module = Module("PartialRMS squareAndLaneSumFree0")
        module.addComment1("PartialRMS step 1 (free0): per-column Σx² over M-rows")
        for n in range(self.mma_n):
            pidx = partials + n
            self._readAccInto(module, accTmp, vgprTiles, 0, n, 0, f"read acc[m=0,n={n},k=0]")
            module.add(
                VMulF32(dst=vgpr(pidx), src0=vgpr(accTmp), src1=vgpr(accTmp),
                        comment=f"partial[n={n}] = acc^2")
            )
            for m in range(self.mma_m):
                for k in range(self.rows_per_lane):
                    if m == 0 and k == 0:
                        continue
                    self._readAccInto(module, accTmp, vgprTiles, m, n, k,
                                      f"read acc[m={m},n={n},k={k}]")
                    module.add(
                        VFmaF32(dst=vgpr(pidx), src0=vgpr(accTmp), src1=vgpr(accTmp),
                                src2=vgpr(pidx), comment=f"partial[n={n}] += acc^2")
                    )
        return module

    def _absAndLaneMaxFree0(self, vgprTiles, amaxPartials: int, accTmp: int) -> Module:
        module = Module("PartialRMS absAndLaneMaxFree0")
        module.addComment1("PartialRMSQuant step 6 (free0): per-column max|D| over M-rows")
        # Materialize the sign-bit mask into a VGPR; VAndB32 cannot encode large literals.
        absMask = self.writer.vgprPool.checkOut(1, tag="pRMS_absMask")
        module.add(VMovB32(dst=vgpr(absMask), src=hex(0x7FFFFFFF), comment="abs mask = 0x7FFFFFFF"))
        for n in range(self.mma_n):
            pidx = amaxPartials + n
            self._readAccInto(module, accTmp, vgprTiles, 0, n, 0, f"read D[m=0,n={n},k=0]")
            module.add(VAndB32(dst=vgpr(pidx), src0=vgpr(accTmp), src1=vgpr(absMask),
                               comment=f"amax[n={n}] = |D|"))
            for m in range(self.mma_m):
                for k in range(self.rows_per_lane):
                    if m == 0 and k == 0:
                        continue
                    self._readAccInto(module, accTmp, vgprTiles, m, n, k,
                                      f"read D[m={m},n={n},k={k}]")
                    module.add(VAndB32(dst=vgpr(accTmp), src0=vgpr(accTmp), src1=vgpr(absMask),
                                       comment="|D|"))
                    module.add(VMaxF32(dst=vgpr(pidx), src0=vgpr(pidx), src1=vgpr(accTmp),
                                       comment=f"amax[n={n}] = max(amax, |D|)"))
        self.writer.vgprPool.checkIn(absMask)
        return module

    def _rowGroupReduceFree0(self, partials: int, op=VAddF32, verb="+") -> Module:
        # Step 2 (free0): all-reduce partial[n] across row groups via ds_bpermute XOR butterfly.
        numRounds = int(math.log2(self.waveSize // self.mfma_n))
        module = Module("PartialRMS rowGroupReduceFree0")
        module.addComment1(
            f"PartialRMS step 2 (free0): XOR butterfly over {self.waveSize // self.mfma_n} row groups"
        )
        if numRounds == 0:
            return module

        laneIdV = self.writer.vgprPool.checkOut(1, tag="pRMS_rgrLaneId")
        addrV = self.writer.vgprPool.checkOut(1, tag="pRMS_rgrAddr")
        tmpV = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_rgrTmp")

        module.add(
            VAndB32(dst=vgpr(laneIdV), src0=vgpr("Serial"), src1=self.waveSize - 1,
                    comment="laneId = Serial & (waveSize-1)")
        )
        for i in range(numRounds):
            xorVal = self.mfma_n << i
            module.add(
                VXorB32(dst=vgpr(addrV), src0=vgpr(laneIdV), src1=xorVal,
                        comment=f"partnerLane = laneId ^ {xorVal}")
            )
            module.add(
                VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(2), src=vgpr(addrV),
                               comment="byteAddr = partnerLane * 4")
            )
            for n in range(self.numPartials):
                module.add(
                    DSBPermuteB32(vgpr(tmpV + n), vgpr(addrV), vgpr(partials + n),
                                  comment=f"fetch partner partial[{n}]")
                )
            module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
            for n in range(self.numPartials):
                module.add(
                    op(dst=vgpr(partials + n), src0=vgpr(partials + n),
                       src1=vgpr(tmpV + n), comment=f"partial[{n}] {verb} partner")
                )

        self.writer.vgprPool.checkIn(tmpV)
        self.writer.vgprPool.checkIn(addrV)
        self.writer.vgprPool.checkIn(laneIdV)
        return module

    def _crossWaveComputeAddrs(self, module, writeAddr: int, readAddr: int) -> None:
        strideW = self.waveSize * self.numPartials * 4
        laneSlotBytes = self.numPartials * 4
        waveId = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WaveId")
        waveM = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WaveM")
        readBaseWave = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadBase")
        laneLoc = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0Lane")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_xwF0Tmp")
        tmpRes = ContinuousRegister(tmpVgpr, 2)
        module.add(VAndB32(dst=vgpr(laneLoc), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="laneId for LDS addressing"))
        module.add(vectorStaticDivide(waveId, "Serial", self.waveSize, tmpRes,
                                      comment="waveId = Serial / WavefrontSize"))
        module.add(VAndB32(dst=vgpr(waveM), src0=vgpr(waveId), src1=self.wg_m - 1,
                           comment=f"waveM = waveId % {self.wg_m}"))
        # readBaseWave = waveId XOR waveM = waveN * wg_m.
        module.add(VXorB32(dst=vgpr(readBaseWave), src0=vgpr(waveId), src1=vgpr(waveM),
                           comment="readBaseWave = waveN * wg_m"))
        with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0AddrSetup") as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(strideW), comment=f"strideW={strideW}"))
            module.add(VMulLOU32(dst=vgpr(writeAddr), src0=sgpr(tmpSgpr), src1=vgpr(waveId),
                                 comment="writeAddr = waveId * strideW"))
            module.add(VMulLOU32(dst=vgpr(readAddr), src0=sgpr(tmpSgpr), src1=vgpr(readBaseWave),
                                 comment="readAddr = readBaseWave * strideW"))
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(laneSlotBytes),
                               comment=f"laneSlotBytes={laneSlotBytes}"))
            module.add(VMulLOU32(dst=vgpr(laneLoc), src0=sgpr(tmpSgpr), src1=vgpr(laneLoc),
                                 comment="lane * laneSlotBytes"))
            module.add(VAddU32(vgpr(writeAddr), vgpr(writeAddr), vgpr(laneLoc),
                               comment="writeAddr += lane*laneSlotBytes"))
            module.add(VAddU32(vgpr(readAddr), vgpr(readAddr), vgpr(laneLoc),
                               comment="readAddr += lane*laneSlotBytes"))
        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(laneLoc)
        self.writer.vgprPool.checkIn(readBaseWave)
        self.writer.vgprPool.checkIn(waveM)
        self.writer.vgprPool.checkIn(waveId)

    def _crossWaveReduceFree0(self, partials: int, op=VAddF32, verb="+") -> Module:
        # Step 3 (free0): LDS reduction across wg_m sibling waves sharing waveN.
        strideW = self.waveSize * self.numPartials * 4
        module = Module("PartialRMS crossWaveReduceFree0")
        module.addComment1(
            f"PartialRMS step 3 (free0): cross-wave LDS reduction over wg_m={self.wg_m}"
        )
        module.add(self.writer._syncThreads(
            self.kernel,
            "partialRMS free0 cross-wave: ensure siblings done reading LDS before scratch write"))
        writeAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WriteAddr")
        readAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadAddr")
        readTmp = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_xwF0ReadTmp")
        self._crossWaveComputeAddrs(module, writeAddr, readAddr)
        for i in range(self.numPartials):
            module.add(DSStoreB32(dstAddr=vgpr(writeAddr), src=vgpr(partials + i),
                                  ds=DSModifiers(offset=i * 4), comment=f"LDS store partial[{i}]"))
        module.add(SWaitCnt(dscnt=0, comment="wait LDS writes"))
        module.add(self.writer._syncThreads(self.kernel, "partialRMS free0 cross-wave write"))
        for j in range(self.wg_m):
            for i in range(self.numPartials):
                module.add(DSLoadB32(dst=vgpr(readTmp + i), src=vgpr(readAddr),
                                     ds=DSModifiers(offset=i * 4),
                                     comment=f"LDS load wave[{j}] partial[{i}]"))
            module.add(SWaitCnt(dscnt=0, comment="wait LDS reads"))
            for i in range(self.numPartials):
                if j == 0:
                    module.add(VMovB32(dst=vgpr(partials + i), src=vgpr(readTmp + i),
                                       comment=f"partial[{i}] = wave[0]"))
                else:
                    module.add(op(dst=vgpr(partials + i), src0=vgpr(partials + i),
                                  src1=vgpr(readTmp + i), comment=f"partial[{i}] {verb} wave[{j}]"))
            if j < self.wg_m - 1:
                with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0Advance") as tmpSgprInfo:
                    module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(strideW),
                                       comment=f"strideW={strideW}"))
                    module.add(VAddU32(vgpr(readAddr), vgpr(readAddr), sgpr(tmpSgprInfo.idx),
                                       comment="advance readAddr to next sibling wave"))
        module.add(self.writer._syncThreads(self.kernel, "partialRMS free0 cross-wave done"))
        self.writer.vgprPool.checkIn(readTmp)
        self.writer.vgprPool.checkIn(readAddr)
        self.writer.vgprPool.checkIn(writeAddr)
        return module

    def _buildWriteMask(self, module, laneMaskSgpr: int, laneId: int) -> None:
        # Active iff rowGroup==0 AND waveM==0 (every lane already holds the all-reduced value).
        log2MfmaN = int(math.log2(self.mfma_n))
        rgV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0RowGroup")
        module.add(VLShiftRightB32(dst=vgpr(rgV), shiftHex=hex(log2MfmaN), src=vgpr(laneId),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        if self.wg_m > 1:
            waveMv = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0WaveM")
            self._computeWaveM(module, waveMv)
            module.add(VOrB32(dst=vgpr(rgV), src0=vgpr(rgV), src1=vgpr(waveMv),
                              comment="selV = rowGroup | waveM (zero iff both zero)"))
            self.writer.vgprPool.checkIn(waveMv)
        module.add(VCmpEQU32(dst=sgpr(laneMaskSgpr, self.lane_sgpr_count), src0=0, src1=vgpr(rgV),
                             comment="laneMask: rowGroup==0 && waveM==0"))
        self.writer.vgprPool.checkIn(rgV)

    def _computeNTiles(self, module, dst: int) -> None:
        # n_d = ceil(SizesFree0 / MT0).
        with self.writer.allocTmpSgpr(1, tag="pRMS_wF0NTilesS") as ntilesS:
            module.add(SAddU32(dst=sgpr(ntilesS.idx), src0=sgpr("SizesFree+0"),
                               src1=self.macro_tile0 - 1,
                               comment=f"N_hidden + MT0-1 (MT0={self.macro_tile0})"))
            mt0 = self.macro_tile0
            if mt0 & (mt0 - 1) == 0:
                module.add(SLShiftRightB32(dst=sgpr(ntilesS.idx), shiftHex=hex(mt0.bit_length() - 1),
                                           src=sgpr(ntilesS.idx),
                                           comment=f"n_d = ceil(SizesFree0 / MT0={mt0})"))
            else:
                magic, postShift = _ceilDivMagic(mt0)
                module.add(SMulHIU32(dst=sgpr(ntilesS.idx), src0=sgpr(ntilesS.idx), src1=hex(magic),
                                     comment=f"n_d magic mul (divisor={mt0})"))
                if postShift:
                    module.add(SLShiftRightB32(dst=sgpr(ntilesS.idx), shiftHex=hex(postShift),
                                               src=sgpr(ntilesS.idx),
                                               comment=f"n_d >> {postShift} (magic post-shift)"))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(ntilesS.idx), comment="ntilesV = n_d"))

    def _computeMPadded(self, module, dst: int) -> None:
        mt1 = self.macro_tile1
        with self.writer.allocTmpSgpr(1, tag="pRMS_mPaddedS") as s:
            module.add(SAddU32(dst=sgpr(s.idx), src0=sgpr("SizesFree+1"), src1=mt1 - 1,
                               comment=f"M + MT1-1 (MT1={mt1})"))
            if mt1 & (mt1 - 1) == 0:
                module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(mt1.bit_length() - 1),
                                           src=sgpr(s.idx), comment=f"mTiles = ceil(M/MT1={mt1})"))
            else:
                magic, postShift = _ceilDivMagic(mt1)
                module.add(SMulHIU32(dst=sgpr(s.idx), src0=sgpr(s.idx), src1=hex(magic),
                                     comment=f"mTiles magic mul (divisor={mt1})"))
                if postShift:
                    module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(postShift),
                                               src=sgpr(s.idx), comment=f"mTiles >> {postShift}"))
            module.add(SMulI32(dst=sgpr(s.idx), src0=sgpr(s.idx), src1=mt1,
                               comment=f"M_padded = mTiles * MT1({mt1})"))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(s.idx), comment="M_paddedV = M_padded"))

    def _amaxEpilogueFree0(self, vgprTiles, amaxPartials: int, accTmp: int, partialSrd: int,
                           laneId: int, savedExec: int, laneMaskSgpr: int, globalAddr: int,
                           colByte: int, mPaddedV: int, scaleV: int) -> Module:
        module = Module("PartialRMSQuant amaxEpilogueFree0")
        module.addComment1("PartialRMSQuant: partial amax(|D|)/fp8_max into partialBuf second half")
        module.add(self._absAndLaneMaxFree0(vgprTiles, amaxPartials, accTmp))
        module.add(self._rowGroupReduceFree0(amaxPartials, op=VMaxF32, verb="max"))
        if self.wg_m > 1:
            module.add(self._crossWaveReduceFree0(amaxPartials, op=VMaxF32, verb="max"))
        self._computeMPadded(module, mPaddedV)
        bits = struct.unpack('<I', struct.pack('<f', 1.0 / _FP8_E4M3_MAX))[0]
        module.add(VMovB32(dst=vgpr(scaleV), src=hex(bits),
                           comment=f"1/fp8_max ({_FP8_E4M3_MAX})"))
        for n in range(self.mma_n):
            module.add(VMulF32(dst=vgpr(amaxPartials + n), src0=vgpr(amaxPartials + n),
                               src1=vgpr(scaleV), comment=f"amax[n={n}] /= fp8_max"))
        module.add(self._writePartialsFree0(
            amaxPartials, partialSrd, laneId, savedExec, laneMaskSgpr,
            globalAddr, colByte, rowOffset=mPaddedV, label="amax/fp8_max"))
        return module

    def _writePartialsFree0(self, partials: int, partialSrd: int, laneId: int, savedExec: int,
                            laneMaskSgpr: int, globalAddr: int, colByte: int,
                            rowOffset: int = None, label: str = "Σx²") -> Module:
        module = Module("PartialRMS writePartialsFree0")
        module.addComment1(
            "PartialRMS step 4 (free0): predicated write of Σx² to partialBuf[token, WG0]")
        lsc = self.lane_sgpr_count
        self._buildWriteMask(module, laneMaskSgpr, laneId)
        ntilesV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0NTiles")
        self._computeNTiles(module, ntilesV)
        tokenBase = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0TokenBase")
        module.add(VLShiftRightB32(dst=vgpr(tokenBase), shiftHex=hex(1), src=vgpr(colByte),
                                   comment="tokenBase = colByte >> 1 (token index for n=0)"))
        module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(laneMaskSgpr, lsc),
                                   comment="save exec; set exec = writing-lane mask"))
        nOffVgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0NOff")
        for n in range(self.mma_n):
            nOff = n * self.mfma_n
            self._addImmU32(module, globalAddr, tokenBase, nOff, nOffVgpr,
                            f"token = tokenBase + {nOff} (n={n})")
            if rowOffset is not None:
                module.add(VAddU32(vgpr(globalAddr), vgpr(globalAddr), vgpr(rowOffset),
                                   comment="token += M_padded (amax second half)"))
            module.add(VMulLOU32(dst=vgpr(globalAddr), src0=vgpr(ntilesV), src1=vgpr(globalAddr),
                                 comment="token * n_d"))
            module.add(VAddU32(vgpr(globalAddr), vgpr(globalAddr), sgpr("WorkGroup0"),
                               comment="+ WorkGroup0 (free0 tile index)"))
            module.add(VLShiftLeftB32(dst=vgpr(globalAddr), shiftHex=hex(2), src=vgpr(globalAddr),
                                      comment="byteOff = (token*n_d + WG0) * 4"))
            module.add(BufferStoreB32(src=vgpr(partials + n), vaddr=vgpr(globalAddr),
                                      saddr=sgpr(partialSrd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"partialBuf[token, WG0] = {label} (n={n})"))
        module.add(SWaitCnt(vscnt=0, comment="wait partialBuf stores"))
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc), comment="restore exec mask"))
        self.writer.vgprPool.checkIn(nOffVgpr)
        self.writer.vgprPool.checkIn(tokenBase)
        self.writer.vgprPool.checkIn(ntilesV)
        return module

    def _applyGammaRow(self, module, vgprTiles, gammaSrd: int, gammaTmp: int, accTmp: int,
                       gammaByteVgpr: int, rowGroupOff: int, wgRowBase: int, scratch: int,
                       m: int, k: int) -> None:
        self._free0RowPos(module, gammaByteVgpr, wgRowBase, rowGroupOff, m, k, scratch)
        module.add(VLShiftLeftB32(dst=vgpr(gammaByteVgpr), shiftHex=hex(1), src=vgpr(gammaByteVgpr),
                                  comment="gammaByte = globalRow * 2"))
        module.add(BufferLoadD16B16(vgpr(gammaTmp), vgpr(gammaByteVgpr), sgpr(gammaSrd, 4), 0,
                                    MUBUFModifiers(offen=True),
                                    comment=f"gamma bf16[globalRow] (m={m},k={k})"))
        module.add(SWaitCnt(vlcnt=0, comment="wait gamma load"))
        module.add(VCvtBF16toFP32(vgpr(gammaTmp), vgpr(gammaTmp), None, 0, comment="gamma bf16 -> fp32"))
        for n in range(self.mma_n):
            self._readAccInto(module, accTmp, vgprTiles, m, n, k, f"read acc[m={m},n={n},k={k}]")
            module.add(VMulF32(dst=vgpr(accTmp), src0=vgpr(accTmp), src1=vgpr(gammaTmp),
                               comment="acc *= gamma"))
            self._writeAccFrom(module, accTmp, vgprTiles, m, n, k, f"write acc[m={m},n={n},k={k}]")

    def _applyGammaFree0(self, vgprTiles, gammaSrd: int, gammaTmp: int, accTmp: int,
                         scratchV: int) -> Module:
        module = Module("PartialRMS applyGammaFree0")
        module.addComment1("PartialRMS step 5 (free0): apply gamma[free0 row] in-place")
        rowGroupOff = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0RGOff")
        wgRowBase = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0WgRowBase")
        self._computeRowGroupOff(module, rowGroupOff)
        self._computeFree0RowBase(module, wgRowBase)
        gammaByteVgpr = scratchV
        mBaseVgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_agF0MBase")
        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                self._applyGammaRow(module, vgprTiles, gammaSrd, gammaTmp, accTmp, gammaByteVgpr,
                                    rowGroupOff, wgRowBase, mBaseVgpr, m, k)
        self.writer.vgprPool.checkIn(mBaseVgpr)
        self.writer.vgprPool.checkIn(wgRowBase)
        self.writer.vgprPool.checkIn(rowGroupOff)
        return module
