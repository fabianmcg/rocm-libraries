# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MXFP8Quant fused epilogue emitter for the Subtile kernel (gfx950).

Runs in the pre-store hook (KernelWriter.py), before VGPR rearrangement and
before the standard fp8 D store. It reads raw f32 accumulators, computes
per-block amax, derives an e8m0 scale byte using ceiling-exponent bit math,
multiplies the accumulators in-place by the power-of-two quantMult, and writes
one e8m0 byte to the MXScale side buffer. The subsequent standard fp8 store
converts the already-in-range values to OCP e4m3 for D.

D is the fp8 (OCP e4m3) output; MXScale is the only new side buffer (uint8),
shape [ceil(M/Q0), ceil(N/Q1)] row-major. One byte encodes 2^(byte-127).

e8m0 math (per block, after alpha applied):
  amax = max(|x|)
  if amax == 0: scaleByte = 0, quantMult = 0
  else:
    scaleF   = amax * (1/448)
    expByte  = (bitcast<u32>(scaleF) >> 23) & 0xFF
    ceilAdj  = (mantissa != 0) ? 1 : 0
    scaleByte = clamp(expByte + ceilAdj, 0, 254)
    qExpField = clamp(254 - scaleByte, 1, 254)
    quantMult = bitcast<float>(qExpField << 23)
  D[i] = SaturateCast<e4m3>(x[i] * quantMult)
  MXScale[block] = uint8(scaleByte)
"""

import math
import struct

from rocisa.code import Module
from rocisa.container import (
    EXEC,
    MUBUFModifiers,
    accvgpr,
    sgpr,
    vgpr,
)
from rocisa.instruction import (
    BufferStoreB8,
    DSBPermuteB32,
    SAddU32,
    SAndB64,
    SAndSaveExecB64,
    SLShiftRightB32,
    SMovB32,
    SMovB64,
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddU32,
    VAndB32,
    VCmpEQF32,
    VCmpEQU32,
    VCmpLtU32,
    VCndMaskB32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMaxF32,
    VMed3I32,
    VMovB32,
    VMulF32,
    VMulLOU32,
    VSubU32,
    VXorB32,
)


# OCP FP8 e4m3 maximum representable magnitude.
_fp8E4m3Max = 448.0


class SubtileMXFP8QuantEmitter:
    """Emit the MXFP8Quant epilogue for the Subtile gfx950 kernel.

    Computes per-block amax from f32 AGPRs, derives an e8m0 scale byte,
    scales accumulators in-place by the power-of-two quantMult, and writes
    one byte to the MXScale side buffer.
    """

    def __init__(self, writer, kernel):
        self.writer   = writer
        self.kernel   = kernel

        self.mfmaM        = kernel["MatrixInstM"]
        self.mfmaN        = kernel["MatrixInstN"]
        self.waveSize      = kernel["WavefrontSize"]
        self.rowsPerLane = (self.mfmaM * self.mfmaN) // self.waveSize

        wg = kernel["MIWaveGroup"]
        self.wgM = wg[0]
        self.wgN = wg[1]

        self.mmaM       = (kernel["MacroTile0"] // self.mfmaM) // self.wgM
        self.mmaN       = (kernel["MacroTile1"] // self.mfmaN) // self.wgN
        self.macroTile0 = kernel["MacroTile0"]
        self.macroTile1 = kernel["MacroTile1"]

        # Resolved quant tile dimensions (from _validateMXFP8Quant).
        self.q0 = kernel["_MXFP8QuantQ0"]
        self.q1 = kernel["_MXFP8QuantQ1"]

        # laneSGPRCount: 1 for wave32, 2 for wave64.
        self.laneSgprCount = writer.states.laneSGPRCount

        # Per-wave quant tile counts (compile-time; Phase-1 bounds these small).
        waveSpanM     = self.mmaM * self.mfmaM
        waveSpanN     = self.mmaN * self.mfmaN
        self.nQTilesM = waveSpanM // self.q0
        self.nQTilesN = waveSpanN // self.q1
        self.subRowQuant = self.q0 < self.mfmaM
        if self.subRowQuant:
            self.kBlocksPerLane = self.rowsPerLane // self.q0
            self.tilesPerMfmaM  = self.mfmaM // self.q0
            self.nLocalQ0       = self.mmaM * self.kBlocksPerLane
            self.tileArrayLen   = self.nLocalQ0 * self.nQTilesN
        else:
            self.tileArrayLen   = self.nQTilesM * self.nQTilesN

    # ------------------------------------------------------------------ #
    # Helpers reused verbatim from SubtileTileQuantEmit.                  #
    # ------------------------------------------------------------------ #

    def _readAccInto(self, module, dst: int, vgprTiles, m: int, n: int, k: int,
                     comment: str) -> None:
        """Copy accumulator element (m, n, k) into VGPR dst."""
        tile = vgprTiles[n * self.mmaM + m]
        reg  = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            module.add(VMovB32(dst=vgpr(dst), src=vgpr(reg), comment=comment))
            return
        module.add(VAccvgprReadB32(vgpr(dst), accvgpr(reg), comment=comment))
        module.add(SNop(waitState=1, comment="s_nop after v_accvgpr_read before VALU (gfx950)."))

    def _writeAccFrom(self, module, src: int, vgprTiles, m: int, n: int, k: int,
                      comment: str) -> None:
        """Write VGPR src back into accumulator element (m, n, k)."""
        tile = vgprTiles[n * self.mmaM + m]
        reg  = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            module.add(VMovB32(dst=vgpr(reg), src=vgpr(src), comment=comment))
            return
        module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(src), comment=comment))

    def _buildBufferSrd(self, module, srd: int, ptrName: str, name: str) -> None:
        """Load a 128-bit buffer descriptor for ptrName into SGPRs [srd, srd+3]."""
        module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr(ptrName, 2), comment=f"{name} SRD base."))
        module.add(SMovB32(dst=sgpr(srd + 2), src="BufferOOB", comment=f"{name} SRD limit."))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96", comment=f"{name} SRD flags."))

    # ------------------------------------------------------------------ #
    # Stage helpers — each ≤40 lines.                                     #
    # ------------------------------------------------------------------ #

    def _computeTotalQTilesN(self, module, dst: int) -> None:
        """Compute ceil(N / Q1) into VGPR dst at runtime from SizesFree+1."""
        with self.writer.allocTmpSgpr(1, tag="mx_nQTNsS") as s:
            module.add(SAddU32(dst=sgpr(s.idx), src0=sgpr("SizesFree+1"),
                               src1=self.q1 - 1,
                               comment=f"N + Q1-1 (Q1={self.q1})."))
            log2q1 = int(math.log2(self.q1))
            module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(log2q1),
                                       src=sgpr(s.idx),
                                       comment=f"totalQTilesN = ceil(N/Q1={self.q1})."))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(s.idx),
                               comment="totalQTilesN into VGPR."))

    def _computeTotalQTilesM(self, module, dst: int) -> None:
        """Compute ceil(nHidden / Q0) into VGPR dst (MXScale row count for sub-row mode)."""
        with self.writer.allocTmpSgpr(1, tag="mx_nQTMsS") as s:
            module.add(SAddU32(dst=sgpr(s.idx), src0=sgpr("SizesFree+0"),
                               src1=self.q0 - 1,
                               comment=f"nHidden + Q0-1 (Q0={self.q0})."))
            log2q0 = int(math.log2(self.q0)) if self.q0 > 1 else 0
            module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(log2q0),
                                       src=sgpr(s.idx),
                                       comment=f"totalQTilesM = ceil(nHidden/Q0={self.q0})."))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(s.idx),
                               comment="totalQTilesM into VGPR."))

    def _setup(self, mxSrd: int, laneId: int, col: int, rowGroup: int) -> Module:
        """Build MXScale SRD; compute laneId, col, rowGroup."""
        module = Module("MXFP8Quant setup")
        module.add(SWaitCnt(kmcnt=0, comment="wait for MXFP8Quant kernarg s_load."))
        self._buildBufferSrd(module, mxSrd, "MXScale", "mxScale")
        module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)."))
        log2N = int(math.log2(self.mfmaN))
        module.add(VAndB32(dst=vgpr(col), src0=vgpr(laneId), src1=self.mfmaN - 1,
                           comment=f"col = laneId & ({self.mfmaN}-1)."))
        module.add(VLShiftRightB32(dst=vgpr(rowGroup), shiftHex=hex(log2N), src=vgpr(laneId),
                                   comment=f"rowGroup = laneId >> {log2N}."))
        return module

    def _accAbsIntoSlot(self, module, accTmp: int, vgprTiles, amaxVgprs: int,
                        m: int, n: int, k: int, slot: int, absMask: int) -> None:
        """Read |acc[m,n,k]| and fold into amax slot."""
        self._readAccInto(module, accTmp, vgprTiles, m, n, k,
                           f"read acc[m={m},n={n},k={k}].")
        module.add(VAndB32(dst=vgpr(accTmp), src0=vgpr(accTmp), src1=vgpr(absMask),
                           comment="|acc|."))
        module.add(VMaxF32(dst=vgpr(amaxVgprs + slot),
                           src0=vgpr(amaxVgprs + slot),
                           src1=vgpr(accTmp),
                           comment=f"amax[tile={slot}] = max(amax, |acc|)."))

    def _accAbsRuntimeQi(self, module, accTmp: int, vgprTiles, amaxVgprs: int,
                          m: int, n: int, k: int, qj: int, absMask: int,
                          rowGroup: int) -> None:
        """Read |acc[m,n,k]|; distribute into correct amax slot via runtime qi."""
        mBase = m * self.mfmaM + k
        self._readAccInto(module, accTmp, vgprTiles, m, n, k,
                           f"read acc[m={m},n={n},k={k}].")
        module.add(VAndB32(dst=vgpr(accTmp), src0=vgpr(accTmp), src1=vgpr(absMask),
                           comment="|acc|."))
        qiV = self.writer.vgprPool.checkOut(1, tag="mx_qiV")
        module.add(VMulLOU32(dst=vgpr(qiV), src0=self.rowsPerLane,
                             src1=vgpr(rowGroup), comment=f"rowGroup * {self.rowsPerLane}."))
        if 0 < mBase <= 64:
            module.add(VAddU32(vgpr(qiV), vgpr(qiV), mBase, comment=f"+ mBase={mBase}."))
        elif mBase > 64:
            tmpM = self.writer.vgprPool.checkOut(1, tag="mx_mBaseV")
            module.add(VMovB32(dst=vgpr(tmpM), src=mBase, comment=f"mBase={mBase}."))
            module.add(VAddU32(vgpr(qiV), vgpr(qiV), vgpr(tmpM), comment="+ mBase."))
            self.writer.vgprPool.checkIn(tmpM)
        log2q0 = int(math.log2(self.q0))
        module.add(VLShiftRightB32(dst=vgpr(qiV), shiftHex=hex(log2q0),
                                   src=vgpr(qiV), comment=f"qi = ... >> {log2q0}."))
        masked = self.writer.vgprPool.checkOut(1, tag="mx_masked")
        cond   = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="mx_qiCond", preventOverflow=False)
        for qiVal in range(self.nQTilesM):
            slot = qiVal * self.nQTilesN + qj
            module.add(VCmpEQU32(dst=sgpr(cond, self.laneSgprCount),
                                  src0=qiVal, src1=vgpr(qiV), comment=f"qi=={qiVal}?."))
            module.add(VCndMaskB32(dst=vgpr(masked), src0=0, src1=vgpr(accTmp),
                                   src2=sgpr(cond, self.laneSgprCount),
                                   comment=f"masked if qi=={qiVal}."))
            module.add(VMaxF32(dst=vgpr(amaxVgprs + slot),
                               src0=vgpr(amaxVgprs + slot),
                               src1=vgpr(masked),
                               comment=f"amax[tile={slot}] = max(amax, masked)."))
        self.writer.sgprPool.checkIn(cond)
        self.writer.vgprPool.checkIn(masked)
        self.writer.vgprPool.checkIn(qiV)

    def _applyAlphaInPlace(self, vgprTiles) -> Module:
        """Scale accumulators by alpha before amax so MXScale reflects alpha*A*B."""
        module = Module("MXFP8Quant applyAlphaInPlace")
        module.addComment1("MXFP8Quant: scale accumulators by alpha before amax.")
        tmp = self.writer.vgprPool.checkOut(1, tag="mx_alphaAcc")
        for n in range(self.mmaN):
            for m in range(self.mmaM):
                for k in range(self.rowsPerLane):
                    self._readAccInto(module, tmp, vgprTiles, m, n, k,
                                      f"read acc[m={m},n={n},k={k}].")
                    module.add(VMulF32(dst=vgpr(tmp), src0=vgpr(tmp), src1=sgpr("Alpha"),
                                      comment="scale acc by alpha."))
                    self._writeAccFrom(module, tmp, vgprTiles, m, n, k,
                                       f"write acc[m={m},n={n},k={k}].")
        self.writer.vgprPool.checkIn(tmp)
        return module

    def _laneTileAmax(self, vgprTiles, amaxVgprs: int, accTmp: int,
                      rowGroup: int) -> Module:
        """Compute per-lane per-quant-tile amax(|acc|)."""
        module = Module("MXFP8Quant laneTileAmax")
        module.addComment1("MXFP8Quant: per-lane per-quant-tile amax(|acc|).")
        totalTiles = self.tileArrayLen
        absMask = self.writer.vgprPool.checkOut(1, tag="mx_absMask")
        module.add(VMovB32(dst=vgpr(absMask), src=hex(0x7FFFFFFF), comment="abs mask."))
        for t in range(totalTiles):
            module.add(VMovB32(dst=vgpr(amaxVgprs + t), src=0, comment=f"amax[tile={t}] = 0."))
        for n in range(self.mmaN):
            qj = (n * self.mfmaN) // self.q1
            for m in range(self.mmaM):
                for k in range(self.rowsPerLane):
                    if self.subRowQuant:
                        slot = (m * self.kBlocksPerLane + k // self.q0) * self.nQTilesN + qj
                        self._accAbsIntoSlot(module, accTmp, vgprTiles, amaxVgprs,
                                             m, n, k, slot, absMask)
                    elif self.nQTilesM == 1:
                        self._accAbsIntoSlot(module, accTmp, vgprTiles, amaxVgprs,
                                              m, n, k, qj, absMask)
                    else:
                        self._accAbsRuntimeQi(module, accTmp, vgprTiles, amaxVgprs,
                                               m, n, k, qj, absMask, rowGroup)
        self.writer.vgprPool.checkIn(absMask)
        return module

    def _butterflyRound(self, module, addrV: int, tmpV: int,
                         amaxVgprs: int, totalTiles: int, laneId: int,
                         xorVal: int) -> None:
        """Emit one XOR-butterfly round: fetch partner amax and fold via VMaxF32."""
        module.add(VXorB32(dst=vgpr(addrV), src0=vgpr(laneId), src1=xorVal,
                           comment=f"partnerLane = laneId ^ {xorVal}."))
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(2), src=vgpr(addrV),
                                  comment="byteAddr = partnerLane * 4."))
        for t in range(totalTiles):
            module.add(DSBPermuteB32(vgpr(tmpV + t), vgpr(addrV), vgpr(amaxVgprs + t),
                                      comment=f"fetch partner amax[tile={t}]."))
        module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute."))
        for t in range(totalTiles):
            module.add(VMaxF32(dst=vgpr(amaxVgprs + t), src0=vgpr(amaxVgprs + t),
                               src1=vgpr(tmpV + t),
                               comment=f"amax[tile={t}] = max(amax, partner)."))

    def _butterflyReduce(self, amaxVgprs: int, laneId: int) -> Module:
        """Reduce per-tile amax across lanes sharing the same quant tile."""
        totalTiles = self.tileArrayLen
        module     = Module("MXFP8Quant butterflyReduce")
        module.addComment1(
            f"MXFP8Quant: butterfly reduce amax (q0={self.q0}, q1={self.q1}).")
        nColRounds = int(math.log2(self.mfmaN))
        rgSpan  = min(self.q0 // self.rowsPerLane, self.waveSize // self.mfmaN)
        log2rg  = int(math.log2(rgSpan)) if rgSpan > 1 else 0
        numRounds  = nColRounds + log2rg
        if numRounds == 0:
            return module
        addrV = self.writer.vgprPool.checkOut(1, tag="mx_bfAddr")
        tmpV  = self.writer.vgprPool.checkOut(totalTiles, tag="mx_bfTmp")
        for i in range(numRounds):
            xorVal = (1 << i) if i < nColRounds else self.mfmaN << (i - nColRounds)
            self._butterflyRound(module, addrV, tmpV, amaxVgprs, totalTiles, laneId, xorVal)
        self.writer.vgprPool.checkIn(tmpV)
        self.writer.vgprPool.checkIn(addrV)
        return module

    def _computeOneMXScale(self, module, slot: int, amaxVgpr: int,
                            quantMultVgpr: int, invFp8V: int,
                            c254V: int, zeroMask: int) -> None:
        """Emit quantMult for one quant tile slot.

        quantMultVgpr is used as a temp for intermediate scaleByte computation
        and then overwritten with the final quantMult = qExpField << 23.
        When amax==0: quantMultVgpr = 0 (scaleByte is naturally 0 in this case).
        c254V and zeroMask are shared across all slots (passed in).
        """
        lsc = self.laneSgprCount
        # scaleF = amax * (1/448) → into quantMultVgpr (temp for scaleByte).
        module.add(VMulF32(dst=vgpr(quantMultVgpr), src0=vgpr(amaxVgpr),
                           src1=vgpr(invFp8V),
                           comment=f"scaleF[{slot}] = amax * (1/448)."))
        # Mantissa check for ceiling via left-shift (avoids literal 0x7FFFFF).
        # mantV = scaleF << 9 discards sign and exponent; mantV!=0 iff mantissa!=0.
        mantV = self.writer.vgprPool.checkOut(1, tag="mx_mant")
        module.add(VLShiftLeftB32(dst=vgpr(mantV), shiftHex=hex(9),
                                  src=vgpr(quantMultVgpr),
                                  comment="mantV = scaleF << 9 (mant != 0 iff mantV != 0)."))
        # expByte = scaleF >> 23; & 0xFF not needed since scaleF >= 0 (sign bit = 0).
        module.add(VLShiftRightB32(dst=vgpr(quantMultVgpr), shiftHex=hex(23),
                                   src=vgpr(quantMultVgpr),
                                   comment="expByte = scaleF >> 23."))
        zmc = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_zmc", preventOverflow=False)
        module.add(VCmpEQU32(dst=sgpr(zmc, lsc), src0=0, src1=vgpr(mantV),
                             comment="mant == 0?."))
        adjV = self.writer.vgprPool.checkOut(1, tag="mx_adj")
        # When zmc TRUE (mant==0): dst=src1=0; FALSE (mant!=0): dst=src0=1.
        module.add(VCndMaskB32(dst=vgpr(adjV), src0=1, src1=0, src2=sgpr(zmc, lsc),
                               comment="adj = (mant!=0) ? 1 : 0."))
        self.writer.sgprPool.checkIn(zmc)
        self.writer.vgprPool.checkIn(mantV)
        # scaleByte = expByte + adj → into quantMultVgpr.
        module.add(VAddU32(vgpr(quantMultVgpr), vgpr(quantMultVgpr), vgpr(adjV),
                           comment=f"scaleByte[{slot}] = expByte + ceilAdj."))
        self.writer.vgprPool.checkIn(adjV)
        # clamp(scaleByte, 0, 254); VMed3I32 requires src2 Container.
        module.add(VMed3I32(dst=vgpr(quantMultVgpr), src0=0,
                            src1=vgpr(quantMultVgpr), src2=vgpr(c254V),
                            comment=f"scaleByte[{slot}] = clamp(scaleByte, 0, 254)."))
        # qExpField = 254 - scaleByte → into quantMultVgpr.
        module.add(VSubU32(vgpr(quantMultVgpr), vgpr(c254V), vgpr(quantMultVgpr),
                           comment=f"qExpField[{slot}] = 254 - scaleByte."))
        # clamp(qExpField, 1, 254).
        module.add(VMed3I32(dst=vgpr(quantMultVgpr), src0=1,
                            src1=vgpr(quantMultVgpr), src2=vgpr(c254V),
                            comment=f"qExpField[{slot}] = clamp(qExpField, 1, 254)."))
        # quantMult = bitcast<float>(qExpField << 23).
        module.add(VLShiftLeftB32(dst=vgpr(quantMultVgpr), shiftHex=hex(23),
                                  src=vgpr(quantMultVgpr),
                                  comment=f"quantMult[{slot}] = qExpField << 23."))
        # amax==0 override: quantMult = 0 when amax==0 (scaleByte is naturally 0).
        module.add(VCmpEQF32(dst=sgpr(zeroMask, lsc), src0=0, src1=vgpr(amaxVgpr),
                             comment=f"amax[{slot}] == 0?."))
        module.add(VCndMaskB32(dst=vgpr(quantMultVgpr), src0=vgpr(quantMultVgpr),
                               src1=0, src2=sgpr(zeroMask, lsc),
                               comment=f"quantMult[{slot}] = 0 if amax==0."))

    def _computeMXScale(self, amaxVgprs: int, quantMultVgprs: int) -> Module:
        """Compute quantMult for every tile from amaxVgprs.

        quantMultVgprs[t] = 0 when amaxVgprs[t] == 0, else
        quantMultVgprs[t] = bitcast<float>(clamp(254 - scaleByte, 1, 254) << 23).
        scaleByte can be recovered later as (quantMultVgprs[t] >> 23); 0 when == 0.
        c254V and zeroMask are shared across all slot computations to reduce overhead.
        """
        module = Module("MXFP8Quant computeMXScale")
        module.addComment1("MXFP8Quant: compute e8m0 quantMult per tile.")
        invFp8Bits = struct.unpack('<I', struct.pack('<f', 1.0 / _fp8E4m3Max))[0]
        totalTiles = self.tileArrayLen
        invFp8V = self.writer.vgprPool.checkOut(1, tag="mx_invFp8")
        module.add(VMovB32(dst=vgpr(invFp8V), src=hex(invFp8Bits),
                           comment=f"1/fp8_max = 1/{_fp8E4m3Max}."))
        c254V = self.writer.vgprPool.checkOut(1, tag="mx_c254")
        module.add(VMovB32(dst=vgpr(c254V), src=254, comment="constant 254."))
        zeroMask = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="mx_zeroMask",
            preventOverflow=False)
        for t in range(totalTiles):
            self._computeOneMXScale(module, t, amaxVgprs + t,
                                    quantMultVgprs + t, invFp8V, c254V, zeroMask)
        self.writer.sgprPool.checkIn(zeroMask)
        self.writer.vgprPool.checkIn(c254V)
        self.writer.vgprPool.checkIn(invFp8V)
        return module

    def _applyScaleInPlace(self, vgprTiles, quantMultVgprs: int,
                            accTmp: int, rowGroup: int) -> Module:
        """Multiply every accumulator element in-place by the tile's quantMult."""
        module = Module("MXFP8Quant applyScaleInPlace")
        module.addComment1("MXFP8Quant: multiply accumulators in place by quantMult.")
        for n in range(self.mmaN):
            qj = (n * self.mfmaN) // self.q1
            for m in range(self.mmaM):
                for k in range(self.rowsPerLane):
                    mBase = m * self.mfmaM + k
                    if self.subRowQuant:
                        slot = (m * self.kBlocksPerLane + k // self.q0) * self.nQTilesN + qj
                        self._readAccInto(module, accTmp, vgprTiles, m, n, k,
                                          f"read acc[m={m},n={n},k={k}].")
                        module.add(VMulF32(dst=vgpr(accTmp), src0=vgpr(accTmp),
                                           src1=vgpr(quantMultVgprs + slot),
                                           comment=f"acc *= quantMult[tile={slot}]."))
                        self._writeAccFrom(module, accTmp, vgprTiles, m, n, k,
                                           f"write acc[m={m},n={n},k={k}].")
                    elif self.nQTilesM == 1:
                        slot = qj
                        self._readAccInto(module, accTmp, vgprTiles, m, n, k,
                                           f"read acc[m={m},n={n},k={k}].")
                        module.add(VMulF32(dst=vgpr(accTmp), src0=vgpr(accTmp),
                                           src1=vgpr(quantMultVgprs + slot),
                                           comment=f"acc *= quantMult[tile={slot}]."))
                        self._writeAccFrom(module, accTmp, vgprTiles, m, n, k,
                                            f"write acc[m={m},n={n},k={k}].")
                    else:
                        self._applyRuntimeQiScale(module, vgprTiles, quantMultVgprs, accTmp,
                                                    m, n, k, qj, mBase, rowGroup)
        return module

    def _applyRuntimeQiScale(self, module, vgprTiles, quantMultVgprs: int, accTmp: int,
                              m: int, n: int, k: int, qj: int, mBase: int,
                              rowGroup: int) -> None:
        """Select the right quantMult at runtime via qi ladder; apply to one acc element."""
        qiRTV = self.writer.vgprPool.checkOut(1, tag="mx_qiRTV")
        module.add(VMulLOU32(dst=vgpr(qiRTV), src0=self.rowsPerLane,
                             src1=vgpr(rowGroup), comment=f"rowGroup * {self.rowsPerLane}."))
        if 0 < mBase <= 64:
            module.add(VAddU32(vgpr(qiRTV), vgpr(qiRTV), mBase, comment=f"+ mBase={mBase}."))
        elif mBase > 64:
            tmpM = self.writer.vgprPool.checkOut(1, tag="mx_mBaseVA")
            module.add(VMovB32(dst=vgpr(tmpM), src=mBase, comment=f"mBase={mBase}."))
            module.add(VAddU32(vgpr(qiRTV), vgpr(qiRTV), vgpr(tmpM), comment="+ mBase."))
            self.writer.vgprPool.checkIn(tmpM)
        log2q0 = int(math.log2(self.q0))
        module.add(VLShiftRightB32(dst=vgpr(qiRTV), shiftHex=hex(log2q0),
                                   src=vgpr(qiRTV), comment=f"qi = ... >> {log2q0}."))
        multSel = self.writer.vgprPool.checkOut(1, tag="mx_multSel")
        module.add(VMovB32(dst=vgpr(multSel), src=0, comment="init multSel=0."))
        condV = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="mx_condVA", preventOverflow=False)
        for qiVal in range(self.nQTilesM):
            slot = qiVal * self.nQTilesN + qj
            module.add(VCmpEQU32(dst=sgpr(condV, self.laneSgprCount),
                                  src0=qiVal, src1=vgpr(qiRTV), comment=f"qi=={qiVal}?."))
            module.add(VCndMaskB32(dst=vgpr(multSel), src0=vgpr(multSel),
                                   src1=vgpr(quantMultVgprs + slot),
                                   src2=sgpr(condV, self.laneSgprCount),
                                   comment=f"multSel = quantMult[{slot}] if qi=={qiVal}."))
        self.writer.sgprPool.checkIn(condV)
        self._readAccInto(module, accTmp, vgprTiles, m, n, k,
                           f"read acc[m={m},n={n},k={k}].")
        module.add(VMulF32(dst=vgpr(accTmp), src0=vgpr(accTmp),
                           src1=vgpr(multSel), comment="acc *= quantMult."))
        self._writeAccFrom(module, accTmp, vgprTiles, m, n, k,
                            f"write acc[m={m},n={n},k={k}].")
        self.writer.vgprPool.checkIn(multSel)
        self.writer.vgprPool.checkIn(qiRTV)

    def _computeWaveIndices(self, module) -> tuple:
        """Compute waveM = waveIdx % wg_m and waveN = (waveIdx // wg_m) % wg_n."""
        if self.wgM <= 1 and self.wgN <= 1:
            return None, None
        log2Wave = int(math.log2(self.waveSize))
        waveIdx = self.writer.vgprPool.checkOut(1, tag="mx_waveIdx")
        module.add(VLShiftRightB32(dst=vgpr(waveIdx), shiftHex=hex(log2Wave),
                                   src=vgpr("Serial"),
                                   comment=f"waveIdx = Serial >> {log2Wave}."))
        waveM = None
        if self.wgM > 1:
            waveM = self.writer.vgprPool.checkOut(1, tag="mx_waveM")
            module.add(VAndB32(dst=vgpr(waveM), src0=vgpr(waveIdx), src1=self.wgM - 1,
                               comment=f"waveM = waveIdx & ({self.wgM}-1)."))
        waveN = None
        if self.wgN > 1:
            log2WgM = int(math.log2(self.wgM))
            waveN = self.writer.vgprPool.checkOut(1, tag="mx_waveN")
            module.add(VLShiftRightB32(dst=vgpr(waveN), shiftHex=hex(log2WgM),
                                       src=vgpr(waveIdx),
                                       comment=f"waveIdx >> {log2WgM}."))
            module.add(VAndB32(dst=vgpr(waveN), src0=vgpr(waveN), src1=self.wgN - 1,
                               comment=f"waveN = (waveIdx >> {log2WgM}) & ({self.wgN}-1)."))
        self.writer.vgprPool.checkIn(waveIdx)
        return waveM, waveN

    def _tileByteOffset(self, module, addrV: int, tmpV: int, tmp2V: int, nQTN: int,
                         qi: int, qj: int, waveM, waveN) -> None:
        """Compute byteOff = linear_tile_idx (1 byte per tile; no shift) into addrV."""
        nQTilesMPerWG = self.nQTilesM * self.wgM
        nQTilesNPerWG = self.nQTilesN * self.wgN
        module.add(VMulLOU32(dst=vgpr(addrV), src0=sgpr("WorkGroup0"), src1=nQTilesMPerWG,
                             comment="WG0 * nQTilesMPerWG."))
        if waveM is not None:
            module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(waveM), src1=self.nQTilesM,
                                 comment="waveM * nQTilesM."))
            module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(tmp2V),
                               comment="+ waveM * nQTilesM."))
        if qi:
            module.add(VAddU32(vgpr(addrV), vgpr(addrV), qi, comment=f"qTileRow += {qi}."))
        module.add(VMulLOU32(dst=vgpr(addrV), src0=vgpr(addrV), src1=vgpr(nQTN),
                             comment="qTileRow * totalQTilesN."))
        module.add(VMulLOU32(dst=vgpr(tmpV), src0=sgpr("WorkGroup1"), src1=nQTilesNPerWG,
                             comment="WG1 * nQTilesNPerWG."))
        if waveN is not None:
            module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(waveN), src1=self.nQTilesN,
                                 comment="waveN * nQTilesN."))
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), vgpr(tmp2V),
                               comment="+ waveN * nQTilesN."))
        if qj:
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), qj, comment=f"qTileCol += {qj}."))
        module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(tmpV),
                           comment="byteOff = linear_tile_idx (1B per tile; no <<2 shift)."))

    def _mulVgprBySgprConst(self, module, dst_vgpr: int, sgpr_name: str,
                             const: int, comment: str) -> None:
        """Emit dst_vgpr = sgpr(sgpr_name) * const, loading const into a tmp VGPR."""
        tmp = self.writer.vgprPool.checkOut(1, tag="mx_mulConst")
        module.add(VMovB32(dst=vgpr(tmp), src=const, comment=f"const={const} into vgpr."))
        module.add(VMulLOU32(dst=vgpr(dst_vgpr), src0=vgpr(tmp),
                             src1=sgpr(sgpr_name), comment=comment))
        self.writer.vgprPool.checkIn(tmp)

    def _tileByteOffsetSubRow(self, module, addrV: int, tmpV: int, tmp2V: int,
                               nQTN: int, m: int, kBlock: int, qj: int,
                               rowGroup: int, waveM, waveN) -> None:
        """Compute byteOff for sub-row MXScale tile (m, kBlock, qj) into addrV."""
        nQTilesMPerWG = self.nQTilesM * self.wgM
        nQTilesNPerWG = self.nQTilesN * self.wgN
        constRow = m * self.tilesPerMfmaM + kBlock
        self._mulVgprBySgprConst(module, addrV, "WorkGroup0", nQTilesMPerWG,
                                  "WG0 * nQTilesMPerWG.")
        if waveM is not None:
            module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(waveM), src1=self.nQTilesM,
                                 comment="waveM * nQTilesM."))
            module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(tmp2V),
                               comment="+ waveM * nQTilesM."))
        module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(rowGroup),
                             src1=self.kBlocksPerLane,
                             comment=f"rowGroup * kBlocksPerLane={self.kBlocksPerLane}."))
        module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(tmp2V),
                           comment="+ rowGroup * kBlocksPerLane."))
        if constRow:
            module.add(VAddU32(vgpr(addrV), vgpr(addrV), constRow,
                               comment=f"qTileRow += constRow={constRow}."))
        module.add(VMulLOU32(dst=vgpr(addrV), src0=vgpr(addrV), src1=vgpr(nQTN),
                             comment="qTileRow * totalQTilesN."))
        self._mulVgprBySgprConst(module, tmpV, "WorkGroup1", nQTilesNPerWG,
                                  "WG1 * nQTilesNPerWG.")
        if waveN is not None:
            module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(waveN), src1=self.nQTilesN,
                                 comment="waveN * nQTilesN."))
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), vgpr(tmp2V),
                               comment="+ waveN * nQTilesN."))
        if qj:
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), qj, comment=f"qTileCol += {qj}."))
        module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(tmpV),
                           comment="byteOff = linear_tile_idx (1B per tile; no <<2 shift)."))

    def _computeScaleByteInline(self, module, slot: int, quantMultVgprs: int,
                                 scaleByteV: int) -> None:
        """Recover e8m0 scaleByte from stored quantMult into scaleByteV (1 VGPR).

        quantMult = qExpField << 23, so qExpField = quantMult >> 23.
        scaleByte = 254 - qExpField, except when quantMult==0 (amax==0) → scaleByte=0.
        """
        lsc = self.laneSgprCount
        module.add(VLShiftRightB32(dst=vgpr(scaleByteV), shiftHex=hex(23),
                                   src=vgpr(quantMultVgprs + slot),
                                   comment=f"qExpField[{slot}] = quantMult >> 23."))
        c254V = self.writer.vgprPool.checkOut(1, tag="mx_sb254")
        module.add(VMovB32(dst=vgpr(c254V), src=254, comment="constant 254."))
        module.add(VSubU32(vgpr(scaleByteV), vgpr(c254V), vgpr(scaleByteV),
                           comment=f"scaleByte[{slot}] = 254 - qExpField."))
        self.writer.vgprPool.checkIn(c254V)
        # When quantMult==0 (amax==0): 254-0=254 is wrong, force to 0.
        zeroMask = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_sbZero",
                                                         preventOverflow=False)
        module.add(VCmpEQU32(dst=sgpr(zeroMask, lsc), src0=0,
                             src1=vgpr(quantMultVgprs + slot),
                             comment=f"quantMult[{slot}] == 0?"))
        module.add(VCndMaskB32(dst=vgpr(scaleByteV), src0=vgpr(scaleByteV), src1=0,
                               src2=sgpr(zeroMask, lsc),
                               comment=f"scaleByte[{slot}] = 0 if amax==0."))
        self.writer.sgprPool.checkIn(zeroMask)

    def _writeTileStore(self, module, qi: int, qj: int, mxSrd: int,
                         quantMultVgprs: int, col: int, rowGroup: int,
                         savedExec: int, laneMask: int,
                         addrV: int, tmpV: int, tmp2V: int, nQTN: int,
                         waveM, waveN) -> None:
        """Predicated write of one MXScale byte for quant tile (qi, qj)."""
        lsc     = self.laneSgprCount
        slot    = qi * self.nQTilesN + qj
        repCol = (qj * self.q1) % self.mfmaN
        repRg  = ((qi * self.q0) % self.mfmaM) // self.rowsPerLane
        module.addComment0(f"  Tile qi={qi}, qj={qj}: repCol={repCol}, repRg={repRg}.")
        colCond = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_wColCond",
                                                        preventOverflow=False)
        rgCond  = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_wRgCond",
                                                        preventOverflow=False)
        module.add(VCmpEQU32(dst=sgpr(colCond, lsc), src0=repCol, src1=vgpr(col),
                             comment=f"col == {repCol}?."))
        module.add(VCmpEQU32(dst=sgpr(rgCond, lsc), src0=repRg, src1=vgpr(rowGroup),
                             comment=f"rowGroup == {repRg}?."))
        module.add(SAndB64(dst=sgpr(laneMask, lsc), src0=sgpr(colCond, lsc),
                           src1=sgpr(rgCond, lsc),
                           comment="writeMask = col==repCol AND rowGroup==repRg."))
        self.writer.sgprPool.checkIn(rgCond)
        self.writer.sgprPool.checkIn(colCond)
        nQTilesNPerWG = self.nQTilesN * self.wgN
        module.add(VMulLOU32(dst=vgpr(tmpV), src0=sgpr("WorkGroup1"), src1=nQTilesNPerWG,
                             comment="WG1 * nQTilesNPerWG."))
        if waveN is not None:
            module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(waveN), src1=self.nQTilesN,
                                 comment="waveN * nQTilesN."))
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), vgpr(tmp2V),
                               comment="+ waveN * nQTilesN."))
        if qj:
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), qj, comment=f"qTileCol += {qj}."))
        colInRange = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_wColInRange",
                                                          preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(colInRange, lsc), src0=vgpr(tmpV), src1=vgpr(nQTN),
                             comment="qTileCol < totalQTilesN?."))
        module.add(SAndB64(dst=sgpr(laneMask, lsc), src0=sgpr(laneMask, lsc),
                           src1=sgpr(colInRange, lsc),
                           comment="AND qTileCol in range."))
        self.writer.sgprPool.checkIn(colInRange)
        module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(laneMask, lsc),
                                   comment="save exec; set exec = write-lane mask."))
        self._tileByteOffset(module, addrV, tmpV, tmp2V, nQTN, qi, qj, waveM, waveN)
        scaleByteV = self.writer.vgprPool.checkOut(1, tag="mx_sbTmp")
        self._computeScaleByteInline(module, slot, quantMultVgprs, scaleByteV)
        module.add(BufferStoreB8(
            src=vgpr(scaleByteV), vaddr=vgpr(addrV),
            saddr=sgpr(mxSrd, 4), soffset=0,
            mubuf=MUBUFModifiers(offen=True),
            comment=f"MXScale[qTileRow, qTileCol] byte (qi={qi}, qj={qj})."))
        self.writer.vgprPool.checkIn(scaleByteV)
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                           comment="restore exec mask."))

    def _writeTileStoreSubRow(self, module, m: int, kBlock: int, qj: int, slot: int,
                               mxSrd: int, quantMultVgprs: int,
                               col: int, rowGroup: int, savedExec: int, laneMask: int,
                               addrV: int, tmpV: int, tmp2V: int,
                               nQTN: int, nQTM: int, waveM, waveN) -> None:
        """Predicated write of sub-row MXScale byte for tile (m, kBlock, qj).

        Recovers scaleByte inline from quantMultVgprs[slot].
        Predicate: col==0 AND qTileCol < totalQTilesN AND qTileRow < totalQTilesM.
        """
        lsc = self.laneSgprCount
        nQTilesMPerWG = self.nQTilesM * self.wgM
        nQTilesNPerWG = self.nQTilesN * self.wgN
        constRow = m * self.tilesPerMfmaM + kBlock
        module.addComment0(f"  SubRow tile m={m}, kBlock={kBlock}, qj={qj}, slot={slot}.")
        module.add(VCmpEQU32(dst=sgpr(laneMask, lsc), src0=0, src1=vgpr(col),
                             comment="col == 0 (representative free1 lane)."))
        self._mulVgprBySgprConst(module, tmpV, "WorkGroup1", nQTilesNPerWG,
                                  "WG1 * nQTilesNPerWG.")
        if waveN is not None:
            module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(waveN), src1=self.nQTilesN,
                                 comment="waveN * nQTilesN."))
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), vgpr(tmp2V),
                               comment="+ waveN * nQTilesN."))
        if qj:
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), qj, comment=f"qTileCol += {qj}."))
        colInRange = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_srColIR",
                                                          preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(colInRange, lsc), src0=vgpr(tmpV), src1=vgpr(nQTN),
                             comment="qTileCol < totalQTilesN?."))
        module.add(SAndB64(dst=sgpr(laneMask, lsc), src0=sgpr(laneMask, lsc),
                           src1=sgpr(colInRange, lsc),
                           comment="AND col in range."))
        self.writer.sgprPool.checkIn(colInRange)
        self._mulVgprBySgprConst(module, tmpV, "WorkGroup0", nQTilesMPerWG,
                                  "WG0 * nQTilesMPerWG.")
        if waveM is not None:
            module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(waveM), src1=self.nQTilesM,
                                 comment="waveM * nQTilesM."))
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), vgpr(tmp2V),
                               comment="+ waveM * nQTilesM."))
        module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(rowGroup),
                             src1=self.kBlocksPerLane,
                             comment=f"rowGroup * kBlocksPerLane={self.kBlocksPerLane}."))
        module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), vgpr(tmp2V),
                           comment="+ rowGroup * kBlocksPerLane."))
        if constRow:
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), constRow,
                               comment=f"qTileRow += constRow={constRow}."))
        rowInRange = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_srRowIR",
                                                          preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(rowInRange, lsc), src0=vgpr(tmpV), src1=vgpr(nQTM),
                             comment="qTileRow < totalQTilesM?."))
        module.add(SAndB64(dst=sgpr(laneMask, lsc), src0=sgpr(laneMask, lsc),
                           src1=sgpr(rowInRange, lsc),
                           comment="AND row in range."))
        self.writer.sgprPool.checkIn(rowInRange)
        module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(laneMask, lsc),
                                   comment="save exec; set exec = write-lane mask."))
        self._tileByteOffsetSubRow(module, addrV, tmpV, tmp2V, nQTN,
                                   m, kBlock, qj, rowGroup, waveM, waveN)
        scaleByteV = self.writer.vgprPool.checkOut(1, tag="mx_sbTmp")
        self._computeScaleByteInline(module, slot, quantMultVgprs, scaleByteV)
        module.add(BufferStoreB8(
            src=vgpr(scaleByteV), vaddr=vgpr(addrV),
            saddr=sgpr(mxSrd, 4), soffset=0,
            mubuf=MUBUFModifiers(offen=True),
            comment=f"MXScale sub-row byte (m={m},kBlock={kBlock},qj={qj})."))
        self.writer.vgprPool.checkIn(scaleByteV)
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                           comment="restore exec mask."))

    def _writeScale(self, mxSrd: int, quantMultVgprs: int,
                    col: int, rowGroup: int, savedExec: int, laneMask: int) -> Module:
        """Write one e8m0 byte per quant tile via predicated BufferStoreB8."""
        module = Module("MXFP8Quant writeScale")
        module.addComment1("MXFP8Quant: write MXScale[qTileRow, qTileCol] byte per tile.")
        addrV = self.writer.vgprPool.checkOut(1, tag="mx_wsAddr")
        tmpV  = self.writer.vgprPool.checkOut(1, tag="mx_wsTmp")
        tmp2V = self.writer.vgprPool.checkOut(1, tag="mx_wsTmp2")
        nQTN  = self.writer.vgprPool.checkOut(1, tag="mx_nQTN")
        self._computeTotalQTilesN(module, nQTN)
        waveM, waveN = self._computeWaveIndices(module)
        for qj in range(self.nQTilesN):
            for qi in range(self.nQTilesM):
                self._writeTileStore(module, qi, qj, mxSrd, quantMultVgprs,
                                     col, rowGroup, savedExec, laneMask,
                                     addrV, tmpV, tmp2V, nQTN, waveM, waveN)
        module.add(SWaitCnt(vscnt=0, comment="wait MXScale stores."))
        if waveN is not None:
            self.writer.vgprPool.checkIn(waveN)
        if waveM is not None:
            self.writer.vgprPool.checkIn(waveM)
        self.writer.vgprPool.checkIn(nQTN)
        self.writer.vgprPool.checkIn(tmp2V)
        self.writer.vgprPool.checkIn(tmpV)
        self.writer.vgprPool.checkIn(addrV)
        return module

    def _writeScaleSubRow(self, mxSrd: int, quantMultVgprs: int,
                          col: int, rowGroup: int, savedExec: int, laneMask: int) -> Module:
        """Write MXScale for sub-row mode: every rowGroup writes its own rows."""
        module = Module("MXFP8Quant writeScaleSubRow")
        module.addComment1("MXFP8Quant sub-row: write MXScale byte per (m, kBlock, qj) tile.")
        addrV = self.writer.vgprPool.checkOut(1, tag="mx_srAddr")
        tmpV  = self.writer.vgprPool.checkOut(1, tag="mx_srTmp")
        tmp2V = self.writer.vgprPool.checkOut(1, tag="mx_srTmp2")
        nQTN  = self.writer.vgprPool.checkOut(1, tag="mx_srNQTN")
        nQTM  = self.writer.vgprPool.checkOut(1, tag="mx_srNQTM")
        self._computeTotalQTilesN(module, nQTN)
        self._computeTotalQTilesM(module, nQTM)
        waveM, waveN = self._computeWaveIndices(module)
        for m in range(self.mmaM):
            for kBlock in range(self.kBlocksPerLane):
                lf0 = m * self.kBlocksPerLane + kBlock
                for qj in range(self.nQTilesN):
                    slot = lf0 * self.nQTilesN + qj
                    self._writeTileStoreSubRow(module, m, kBlock, qj, slot, mxSrd,
                                               quantMultVgprs, col, rowGroup,
                                               savedExec, laneMask,
                                               addrV, tmpV, tmp2V, nQTN, nQTM,
                                               waveM, waveN)
        module.add(SWaitCnt(vscnt=0, comment="wait MXScale sub-row stores."))
        if waveN is not None:
            self.writer.vgprPool.checkIn(waveN)
        if waveM is not None:
            self.writer.vgprPool.checkIn(waveM)
        for r in (nQTM, nQTN, tmp2V, tmpV, addrV):
            self.writer.vgprPool.checkIn(r)
        return module

    def _freeEmitRegs(self, amaxVgprs: int, accTmp: int, quantMultVgprs: int,
                       laneId: int, col: int, rowGroup: int,
                       mxSrd: int, savedExec: int, laneMask: int) -> None:
        """Return all emit-phase VGPR/SGPR allocations to their pools."""
        self.writer.sgprPool.checkIn(laneMask)
        self.writer.sgprPool.checkIn(savedExec)
        self.writer.sgprPool.checkIn(mxSrd)
        self.writer.vgprPool.checkIn(rowGroup)
        self.writer.vgprPool.checkIn(col)
        self.writer.vgprPool.checkIn(laneId)
        self.writer.vgprPool.checkIn(quantMultVgprs)
        self.writer.vgprPool.checkIn(accTmp)
        self.writer.vgprPool.checkIn(amaxVgprs)

    def emit(self, vgprTiles) -> Module:
        """Return the full MXFP8Quant epilogue module."""
        totalTiles = self.tileArrayLen
        module = Module("MXFP8Quant epilogue")
        module.addComment1("MXFP8Quant: per-block e8m0 dynamic quant for fp8 D output.")
        module.addComment0(
            f"  q0={self.q0}, q1={self.q1}, nQTilesM={self.nQTilesM}, nQTilesN={self.nQTilesN}.")
        module.add(SWaitCnt(waitAll=True, comment="flush MFMA pipeline before MXFP8Quant."))
        amaxVgprs      = self.writer.vgprPool.checkOut(totalTiles, tag="mx_amaxVgprs")
        accTmp         = self.writer.vgprPool.checkOut(1,           tag="mx_accTmp")
        quantMultVgprs = self.writer.vgprPool.checkOut(totalTiles,  tag="mx_quantMultVgprs")
        laneId         = self.writer.vgprPool.checkOut(1,           tag="mx_laneId")
        col            = self.writer.vgprPool.checkOut(1,           tag="mx_col")
        rowGroup       = self.writer.vgprPool.checkOut(1,           tag="mx_rowGroup")
        mxSrd  = self.writer.sgprPool.checkOutAligned(4, 4, tag="mx_mxSrd",
                                                        preventOverflow=False)
        savedExec = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="mx_savedExec", preventOverflow=False)
        laneMask  = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="mx_laneMask", preventOverflow=False)
        module.add(self._setup(mxSrd, laneId, col, rowGroup))
        module.add(self._applyAlphaInPlace(vgprTiles))
        module.add(self._laneTileAmax(vgprTiles, amaxVgprs, accTmp, rowGroup))
        module.add(self._butterflyReduce(amaxVgprs, laneId))
        # Compute quantMult per tile; scaleByte is recovered inline during writes.
        module.add(self._computeMXScale(amaxVgprs, quantMultVgprs))
        module.add(self._applyScaleInPlace(vgprTiles, quantMultVgprs, accTmp, rowGroup))
        if self.subRowQuant:
            module.add(self._writeScaleSubRow(mxSrd, quantMultVgprs,
                                              col, rowGroup, savedExec, laneMask))
        else:
            module.add(self._writeScale(mxSrd, quantMultVgprs, col, rowGroup, savedExec, laneMask))
        self._freeEmitRegs(amaxVgprs, accTmp, quantMultVgprs,
                           laneId, col, rowGroup, mxSrd, savedExec, laneMask)
        return module
