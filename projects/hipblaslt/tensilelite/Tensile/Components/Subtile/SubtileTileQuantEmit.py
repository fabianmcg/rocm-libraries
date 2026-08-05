# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TileQuant fused epilogue emitter for the Subtile kernel (gfx950).

Runs in the pre-store hook (KernelWriter.py ~line 5226), before VGPR
rearrangement and before the standard fp8 D store. It reads raw f32
accumulators, computes per-tile amax, multiplies the accumulators in
place by 448/amax (the RstdScale in-place pattern), and writes
QuantScale = amax/448 to the side buffer. The subsequent standard fp8
store converts the already-in-range values to OCP e4m3 for D.

D is the fp8 (OCP e4m3) output — no separate QuantOut tensor.
QuantScale is the only new side buffer (fp32), shape [ceil(M/Q0), ceil(N/Q1)].

Phase 1 constraint: quant tile must fit within a single wave sub-tile
(no cross-wave LDS reduction needed).

Config: DataType=B (bf16 A/B), DestDataType=F8 (OCP e4m3 D), ComputeDataType=S.

MFMA layout (gfx950, waveSize=64, 16x16 MFMA):
  lane % mfma_n = N-column within MMA tile.
  lane // mfma_n = row_group (which set of rows this lane owns).
  rows_per_lane = (mfma_m * mfma_n) // waveSize.

Acc VGPR ordering (n-outer mmaM-stride, m-inner):
  acc_idx(n, m) -> vgprTiles[n * mma_m + m].regList.indices[k].
  n is the M-tile index (outer, 0..mmaN-1), m is the N-tile index (inner, 0..mmaM-1).
  mmaM = MIWaveTile[0], mmaN = MIWaveTile[1].
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
    BufferStoreB32,
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
    VMovB32,
    VMulF32,
    VMulLOU32,
    VRcpF32,
    VXorB32,
)


# OCP FP8 e4m3 maximum representable magnitude.
_fp8E4m3Max = 448.0


class SubtileTileQuantEmitter:
    """Emit the TileQuant epilogue for the Subtile gfx950 kernel.

    Computes per-tile amax from f32 AGPRs, scales accumulators in place
    by 448/amax, and writes QuantScale=amax/448 to a side buffer.
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

        # Resolved quant tile dimensions (from _validateTileQuant).
        self.q0 = kernel["_TileQuantQ0"]
        self.q1 = kernel["_TileQuantQ1"]

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
    # Helpers reused from SubtilePartialRMSEmit (verbatim logic).         #
    # ------------------------------------------------------------------ #

    def _readAccInto(self, module, dst: int, vgprTiles, m: int, n: int, k: int,
                     comment: str) -> None:
        """Copy accumulator element (m, n, k) into VGPR dst.

        vgprTiles is indexed as n * mmaM + m, where:
          - n is the outer (M-tile) index, ranging 0..mmaN-1
          - m is the inner (N-tile) index, ranging 0..mmaM-1
        This matches the store path's tile ordering (tt0 + outerTT0 * tt1) =
        (m + mmaM * n), since mmaM = outerTT0 = MIWaveTile[0].
        """
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
        """Compute ceil(N / Q1) into VGPR dst at runtime from SizesFree+1.

        QuantScale has shape [ceil(M/Q0), ceil(N/Q1)] row-major; the row stride
        is ceil(N/Q1) and must reflect the full runtime N, not just MacroTile1.
        """
        with self.writer.allocTmpSgpr(1, tag="tq_nQTNsS") as s:
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
        """Compute ceil(nHidden / Q0) into VGPR dst (QuantScale row count for sub-row mode)."""
        with self.writer.allocTmpSgpr(1, tag="tq_nQTMsS") as s:
            module.add(SAddU32(dst=sgpr(s.idx), src0=sgpr("SizesFree+0"),
                               src1=self.q0 - 1,
                               comment=f"nHidden + Q0-1 (Q0={self.q0})."))
            log2q0 = int(math.log2(self.q0)) if self.q0 > 1 else 0
            module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(log2q0),
                                       src=sgpr(s.idx),
                                       comment=f"totalQTilesM = ceil(nHidden/Q0={self.q0})."))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(s.idx),
                               comment="totalQTilesM into VGPR."))

    def _setup(self, quantSrd: int, laneId: int, col: int, rowGroup: int) -> Module:
        """Build QuantScale SRD; compute laneId, col, rowGroup."""
        module = Module("TileQuant setup")
        module.add(SWaitCnt(kmcnt=0, comment="wait for TileQuant kernarg s_load."))
        self._buildBufferSrd(module, quantSrd, "QuantScale", "quantScale")
        module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)."))
        log2N = int(math.log2(self.mfmaN))
        module.add(VAndB32(dst=vgpr(col), src0=vgpr(laneId), src1=self.mfmaN - 1,
                           comment=f"col = laneId & ({self.mfmaN}-1)."))
        module.add(VLShiftRightB32(dst=vgpr(rowGroup), shiftHex=hex(log2N), src=vgpr(laneId),
                                   comment=f"rowGroup = laneId >> {log2N}."))
        return module

    def _accAbsIntoSlot(self, module, accTmp: int, vgprTiles, amaxVgprs: int,
                        m: int, n: int, k: int, qj: int, absMask: int) -> None:
        """Read |acc[m,n,k]| and fold into amax slot for single-tile M case (qi=0)."""
        slot = qj
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
        """Read |acc[m,n,k]|; distribute into the correct amax slot via runtime qi."""
        mBase = m * self.mfmaM + k
        self._readAccInto(module, accTmp, vgprTiles, m, n, k,
                           f"read acc[m={m},n={n},k={k}].")
        module.add(VAndB32(dst=vgpr(accTmp), src0=vgpr(accTmp), src1=vgpr(absMask),
                           comment="|acc|."))
        qiV = self.writer.vgprPool.checkOut(1, tag="tq_qiV")
        module.add(VMulLOU32(dst=vgpr(qiV), src0=self.rowsPerLane,
                             src1=vgpr(rowGroup), comment=f"rowGroup * {self.rowsPerLane}."))
        if 0 < mBase <= 64:
            module.add(VAddU32(vgpr(qiV), vgpr(qiV), mBase, comment=f"+ mBase={mBase}."))
        elif mBase > 64:
            tmpM = self.writer.vgprPool.checkOut(1, tag="tq_mBaseV")
            module.add(VMovB32(dst=vgpr(tmpM), src=mBase, comment=f"mBase={mBase}."))
            module.add(VAddU32(vgpr(qiV), vgpr(qiV), vgpr(tmpM), comment="+ mBase."))
            self.writer.vgprPool.checkIn(tmpM)
        log2q0 = int(math.log2(self.q0))
        module.add(VLShiftRightB32(dst=vgpr(qiV), shiftHex=hex(log2q0),
                                   src=vgpr(qiV), comment=f"qi = ... >> {log2q0}."))
        masked = self.writer.vgprPool.checkOut(1, tag="tq_masked")
        cond   = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="tq_qiCond", preventOverflow=False)
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
        # alpha must be applied before amax so QuantScale = amax(alpha*A*B)/448.
        module = Module("TileQuant applyAlphaInPlace")
        module.addComment1("TileQuant: scale accumulators by alpha before amax.")
        tmp = self.writer.vgprPool.checkOut(1, tag="tq_alphaAcc")
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
        """Compute per-lane per-quant-tile amax(|acc|).

        qj = (n * mfma_n) // q1 is compile-time.
        qi = (m*mfma_m + rowGroup*rows_per_lane + k) // q0 is runtime when nQTilesM > 1.
        """
        module = Module("TileQuant laneTileAmax")
        module.addComment1("TileQuant: per-lane per-quant-tile amax(|acc|).")
        totalTiles = self.tileArrayLen
        absMask = self.writer.vgprPool.checkOut(1, tag="tq_absMask")
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
        """Emit one XOR-butterfly round: fetch partner amax and fold via VMaxF32.

        N-butterfly xorVals (< mfma_n) flip col bits; M-butterfly xorVals
        (>= mfma_n, multiples of mfma_n) flip rowGroup bits. The two sets are
        disjoint by construction so there is no collision.
        """
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
        """Reduce per-tile amax across lanes sharing the same quant tile.

        Correctness (q0=32,q1=32,mfma_n=16,waveSize=64,rows_per_lane=4):
          N: nColRounds=4, xorVals {1,2,4,8}; M: rg_span=4, xorVals {16,32}.
          Full set {1,2,4,8,16,32}: all distinct, max=32<64.
        """
        totalTiles = self.tileArrayLen
        module     = Module("TileQuant butterflyReduce")
        module.addComment1(
            f"TileQuant: butterfly reduce amax (q0={self.q0}, q1={self.q1}).")
        # _resolveTileQuantShape enforces q1 >= mfmaN >= 2, so min(q1, mfmaN) == mfmaN always.
        nColRounds = int(math.log2(self.mfmaN))
        rgSpan  = min(self.q0 // self.rowsPerLane, self.waveSize // self.mfmaN)
        log2rg  = int(math.log2(rgSpan)) if rgSpan > 1 else 0
        numRounds  = nColRounds + log2rg
        if numRounds == 0:
            return module
        addrV = self.writer.vgprPool.checkOut(1, tag="tq_bfAddr")
        tmpV  = self.writer.vgprPool.checkOut(totalTiles, tag="tq_bfTmp")
        for i in range(numRounds):
            xorVal = (1 << i) if i < nColRounds else self.mfmaN << (i - nColRounds)
            self._butterflyRound(module, addrV, tmpV, amaxVgprs, totalTiles, laneId, xorVal)
        self.writer.vgprPool.checkIn(tmpV)
        self.writer.vgprPool.checkIn(addrV)
        return module

    def _computeOneTileScale(self, module, slot: int, amaxVgpr: int,
                              quantMultVgpr: int, scaleDequantVgpr: int,
                              fp8MaxV: int, invFp8V: int, zeroMask: int) -> None:
        """Emit scale instructions for one quant tile slot."""
        module.add(VRcpF32(dst=vgpr(quantMultVgpr), src=vgpr(amaxVgpr),
                           comment=f"quantMult[{slot}] = 1/amax[{slot}]."))
        if self.writer.states.archCaps.get("TransOpWait", False):
            module.add(SNop(waitState=0, comment="wait state after v_rcp_f32 transcendental (gfx950)."))
        module.add(VMulF32(dst=vgpr(quantMultVgpr),
                           src0=vgpr(quantMultVgpr), src1=vgpr(fp8MaxV),
                           comment=f"quantMult[{slot}] *= 448."))
        module.add(VCmpEQF32(dst=sgpr(zeroMask, self.laneSgprCount),
                              src0=0, src1=vgpr(amaxVgpr),
                              comment=f"amax[{slot}] == 0?."))
        module.add(VCndMaskB32(dst=vgpr(quantMultVgpr),
                               src0=vgpr(quantMultVgpr), src1=0,
                               src2=sgpr(zeroMask, self.laneSgprCount),
                               comment=f"quantMult[{slot}] = 0 if amax==0."))
        if scaleDequantVgpr is not None:
            module.add(VMulF32(dst=vgpr(scaleDequantVgpr),
                               src0=vgpr(amaxVgpr), src1=vgpr(invFp8V),
                               comment=f"scaleDequant[{slot}] = amax/{_fp8E4m3Max}."))

    def _computePerTileScale(self, amaxVgprs: int, quantMultVgprs: int,
                              scaleDequantVgprs: int,
                              computeScaleDequant: bool = True) -> Module:
        """Compute quantMult=448/amax and scaleDequant=amax/448 per tile.

        Guards amax==0 to avoid inf: sets quantMult=0 when amax==0.
        """
        module = Module("TileQuant computePerTileScale")
        module.addComment1("TileQuant: compute quantMult=448/amax and scaleDequant=amax/448 per tile.")
        fp8MaxBits = struct.unpack('<I', struct.pack('<f', _fp8E4m3Max))[0]
        invFp8Bits = struct.unpack('<I', struct.pack('<f', 1.0 / _fp8E4m3Max))[0]
        totalTiles = self.tileArrayLen
        fp8MaxV = self.writer.vgprPool.checkOut(1, tag="tq_fp8Max")
        invFp8V = self.writer.vgprPool.checkOut(1, tag="tq_invFp8") if computeScaleDequant else None
        module.add(VMovB32(dst=vgpr(fp8MaxV), src=hex(fp8MaxBits),
                           comment=f"fp8_max = {_fp8E4m3Max}."))
        if computeScaleDequant:
            module.add(VMovB32(dst=vgpr(invFp8V), src=hex(invFp8Bits),
                               comment=f"1/fp8_max = 1/{_fp8E4m3Max}."))
        zeroMask = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="tq_zeroMask",
            preventOverflow=False)
        for t in range(totalTiles):
            scaleDequantVgpr = (scaleDequantVgprs + t) if computeScaleDequant else None
            self._computeOneTileScale(module, t, amaxVgprs + t,
                                      quantMultVgprs + t, scaleDequantVgpr,
                                      fp8MaxV, invFp8V, zeroMask)
        self.writer.sgprPool.checkIn(zeroMask)
        if invFp8V is not None:
            self.writer.vgprPool.checkIn(invFp8V)
        self.writer.vgprPool.checkIn(fp8MaxV)
        return module

    def _applyScaleInPlace(self, vgprTiles, quantMultVgprs: int,
                            accTmp: int, rowGroup: int) -> Module:
        """Multiply every accumulator element in-place by the tile's quantMult."""
        module = Module("TileQuant applyScaleInPlace")
        module.addComment1("TileQuant: multiply accumulators in place by quantMult.")
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
        qiRTV = self.writer.vgprPool.checkOut(1, tag="tq_qiRTV")
        module.add(VMulLOU32(dst=vgpr(qiRTV), src0=self.rowsPerLane,
                             src1=vgpr(rowGroup), comment=f"rowGroup * {self.rowsPerLane}."))
        if 0 < mBase <= 64:
            module.add(VAddU32(vgpr(qiRTV), vgpr(qiRTV), mBase, comment=f"+ mBase={mBase}."))
        elif mBase > 64:
            tmpM = self.writer.vgprPool.checkOut(1, tag="tq_mBaseVA")
            module.add(VMovB32(dst=vgpr(tmpM), src=mBase, comment=f"mBase={mBase}."))
            module.add(VAddU32(vgpr(qiRTV), vgpr(qiRTV), vgpr(tmpM), comment="+ mBase."))
            self.writer.vgprPool.checkIn(tmpM)
        log2q0 = int(math.log2(self.q0))
        module.add(VLShiftRightB32(dst=vgpr(qiRTV), shiftHex=hex(log2q0),
                                   src=vgpr(qiRTV), comment=f"qi = ... >> {log2q0}."))
        multSel = self.writer.vgprPool.checkOut(1, tag="tq_multSel")
        module.add(VMovB32(dst=vgpr(multSel), src=0, comment="init multSel=0."))
        condV = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="tq_condVA", preventOverflow=False)
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
        """Compute waveM = waveIdx % wg_m and waveN = (waveIdx // wg_m) % wg_n.

        waveIdx = Serial >> log2(waveSize) is the flat wave index in the
        workgroup. Returns (waveM, waveN) VGPR indices, each None when its
        dimension has a single wave and needs no offset.
        """
        if self.wgM <= 1 and self.wgN <= 1:
            return None, None
        log2Wave = int(math.log2(self.waveSize))
        waveIdx = self.writer.vgprPool.checkOut(1, tag="tq_waveIdx")
        module.add(VLShiftRightB32(dst=vgpr(waveIdx), shiftHex=hex(log2Wave),
                                   src=vgpr("Serial"),
                                   comment=f"waveIdx = Serial >> {log2Wave}."))
        waveM = None
        if self.wgM > 1:
            waveM = self.writer.vgprPool.checkOut(1, tag="tq_waveM")
            module.add(VAndB32(dst=vgpr(waveM), src0=vgpr(waveIdx), src1=self.wgM - 1,
                               comment=f"waveM = waveIdx & ({self.wgM}-1)."))
        waveN = None
        if self.wgN > 1:
            log2WgM = int(math.log2(self.wgM))
            waveN = self.writer.vgprPool.checkOut(1, tag="tq_waveN")
            module.add(VLShiftRightB32(dst=vgpr(waveN), shiftHex=hex(log2WgM),
                                       src=vgpr(waveIdx),
                                       comment=f"waveIdx >> {log2WgM}."))
            module.add(VAndB32(dst=vgpr(waveN), src0=vgpr(waveN), src1=self.wgN - 1,
                               comment=f"waveN = (waveIdx >> {log2WgM}) & ({self.wgN}-1)."))
        self.writer.vgprPool.checkIn(waveIdx)
        return waveM, waveN

    def _tileByteOffset(self, module, addrV: int, tmpV: int, tmp2V: int, nQTN: int,
                         qi: int, qj: int, waveM, waveN) -> None:
        """Compute byteOff = (qTileRow * totalQTilesN + qTileCol) * 4 into addrV.

        qTileRow = WorkGroup0 * nQTilesMPerWG + waveM * nQTilesM + qi.
        qTileCol = WorkGroup1 * nQTilesNPerWG + waveN * nQTilesN + qj.
        nQTilesMPerWG = nQTilesM * wg_m; nQTilesNPerWG = nQTilesN * wg_n.
        totalQTilesN = ceil(N/Q1) is pre-loaded into VGPR nQTN.
        """
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
                           comment="qTileRow * totalQTilesN + qTileCol."))
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(2), src=vgpr(addrV),
                                  comment="byteOff = linear_tile_idx * 4."))

    def _mulVgprBySgprConst(self, module, dst_vgpr: int, sgpr_name: str,
                             const: int, comment: str) -> None:
        """Emit dst_vgpr = sgpr(sgpr_name) * const, loading const into a tmp VGPR.

        v_mul_lo_u32 does not accept literal operands on gfx950 when src0 is an
        SGPR, and cannot have two SGPR sources. Load the constant into a VGPR so
        gfx950 sees one SGPR source and one VGPR source.
        """
        tmp = self.writer.vgprPool.checkOut(1, tag="tq_mulConst")
        module.add(VMovB32(dst=vgpr(tmp), src=const, comment=f"const={const} into vgpr."))
        module.add(VMulLOU32(dst=vgpr(dst_vgpr), src0=vgpr(tmp),
                             src1=sgpr(sgpr_name), comment=comment))
        self.writer.vgprPool.checkIn(tmp)

    def _tileByteOffsetSubRow(self, module, addrV: int, tmpV: int, tmp2V: int,
                               nQTN: int, m: int, kBlock: int, qj: int,
                               rowGroup: int, waveM, waveN) -> None:
        """Compute byteOff for sub-row QuantScale tile (m, kBlock, qj) into addrV.

        qTileRow = WG0*nQTilesMPerWG + waveM*nQTilesM + rowGroup*kBlocksPerLane + constRow.
        qTileCol = WG1*nQTilesNPerWG + waveN*nQTilesN + qj.
        """
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
                           comment="qTileRow * totalQTilesN + qTileCol."))
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(2), src=vgpr(addrV),
                                  comment="byteOff = linear_tile_idx * 4."))

    def _writeTileStore(self, module, qi: int, qj: int, quantSrd: int,
                         scaleDequantVgprs: int, col: int, rowGroup: int,
                         savedExec: int, laneMask: int,
                         addrV: int, tmpV: int, tmp2V: int, nQTN: int,
                         waveM, waveN) -> None:
        """Predicated write of QuantScale for one quant tile (qi, qj)."""
        lsc     = self.laneSgprCount
        slot    = qi * self.nQTilesN + qj
        repCol = (qj * self.q1) % self.mfmaN
        # rowGroup is in [0, waveSize/mfma_n - 1]; use the position within the
        # first mfma tile that this quant tile starts in.
        repRg  = ((qi * self.q0) % self.mfmaM) // self.rowsPerLane
        module.addComment0(f"  Tile qi={qi}, qj={qj}: repCol={repCol}, repRg={repRg}.")
        colCond = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="tq_wColCond",
                                                        preventOverflow=False)
        rgCond  = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="tq_wRgCond",
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
        # Suppress stores whose column tile is beyond the runtime column count
        # (totalQTilesN = ceil(M/Q1)); such tiles would otherwise alias into the
        # next QuantScale row or overrun the buffer. qTileCol is wave-uniform.
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
        colInRange = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="tq_wColInRange",
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
        module.add(BufferStoreB32(
            src=vgpr(scaleDequantVgprs + slot), vaddr=vgpr(addrV),
            saddr=sgpr(quantSrd, 4), soffset=0,
            mubuf=MUBUFModifiers(offen=True),
            comment=f"QuantScale[qTileRow, qTileCol] (qi={qi}, qj={qj})."))
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                           comment="restore exec mask."))

    def _writeTileStoreSubRow(self, module, m: int, kBlock: int, qj: int, slot: int,
                               quantSrd: int, amaxVgprs: int, invFp8V: int, scaleTmp: int,
                               col: int, rowGroup: int, savedExec: int, laneMask: int,
                               addrV: int, tmpV: int, tmp2V: int,
                               nQTN: int, nQTM: int, waveM, waveN) -> None:
        """Predicated write of sub-row QuantScale for tile (m, kBlock, qj).

        Predicate: col==0 AND qTileCol < totalQTilesN AND qTileRow < totalQTilesM.
        Every rowGroup writes its own row (no single representative rowGroup).
        """
        lsc = self.laneSgprCount
        nQTilesMPerWG = self.nQTilesM * self.wgM
        nQTilesNPerWG = self.nQTilesN * self.wgN
        constRow = m * self.tilesPerMfmaM + kBlock
        module.addComment0(f"  SubRow tile m={m}, kBlock={kBlock}, qj={qj}, slot={slot}.")
        module.add(VCmpEQU32(dst=sgpr(laneMask, lsc), src0=0, src1=vgpr(col),
                             comment="col == 0 (representative free1 lane)."))
        # qTileCol < totalQTilesN guard.
        self._mulVgprBySgprConst(module, tmpV, "WorkGroup1", nQTilesNPerWG,
                                  "WG1 * nQTilesNPerWG.")
        if waveN is not None:
            module.add(VMulLOU32(dst=vgpr(tmp2V), src0=vgpr(waveN), src1=self.nQTilesN,
                                 comment="waveN * nQTilesN."))
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), vgpr(tmp2V),
                               comment="+ waveN * nQTilesN."))
        if qj:
            module.add(VAddU32(vgpr(tmpV), vgpr(tmpV), qj, comment=f"qTileCol += {qj}."))
        colInRange = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="tq_srColIR",
                                                          preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(colInRange, lsc), src0=vgpr(tmpV), src1=vgpr(nQTN),
                             comment="qTileCol < totalQTilesN?."))
        module.add(SAndB64(dst=sgpr(laneMask, lsc), src0=sgpr(laneMask, lsc),
                           src1=sgpr(colInRange, lsc),
                           comment="AND col in range."))
        self.writer.sgprPool.checkIn(colInRange)
        # qTileRow < totalQTilesM guard.
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
        rowInRange = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="tq_srRowIR",
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
        module.add(VMulF32(dst=vgpr(scaleTmp), src0=vgpr(amaxVgprs + slot),
                           src1=vgpr(invFp8V),
                           comment=f"scaleDequant = amax[slot={slot}]/{_fp8E4m3Max} (inline)."))
        module.add(BufferStoreB32(
            src=vgpr(scaleTmp), vaddr=vgpr(addrV),
            saddr=sgpr(quantSrd, 4), soffset=0,
            mubuf=MUBUFModifiers(offen=True),
            comment=f"QuantScale sub-row (m={m},kBlock={kBlock},qj={qj})."))
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                           comment="restore exec mask."))

    def _writeScale(self, quantSrd: int, scaleDequantVgprs: int,
                    col: int, rowGroup: int, savedExec: int, laneMask: int) -> Module:
        """Write one fp32 QuantScale per quant tile via predicated BufferStoreB32."""
        module = Module("TileQuant writeScale")
        module.addComment1("TileQuant: write QuantScale[qTileRow, qTileCol] per tile.")
        addrV = self.writer.vgprPool.checkOut(1, tag="tq_wsAddr")
        tmpV  = self.writer.vgprPool.checkOut(1, tag="tq_wsTmp")
        tmp2V = self.writer.vgprPool.checkOut(1, tag="tq_wsTmp2")
        nQTN  = self.writer.vgprPool.checkOut(1, tag="tq_nQTN")
        self._computeTotalQTilesN(module, nQTN)
        waveM, waveN = self._computeWaveIndices(module)
        for qj in range(self.nQTilesN):
            for qi in range(self.nQTilesM):
                self._writeTileStore(module, qi, qj, quantSrd, scaleDequantVgprs,
                                     col, rowGroup, savedExec, laneMask,
                                     addrV, tmpV, tmp2V, nQTN, waveM, waveN)
        module.add(SWaitCnt(vscnt=0, comment="wait QuantScale stores."))
        if waveN is not None:
            self.writer.vgprPool.checkIn(waveN)
        if waveM is not None:
            self.writer.vgprPool.checkIn(waveM)
        self.writer.vgprPool.checkIn(nQTN)
        self.writer.vgprPool.checkIn(tmp2V)
        self.writer.vgprPool.checkIn(tmpV)
        self.writer.vgprPool.checkIn(addrV)
        return module

    def _writeScaleSubRow(self, quantSrd: int, amaxVgprs: int,
                          col: int, rowGroup: int, savedExec: int, laneMask: int) -> Module:
        """Write QuantScale for sub-row mode: every rowGroup writes its own rows."""
        module = Module("TileQuant writeScaleSubRow")
        module.addComment1("TileQuant sub-row: write QuantScale per (m, kBlock, qj) tile.")
        addrV = self.writer.vgprPool.checkOut(1, tag="tq_srAddr")
        tmpV  = self.writer.vgprPool.checkOut(1, tag="tq_srTmp")
        tmp2V = self.writer.vgprPool.checkOut(1, tag="tq_srTmp2")
        nQTN  = self.writer.vgprPool.checkOut(1, tag="tq_srNQTN")
        nQTM  = self.writer.vgprPool.checkOut(1, tag="tq_srNQTM")
        invFp8V  = self.writer.vgprPool.checkOut(1, tag="tq_srInvFp8")
        scaleTmp = self.writer.vgprPool.checkOut(1, tag="tq_srScaleDequant")
        invFp8Bits = struct.unpack('<I', struct.pack('<f', 1.0 / _fp8E4m3Max))[0]
        module.add(VMovB32(dst=vgpr(invFp8V), src=hex(invFp8Bits),
                           comment=f"1/fp8_max = 1/{_fp8E4m3Max}."))
        self._computeTotalQTilesN(module, nQTN)
        self._computeTotalQTilesM(module, nQTM)
        waveM, waveN = self._computeWaveIndices(module)
        for m in range(self.mmaM):
            for kBlock in range(self.kBlocksPerLane):
                lf0 = m * self.kBlocksPerLane + kBlock
                for qj in range(self.nQTilesN):
                    slot = lf0 * self.nQTilesN + qj
                    self._writeTileStoreSubRow(module, m, kBlock, qj, slot, quantSrd,
                                               amaxVgprs, invFp8V, scaleTmp, col, rowGroup,
                                               savedExec, laneMask,
                                               addrV, tmpV, tmp2V, nQTN, nQTM,
                                               waveM, waveN)
        module.add(SWaitCnt(vscnt=0, comment="wait QuantScale sub-row stores."))
        if waveN is not None:
            self.writer.vgprPool.checkIn(waveN)
        if waveM is not None:
            self.writer.vgprPool.checkIn(waveM)
        for r in (scaleTmp, invFp8V, nQTM, nQTN, tmp2V, tmpV, addrV):
            self.writer.vgprPool.checkIn(r)
        return module

    def _injectKnownValues(self, module, vgprTiles, quantMultVgprs: int,
                            scaleDequantVgprs: int, amaxVgprs: int) -> None:
        """Replace accumulators and scales with deterministic values for store-order testing.

        Each accumulator element (n, m, k) receives float(n * mmaM * rowsPerLane + m *
        rowsPerLane + k + 1), a unique positive integer in [1, mmaN*mmaM*rowsPerLane].
        All quantMult slots are set to 1.0 and all scaleDequant slots to 1/448 so
        the store path writes fp8(known_value) without any further scaling.
        """
        module.addComment1("TileQuant DEBUG: inject position-encoded values into accumulators.")
        totalTiles = self.tileArrayLen
        constV = self.writer.vgprPool.checkOut(1, tag="tq_dbgConst")
        oneBits    = struct.unpack('<I', struct.pack('<f', 1.0))[0]
        invFp8Bits = struct.unpack('<I', struct.pack('<f', 1.0 / _fp8E4m3Max))[0]
        for t in range(totalTiles):
            module.add(VMovB32(dst=vgpr(quantMultVgprs + t), src=hex(oneBits),
                               comment=f"quantMult[{t}] = 1.0 (debug inject)."))
            if self.subRowQuant:
                module.add(VMovB32(dst=vgpr(amaxVgprs + t), src=hex(oneBits),
                                   comment=f"amax[{t}] = 1.0 (debug inject, sub-row)."))
            else:
                module.add(VMovB32(dst=vgpr(scaleDequantVgprs + t), src=hex(invFp8Bits),
                                   comment=f"scaleDequant[{t}] = 1/448 (debug inject)."))
        for n in range(self.mmaN):
            for m in range(self.mmaM):
                for k in range(self.rowsPerLane):
                    linId = n * self.mmaM * self.rowsPerLane + m * self.rowsPerLane + k + 1
                    valBits = struct.unpack('<I', struct.pack('<f', float(linId)))[0]
                    module.add(VMovB32(dst=vgpr(constV), src=hex(valBits),
                                       comment=f"inject acc[n={n},m={m},k={k}]={linId}."))
                    self._writeAccFrom(module, constV, vgprTiles, m, n, k,
                                       f"write known acc[n={n},m={m},k={k}].")
        self.writer.vgprPool.checkIn(constV)

    def _freeEmitRegs(self, amaxVgprs: int, accTmp: int, quantMultVgprs: int,
                       scaleDequantVgprs: int, laneId: int, col: int, rowGroup: int,
                       quantSrd: int, savedExec: int, laneMask: int) -> None:
        """Return all emit-phase VGPR/SGPR allocations to their pools."""
        self.writer.sgprPool.checkIn(laneMask)
        self.writer.sgprPool.checkIn(savedExec)
        self.writer.sgprPool.checkIn(quantSrd)
        self.writer.vgprPool.checkIn(rowGroup)
        self.writer.vgprPool.checkIn(col)
        self.writer.vgprPool.checkIn(laneId)
        self.writer.vgprPool.checkIn(scaleDequantVgprs)
        self.writer.vgprPool.checkIn(quantMultVgprs)
        self.writer.vgprPool.checkIn(accTmp)
        self.writer.vgprPool.checkIn(amaxVgprs)

    def emit(self, vgprTiles) -> Module:
        """Return the full TileQuant epilogue module."""
        totalTiles = self.tileArrayLen
        module = Module("TileQuant epilogue")
        module.addComment1("TileQuant: per-tile amax pre-scale for fp8 D output.")
        module.addComment0(
            f"  q0={self.q0}, q1={self.q1}, nQTilesM={self.nQTilesM}, nQTilesN={self.nQTilesN}.")
        module.add(SWaitCnt(waitAll=True, comment="flush MFMA pipeline before TileQuant."))
        amaxVgprs         = self.writer.vgprPool.checkOut(totalTiles, tag="tq_amaxVgprs")
        accTmp            = self.writer.vgprPool.checkOut(1,           tag="tq_accTmp")
        quantMultVgprs    = self.writer.vgprPool.checkOut(totalTiles,  tag="tq_quantMultVgprs")
        scaleDequantAllocLen = 1 if self.subRowQuant else totalTiles
        scaleDequantVgprs = self.writer.vgprPool.checkOut(scaleDequantAllocLen, tag="tq_scaleDequantVgprs")
        laneId            = self.writer.vgprPool.checkOut(1,           tag="tq_laneId")
        col               = self.writer.vgprPool.checkOut(1,           tag="tq_col")
        rowGroup          = self.writer.vgprPool.checkOut(1,           tag="tq_rowGroup")
        quantSrd  = self.writer.sgprPool.checkOutAligned(4, 4, tag="tq_quantSrd",
                                                          preventOverflow=False)
        savedExec = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="tq_savedExec", preventOverflow=False)
        laneMask  = self.writer.sgprPool.checkOutAligned(
            self.laneSgprCount, self.laneSgprCount, tag="tq_laneMask", preventOverflow=False)
        debugInject = self.kernel.get("_TQ_DebugInjectKnown", False)
        module.add(self._setup(quantSrd, laneId, col, rowGroup))
        if debugInject:
            # Skip all epilogue math; overwrite accumulators with position-encoded
            # constants and set scale factors to identity so fp8 stores the raw value.
            module.addComment1("TileQuant DEBUG: _TQ_DebugInjectKnown=True — skipping epilogue math.")
            self._injectKnownValues(module, vgprTiles, quantMultVgprs, scaleDequantVgprs, amaxVgprs)
        else:
            module.add(self._applyAlphaInPlace(vgprTiles))
            module.add(self._laneTileAmax(vgprTiles, amaxVgprs, accTmp, rowGroup))
            module.add(self._butterflyReduce(amaxVgprs, laneId))
            module.add(self._computePerTileScale(amaxVgprs, quantMultVgprs, scaleDequantVgprs,
                                                 computeScaleDequant=not self.subRowQuant))
            module.add(self._applyScaleInPlace(vgprTiles, quantMultVgprs, accTmp, rowGroup))
        if self.subRowQuant:
            module.add(self._writeScaleSubRow(quantSrd, amaxVgprs, col, rowGroup, savedExec, laneMask))
        else:
            module.add(self._writeScale(quantSrd, scaleDequantVgprs, col, rowGroup, savedExec, laneMask))
        self._freeEmitRegs(amaxVgprs, accTmp, quantMultVgprs, scaleDequantVgprs,
                           laneId, col, rowGroup, quantSrd, savedExec, laneMask)
        return module
