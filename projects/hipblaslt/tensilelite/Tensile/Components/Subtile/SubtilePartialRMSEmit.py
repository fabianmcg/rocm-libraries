# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PartialRMS fused epilogue emitter for the Subtile kernel (gfx950, bf16).

Computes per-row sum-of-squares from the fp32 GEMM accumulator (AGPRs) and
writes one fp32 value per (row, N-tile) to a global partialBuf. Also applies
the gamma weight (bf16) to the accumulator in-place. The downstream global-write
path then stores D as bf16.

This is Phase 1 (K1) of a two-kernel RMSNorm implementation:
  - K1 (this kernel): computes partial Σx² over MT1 columns owned by this WG
    and writes to partialBuf[m, WorkGroup1]. K1 does NOT divide by N_hidden.
  - K2: reads all partialBuf columns per row, reduces, computes
    rstd = rsqrt(Σx²/N + eps), and writes rstdBuf.

partialBuf layout contract (2D, row-major):
  - Logical shape [M_padded, N_tiles_N], N_tiles_N = ceil(N_hidden / MT1).
  - partialBuf[m, t] = Σ_{n in WG t's columns} h1[m,n]²  (raw sum; K2 divides by N_hidden).
  - Byte offset for (m, t) = (m * N_tiles_N + t) * 4.
  - Row index m = WorkGroup0 * MT0 + intra-tile row (as before).
  - Tile column t = WorkGroup1 (the WG's index along the N axis).
  - N_tiles_N is computed on device as ceil(SizesFree[1] / MT1); it is not a kernarg.
  - N_hidden may be any multiple of MT1; the WG owns one MT1-wide N-tile.
  - N_hidden need not divide MT1; the trailing partial N-tile is GEMM-zero-padded
    and contributes 0 to Σx². The host passes N_tiles_N = ceil(N_hidden / MT1).

Reduction stages:
  1. Within-wave butterfly across the mfma_n column-lanes.
  2. (When wg_n > 1) LDS cross-wave reduction across wg_n sibling waves.
  3. Lanes where laneId % mfma_n == 0 write partialBuf. Exec mask is
     narrowed to those lanes for the stores, then restored.

Gamma application:
  Loads gamma (bf16) for each MMA N-tile via BufferLoadD16B16, converts to
  fp32, and multiplies each accumulator element in-place. No rstd multiply.
  The result is h1 * gamma[n], written back to AGPRs for the store path.

MFMA layout (gfx950, waveSize=64, 16x16 MFMA):
  - lane % mfma_n = N-column within MMA tile
  - rows_per_lane = (mfma_m * mfma_n) // waveSize

Acc VGPR ordering (N-outer, M-inner):
  acc_idx(base, m, n, k) = base + (n*mma_m + m)*rows_per_lane + k

Alpha=1, beta=0 must be passed by the host.
"""

import math

from rocisa.code import Module
from rocisa.container import (
    ContinuousRegister,
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
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddF32,
    VAddU32,
    VAndB32,
    VCmpEQU32,
    VCvtBF16toFP32,
    VFmaF32,
    VLShiftLeftB32,
    SAddU32,
    SLShiftLeftB32,
    SLShiftRightB32,
    VMulF32,
    VMulLOU32,
    VMovB32,
    VLShiftRightB32,
    VXorB32,
)


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

        # laneSGPRCount: 1 for wave32, 2 for wave64.
        self.lane_sgpr_count = writer.states.laneSGPRCount

    def _acc_idx(self, base: int, m: int, n: int, k: int) -> int:
        """AGPR index for accumulator element at M-tile m, N-tile n, row-offset k."""
        return base + (n * self.mma_m + m) * self.rows_per_lane + k

    def _partial_idx(self, m: int, k: int) -> int:
        """Index into partials array for M-tile m, row-offset k."""
        return m * self.rows_per_lane + k

    def emit(self, accVgprBase: int) -> Module:
        """Return the full PartialRMS epilogue module.

        accVgprBase: AGPR index of the first D-tile accumulator.
        """
        numAccVgpr = self.mma_m * self.mma_n * self.rows_per_lane
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
        partials = self.writer.vgprPool.checkOut(self.numRows, tag="pRMS_partials")
        accTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_accTmp")
        gammaTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_gammaTmp")
        bflyTmp = self.writer.vgprPool.checkOut(1, tag="pRMS_bflyTmp")
        permAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_permAddr")
        laneId = self.writer.vgprPool.checkOut(1, tag="pRMS_laneId")
        # colByte is computed and consumed outside the EXEC-narrowed window in _writePartials.
        colByte = self.writer.vgprPool.checkOut(1, tag="pRMS_colByte")
        globalAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_globalAddr")

        # Allocate SGPRs: gamma SRD, partialBuf SRD, saved exec.
        # savedExec and laneMaskSgpr must be 2-aligned for 64-bit EXEC operations.
        # rowBase = WorkGroup0 * MT0 is computed into globalAddr on demand (no SGPR).
        # tileCol = sgpr("WorkGroup1"), live named SGPR, no allocation needed.
        gammaSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_gammaSrd")
        partialSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_partialSrd")
        savedExec = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_savedExec"
        )
        laneMaskSgpr = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_laneMask"
        )

        # Flush MFMA pipeline before reading AGPRs.
        module.add(
            SWaitCnt(waitAll=True, comment="flush MFMA pipeline before PartialRMS")
        )

        module.add(self._setup(gammaSrd, partialSrd, laneId, colByte))
        module.add(self._squareAndLaneSum(accVgprBase, partials, accTmp))
        module.add(self._butterflyReduce(partials, permAddr, laneId, bflyTmp))
        if self.wg_n > 1:
            module.add(self._crossWaveReduce(partials))
        module.add(
            self._writePartials(
                partials, partialSrd, laneId, savedExec, laneMaskSgpr, globalAddr
            )
        )
        module.add(
            self._applyGammaOnly(accVgprBase, gammaSrd, gammaTmp, accTmp, colByte)
        )

        self.writer.sgprPool.checkIn(laneMaskSgpr)
        self.writer.sgprPool.checkIn(savedExec)
        self.writer.sgprPool.checkIn(partialSrd)
        self.writer.sgprPool.checkIn(gammaSrd)
        self.writer.vgprPool.checkIn(globalAddr)
        self.writer.vgprPool.checkIn(colByte)
        self.writer.vgprPool.checkIn(laneId)
        self.writer.vgprPool.checkIn(permAddr)
        self.writer.vgprPool.checkIn(bflyTmp)
        self.writer.vgprPool.checkIn(gammaTmp)
        self.writer.vgprPool.checkIn(accTmp)
        self.writer.vgprPool.checkIn(partials)

        return module

    def _setup(
        self,
        gammaSrd: int,
        partialSrd: int,
        laneId: int,
        colByte: int,
    ) -> Module:
        """Build gamma SRD, partialBuf SRD; derive laneId and colByte.

        Signature append order (matches Signature.py additions):
          slot N+0: RMSNormGamma  (bf16 global buffer pointer, 8 bytes)
          slot N+1: PartialBuf    (fp32 global buffer pointer, 8 bytes) [InOutArray]

        rowBase = WorkGroup0 * MT0 is computed on demand in _writePartials (no SGPR).
        tileCol = sgpr("WorkGroup1"), live named SGPR, no extra allocation needed.
        """
        module = Module("PartialRMS setup")
        module.add(SWaitCnt(kmcnt=0, comment="wait for PartialRMS kernarg s_load"))

        # Gamma SRD (bf16 global buffer).
        module.add(
            SMovB64(
                dst=sgpr(gammaSrd, 2),
                src=sgpr("RMSNormGamma", 2),
                comment="gamma SRD base",
            )
        )
        module.add(
            SMovB32(dst=sgpr(gammaSrd + 2), src="BufferOOB", comment="gamma SRD limit")
        )
        module.add(
            SMovB32(dst=sgpr(gammaSrd + 3), src="Srd127_96", comment="gamma SRD flags")
        )

        # PartialBuf SRD (fp32 global buffer).
        module.add(
            SMovB64(
                dst=sgpr(partialSrd, 2),
                src=sgpr("PartialBuf", 2),
                comment="partialBuf SRD base",
            )
        )
        module.add(
            SMovB32(
                dst=sgpr(partialSrd + 2),
                src="BufferOOB",
                comment="partialBuf SRD limit",
            )
        )
        module.add(
            SMovB32(
                dst=sgpr(partialSrd + 3),
                src="Srd127_96",
                comment="partialBuf SRD flags",
            )
        )

        # laneId = Serial & (waveSize - 1).
        module.add(
            VAndB32(
                dst=vgpr(laneId),
                src0=vgpr("Serial"),
                src1=self.waveSize - 1,
                comment="laneId = Serial & (waveSize-1)",
            )
        )

        # colByte = (laneId % mfma_n) * 2  (byte offset into bf16 gamma per-lane).
        module.add(
            VAndB32(
                dst=vgpr(colByte),
                src0=vgpr(laneId),
                src1=self.mfma_n - 1,
                comment=f"colInMma = laneId % {self.mfma_n}",
            )
        )
        module.add(
            VLShiftLeftB32(
                dst=vgpr(colByte),
                shiftHex=hex(1),
                src=vgpr(colByte),
                comment="colByte = colInMma * 2 (bf16 size)",
            )
        )

        # When wg_n > 1, shift colByte by the wave's column base.
        if self.wg_n > 1:
            waveN = self.writer.vgprPool.checkOut(1, tag="pRMS_setupWaveN")
            tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_setupTmp")
            tmpRes = ContinuousRegister(tmpVgpr, 2)
            module.add(
                vectorStaticDivide(
                    waveN,
                    "Serial",
                    self.waveSize * self.wg_m,
                    tmpRes,
                    comment=f"waveN = Serial / {self.waveSize * self.wg_m}",
                )
            )
            colBaseBytes = self.mma_n * self.mfma_n * 2
            with self.writer.allocTmpSgpr(1, tag="pRMS_setupColBase") as tmpSgprInfo:
                module.add(
                    SMovB32(
                        dst=sgpr(tmpSgprInfo.idx),
                        src=hex(colBaseBytes),
                        comment=f"col base bytes per wave ({colBaseBytes})",
                    )
                )
                module.add(
                    VMulLOU32(
                        dst=vgpr(waveN),
                        src0=sgpr(tmpSgprInfo.idx),
                        src1=vgpr(waveN),
                        comment="waveN * mma_n * mfma_n * 2",
                    )
                )
            module.add(
                VAddU32(
                    vgpr(colByte),
                    vgpr(colByte),
                    vgpr(waveN),
                    comment="colByte += wave column base",
                )
            )
            self.writer.vgprPool.checkIn(tmpVgpr)
            self.writer.vgprPool.checkIn(waveN)

        # Add WorkGroup1 * MT1 * 2 to colByte so each WG addresses its own gamma tile.
        # MT1 * 2 is a power-of-2 because MT1 is a power-of-2 and bf16 is 2 bytes.
        wg1Shift = int(math.log2(self.macro_tile1 * 2))
        with self.writer.allocTmpSgpr(1, tag="pRMS_setupWG1") as wg1S:
            module.add(
                SLShiftLeftB32(
                    dst=sgpr(wg1S.idx),
                    src=sgpr("WorkGroup1"),
                    shiftHex=hex(wg1Shift),
                    comment=f"wg1ColByte = WorkGroup1 * MT1*2 (MT1={self.macro_tile1})",
                )
            )
            module.add(
                VAddU32(
                    vgpr(colByte),
                    vgpr(colByte),
                    sgpr(wg1S.idx),
                    comment="colByte += WorkGroup1 * MT1 * 2",
                )
            )

        return module

    def _squareAndLaneSum(
        self, accVgprBase: int, partials: int, accTmp: int
    ) -> Module:
        """Step 1: compute per-row Σx² from fp32 AGPRs across all N-tiles.

        For each (m, k): partial[m*rows_per_lane+k] = Σ_n acc[m,n,k]²
        """
        module = Module("PartialRMS squareAndLaneSum")
        module.addComment1("PartialRMS step 1: per-row partial Σx² from AGPRs")
        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                pidx = partials + self._partial_idx(m, k)
                first = self._acc_idx(accVgprBase, m, 0, k)
                module.add(
                    VAccvgprReadB32(
                        vgpr(accTmp),
                        accvgpr(first),
                        comment=f"read acc[m={m},n=0,k={k}]",
                    )
                )
                module.add(
                    VMulF32(
                        dst=vgpr(pidx),
                        src0=vgpr(accTmp),
                        src1=vgpr(accTmp),
                        comment=f"partial[m={m},k={k}] = acc^2",
                    )
                )
                for n in range(1, self.mma_n):
                    a = self._acc_idx(accVgprBase, m, n, k)
                    module.add(
                        VAccvgprReadB32(
                            vgpr(accTmp),
                            accvgpr(a),
                            comment=f"read acc[m={m},n={n},k={k}]",
                        )
                    )
                    module.add(
                        VFmaF32(
                            dst=vgpr(pidx),
                            src0=vgpr(accTmp),
                            src1=vgpr(accTmp),
                            src2=vgpr(pidx),
                            comment=f"partial[m={m},k={k}] += acc^2",
                        )
                    )
        return module

    def _butterflyReduce(
        self, partials: int, permAddr: int, laneId: int, bflyTmp: int
    ) -> Module:
        """Step 2: butterfly across mfma_n column-sharing lanes.

        Strides computed as [mfma_n >> i for i in range(1, mfma_n.bit_length())].
        For mfma_n=16 → [8, 4, 2, 1] (4 stages).
        """
        module = Module("PartialRMS butterflyReduce")
        module.addComment1(
            f"PartialRMS step 2: butterfly Σx² across {self.mfma_n} column lanes"
        )
        strides = [self.mfma_n >> i for i in range(1, self.mfma_n.bit_length())]
        for stride in strides:
            module.addComment0(f"  butterfly stride={stride}")
            module.add(
                VXorB32(
                    dst=vgpr(permAddr),
                    src0=vgpr(laneId),
                    src1=stride,
                    comment=f"partner = laneId ^ {stride}",
                )
            )
            module.add(
                VLShiftLeftB32(
                    dst=vgpr(permAddr),
                    shiftHex=hex(2),
                    src=vgpr(permAddr),
                    comment="partnerByteAddr = partner * 4",
                )
            )
            for i in range(self.numRows):
                module.add(
                    DSBPermuteB32(
                        dst=vgpr(bflyTmp),
                        src0=vgpr(permAddr),
                        src1=vgpr(partials + i),
                        comment=f"bflyTmp = partner's partial[{i}]",
                    )
                )
                module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
                module.add(
                    VAddF32(
                        dst=vgpr(partials + i),
                        src0=vgpr(partials + i),
                        src1=vgpr(bflyTmp),
                        comment=f"partial[{i}] += partner's value",
                    )
                )
        return module

    def _crossWaveReduce(self, partials: int) -> Module:
        """Step 3: LDS reduction across wg_n sibling waves.

        After the butterfly, each wave holds Σx² over its N-column slice.
        Combines the wg_n sibling-wave partials through LDS so every lane
        holds the full per-row Σx².

        Wave-id convention: waveId = waveN*wg_m + waveM.
        Slot stride: waveSize * numRows * 4 bytes per wave slot.
        """
        strideW = self.waveSize * self.numRows * 4
        # laneSlotBytes: byte stride between adjacent lane slots within a wave slot.
        laneSlotBytes = self.numRows * 4
        groupStride = self.wg_m * strideW

        module = Module("PartialRMS crossWaveReduce")
        module.addComment1(
            f"PartialRMS step 3: cross-wave LDS reduction over wg_n={self.wg_n}"
        )

        module.add(
            self.writer._syncThreads(
                self.kernel,
                "partialRMS cross-wave: ensure siblings done reading LDS before scratch write",
            )
        )

        waveId = self.writer.vgprPool.checkOut(1, tag="pRMS_xwWaveId")
        waveM = self.writer.vgprPool.checkOut(1, tag="pRMS_xwWaveM")
        laneLoc = self.writer.vgprPool.checkOut(1, tag="pRMS_xwLane")
        writeAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwWriteAddr")
        readAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwReadAddr")
        readTmp = self.writer.vgprPool.checkOut(self.numRows, tag="pRMS_xwReadTmp")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_xwTmp")
        tmpRes = ContinuousRegister(tmpVgpr, 2)

        module.add(
            VAndB32(
                dst=vgpr(laneLoc),
                src0=vgpr("Serial"),
                src1=self.waveSize - 1,
                comment="laneId for LDS addressing",
            )
        )
        module.add(
            vectorStaticDivide(
                waveId,
                "Serial",
                self.waveSize,
                tmpRes,
                comment="waveId = Serial / WavefrontSize",
            )
        )
        module.add(
            VAndB32(
                dst=vgpr(waveM),
                src0=vgpr(waveId),
                src1=self.wg_m - 1,
                comment=f"waveM = waveId %% {self.wg_m} (bitmask, wg_m pow2)",
            )
        )

        # Compute writeAddr and readAddr base (wave-level byte offsets into LDS).
        with self.writer.allocTmpSgpr(1, tag="pRMS_xwAddrSetup") as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            module.add(
                SMovB32(
                    dst=sgpr(tmpSgpr), src=hex(strideW), comment=f"strideW={strideW}"
                )
            )
            module.add(
                VMulLOU32(
                    dst=vgpr(writeAddr),
                    src0=sgpr(tmpSgpr),
                    src1=vgpr(waveId),
                    comment="writeAddr = waveId * strideW",
                )
            )
            module.add(
                VMulLOU32(
                    dst=vgpr(readAddr),
                    src0=sgpr(tmpSgpr),
                    src1=vgpr(waveM),
                    comment="readAddr = waveM * strideW",
                )
            )
            module.add(
                SMovB32(
                    dst=sgpr(tmpSgpr),
                    src=hex(laneSlotBytes),
                    comment=f"laneSlotBytes={laneSlotBytes}",
                )
            )
            module.add(
                VMulLOU32(
                    dst=vgpr(laneLoc),
                    src0=sgpr(tmpSgpr),
                    src1=vgpr(laneLoc),
                    comment="lane * laneSlotBytes",
                )
            )
            module.add(
                VAddU32(
                    vgpr(writeAddr),
                    vgpr(writeAddr),
                    vgpr(laneLoc),
                    comment="writeAddr += lane*laneSlotBytes",
                )
            )
            module.add(
                VAddU32(
                    vgpr(readAddr),
                    vgpr(readAddr),
                    vgpr(laneLoc),
                    comment="readAddr += lane*laneSlotBytes",
                )
            )

        for i in range(self.numRows):
            module.add(
                DSStoreB32(
                    dstAddr=vgpr(writeAddr),
                    src=vgpr(partials + i),
                    ds=DSModifiers(offset=i * 4),
                    comment=f"LDS store partial[{i}]",
                )
            )
        module.add(SWaitCnt(dscnt=0, comment="wait LDS writes"))
        module.add(self.writer._syncThreads(self.kernel, "partialRMS cross-wave write"))

        for j in range(self.wg_n):
            for i in range(self.numRows):
                module.add(
                    DSLoadB32(
                        dst=vgpr(readTmp + i),
                        src=vgpr(readAddr),
                        ds=DSModifiers(offset=i * 4),
                        comment=f"LDS load wave[{j}] partial[{i}]",
                    )
                )
            module.add(SWaitCnt(dscnt=0, comment="wait LDS reads"))
            for i in range(self.numRows):
                if j == 0:
                    module.add(
                        VMovB32(
                            dst=vgpr(partials + i),
                            src=vgpr(readTmp + i),
                            comment=f"partial[{i}] = wave[0]",
                        )
                    )
                else:
                    module.add(
                        VAddF32(
                            dst=vgpr(partials + i),
                            src0=vgpr(partials + i),
                            src1=vgpr(readTmp + i),
                            comment=f"partial[{i}] += wave[{j}]",
                        )
                    )
            if j < self.wg_n - 1:
                with self.writer.allocTmpSgpr(1, tag="pRMS_xwAdvance") as tmpSgprInfo:
                    module.add(
                        SMovB32(
                            dst=sgpr(tmpSgprInfo.idx),
                            src=hex(groupStride),
                            comment=f"groupStride={groupStride}",
                        )
                    )
                    module.add(
                        VAddU32(
                            vgpr(readAddr),
                            vgpr(readAddr),
                            sgpr(tmpSgprInfo.idx),
                            comment="advance readAddr to next sibling",
                        )
                    )

        module.add(self.writer._syncThreads(self.kernel, "partialRMS cross-wave done"))

        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(readTmp)
        self.writer.vgprPool.checkIn(readAddr)
        self.writer.vgprPool.checkIn(writeAddr)
        self.writer.vgprPool.checkIn(laneLoc)
        self.writer.vgprPool.checkIn(waveM)
        self.writer.vgprPool.checkIn(waveId)

        return module

    def _writePartials(
        self,
        partials: int,
        partialSrd: int,
        laneId: int,
        savedExec: int,
        laneMaskSgpr: int,
        globalAddr: int,
    ) -> Module:
        """Step 4: write per-row Σx² to global partialBuf (2D layout).

        MFMA row-group layout (16x16 MFMA, wave64):
          - row group g = laneId // mfma_n  (0..waveSize//mfma_n - 1)
          - Within M-tile m, row groups are interleaved:
              row group g owns rows m*mfma_m + g*rows_per_lane .. m*mfma_m + g*rows_per_lane + rows_per_lane-1

        Each row group selects one writing lane (colInMma == 0, i.e., laneId % mfma_n == 0).
        Writing lane with row group g writes partial[m*rows_per_lane+k] to 2D address:
          byteOff = (globalRow * N_tiles_N + tileCol) * 4
          where globalRow = WorkGroup0 * MT0 + m*mfma_m + k + rowGroup*rows_per_lane
                N_tiles_N  = ceil(SizesFree[1] / MT1), computed on device
                tileCol    = sgpr("WorkGroup1") (live named SGPR, no extra allocation)

        Row base is computed directly into globalAddr per iteration (no SGPR needed).
        N_tiles_N is moved into a VGPR once to avoid SGPR-src0 restrictions on VMulLOU32.
        """
        module = Module("PartialRMS writePartials")
        module.addComment1(
            "PartialRMS step 4: predicated 2D write of Σx² to partialBuf"
        )
        module.addComment0(
            f"  Writing lanes: laneId % {self.mfma_n} == 0; 2D addr = (row*NTilesN+tileCol)*4"
        )

        lsc = self.lane_sgpr_count

        # Compute rowGroup = laneId // mfma_n (runtime, per-lane).
        rowGroup = self.writer.vgprPool.checkOut(1, tag="pRMS_rowGroup")
        rowGroupOff = self.writer.vgprPool.checkOut(1, tag="pRMS_rowGroupOff")
        ntilesVgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_nTilesV")

        # rowGroup = laneId >> log2(mfma_n)  (mfma_n must be power of 2).
        log2MfmaN = int(math.log2(self.mfma_n))
        module.add(
            VLShiftRightB32(
                dst=vgpr(rowGroup),
                shiftHex=hex(log2MfmaN),
                src=vgpr(laneId),
                comment=f"rowGroup = laneId >> {log2MfmaN} (= laneId // {self.mfma_n})",
            )
        )
        # rowGroupOff = rowGroup * rows_per_lane.
        module.add(
            VMulLOU32(
                dst=vgpr(rowGroupOff),
                src0=self.rows_per_lane,
                src1=vgpr(rowGroup),
                comment=f"rowGroupOff = rowGroup * {self.rows_per_lane}",
            )
        )

        # Compute N_tiles_N = ceil(SizesFree[1] / MT1) into ntilesVgpr.
        # NTilesN is not loaded as a named SGPR (to avoid SGPR pool pressure).
        log2Mt1 = int(math.log2(self.macro_tile1))
        with self.writer.allocTmpSgpr(1, tag="pRMS_nTilesS") as ntilesS:
            module.add(
                SAddU32(
                    dst=sgpr(ntilesS.idx),
                    src0=sgpr("SizesFree+1"),
                    src1=self.macro_tile1 - 1,
                    comment=f"N + MT1-1  (MT1={self.macro_tile1})",
                )
            )
            module.add(
                SLShiftRightB32(
                    dst=sgpr(ntilesS.idx),
                    shiftHex=hex(log2Mt1),
                    src=sgpr(ntilesS.idx),
                    comment=f"N_tiles_N = ceil(N / MT1={self.macro_tile1})",
                )
            )
            module.add(
                VMovB32(
                    dst=vgpr(ntilesVgpr),
                    src=sgpr(ntilesS.idx),
                    comment="ntilesVgpr = N_tiles_N",
                )
            )

        # Compute lane mask: active iff laneId % mfma_n == 0.
        colInMma = self.writer.vgprPool.checkOut(1, tag="pRMS_colInMma")
        module.add(
            VAndB32(
                dst=vgpr(colInMma),
                src0=vgpr(laneId),
                src1=self.mfma_n - 1,
                comment="colInMma = laneId % mfma_n",
            )
        )
        module.add(
            VCmpEQU32(
                dst=sgpr(laneMaskSgpr, lsc),
                src0=0,
                src1=vgpr(colInMma),
                comment="laneMask: lanes where colInMma == 0",
            )
        )
        self.writer.vgprPool.checkIn(colInMma)

        # Hoist wgRowBase = WorkGroup0 * MT0 once before the exec-narrowing.
        # VMulLOU32 cannot encode large integer literals in src0; move MT0 into a
        # VGPR first so both sources are registers.
        wgRowBase = self.writer.vgprPool.checkOut(1, tag="pRMS_wgRowBase")
        mt0Vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_mt0V")
        module.add(
            VMovB32(
                dst=vgpr(mt0Vgpr),
                src=self.macro_tile0,
                comment=f"MT0={self.macro_tile0} → vgpr for VMulLOU32",
            )
        )
        module.add(
            VMulLOU32(
                dst=vgpr(wgRowBase),
                src0=vgpr(mt0Vgpr),
                src1=sgpr("WorkGroup0"),
                comment=f"wgRowBase = WorkGroup0 * MT0={self.macro_tile0}",
            )
        )
        self.writer.vgprPool.checkIn(mt0Vgpr)

        # Add waveM * mma_m * mfma_m to wgRowBase so each wave addresses its
        # own M-row slice when wg_m > 1.
        if self.wg_m > 1:
            waveId = self.writer.vgprPool.checkOut(1, tag="pRMS_wpWaveId")
            waveM = self.writer.vgprPool.checkOut(1, tag="pRMS_wpWaveM")
            tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_wpTmp")
            tmpRes = ContinuousRegister(tmpVgpr, 2)
            module.add(
                vectorStaticDivide(
                    waveId,
                    "Serial",
                    self.waveSize,
                    tmpRes,
                    comment="waveId = Serial / WavefrontSize",
                )
            )
            module.add(
                VAndB32(
                    dst=vgpr(waveM),
                    src0=vgpr(waveId),
                    src1=self.wg_m - 1,
                    comment=f"waveM = waveId %% {self.wg_m} (bitmask, wg_m pow2)",
                )
            )
            waveStride = self.mma_m * self.mfma_m
            waveStrideVgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_wpWaveStride")
            module.add(
                VMovB32(
                    dst=vgpr(waveStrideVgpr),
                    src=waveStride,
                    comment=f"waveStride = mma_m * mfma_m = {waveStride}",
                )
            )
            module.add(
                VMulLOU32(
                    dst=vgpr(waveM),
                    src0=vgpr(waveStrideVgpr),
                    src1=vgpr(waveM),
                    comment="waveMOff = waveM * (mma_m * mfma_m)",
                )
            )
            module.add(
                VAddU32(
                    vgpr(wgRowBase),
                    vgpr(wgRowBase),
                    vgpr(waveM),
                    comment="wgRowBase += waveM * mma_m * mfma_m",
                )
            )
            self.writer.vgprPool.checkIn(waveStrideVgpr)
            self.writer.vgprPool.checkIn(tmpVgpr)
            self.writer.vgprPool.checkIn(waveM)
            self.writer.vgprPool.checkIn(waveId)

        # Save exec and narrow to writing lanes.
        module.add(
            SAndSaveExecB64(
                dst=sgpr(savedExec, lsc),
                src=sgpr(laneMaskSgpr, lsc),
                comment="save exec; set exec = writing-lane mask",
            )
        )

        # Write each partial[m*rows_per_lane+k] to 2D address in partialBuf.
        # 2D address: byteOff = (globalRow * N_tiles_N + tileCol) * 4
        # globalRow = WorkGroup0 * MT0 + m*mfma_m + k + rowGroup * rows_per_lane
        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                i = self._partial_idx(m, k)
                mBase = m * self.mfma_m + k
                # mBase is small (< MT0), so this never wraps for M < 2^31.
                module.add(
                    VAddU32(
                        vgpr(globalAddr),
                        vgpr(wgRowBase),
                        mBase,
                        comment=f"globalRow = wgRowBase + {mBase} (m*mfma_m + k)",
                    )
                )
                module.add(
                    VAddU32(
                        vgpr(globalAddr),
                        vgpr(globalAddr),
                        vgpr(rowGroupOff),
                        comment="globalRow += rowGroup * rows_per_lane",
                    )
                )
                module.add(
                    VMulLOU32(
                        dst=vgpr(globalAddr),
                        src0=vgpr(ntilesVgpr),
                        src1=vgpr(globalAddr),
                        comment="globalRow * N_tiles_N",
                    )
                )
                module.add(
                    VAddU32(
                        vgpr(globalAddr),
                        vgpr(globalAddr),
                        sgpr("WorkGroup1"),
                        comment="+ tileCol = WorkGroup1",
                    )
                )
                module.add(
                    VLShiftLeftB32(
                        dst=vgpr(globalAddr),
                        shiftHex=hex(2),
                        src=vgpr(globalAddr),
                        comment="byteOff = (row*N_tiles_N + tileCol) * 4",
                    )
                )
                module.add(
                    BufferStoreB32(
                        src=vgpr(partials + i),
                        vaddr=vgpr(globalAddr),
                        saddr=sgpr(partialSrd, 4),
                        soffset=0,
                        mubuf=MUBUFModifiers(offen=True),
                        comment=f"partialBuf[row, tileCol] = Σx² (m={m},k={k})",
                    )
                )
        module.add(SWaitCnt(vlcnt=0, comment="wait partialBuf stores"))

        module.add(
            SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc), comment="restore exec mask")
        )

        self.writer.vgprPool.checkIn(wgRowBase)
        self.writer.vgprPool.checkIn(ntilesVgpr)
        self.writer.vgprPool.checkIn(rowGroupOff)
        self.writer.vgprPool.checkIn(rowGroup)

        return module

    def _applyGammaOnly(
        self,
        accVgprBase: int,
        gammaSrd: int,
        gammaTmp: int,
        accTmp: int,
        colByte: int,
    ) -> Module:
        """Step 5: load gamma (bf16) and multiply each accumulator element.

        For each MMA N-tile n:
          gammaFp32 = bf16_to_fp32(gamma[n*mfma_n + lane%mfma_n])
          for (m, k): acc[m, n, k] *= gammaFp32

        No rstd multiply here; K2 applies rstd. The store path writes D as bf16.
        Gamma byte offset for N-tile n: n * mfma_n * 2 (bf16 element size = 2).
        """
        module = Module("PartialRMS applyGammaOnly")
        module.addComment1("PartialRMS step 5: apply gamma in-place (no rstd)")

        for n in range(self.mma_n):
            mmaBaseByte = n * self.mfma_n * 2
            module.add(
                BufferLoadD16B16(
                    vgpr(gammaTmp),
                    vgpr(colByte),
                    sgpr(gammaSrd, 4),
                    0,
                    mubuf=MUBUFModifiers(offen=True, offset12=mmaBaseByte),
                    comment=f"gammaBf16[n={n}]",
                )
            )
            module.add(SWaitCnt(vlcnt=0, comment="wait gamma load"))
            module.add(
                VCvtBF16toFP32(
                    vgpr(gammaTmp),
                    vgpr(gammaTmp),
                    None,
                    0,
                    comment="gamma bf16 → fp32",
                )
            )
            for m in range(self.mma_m):
                for k in range(self.rows_per_lane):
                    a = self._acc_idx(accVgprBase, m, n, k)
                    module.add(
                        VAccvgprReadB32(
                            vgpr(accTmp),
                            accvgpr(a),
                            comment=f"read acc[m={m},n={n},k={k}]",
                        )
                    )
                    module.add(
                        VMulF32(
                            dst=vgpr(accTmp),
                            src0=vgpr(accTmp),
                            src1=vgpr(gammaTmp),
                            comment="acc *= gamma",
                        )
                    )
                    module.add(
                        VAccvgprWriteB32(
                            accvgpr(a),
                            vgpr(accTmp),
                            comment=f"write acc[m={m},n={n},k={k}]",
                        )
                    )

        return module
