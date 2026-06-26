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
  3. Lanes where lane_id % mfma_n == 0 write partialBuf. Exec mask is
     narrowed to those lanes for the stores, then restored.

Gamma application:
  Loads gamma (bf16) for each MMA N-tile via BufferLoadD16B16, converts to
  fp32, and multiplies each accumulator element in-place. No rstd multiply.
  The result is h1 * gamma[n], written back to AGPRs for the store path.

MFMA layout (gfx950, waveSize=64, 16x16 MFMA):
  - lane % mfma_n = N-column within MMA tile
  - rows_per_lane = (mfma_m * mfma_n) // wave_size

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
        self.wave_size = kernel["WavefrontSize"]
        self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.wave_size

        wg = kernel["MIWaveGroup"]
        self.wg_m = wg[0]
        self.wg_n = wg[1]

        self.mma_m = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
        self.mma_n = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
        self.macro_tile0 = kernel["MacroTile0"]
        self.macro_tile1 = kernel["MacroTile1"]
        self.num_rows = self.mma_m * self.rows_per_lane

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
        partials = self.writer.vgprPool.checkOut(self.num_rows, tag="pRMS_partials")
        acc_tmp = self.writer.vgprPool.checkOut(1, tag="pRMS_accTmp")
        gamma_tmp = self.writer.vgprPool.checkOut(1, tag="pRMS_gammaTmp")
        bfly_tmp = self.writer.vgprPool.checkOut(1, tag="pRMS_bflyTmp")
        perm_addr = self.writer.vgprPool.checkOut(1, tag="pRMS_permAddr")
        lane_id = self.writer.vgprPool.checkOut(1, tag="pRMS_laneId")
        col_byte = self.writer.vgprPool.checkOut(1, tag="pRMS_colByte")
        global_addr = self.writer.vgprPool.checkOut(1, tag="pRMS_globalAddr")

        # Allocate SGPRs: gamma SRD, partialBuf SRD, saved exec.
        # saved_exec and lane_mask_sgpr must be 2-aligned for 64-bit EXEC operations.
        # row_base = WorkGroup0 * MT0 is computed into global_addr on demand (no SGPR).
        # tile_col = sgpr("WorkGroup1"), live named SGPR, no allocation needed.
        gamma_srd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_gammaSrd")
        partial_srd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_partialSrd")
        saved_exec = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_savedExec"
        )
        lane_mask_sgpr = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_laneMask"
        )

        # Flush MFMA pipeline before reading AGPRs.
        module.add(
            SWaitCnt(waitAll=True, comment="flush MFMA pipeline before PartialRMS")
        )

        module.add(self._setup(gamma_srd, partial_srd, lane_id, col_byte))
        module.add(self._squareAndLaneSum(accVgprBase, partials, acc_tmp))
        module.add(self._butterflyReduce(partials, perm_addr, lane_id, bfly_tmp))
        if self.wg_n > 1:
            module.add(self._crossWaveReduce(partials))
        module.add(
            self._writePartials(
                partials, partial_srd, lane_id, saved_exec, lane_mask_sgpr, global_addr
            )
        )
        module.add(
            self._applyGammaOnly(accVgprBase, gamma_srd, gamma_tmp, acc_tmp, col_byte)
        )

        self.writer.sgprPool.checkIn(lane_mask_sgpr)
        self.writer.sgprPool.checkIn(saved_exec)
        self.writer.sgprPool.checkIn(partial_srd)
        self.writer.sgprPool.checkIn(gamma_srd)
        self.writer.vgprPool.checkIn(global_addr)
        self.writer.vgprPool.checkIn(col_byte)
        self.writer.vgprPool.checkIn(lane_id)
        self.writer.vgprPool.checkIn(perm_addr)
        self.writer.vgprPool.checkIn(bfly_tmp)
        self.writer.vgprPool.checkIn(gamma_tmp)
        self.writer.vgprPool.checkIn(acc_tmp)
        self.writer.vgprPool.checkIn(partials)

        return module

    def _setup(
        self,
        gamma_srd: int,
        partial_srd: int,
        lane_id: int,
        col_byte: int,
    ) -> Module:
        """Build gamma SRD, partialBuf SRD; derive lane_id and col_byte.

        Signature append order (matches Signature.py additions):
          slot N+0: RMSNormGamma  (bf16 global buffer pointer, 8 bytes)
          slot N+1: PartialBuf    (fp32 global buffer pointer, 8 bytes) [InOutArray]

        row_base = WorkGroup0 * MT0 is computed on demand in _writePartials (no SGPR).
        tile_col = sgpr("WorkGroup1"), live named SGPR, no extra allocation needed.
        """
        module = Module("PartialRMS setup")
        module.add(SWaitCnt(kmcnt=0, comment="wait for PartialRMS kernarg s_load"))

        # Gamma SRD (bf16 global buffer).
        module.add(
            SMovB64(
                dst=sgpr(gamma_srd, 2),
                src=sgpr("RMSNormGamma", 2),
                comment="gamma SRD base",
            )
        )
        module.add(
            SMovB32(dst=sgpr(gamma_srd + 2), src="BufferOOB", comment="gamma SRD limit")
        )
        module.add(
            SMovB32(dst=sgpr(gamma_srd + 3), src="Srd127_96", comment="gamma SRD flags")
        )

        # PartialBuf SRD (fp32 global buffer).
        module.add(
            SMovB64(
                dst=sgpr(partial_srd, 2),
                src=sgpr("PartialBuf", 2),
                comment="partialBuf SRD base",
            )
        )
        module.add(
            SMovB32(
                dst=sgpr(partial_srd + 2),
                src="BufferOOB",
                comment="partialBuf SRD limit",
            )
        )
        module.add(
            SMovB32(
                dst=sgpr(partial_srd + 3),
                src="Srd127_96",
                comment="partialBuf SRD flags",
            )
        )

        # lane_id = Serial & (wave_size - 1).
        module.add(
            VAndB32(
                dst=vgpr(lane_id),
                src0=vgpr("Serial"),
                src1=self.wave_size - 1,
                comment="lane_id = Serial & (wave_size-1)",
            )
        )

        # col_byte = (lane_id % mfma_n) * 2  (byte offset into bf16 gamma per-lane).
        module.add(
            VAndB32(
                dst=vgpr(col_byte),
                src0=vgpr(lane_id),
                src1=self.mfma_n - 1,
                comment=f"col_in_mma = lane_id % {self.mfma_n}",
            )
        )
        module.add(
            VLShiftLeftB32(
                dst=vgpr(col_byte),
                shiftHex=hex(1),
                src=vgpr(col_byte),
                comment="col_byte = col_in_mma * 2 (bf16 size)",
            )
        )

        # When wg_n > 1, shift col_byte by the wave's column base.
        if self.wg_n > 1:
            wave_n = self.writer.vgprPool.checkOut(1, tag="pRMS_setupWaveN")
            tmp_vgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_setupTmp")
            tmp_res = ContinuousRegister(tmp_vgpr, 2)
            module.add(
                vectorStaticDivide(
                    wave_n,
                    "Serial",
                    self.wave_size * self.wg_m,
                    tmp_res,
                    comment=f"wave_n = Serial / {self.wave_size * self.wg_m}",
                )
            )
            col_base_bytes = self.mma_n * self.mfma_n * 2
            with self.writer.allocTmpSgpr(1, tag="pRMS_setupColBase") as tmpSgprInfo:
                module.add(
                    SMovB32(
                        dst=sgpr(tmpSgprInfo.idx),
                        src=hex(col_base_bytes),
                        comment=f"col base bytes per wave ({col_base_bytes})",
                    )
                )
                module.add(
                    VMulLOU32(
                        dst=vgpr(wave_n),
                        src0=sgpr(tmpSgprInfo.idx),
                        src1=vgpr(wave_n),
                        comment="wave_n * mma_n * mfma_n * 2",
                    )
                )
            module.add(
                VAddU32(
                    vgpr(col_byte),
                    vgpr(col_byte),
                    vgpr(wave_n),
                    comment="col_byte += wave column base",
                )
            )
            self.writer.vgprPool.checkIn(tmp_vgpr)
            self.writer.vgprPool.checkIn(wave_n)

        # Add WorkGroup1 * MT1 * 2 to col_byte so each WG addresses its own gamma tile.
        # MT1 * 2 is a power-of-2 because MT1 is a power-of-2 and bf16 is 2 bytes.
        wg1_shift = int(math.log2(self.macro_tile1 * 2))
        with self.writer.allocTmpSgpr(1, tag="pRMS_setupWG1") as wg1_s:
            module.add(
                SLShiftLeftB32(
                    dst=sgpr(wg1_s.idx),
                    src=sgpr("WorkGroup1"),
                    shiftHex=hex(wg1_shift),
                    comment=f"wg1_col_byte = WorkGroup1 * MT1*2 (MT1={self.macro_tile1})",
                )
            )
            module.add(
                VAddU32(
                    vgpr(col_byte),
                    vgpr(col_byte),
                    sgpr(wg1_s.idx),
                    comment="col_byte += WorkGroup1 * MT1 * 2",
                )
            )

        return module

    def _squareAndLaneSum(
        self, accVgprBase: int, partials: int, acc_tmp: int
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
                        vgpr(acc_tmp),
                        accvgpr(first),
                        comment=f"read acc[m={m},n=0,k={k}]",
                    )
                )
                module.add(
                    VMulF32(
                        dst=vgpr(pidx),
                        src0=vgpr(acc_tmp),
                        src1=vgpr(acc_tmp),
                        comment=f"partial[m={m},k={k}] = acc^2",
                    )
                )
                for n in range(1, self.mma_n):
                    a = self._acc_idx(accVgprBase, m, n, k)
                    module.add(
                        VAccvgprReadB32(
                            vgpr(acc_tmp),
                            accvgpr(a),
                            comment=f"read acc[m={m},n={n},k={k}]",
                        )
                    )
                    module.add(
                        VFmaF32(
                            dst=vgpr(pidx),
                            src0=vgpr(acc_tmp),
                            src1=vgpr(acc_tmp),
                            src2=vgpr(pidx),
                            comment=f"partial[m={m},k={k}] += acc^2",
                        )
                    )
        return module

    def _butterflyReduce(
        self, partials: int, perm_addr: int, lane_id: int, bfly_tmp: int
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
                    dst=vgpr(perm_addr),
                    src0=vgpr(lane_id),
                    src1=stride,
                    comment=f"partner = lane_id ^ {stride}",
                )
            )
            module.add(
                VLShiftLeftB32(
                    dst=vgpr(perm_addr),
                    shiftHex=hex(2),
                    src=vgpr(perm_addr),
                    comment="partner_byte_addr = partner * 4",
                )
            )
            for i in range(self.num_rows):
                module.add(
                    DSBPermuteB32(
                        dst=vgpr(bfly_tmp),
                        src0=vgpr(perm_addr),
                        src1=vgpr(partials + i),
                        comment=f"bfly_tmp = partner's partial[{i}]",
                    )
                )
                module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
                module.add(
                    VAddF32(
                        dst=vgpr(partials + i),
                        src0=vgpr(partials + i),
                        src1=vgpr(bfly_tmp),
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
        Slot stride: wave_size * num_rows * 4 bytes per wave slot.
        """
        stride_w = self.wave_size * self.num_rows * 4
        row_stride = self.num_rows * 4
        group_stride = self.wg_m * stride_w

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

        wave_id = self.writer.vgprPool.checkOut(1, tag="pRMS_xwWaveId")
        wave_m = self.writer.vgprPool.checkOut(1, tag="pRMS_xwWaveM")
        lane_loc = self.writer.vgprPool.checkOut(1, tag="pRMS_xwLane")
        write_addr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwWriteAddr")
        read_addr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwReadAddr")
        read_tmp = self.writer.vgprPool.checkOut(self.num_rows, tag="pRMS_xwReadTmp")
        tmp_vgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_xwTmp")
        tmp_res = ContinuousRegister(tmp_vgpr, 2)

        module.add(
            VAndB32(
                dst=vgpr(lane_loc),
                src0=vgpr("Serial"),
                src1=self.wave_size - 1,
                comment="lane_id for LDS addressing",
            )
        )
        module.add(
            vectorStaticDivide(
                wave_id,
                "Serial",
                self.wave_size,
                tmp_res,
                comment="waveId = Serial / WavefrontSize",
            )
        )
        module.add(
            VAndB32(
                dst=vgpr(wave_m),
                src0=vgpr(wave_id),
                src1=self.wg_m - 1,
                comment=f"waveM = waveId %% {self.wg_m} (bitmask, wg_m pow2)",
            )
        )

        # Compute write_addr and read_addr base (wave-level byte offsets into LDS).
        with self.writer.allocTmpSgpr(1, tag="pRMS_xwAddrSetup") as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            module.add(
                SMovB32(
                    dst=sgpr(tmpSgpr), src=hex(stride_w), comment=f"stride_w={stride_w}"
                )
            )
            module.add(
                VMulLOU32(
                    dst=vgpr(write_addr),
                    src0=sgpr(tmpSgpr),
                    src1=vgpr(wave_id),
                    comment="write_addr = waveId * stride_w",
                )
            )
            module.add(
                VMulLOU32(
                    dst=vgpr(read_addr),
                    src0=sgpr(tmpSgpr),
                    src1=vgpr(wave_m),
                    comment="read_addr = waveM * stride_w",
                )
            )
            module.add(
                SMovB32(
                    dst=sgpr(tmpSgpr),
                    src=hex(row_stride),
                    comment=f"row_stride={row_stride}",
                )
            )
            module.add(
                VMulLOU32(
                    dst=vgpr(lane_loc),
                    src0=sgpr(tmpSgpr),
                    src1=vgpr(lane_loc),
                    comment="lane * row_stride",
                )
            )
            module.add(
                VAddU32(
                    vgpr(write_addr),
                    vgpr(write_addr),
                    vgpr(lane_loc),
                    comment="write_addr += lane*row_stride",
                )
            )
            module.add(
                VAddU32(
                    vgpr(read_addr),
                    vgpr(read_addr),
                    vgpr(lane_loc),
                    comment="read_addr += lane*row_stride",
                )
            )

        for i in range(self.num_rows):
            module.add(
                DSStoreB32(
                    dstAddr=vgpr(write_addr),
                    src=vgpr(partials + i),
                    ds=DSModifiers(offset=i * 4),
                    comment=f"LDS store partial[{i}]",
                )
            )
        module.add(SWaitCnt(dscnt=0, comment="wait LDS writes"))
        module.add(self.writer._syncThreads(self.kernel, "partialRMS cross-wave write"))

        for j in range(self.wg_n):
            for i in range(self.num_rows):
                module.add(
                    DSLoadB32(
                        dst=vgpr(read_tmp + i),
                        src=vgpr(read_addr),
                        ds=DSModifiers(offset=i * 4),
                        comment=f"LDS load wave[{j}] partial[{i}]",
                    )
                )
            module.add(SWaitCnt(dscnt=0, comment="wait LDS reads"))
            for i in range(self.num_rows):
                if j == 0:
                    module.add(
                        VMovB32(
                            dst=vgpr(partials + i),
                            src=vgpr(read_tmp + i),
                            comment=f"partial[{i}] = wave[0]",
                        )
                    )
                else:
                    module.add(
                        VAddF32(
                            dst=vgpr(partials + i),
                            src0=vgpr(partials + i),
                            src1=vgpr(read_tmp + i),
                            comment=f"partial[{i}] += wave[{j}]",
                        )
                    )
            if j < self.wg_n - 1:
                with self.writer.allocTmpSgpr(1, tag="pRMS_xwAdvance") as tmpSgprInfo:
                    module.add(
                        SMovB32(
                            dst=sgpr(tmpSgprInfo.idx),
                            src=hex(group_stride),
                            comment=f"group_stride={group_stride}",
                        )
                    )
                    module.add(
                        VAddU32(
                            vgpr(read_addr),
                            vgpr(read_addr),
                            sgpr(tmpSgprInfo.idx),
                            comment="advance read_addr to next sibling",
                        )
                    )

        module.add(self.writer._syncThreads(self.kernel, "partialRMS cross-wave done"))

        self.writer.vgprPool.checkIn(tmp_vgpr)
        self.writer.vgprPool.checkIn(read_tmp)
        self.writer.vgprPool.checkIn(read_addr)
        self.writer.vgprPool.checkIn(write_addr)
        self.writer.vgprPool.checkIn(lane_loc)
        self.writer.vgprPool.checkIn(wave_m)
        self.writer.vgprPool.checkIn(wave_id)

        return module

    def _writePartials(
        self,
        partials: int,
        partial_srd: int,
        lane_id: int,
        saved_exec: int,
        lane_mask_sgpr: int,
        global_addr: int,
    ) -> Module:
        """Step 4: write per-row Σx² to global partialBuf (2D layout).

        MFMA row-group layout (16x16 MFMA, wave64):
          - row group g = lane_id // mfma_n  (0..wave_size//mfma_n - 1)
          - Within M-tile m, row groups are interleaved:
              row group g owns rows m*mfma_m + g*rows_per_lane .. m*mfma_m + g*rows_per_lane + rows_per_lane-1

        Each row group selects one writing lane (col_in_mma == 0, i.e., lane_id % mfma_n == 0).
        Writing lane with row group g writes partial[m*rows_per_lane+k] to 2D address:
          byte_off = (global_row * N_tiles_N + tile_col) * 4
          where global_row = WorkGroup0 * MT0 + m*mfma_m + k + row_group*rows_per_lane
                N_tiles_N  = ceil(SizesFree[1] / MT1), computed on device
                tile_col   = sgpr("WorkGroup1") (live named SGPR, no extra allocation)

        Row base is computed directly into global_addr per iteration (no SGPR needed).
        N_tiles_N is moved into a VGPR once to avoid SGPR-src0 restrictions on VMulLOU32.
        """
        module = Module("PartialRMS writePartials")
        module.addComment1(
            "PartialRMS step 4: predicated 2D write of Σx² to partialBuf"
        )
        module.addComment0(
            f"  Writing lanes: lane_id % {self.mfma_n} == 0; 2D addr = (row*NTilesN+tile_col)*4"
        )

        lsc = self.lane_sgpr_count

        # Compute row_group = lane_id // mfma_n (runtime, per-lane).
        row_group = self.writer.vgprPool.checkOut(1, tag="pRMS_rowGroup")
        row_group_off = self.writer.vgprPool.checkOut(1, tag="pRMS_rowGroupOff")
        ntiles_vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_nTilesV")

        # row_group = lane_id >> log2(mfma_n)  (mfma_n must be power of 2).
        log2_mfma_n = int(math.log2(self.mfma_n))
        module.add(
            VLShiftRightB32(
                dst=vgpr(row_group),
                shiftHex=hex(log2_mfma_n),
                src=vgpr(lane_id),
                comment=f"row_group = lane_id >> {log2_mfma_n} (= lane_id // {self.mfma_n})",
            )
        )
        # row_group_off = row_group * rows_per_lane.
        module.add(
            VMulLOU32(
                dst=vgpr(row_group_off),
                src0=self.rows_per_lane,
                src1=vgpr(row_group),
                comment=f"row_group_off = row_group * {self.rows_per_lane}",
            )
        )

        # Compute N_tiles_N = ceil(SizesFree[1] / MT1) into ntiles_vgpr.
        # NTilesN is not loaded as a named SGPR (to avoid SGPR pool pressure).
        log2_mt1 = int(math.log2(self.macro_tile1))
        with self.writer.allocTmpSgpr(1, tag="pRMS_nTilesS") as ntiles_s:
            module.add(
                SAddU32(
                    dst=sgpr(ntiles_s.idx),
                    src0=sgpr("SizesFree+1"),
                    src1=self.macro_tile1 - 1,
                    comment=f"N + MT1-1  (MT1={self.macro_tile1})",
                )
            )
            module.add(
                SLShiftRightB32(
                    dst=sgpr(ntiles_s.idx),
                    shiftHex=hex(log2_mt1),
                    src=sgpr(ntiles_s.idx),
                    comment=f"N_tiles_N = ceil(N / MT1={self.macro_tile1})",
                )
            )
            module.add(
                VMovB32(
                    dst=vgpr(ntiles_vgpr),
                    src=sgpr(ntiles_s.idx),
                    comment="ntiles_vgpr = N_tiles_N",
                )
            )

        # Compute lane mask: active iff lane_id % mfma_n == 0.
        col_in_mma = self.writer.vgprPool.checkOut(1, tag="pRMS_colInMma")
        module.add(
            VAndB32(
                dst=vgpr(col_in_mma),
                src0=vgpr(lane_id),
                src1=self.mfma_n - 1,
                comment="col_in_mma = lane_id % mfma_n",
            )
        )
        module.add(
            VCmpEQU32(
                dst=sgpr(lane_mask_sgpr, lsc),
                src0=0,
                src1=vgpr(col_in_mma),
                comment="lane_mask: lanes where col_in_mma == 0",
            )
        )
        self.writer.vgprPool.checkIn(col_in_mma)

        # Hoist wg_row_base = WorkGroup0 * MT0 once before the exec-narrowing.
        # VMulLOU32 cannot encode large integer literals in src0; move MT0 into a
        # VGPR first so both sources are registers.
        wg_row_base = self.writer.vgprPool.checkOut(1, tag="pRMS_wgRowBase")
        mt0_vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_mt0V")
        module.add(
            VMovB32(
                dst=vgpr(mt0_vgpr),
                src=self.macro_tile0,
                comment=f"MT0={self.macro_tile0} → vgpr for VMulLOU32",
            )
        )
        module.add(
            VMulLOU32(
                dst=vgpr(wg_row_base),
                src0=vgpr(mt0_vgpr),
                src1=sgpr("WorkGroup0"),
                comment=f"wg_row_base = WorkGroup0 * MT0={self.macro_tile0}",
            )
        )
        self.writer.vgprPool.checkIn(mt0_vgpr)

        # Add waveM * mma_m * mfma_m to wg_row_base so each wave addresses its
        # own M-row slice when wg_m > 1.
        if self.wg_m > 1:
            wave_id = self.writer.vgprPool.checkOut(1, tag="pRMS_wpWaveId")
            wave_m = self.writer.vgprPool.checkOut(1, tag="pRMS_wpWaveM")
            tmp_vgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_wpTmp")
            tmp_res = ContinuousRegister(tmp_vgpr, 2)
            module.add(
                vectorStaticDivide(
                    wave_id,
                    "Serial",
                    self.wave_size,
                    tmp_res,
                    comment="waveId = Serial / WavefrontSize",
                )
            )
            module.add(
                VAndB32(
                    dst=vgpr(wave_m),
                    src0=vgpr(wave_id),
                    src1=self.wg_m - 1,
                    comment=f"waveM = waveId %% {self.wg_m} (bitmask, wg_m pow2)",
                )
            )
            wave_stride = self.mma_m * self.mfma_m
            wave_stride_vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_wpWaveStride")
            module.add(
                VMovB32(
                    dst=vgpr(wave_stride_vgpr),
                    src=wave_stride,
                    comment=f"wave_stride = mma_m * mfma_m = {wave_stride}",
                )
            )
            module.add(
                VMulLOU32(
                    dst=vgpr(wave_m),
                    src0=vgpr(wave_stride_vgpr),
                    src1=vgpr(wave_m),
                    comment="waveM_off = waveM * (mma_m * mfma_m)",
                )
            )
            module.add(
                VAddU32(
                    vgpr(wg_row_base),
                    vgpr(wg_row_base),
                    vgpr(wave_m),
                    comment="wg_row_base += waveM * mma_m * mfma_m",
                )
            )
            self.writer.vgprPool.checkIn(wave_stride_vgpr)
            self.writer.vgprPool.checkIn(tmp_vgpr)
            self.writer.vgprPool.checkIn(wave_m)
            self.writer.vgprPool.checkIn(wave_id)

        # Save exec and narrow to writing lanes.
        module.add(
            SAndSaveExecB64(
                dst=sgpr(saved_exec, lsc),
                src=sgpr(lane_mask_sgpr, lsc),
                comment="save exec; set exec = writing-lane mask",
            )
        )

        # Write each partial[m*rows_per_lane+k] to 2D address in partialBuf.
        # 2D address: byte_off = (global_row * N_tiles_N + tile_col) * 4
        # global_row = WorkGroup0 * MT0 + m*mfma_m + k + row_group * rows_per_lane
        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                i = self._partial_idx(m, k)
                m_base = m * self.mfma_m + k
                module.add(
                    VAddU32(
                        vgpr(global_addr),
                        vgpr(wg_row_base),
                        m_base,
                        comment=f"global_row = wg_row_base + {m_base} (m*mfma_m + k)",
                    )
                )
                module.add(
                    VAddU32(
                        vgpr(global_addr),
                        vgpr(global_addr),
                        vgpr(row_group_off),
                        comment="global_row += row_group * rows_per_lane",
                    )
                )
                module.add(
                    VMulLOU32(
                        dst=vgpr(global_addr),
                        src0=vgpr(ntiles_vgpr),
                        src1=vgpr(global_addr),
                        comment="global_row * N_tiles_N",
                    )
                )
                module.add(
                    VAddU32(
                        vgpr(global_addr),
                        vgpr(global_addr),
                        sgpr("WorkGroup1"),
                        comment="+ tile_col = WorkGroup1",
                    )
                )
                module.add(
                    VLShiftLeftB32(
                        dst=vgpr(global_addr),
                        shiftHex=hex(2),
                        src=vgpr(global_addr),
                        comment="byte_off = (row*N_tiles_N + tile_col) * 4",
                    )
                )
                module.add(
                    BufferStoreB32(
                        src=vgpr(partials + i),
                        vaddr=vgpr(global_addr),
                        saddr=sgpr(partial_srd, 4),
                        soffset=0,
                        mubuf=MUBUFModifiers(offen=True),
                        comment=f"partialBuf[row, tile_col] = Σx² (m={m},k={k})",
                    )
                )
        module.add(SWaitCnt(vlcnt=0, comment="wait partialBuf stores"))

        module.add(
            SMovB64(dst=EXEC(), src=sgpr(saved_exec, lsc), comment="restore exec mask")
        )

        self.writer.vgprPool.checkIn(wg_row_base)
        self.writer.vgprPool.checkIn(ntiles_vgpr)
        self.writer.vgprPool.checkIn(row_group_off)
        self.writer.vgprPool.checkIn(row_group)

        return module

    def _applyGammaOnly(
        self,
        accVgprBase: int,
        gamma_srd: int,
        gamma_tmp: int,
        acc_tmp: int,
        col_byte: int,
    ) -> Module:
        """Step 5: load gamma (bf16) and multiply each accumulator element.

        For each MMA N-tile n:
          gamma_fp32 = bf16_to_fp32(gamma[n*mfma_n + lane%mfma_n])
          for (m, k): acc[m, n, k] *= gamma_fp32

        No rstd multiply here; K2 applies rstd. The store path writes D as bf16.
        Gamma byte offset for N-tile n: n * mfma_n * 2 (bf16 element size = 2).
        """
        module = Module("PartialRMS applyGammaOnly")
        module.addComment1("PartialRMS step 5: apply gamma in-place (no rstd)")

        for n in range(self.mma_n):
            mma_base_byte = n * self.mfma_n * 2
            module.add(
                BufferLoadD16B16(
                    vgpr(gamma_tmp),
                    vgpr(col_byte),
                    sgpr(gamma_srd, 4),
                    0,
                    mubuf=MUBUFModifiers(offen=True, offset12=mma_base_byte),
                    comment=f"gamma_bf16[n={n}]",
                )
            )
            module.add(SWaitCnt(vlcnt=0, comment="wait gamma load"))
            module.add(
                VCvtBF16toFP32(
                    vgpr(gamma_tmp),
                    vgpr(gamma_tmp),
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
                            vgpr(acc_tmp),
                            accvgpr(a),
                            comment=f"read acc[m={m},n={n},k={k}]",
                        )
                    )
                    module.add(
                        VMulF32(
                            dst=vgpr(acc_tmp),
                            src0=vgpr(acc_tmp),
                            src1=vgpr(gamma_tmp),
                            comment="acc *= gamma",
                        )
                    )
                    module.add(
                        VAccvgprWriteB32(
                            accvgpr(a),
                            vgpr(acc_tmp),
                            comment=f"write acc[m={m},n={n},k={k}]",
                        )
                    )

        return module
