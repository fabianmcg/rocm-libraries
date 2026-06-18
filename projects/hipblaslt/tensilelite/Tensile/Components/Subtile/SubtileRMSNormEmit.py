# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RMSNorm fused epilogue emitter for the Subtile kernel (gfx950, bf16).

Inserts a row-wise RMSNorm normalization between the GEMM accumulator (fp32)
and the global write. Assumes:
  - UseSubtileImpl=True, ISA=gfx950, DataType=bf16
  - StreamKForceDPOnly=1  (every WG holds a complete tile at the hook)
  - N == MacroTile1        (WG owns full output rows; validated at launch time)

Row reduction is two-stage:
  1. within-wave butterfly across the 16 column-lanes of a single wave
     (`_butterflyReduce`), and
  2. when MIWaveGroup[1] (wg_n) > 1, an LDS-backed reduction across the wg_n
     sibling waves that share the same wave_m (same output rows, different
     N-column slices) (`_crossWaveReduce`).
After both stages every lane holds the full per-row Σx² over the whole N tile.

Alpha/beta ordering note:
  This hook runs on the raw GEMM accumulator *before* the global write path
  applies alpha and beta (notLocalSplitUGlobalWrite → GlobalWriteBatch).  The
  effective output is therefore:

      D = alpha * RMSNorm(A*B) + beta * C

  For the only supported values (alpha=1, beta=0) this is mathematically
  correct.  The host dispatch layer must guarantee these runtime values.

MFMA 16x16 output layout (gfx950, waveSize=64):
  - lane % 16 = N-column within the MMA tile (0..15)
  - lane // 16 = row-group g (0..3); holds rows {4g, 4g+1, 4g+2, 4g+3} within MMA tile

Acc VGPR ordering (localMMATileGrid = [mma_m, mma_n]):
  - Total tiles = mma_m × mma_n
  - linearId = n * mma_m + m  (N-outer, M-inner ordering)
  - Tile (m, n) occupies 4 agprs starting at accVgprBase + linearId * 4

On gfx950 (ArchAccUnifiedRegs), the D-tile accumulator lives in AGPRs (acc registers).
All arithmetic must be done in regular VGPRs. This emitter reads AGPRs to temp VGPRs,
performs the normalization in VGPRs, and writes results back to AGPRs.

For each RMSNorm row (m, k) across all N columns:
  partial[m, k] = sum_{n=0}^{mma_n-1} acc[m, n, k]^2
  rstd[m, k]    = rsqrt(partial[m, k] / N + eps)
  acc[m, n, k] *= rstd[m, k] * gamma[n*16 + lane%16]  (in-place)
"""
import struct

from rocisa.code import Module
from rocisa.container import vgpr, sgpr, accvgpr, MUBUFModifiers, DSModifiers, ContinuousRegister
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadD16B16,
    DSBPermuteB32,
    DSLoadB32,
    DSStoreB32,
    SMovB32,
    SMovB64,
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddF32,
    VAddU32,
    VAndB32,
    VCvtBF16toFP32,
    VFmaF32,
    VLShiftLeftB32,
    VMovB32,
    VMulF32,
    VMulLOU32,
    VRsqF32,
    VXorB32,
)


_MFMA_M        = 16  # MMA instruction M and N dimension
_ROWS_PER_LANE  = 4  # fp32 acc elements per lane per MMA tile


class SubtileRMSNormEmitter:
    """Emit the fused RMSNorm epilogue for the Subtile gfx950 bf16 kernel.

    Reads accumulator values from AGPRs (gfx950 ArchAccUnifiedRegs), performs
    all arithmetic in regular VGPRs, and writes normalized results back to AGPRs.
    """

    def __init__(self, writer, kernel):
        self.writer   = writer
        self.kernel   = kernel
        self.archCaps = writer.states.archCaps

        mt0 = kernel["MacroTile0"]
        mt1 = kernel["MacroTile1"]
        # Per-wave local MMA tile grid: divide global grid by the wave group size.
        # Using the global grid (mt0//16, mt1//16) would over-read AGPRs when
        # MIWaveGroup[i] > 1, because each wave only owns a fraction of the tile.
        wg = kernel["MIWaveGroup"]   # [wg_m, wg_n]
        self.wg_m = wg[0]
        self.wg_n = wg[1]
        self.num_waves = self.wg_m * self.wg_n
        self.wave_size = kernel["WavefrontSize"]
        self.mma_m = (mt0 // _MFMA_M) // wg[0]
        self.mma_n = (mt1 // _MFMA_M) // wg[1]
        self.macro_tile_n = mt1
        self.num_rows = self.mma_m * _ROWS_PER_LANE

    def _acc_idx(self, accVgprBase: int, m: int, n: int, k: int) -> int:
        """AGPR index for acc element at M-tile m, N-tile n, row-offset k."""
        linear_id = n * self.mma_m + m
        return accVgprBase + linear_id * _ROWS_PER_LANE + k

    def _partial_idx(self, m: int, k: int) -> int:
        """Index into partials array for M-tile m, row-offset k."""
        return m * _ROWS_PER_LANE + k

    def emit(self, accVgprBase: int) -> Module:
        """Return the full RMSNorm epilogue module.

        accVgprBase: AGPR index of the first D-tile accumulator.
        The total accumulator count is derived internally as mma_m * mma_n * 4.
        """
        numAccVgpr = self.mma_m * self.mma_n * _ROWS_PER_LANE
        module = Module("RMSNorm epilogue")
        module.addComment1("RMSNorm: fused row-normalization epilogue")
        module.addComment0(
            f"  Acc AGPRs [{accVgprBase}, {accVgprBase+numAccVgpr}), "
            f"mma_m={self.mma_m}, mma_n={self.mma_n}, N={self.macro_tile_n}"
        )

        # Allocate regular VGPRs for all temporaries
        partials    = self.writer.vgprPool.checkOut(self.num_rows, tag="rmsNorm_partials")
        acc_tmp     = self.writer.vgprPool.checkOut(1,             tag="rmsNorm_accTmp")
        gamma_tmp   = self.writer.vgprPool.checkOut(1,             tag="rmsNorm_gammaTmp")
        bfly_tmp    = self.writer.vgprPool.checkOut(1,             tag="rmsNorm_bflyTmp")
        perm_addr   = self.writer.vgprPool.checkOut(1,             tag="rmsNorm_permAddr")
        lane_id     = self.writer.vgprPool.checkOut(1,             tag="rmsNorm_laneId")
        col_byte    = self.writer.vgprPool.checkOut(1,             tag="rmsNorm_colByte")
        inv_n_vgpr  = self.writer.vgprPool.checkOut(1,             tag="rmsNorm_invN")
        eps_vgpr    = self.writer.vgprPool.checkOut(1,             tag="rmsNorm_eps")

        gamma_srd   = self.writer.sgprPool.checkOutAligned(4, 4,   tag="rmsNorm_gammaSrd")

        # Ensure all MFMA results are fully committed to AGPRs before reading them.
        module.add(SWaitCnt(waitAll=True, comment="flush MFMA pipeline before RMSNorm"))
        module.add(self._setup(inv_n_vgpr, eps_vgpr, gamma_srd, lane_id, col_byte))
        module.add(self._squareAndLaneSum(accVgprBase, partials, acc_tmp))
        module.add(self._butterflyReduce(partials, perm_addr, lane_id, bfly_tmp))
        # When N is split across waves (wg_n > 1), each wave only summed its own
        # column slice; combine the wg_n sibling-wave partials through LDS so every
        # lane holds the full per-row Σx² before computing rstd.
        if self.wg_n > 1:
            module.add(self._crossWaveReduce(partials))
        module.add(self._computeRstd(partials, inv_n_vgpr, eps_vgpr))
        module.add(self._loadGammaAndApplyScale(
            accVgprBase, partials, gamma_srd, gamma_tmp, acc_tmp, col_byte, lane_id
        ))

        self.writer.sgprPool.checkIn(gamma_srd)
        self.writer.vgprPool.checkIn(eps_vgpr)
        self.writer.vgprPool.checkIn(inv_n_vgpr)
        self.writer.vgprPool.checkIn(col_byte)
        self.writer.vgprPool.checkIn(lane_id)
        self.writer.vgprPool.checkIn(perm_addr)
        self.writer.vgprPool.checkIn(bfly_tmp)
        self.writer.vgprPool.checkIn(gamma_tmp)
        self.writer.vgprPool.checkIn(acc_tmp)
        self.writer.vgprPool.checkIn(partials)

        return module

    def _setup(self, inv_n_vgpr, eps_vgpr, gamma_srd, lane_id, col_byte) -> Module:
        """Load constants and build the gamma SRD."""
        module = Module("RMSNorm setup")

        # Load 1/N as a compile-time f32 immediate directly into a VGPR.
        inv_n_bits = struct.unpack('I', struct.pack('f', 1.0 / self.macro_tile_n))[0]
        module.add(VMovB32(dst=vgpr(inv_n_vgpr), src=hex(inv_n_bits),
                           comment=f"1/{self.macro_tile_n} as f32 immediate"))
        module.add(VMovB32(dst=vgpr(eps_vgpr), src=sgpr("RMSNormEps"),
                           comment="eps to VGPR"))

        # Wait for late-loaded RMSNorm kernel args (s_load_dwordx2/s_load_dword)
        # to complete. These are issued after Summation_End and must be available.
        module.add(SWaitCnt(kmcnt=0, comment="wait for RMSNormGamma/Eps s_load"))
        module.add(SMovB64(dst=sgpr(gamma_srd, 2), src=sgpr("RMSNormGamma", 2),
                           comment="gamma SRD base"))
        module.add(SMovB32(dst=sgpr(gamma_srd + 2), src="BufferOOB",
                           comment="gamma SRD limit"))
        module.add(SMovB32(dst=sgpr(gamma_srd + 3), src="Srd127_96",
                           comment="gamma SRD flags"))

        wave_size = self.kernel["WavefrontSize"]
        module.add(VAndB32(dst=vgpr(lane_id), src0=vgpr("Serial"), src1=wave_size - 1,
                           comment="lane_id"))
        module.add(VAndB32(dst=vgpr(col_byte), src0=vgpr(lane_id), src1=_MFMA_M - 1,
                           comment="col_in_mma = lane_id % 16"))
        module.add(VLShiftLeftB32(dst=vgpr(col_byte), shiftHex=hex(1), src=vgpr(col_byte),
                                  comment="col_byte = col_in_mma * 2"))

        # When N is split across waves, this wave owns global N columns
        # [wave_n * mma_n * 16, ...).  The gamma global-load offset12 only encodes
        # the local N-tile (n*16); add the wave's column base to col_byte so each
        # wave reads its own gamma slice.  wave_n = Serial / (wave_size * wg_m).
        if self.wg_n > 1:
            wave_n   = self.writer.vgprPool.checkOut(1, tag="rmsNorm_setupWaveN")
            tmp_vgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="rmsNorm_setupTmp")
            tmp_res  = ContinuousRegister(tmp_vgpr, 2)
            module.add(vectorStaticDivide(wave_n, "Serial", wave_size * self.wg_m, tmp_res,
                                          comment=f"wave_n = Serial / {wave_size * self.wg_m}"))
            col_base_bytes = self.mma_n * _MFMA_M * 2
            with self.writer.allocTmpSgpr(1, tag="rmsNorm_setupSgpr") as tmpSgprInfo:
                module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(col_base_bytes),
                                   comment=f"gamma col base bytes ({col_base_bytes})"))
                module.add(VMulLOU32(dst=vgpr(wave_n), src0=sgpr(tmpSgprInfo.idx), src1=vgpr(wave_n),
                                     comment="wave_n * mma_n*16*2"))
            module.add(VAddU32(vgpr(col_byte), vgpr(col_byte), vgpr(wave_n),
                               comment="col_byte += wave column base"))
            self.writer.vgprPool.checkIn(tmp_vgpr)
            self.writer.vgprPool.checkIn(wave_n)

        return module

    def _squareAndLaneSum(self, accVgprBase: int, partials: int, acc_tmp: int) -> Module:
        """Step 1: per-row partial Σx² via AGPR reads.

        For each (m, k): partial[m*4+k] = Σ_n acc[m,n,k]²
        Reads from AGPRs into acc_tmp VGPR for computation.
        """
        module = Module("RMSNorm squareAndLaneSum")
        module.addComment1("RMSNorm step 1: per-row partial sum-of-squares (from AGPRs)")
        for m in range(self.mma_m):
            for k in range(_ROWS_PER_LANE):
                pidx = partials + self._partial_idx(m, k)
                # First n: square into partial
                first = self._acc_idx(accVgprBase, m, 0, k)
                module.add(VAccvgprReadB32(vgpr(acc_tmp), accvgpr(first),
                                           comment=f"read acc[m={m},n=0,k={k}]"))
                module.add(VMulF32(dst=vgpr(pidx), src0=vgpr(acc_tmp), src1=vgpr(acc_tmp),
                                   comment=f"partial[m={m},k={k}] = acc^2"))
                # Subsequent n: fma
                for n in range(1, self.mma_n):
                    a = self._acc_idx(accVgprBase, m, n, k)
                    module.add(VAccvgprReadB32(vgpr(acc_tmp), accvgpr(a),
                                               comment=f"read acc[m={m},n={n},k={k}]"))
                    module.add(VFmaF32(dst=vgpr(pidx),
                                       src0=vgpr(acc_tmp), src1=vgpr(acc_tmp), src2=vgpr(pidx),
                                       comment=f"partial[m={m},k={k}] += acc^2"))
        return module

    def _butterflyReduce(self, partials: int, perm_addr: int, lane_id: int,
                         bfly_tmp: int) -> Module:
        """Step 2: butterfly across 16 column-sharing lanes.

        Fetches partner's partial into bfly_tmp, then adds to own partial.
        """
        module = Module("RMSNorm butterflyReduce")
        module.addComment1("RMSNorm step 2: butterfly sum across 16 column lanes")
        for stride in (8, 4, 2, 1):
            module.addComment0(f"  butterfly stride={stride}")
            module.add(VXorB32(dst=vgpr(perm_addr), src0=vgpr(lane_id), src1=stride,
                               comment=f"partner = lane_id ^ {stride}"))
            module.add(VLShiftLeftB32(dst=vgpr(perm_addr), shiftHex=hex(2), src=vgpr(perm_addr),
                                      comment="partner_byte_addr = partner * 4"))
            for i in range(self.num_rows):
                module.add(DSBPermuteB32(
                    dst=vgpr(bfly_tmp),
                    src0=vgpr(perm_addr),
                    src1=vgpr(partials + i),
                    comment=f"bfly_tmp = partner's partial[{i}]",
                ))
                module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
                module.add(VAddF32(
                    dst=vgpr(partials + i),
                    src0=vgpr(partials + i),
                    src1=vgpr(bfly_tmp),
                    comment=f"partial[{i}] += partner's value",
                ))
        return module

    def _crossWaveReduce(self, partials: int) -> Module:
        """Step 3: sum per-wave partials across the wg_n sibling waves via LDS.

        After the within-wave butterfly, each wave holds Σx² over only its own
        N-column slice. Waves sharing the same wave_m own the same output rows but
        different N columns, so their partials must be summed to obtain the full
        per-row Σx².

        Wave-id convention (matches SubtileGREmit): waveId = waveN*wg_m + waveM,
        so the wg_n siblings of (waveM, waveN) are waveId = j*wg_m + waveM for
        j in [0, wg_n). LDS slots are keyed by (waveId, lane, row), so each lane
        reads back the same lane's row partials from every sibling wave.

        Each lane writes its `num_rows` partials, barriers, then reads and sums
        all wg_n sibling slots (its own slot included, so the sum overwrites the
        local partial without double-counting).
        """
        stride_w     = self.wave_size * self.num_rows * 4   # bytes per wave slot
        row_stride   = self.num_rows * 4                    # bytes per lane
        group_stride = self.wg_m * stride_w                 # bytes between siblings

        module = Module("RMSNorm crossWaveReduce")
        module.addComment1(
            f"RMSNorm step 3: cross-wave sum over wg_n={self.wg_n} waves via LDS"
        )

        # Cross-wave WAR hazard: sibling waves may still be executing their
        # final main-loop local_read from the LDS region we are about to
        # overwrite with cross-wave scratch data.  endSummation / globalWrite-
        # WorkGroupInit contain no unconditional barrier, so we must insert one
        # here before the first DSStoreB32.
        module.add(self.writer._syncThreads(
            self.kernel, "rmsnorm cross-wave: ensure siblings done reading LDS before scratch write"
        ))

        wave_id   = self.writer.vgprPool.checkOut(1, tag="rmsNorm_xwWaveId")
        wave_m    = self.writer.vgprPool.checkOut(1, tag="rmsNorm_xwWaveM")
        lane_loc  = self.writer.vgprPool.checkOut(1, tag="rmsNorm_xwLane")
        write_addr= self.writer.vgprPool.checkOut(1, tag="rmsNorm_xwWriteAddr")
        read_addr = self.writer.vgprPool.checkOut(1, tag="rmsNorm_xwReadAddr")
        read_tmp  = self.writer.vgprPool.checkOut(self.num_rows, tag="rmsNorm_xwReadTmp")
        tmp_vgpr  = self.writer.vgprPool.checkOutAligned(2, 2, tag="rmsNorm_xwTmp")
        tmp_res   = ContinuousRegister(tmp_vgpr, 2)

        module.add(VAndB32(dst=vgpr(lane_loc), src0=vgpr("Serial"), src1=self.wave_size - 1,
                           comment="lane_id"))
        module.add(vectorStaticDivide(wave_id, "Serial", self.wave_size, tmp_res,
                                      comment="waveId = Serial / WavefrontSize"))
        # wg_m - 1 bitmask is valid because _validateRMSNorm in Solution.py
        # rejects non-power-of-two MIWaveGroup[0] when wg_n > 1.
        module.add(VAndB32(dst=vgpr(wave_m), src0=vgpr(wave_id), src1=self.wg_m - 1,
                           comment=f"waveM = waveId %% {self.wg_m} (wg_m is pow2, bitmask valid)"))

        with self.writer.allocTmpSgpr(1, tag="rmsNorm_xwSgpr") as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            # write_addr = waveId*stride_w + lane*row_stride
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(stride_w),
                               comment=f"stride_w ({stride_w})"))
            module.add(VMulLOU32(dst=vgpr(write_addr), src0=sgpr(tmpSgpr), src1=vgpr(wave_id),
                                 comment="write_addr = waveId * stride_w"))
            # read_base = waveM*stride_w + lane*row_stride (waveN folded into DS loop)
            module.add(VMulLOU32(dst=vgpr(read_addr), src0=sgpr(tmpSgpr), src1=vgpr(wave_m),
                                 comment="read_addr = waveM * stride_w"))
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(row_stride),
                               comment=f"row_stride ({row_stride})"))
            module.add(VMulLOU32(dst=vgpr(lane_loc), src0=sgpr(tmpSgpr), src1=vgpr(lane_loc),
                                 comment="lane * row_stride"))
            module.add(VAddU32(vgpr(write_addr), vgpr(write_addr), vgpr(lane_loc),
                               comment="write_addr += lane*row_stride"))
            module.add(VAddU32(vgpr(read_addr), vgpr(read_addr), vgpr(lane_loc),
                               comment="read_addr += lane*row_stride"))

            # Write own partials to LDS.
            for i in range(self.num_rows):
                module.add(DSStoreB32(dstAddr=vgpr(write_addr), src=vgpr(partials + i),
                                      ds=DSModifiers(offset=i * 4),
                                      comment=f"LDS store partial[{i}]"))
            module.add(SWaitCnt(dscnt=0, comment="wait for LDS writes"))
            module.add(self.writer._syncThreads(self.kernel, "rmsnorm cross-wave write"))

            # Read+accumulate every sibling slot (own slot included → exact sum).
            for j in range(self.wg_n):
                for i in range(self.num_rows):
                    module.add(DSLoadB32(dst=vgpr(read_tmp + i), src=vgpr(read_addr),
                                         ds=DSModifiers(offset=i * 4),
                                         comment=f"LDS load wave[j={j}] partial[{i}]"))
                module.add(SWaitCnt(dscnt=0, comment="wait for LDS reads"))
                for i in range(self.num_rows):
                    if j == 0:
                        module.add(VMovB32(dst=vgpr(partials + i), src=vgpr(read_tmp + i),
                                           comment=f"partial[{i}] = wave[0]"))
                    else:
                        module.add(VAddF32(dst=vgpr(partials + i), src0=vgpr(partials + i),
                                           src1=vgpr(read_tmp + i),
                                           comment=f"partial[{i}] += wave[{j}]"))
                if j < self.wg_n - 1:
                    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(group_stride),
                                       comment=f"group_stride ({group_stride})"))
                    module.add(VAddU32(vgpr(read_addr), vgpr(read_addr), sgpr(tmpSgpr),
                                       comment="advance read_addr to next sibling wave"))

        # Barrier so LDS is safe for the downstream global-write path to reuse.
        module.add(self.writer._syncThreads(self.kernel, "rmsnorm cross-wave done"))

        self.writer.vgprPool.checkIn(tmp_vgpr)
        self.writer.vgprPool.checkIn(read_tmp)
        self.writer.vgprPool.checkIn(read_addr)
        self.writer.vgprPool.checkIn(write_addr)
        self.writer.vgprPool.checkIn(lane_loc)
        self.writer.vgprPool.checkIn(wave_m)
        self.writer.vgprPool.checkIn(wave_id)

        return module

    def _computeRstd(self, partials: int, inv_n_vgpr: int, eps_vgpr: int) -> Module:
        """Step 4: rstd = rsqrt(sum_sq / N + eps), stored in-place in partials."""
        module = Module("RMSNorm computeRstd")
        module.addComment1(
            f"RMSNorm step 4: rstd = rsqrt(sum_sq / {self.macro_tile_n} + eps)"
        )
        for i in range(self.num_rows):
            module.add(VMulF32(dst=vgpr(partials + i), src0=vgpr(partials + i),
                               src1=vgpr(inv_n_vgpr), comment=f"partial[{i}] /= N"))
            module.add(VAddF32(dst=vgpr(partials + i), src0=vgpr(partials + i),
                               src1=vgpr(eps_vgpr), comment=f"partial[{i}] += eps"))
            module.add(VRsqF32(dst=vgpr(partials + i), src=vgpr(partials + i),
                               comment=f"rstd[{i}] = rsqrt"))
            if self.archCaps.get("TransOpWait"):
                module.add(SNop(waitState=0, comment="1 wait state after v_rsq_f32"))
        return module

    def _loadGammaAndApplyScale(
        self,
        accVgprBase: int,
        rstd: int,
        gamma_srd: int,
        gamma_tmp: int,
        acc_tmp: int,
        col_byte: int,
        lane_id: int,
    ) -> Module:
        """Steps 5+6: load gamma per N-tile, read AGPR, scale, write back to AGPR."""
        module = Module("RMSNorm loadGammaAndApplyScale")
        module.addComment1("RMSNorm steps 5+6: load gamma and scale (read/write AGPRs)")

        for n in range(self.mma_n):
            mma_base_byte = n * _MFMA_M * 2
            module.add(BufferLoadD16B16(
                vgpr(gamma_tmp), vgpr(col_byte), sgpr(gamma_srd, 4), 0,
                mubuf=MUBUFModifiers(offen=True, offset12=mma_base_byte),
                comment=f"gamma_bf16[n={n}]",
            ))
            module.add(SWaitCnt(vlcnt=0, comment="wait for gamma"))
            module.add(VCvtBF16toFP32(vgpr(gamma_tmp), vgpr(gamma_tmp), None, 0,
                                       comment="gamma bf16 → fp32"))
            for m in range(self.mma_m):
                for k in range(_ROWS_PER_LANE):
                    a    = self._acc_idx(accVgprBase, m, n, k)
                    ridx = rstd + self._partial_idx(m, k)
                    # Read from AGPR
                    module.add(VAccvgprReadB32(vgpr(acc_tmp), accvgpr(a),
                                               comment=f"read acc[m={m},n={n},k={k}]"))
                    # Scale: acc *= rstd * gamma
                    module.add(VMulF32(dst=vgpr(acc_tmp), src0=vgpr(acc_tmp), src1=vgpr(ridx),
                                       comment="acc *= rstd"))
                    module.add(VMulF32(dst=vgpr(acc_tmp), src0=vgpr(acc_tmp), src1=vgpr(gamma_tmp),
                                       comment="acc *= gamma"))
                    # Write back to AGPR
                    module.add(VAccvgprWriteB32(accvgpr(a), vgpr(acc_tmp),
                                                comment=f"write acc[m={m},n={n},k={k}]"))

        return module
