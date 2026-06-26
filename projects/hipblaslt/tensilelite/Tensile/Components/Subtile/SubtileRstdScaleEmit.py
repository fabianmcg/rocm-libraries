# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RstdScale fused epilogue emitter for the Subtile kernel (gfx950, bf16).

Applies a pre-computed per-row rstd scalar to the fp32 GEMM accumulator (AGPRs).
This is Phase 3 (K3) of the two-kernel RMSNorm pipeline:
  - K1 (PartialRMS): computes raw Σx² and applies gamma, writes h2 as bf16.
  - K2 (aux reduction): reads partialBuf, computes rstd = rsqrt(Σx²/N + eps).
  - K3 (this kernel, RstdScale): performs GEMM2 (h2 @ W1), then for each output
    row multiplies every accumulator element by rstd[row] loaded from rstdBuf.

No reduction, no LDS, no butterfly — simplest of the three emitters.

rstdBuf layout contract:
  - One fp32 per output row, indexed by GLOBAL row m.
  - WG with WorkGroup0 = t reads rows [t*MT0, t*MT0 + MT0).

Key property: every lane loads the SAME row's rstd value (broadcast load).
partial_idx(m, k) spans all num_rows rows regardless of which N-column lane
covers. The byte offset is (m_tile_start + m*rows_per_lane + k) * 4 with NO
lane_id // mfma_n term.

MFMA layout (gfx950, waveSize=64, 16x16 MFMA):
  - lane % mfma_n = N-column within MMA tile
  - rows_per_lane = (mfma_m * mfma_n) // wave_size

Acc VGPR ordering (N-outer, M-inner):
  acc_idx(base, m, n, k) = base + (n*mma_m + m)*rows_per_lane + k

Alpha=1, beta=0 must be passed by the host.
"""

from rocisa.code import Module
from rocisa.container import (
    MUBUFModifiers,
    accvgpr,
    sgpr,
    vgpr,
)
from rocisa.instruction import (
    BufferLoadB32,
    SMovB32,
    SMovB64,
    SMulI32,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddU32,
    VAndB32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMovB32,
    VMulF32,
    VMulLOU32,
)


class SubtileRstdScaleEmitter:
    """Emit the RstdScale epilogue for the Subtile gfx950 bf16 kernel.

    Loads per-row rstd from rstdBuf (fp32) and multiplies every accumulator
    element of that row in-place. No reduction, no LDS.
    """

    def __init__(self, writer, kernel):
        self.writer   = writer
        self.kernel   = kernel
        self.archCaps = writer.states.archCaps

        # Derive all geometry from kernel params; no module-level constants.
        self.mfma_m        = kernel["MatrixInstM"]
        self.mfma_n        = kernel["MatrixInstN"]
        self.wave_size     = kernel["WavefrontSize"]
        self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.wave_size

        wg = kernel["MIWaveGroup"]
        self.wg_m, self.wg_n = wg[0], wg[1]

        self.mma_m       = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
        self.mma_n       = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
        self.macro_tile0 = kernel["MacroTile0"]
        self.num_rows    = self.mma_m * self.rows_per_lane

    def _acc_idx(self, base: int, m: int, n: int, k: int) -> int:
        """AGPR index for accumulator element at M-tile m, N-tile n, row-offset k."""
        return base + (n * self.mma_m + m) * self.rows_per_lane + k

    def _partial_idx(self, m: int, k: int) -> int:
        """Index into rstd_vgprs for M-tile m, row-offset k."""
        return m * self.rows_per_lane + k

    def _setup(self, rstd_srd: int, row_base_sgpr: int, lane_id: int,
               row_group: int, row_group_byte: int) -> Module:
        """Build rstdBuf SRD, compute m_tile_start, derive lane_id and row_group.

        row_group = lane_id // mfma_n (number of the row-group this lane belongs to).
        row_group_byte = row_group * rows_per_lane * 4 (byte contribution of g to address).

        Signature append order:
          slot N+0: RstdBuf (fp32 global buffer pointer, 8 bytes)
        """
        import math as _math
        module = Module("RstdScale setup")
        module.add(SWaitCnt(kmcnt=0, comment="wait for RstdScale kernarg s_load"))

        # RstdBuf SRD (fp32 global buffer).
        module.add(SMovB64(dst=sgpr(rstd_srd, 2), src=sgpr("RstdBuf", 2),
                           comment="rstdBuf SRD base"))
        module.add(SMovB32(dst=sgpr(rstd_srd + 2), src="BufferOOB",
                           comment="rstdBuf SRD limit"))
        module.add(SMovB32(dst=sgpr(rstd_srd + 3), src="Srd127_96",
                           comment="rstdBuf SRD flags"))

        # row_base_sgpr = WorkGroup0 * MT0.
        module.add(SMulI32(
            dst=sgpr(row_base_sgpr),
            src0=sgpr("WorkGroup0"),
            src1=self.macro_tile0,
            comment=f"row_base = WorkGroup0 * MT0={self.macro_tile0}",
        ))

        # lane_id = Serial & (wave_size - 1).
        module.add(VAndB32(dst=vgpr(lane_id), src0=vgpr("Serial"), src1=self.wave_size - 1,
                           comment="lane_id = Serial & (wave_size-1)"))

        # row_group = lane_id >> log2(mfma_n)  (mfma_n must be power of two).
        log2_mfma_n = int(_math.log2(self.mfma_n))
        module.add(VLShiftRightB32(
            dst=vgpr(row_group),
            shiftHex=hex(log2_mfma_n),
            src=vgpr(lane_id),
            comment=f"row_group = lane_id >> {log2_mfma_n} (= lane_id // {self.mfma_n})",
        ))

        # row_group_byte = row_group * rows_per_lane * 4.
        module.add(VMulLOU32(
            dst=vgpr(row_group_byte),
            src0=self.rows_per_lane * 4,
            src1=vgpr(row_group),
            comment=f"row_group_byte = row_group * {self.rows_per_lane * 4}",
        ))

        return module

    def _loadRstd(self, rstd_srd: int, row_base_sgpr: int, row_group_byte: int,
                  rstd_vgprs: int) -> Module:
        """Load per-row rstd from rstdBuf into rstd_vgprs.

        Global row for (m, k) = row_base + m * mfma_m + g * rows_per_lane + k,
        where g = row_group (runtime, derived from lane_id).

        All N-column lanes in the same row group share the same g, so the rstd
        load is effectively "broadcast" across the mfma_n N-column lanes.
        """
        module = Module("RstdScale loadRstd")
        module.addComment1("RstdScale: load per-row rstd from rstdBuf")

        global_addr = self.writer.vgprPool.checkOut(1, tag="rstdScale_globalAddr")

        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                i = self._partial_idx(m, k)
                # Byte offset = (row_base + m*mfma_m + g*rows_per_lane + k) * 4.
                # m*mfma_m + k is the compile-time part; g*rows_per_lane is runtime.
                m_k_offset = m * self.mfma_m + k
                module.add(VMovB32(dst=vgpr(global_addr), src=sgpr(row_base_sgpr),
                                   comment=f"global_addr = row_base for rstd[{i}]"))
                module.add(VAddU32(vgpr(global_addr), vgpr(global_addr), m_k_offset,
                                   comment=f"global_addr += {m_k_offset} (m*mfma_m + k)"))
                # Multiply by 4 before adding row_group_byte (already in bytes).
                module.add(VLShiftLeftB32(dst=vgpr(global_addr), shiftHex=hex(2),
                                          src=vgpr(global_addr),
                                          comment="global_addr_bytes = row_int * 4"))
                module.add(VAddU32(vgpr(global_addr), vgpr(global_addr), vgpr(row_group_byte),
                                   comment="global_addr_bytes += g * rows_per_lane * 4"))
                module.add(BufferLoadB32(
                    dst=vgpr(rstd_vgprs + i),
                    vaddr=vgpr(global_addr),
                    saddr=sgpr(rstd_srd, 4),
                    soffset=0,
                    mubuf=MUBUFModifiers(offen=True),
                    comment=f"rstd[m={m},k={k}] = rstdBuf[row_base+m*mfma_m+g*rpl+k]",
                ))

        module.add(SWaitCnt(vlcnt=0, comment="wait for all rstd loads"))

        self.writer.vgprPool.checkIn(global_addr)
        return module

    def _applyScale(self, accVgprBase: int, rstd_vgprs: int) -> Module:
        """Multiply every accumulator element by the corresponding rstd value.

        For each (m, n, k): acc[m,n,k] *= rstd[partial_idx(m,k)]
        """
        module = Module("RstdScale applyScale")
        module.addComment1("RstdScale: multiply each acc element by rstd[row]")

        acc_tmp = self.writer.vgprPool.checkOut(1, tag="rstdScale_accTmp")

        for m in range(self.mma_m):
            for n in range(self.mma_n):
                for k in range(self.rows_per_lane):
                    a    = self._acc_idx(accVgprBase, m, n, k)
                    ridx = rstd_vgprs + self._partial_idx(m, k)
                    module.add(VAccvgprReadB32(vgpr(acc_tmp), accvgpr(a),
                                               comment=f"read acc[m={m},n={n},k={k}]"))
                    module.add(VMulF32(dst=vgpr(acc_tmp), src0=vgpr(acc_tmp),
                                       src1=vgpr(ridx),
                                       comment=f"acc *= rstd[m={m},k={k}]"))
                    module.add(VAccvgprWriteB32(accvgpr(a), vgpr(acc_tmp),
                                                comment=f"write acc[m={m},n={n},k={k}]"))

        self.writer.vgprPool.checkIn(acc_tmp)
        return module

    def emit(self, accVgprBase: int) -> Module:
        """Return the full RstdScale epilogue module.

        accVgprBase: AGPR index of the first D-tile accumulator.
        """
        module = Module("RstdScale epilogue")
        module.addComment1("RstdScale: per-row rstd scale epilogue")
        module.addComment0(
            f"  mma_m={self.mma_m}, mma_n={self.mma_n}, "
            f"MT0={self.macro_tile0}, num_rows={self.num_rows}"
        )

        # Allocate VGPRs for rstd values (one per row owned by this wave).
        rstd_vgprs    = self.writer.vgprPool.checkOut(self.num_rows, tag="rstdScale_rstdVgprs")
        lane_id       = self.writer.vgprPool.checkOut(1,             tag="rstdScale_laneId")
        row_group     = self.writer.vgprPool.checkOut(1,             tag="rstdScale_rowGroup")
        row_group_byte = self.writer.vgprPool.checkOut(1,            tag="rstdScale_rowGroupByte")

        # Allocate SGPRs: rstdBuf SRD (4-aligned) + row base.
        rstd_srd      = self.writer.sgprPool.checkOutAligned(4, 4, tag="rstdScale_rstdSrd")
        row_base_sgpr = self.writer.sgprPool.checkOut(1,           tag="rstdScale_rowBase")

        # Flush MFMA pipeline before reading AGPRs.
        module.add(SWaitCnt(waitAll=True, comment="flush MFMA pipeline before RstdScale"))

        module.add(self._setup(rstd_srd, row_base_sgpr, lane_id, row_group, row_group_byte))
        module.add(self._loadRstd(rstd_srd, row_base_sgpr, row_group_byte, rstd_vgprs))
        module.add(self._applyScale(accVgprBase, rstd_vgprs))

        self.writer.sgprPool.checkIn(row_base_sgpr)
        self.writer.sgprPool.checkIn(rstd_srd)
        self.writer.vgprPool.checkIn(row_group_byte)
        self.writer.vgprPool.checkIn(row_group)
        self.writer.vgprPool.checkIn(lane_id)
        self.writer.vgprPool.checkIn(rstd_vgprs)

        return module
