# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DeepseekScale fused epilogue emitter for the Subtile kernel (gfx950, fp8 in / f32 out).

Applies per-row fp32 scaleA from scaleA[M, 1] and/or per-128col-block fp32
scaleB from scaleB[1, ceil(N/128)] to the fp32 GEMM accumulator (AGPRs).
This implements the Deepseek V3 dequantization pattern restricted to a single
K-block (DepthU <= DeepseekScaleBlockK).

Full formula:
  D[m,n] = alpha * scaleA[m] * scaleB[n//128] * (A_fp8 @ B_fp8)[m,n] + beta*C

If only scaleA: scaleB factor is omitted.
If only scaleB: scaleA factor is omitted.

scaleA layout: one fp32 per output row, indexed by global row m.
scaleB layout: one fp32 per 128-column N-block, indexed by global block n//128.

MFMA output accumulator layout (gfx950, waveSize=64, 16x16 MFMA):
  lane % mfma_n = N-column within MMA tile.
  lane // mfma_n = row_group (which set of rows this lane owns).
  rows_per_lane = (mfma_m * mfma_n) // wave_size.

Acc VGPR ordering (N-outer, M-inner):
  acc_idx(base, m, n, k) = base + (n*mma_m + m)*rows_per_lane + k.

MFMA A-input lane distribution (strided M, standard AMD ISA):
  A-input row for lane l = l % mfma_m.
  All mfma_m groups of (wave_size / mfma_m) lanes share K-segment (l // mfma_m).
  For v_mfma_scale: scale_a[l] must equal scaleA[l % mfma_m, kb].
  The single-K epilogue uses the accumulator row_group (l // mfma_n) instead,
  because it applies scale post-MFMA to the output accumulator elements.

N-block assignment (compile-time): each accumulator tile position n maps to
  local_n_block = (n * mfma_n) // 128.
For all existing configs (MT1=128, mma_n*mfma_n<=128), local_n_block=0 always,
so every wave uses a single scaleB value determined by WorkGroup1.
"""

import math as _math

from rocisa.code import Module
from rocisa.container import (
    ContinuousRegister,
    MUBUFModifiers,
    VCC,
    accvgpr,
    sgpr,
    vgpr,
)
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadU8,
    FlatLoadD16U8,
    SAddU32,
    SLoadB64,
    SMulI32,
    SMovB32,
    SMovB64,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddCCOU32,
    VAddCOU32,
    VAddU32,
    VAndB32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMovB32,
    VMulF32,
    VMulLOU32,
    VReadfirstlaneB32,
)


def _computeWaveBase(module, writer, kernel, mma_m, mfma_m, wg_m, macro_tile0, wave_size):
    """Return an SGPR holding WG0*MT0 + waveM*mma_m*mfma_m. Caller frees it."""
    wave_base = writer.sgprPool.checkOut(1, tag="dml_waveBase")
    module.add(SMulI32(dst=sgpr(wave_base), src0=sgpr("WorkGroup0"), src1=macro_tile0,
                       comment=f"waveBase = WorkGroup0 * {macro_tile0}."))
    if wg_m == 1:
        return wave_base
    wave_m_vgpr = writer.vgprPool.checkOut(1, tag="dml_waveM")
    log2ws = int(_math.log2(wave_size))
    module.add(VLShiftRightB32(dst=vgpr(wave_m_vgpr), shiftHex=hex(log2ws),
                               src=vgpr("Serial"), comment="waveId = Serial >> log2(waveSize)."))
    module.add(VAndB32(dst=vgpr(wave_m_vgpr), src0=vgpr(wave_m_vgpr), src1=wg_m - 1,
                       comment=f"waveM = waveId & {wg_m - 1}."))
    wave_m_sgpr = writer.sgprPool.checkOut(1, tag="dml_waveMSgpr")
    module.add(VReadfirstlaneB32(dst=sgpr(wave_m_sgpr), src=vgpr(wave_m_vgpr),
                                 comment="waveM scalar."))
    writer.vgprPool.checkIn(wave_m_vgpr)
    wave_stride = mma_m * mfma_m
    module.add(SMulI32(dst=sgpr(wave_m_sgpr), src0=sgpr(wave_m_sgpr), src1=wave_stride,
                       comment=f"waveMOff = waveM * {wave_stride}."))
    module.add(SAddU32(dst=sgpr(wave_base), src0=sgpr(wave_base), src1=sgpr(wave_m_sgpr),
                       comment="waveBase += waveMOff."))
    writer.sgprPool.checkIn(wave_m_sgpr)
    return wave_base


def _computeScaleAVaddrs(module, writer, kernel, dml, mma_m, mfma_m, wg_m, macro_tile0, wave_size):
    """Precompute per-lane scaleA vaddr VGPRs (row * nKBlocks) for each mma0.

    Scale is E8M0: 1 byte per (row, K-block) entry, so the byte offset is
    row * nKBlocks + kbOffset (kbOffset advances by 1 each iteration).
    """
    # nKBlocks = LoopCounterL (PGR=0: LoopCounterL == nKBlocks initially).
    nkb_vgpr = writer.vgprPool.checkOut(1, tag="dml_nkb")
    module.add(VMovB32(dst=vgpr(nkb_vgpr), src=sgpr("LoopCounterL"),
                       comment="nKBlocks = LoopCounterL (E8M0: 1 byte per K-block)."))
    # AMD MFMA uses strided M-lane distribution: lane l feeds A-input row l % mfma_m.
    # For mfma_m=16: A-row = lane & 0xF, giving 16 rows × 4 K-segments across 64 lanes.
    lane_row = writer.vgprPool.checkOut(1, tag="dml_laneRow")
    module.add(VAndB32(dst=vgpr(lane_row), src0=vgpr("Serial"), src1=mfma_m - 1,
                       comment=f"lane_row = lane_id % {mfma_m} (A-input row, strided MFMA distribution)."))
    wave_base = _computeWaveBase(module, writer, kernel, mma_m, mfma_m, wg_m, macro_tile0, wave_size)
    tmp_row = writer.vgprPool.checkOut(1, tag="dml_rowTmp")
    vaddrs = dml["scaleAVaddrs"]
    for i in range(mma_m):
        # row_i[lane] = (waveBase + i*mfma_m) + lane_row[lane].
        module.add(VAddU32(vgpr(tmp_row), sgpr(wave_base), vgpr(lane_row),
                           comment=f"row_{i} = waveBase + lane_row."))
        if i > 0:
            module.add(VAddU32(vgpr(tmp_row), vgpr(tmp_row), i * mfma_m,
                               comment=f"row_{i} += {i * mfma_m} (mma0 offset)."))
        module.add(VMulLOU32(vgpr(vaddrs + i), vgpr(tmp_row), vgpr(nkb_vgpr),
                             comment=f"scaleAVaddr[{i}] = row_{i} * nKBlocks (byte stride)."))
    writer.sgprPool.checkIn(wave_base)
    writer.vgprPool.checkIn(tmp_row)
    writer.vgprPool.checkIn(lane_row)
    writer.vgprPool.checkIn(nkb_vgpr)


def _scaleBufKernArgOffsets(writer):
    """Return (offset_a_or_None, offset_b_or_None) byte offsets of ScaleABuf/ScaleBBuf.

    Byte offsets are relative to the per-GEMM kernel arg base (KernArgAddress after
    the common-args shift), computed by walking numStoreSgprNames from the argLoader
    current position.
    """
    base = writer.argLoader.getOffset()
    names = writer.states.numStoreSgprNames
    sizes = writer.states.numStoreSgprNameSizes
    off_a = off_b = None
    cur = base
    for name, size in zip(names, sizes):
        if name == "ScaleABuf":
            off_a = cur
        elif name == "ScaleBBuf":
            off_b = cur
        cur += size * 4
    return off_a, off_b


def setupDeepseekMainloopScale(module, writer, kernel, mma_m):
    """Allocate scale VGPRs/SGPRs and precompute per-lane scaleA vaddrs.

    Uses flat_load (no buffer SRD) to avoid SGPR pressure in StreamK kernels.
    Emits preloop setup code into module and stores VGPR/SGPR indices in
    kernel["_deepseekML"] for use by emitDeepseekScaleGR and emit_mfma.
    Called from mainLoop() only when PrefetchGlobalRead == 0.

    kbBOffset and nNBlocksStride are VGPRs (not SGPRs) to keep SGPR pressure low.
    A single shared 2-SGPR pair (sharedBufPtrSgpr) is loaded each iteration with
    the ScaleABuf or ScaleBBuf pointer immediately before the corresponding flat_loads.
    The post-loop store-SGPR symbols (sgprScaleABuf, sgprScaleBBuf) cannot be
    forward-referenced from the mainloop, so we load pointers directly from the
    kernel arg area using KernArgAddress.
    """
    mfma_m    = kernel["MatrixInstM"]
    wg_m      = kernel["MIWaveGroup"][0]
    mt0       = kernel["MacroTile0"]
    mt1       = kernel["MacroTile1"]
    wave_size = kernel["WavefrontSize"]
    use_a     = kernel.get("UseDeepseekScaleA", False)
    use_b     = kernel.get("UseDeepseekScaleB", False)
    dml = {}

    # Compute scale buffer kernel arg offsets now (before allocations shift state).
    off_a, off_b = _scaleBufKernArgOffsets(writer)
    assert not use_a or off_a is not None, "ScaleABuf not found in numStoreSgprNames"
    assert not use_b or off_b is not None, "ScaleBBuf not found in numStoreSgprNames"
    dml["scaleBufOffsets"] = (off_a, off_b)

    # Unit E8M0 = 1.0 (byte 0x7f in all 4 positions): neutral scale for inactive side.
    # v_mfma_scale uses bytes 0-3 for the 4 sub-K-blocks of the 128-element K-block;
    # all 4 bytes must hold 0x7f so every sub-K-block scales by 1.0.
    unit_e8m0 = writer.vgprPool.checkOut(1)
    module.add(VMovB32(dst=vgpr(unit_e8m0), src=hex(0x7f7f7f7f),
                       comment="unit E8M0 = 1.0 in all 4 byte lanes for Deepseek mainloop MFMA."))
    dml["unitE8m0"] = unit_e8m0

    if use_a:
        dml["scaleAVaddrs"] = writer.vgprPool.checkOut(mma_m)
        dml["scaleAVgprs"]  = writer.vgprPool.checkOut(mma_m)
        kb_off = writer.sgprPool.checkOut(1)
        module.add(SMovB32(dst=sgpr(kb_off), src=0, comment="kbOffset = 0."))
        dml["kbOffset"] = kb_off
        _computeScaleAVaddrs(module, writer, kernel, dml, mma_m, mfma_m, wg_m, mt0, wave_size)

    if use_b:
        dml["scaleBVaddr"] = writer.vgprPool.checkOut(1)
        dml["scaleBVgpr"]  = writer.vgprPool.checkOut(1)
        # kbBOffset and nNBlocksStride live in VGPRs to avoid consuming scarce SGPRs.
        kb_b_off = writer.vgprPool.checkOut(1)
        module.add(VMovB32(dst=vgpr(kb_b_off), src=0, comment="kbBOffset = 0."))
        dml["kbBOffset"] = kb_b_off
        nn_stride = writer.vgprPool.checkOut(1)
        module.add(VMovB32(dst=vgpr(nn_stride), src=sgpr("SizesFree+1"),
                           comment="nNBlocksStride = N."))
        module.add(VLShiftRightB32(dst=vgpr(nn_stride), shiftHex=hex(7), src=vgpr(nn_stride),
                                   comment="nNBlocksStride = N >> 7 (= N/128, 1 byte per N-block)."))
        dml["nNBlocksStride"] = nn_stride
        # scaleBVaddr_base: byte offset for WG1 * mt1_blocks (E8M0: 1 byte per N-block).
        mt1_blocks = mt1 // 128
        module.add(VMovB32(dst=vgpr(dml["scaleBVaddr"]), src=sgpr("WorkGroup1"),
                           comment="scaleBVaddr = WorkGroup1."))
        if mt1_blocks != 1:
            module.add(VMulLOU32(dst=vgpr(dml["scaleBVaddr"]),
                                 src0=vgpr(dml["scaleBVaddr"]), src1=mt1_blocks,
                                 comment=f"scaleBVaddr *= {mt1_blocks} (N-block index)."))

    # Allocate a shared 2-SGPR pair for the scale buffer pointer. It is reloaded
    # each iteration with the ScaleABuf or ScaleBBuf address immediately before use,
    # so only one pair is needed even when both scales are active.
    shared_ptr = writer.sgprPool.checkOutAligned(2, 2, tag="dml_sharedBufPtr")
    dml["sharedBufPtrSgpr"] = shared_ptr

    kernel["_deepseekML"] = dml


def _emitFlatLoad(module, writer, base_lo_sgpr, base_hi_sgpr, offset_vgpr, extra_reg, dst_vgpr, comment):
    """Emit a flat_load_d16_u8 to read one E8M0 byte into bits [7:0] of dst_vgpr.

    base_lo/hi_sgpr: 64-bit base pointer split across two SGPRs.
    offset_vgpr: 32-bit per-lane byte offset VGPR index (pre-computed).
    extra_reg: rocisa register expression (sgpr(...) or vgpr(...)) for a uniform
               addend such as kbOffset or kbBOffset, or None if unused.
    dst_vgpr: destination VGPR index for the loaded E8M0 byte (byte in bits [7:0]).
    Allocates a 2-VGPR even-aligned pair (flat_load requires aligned pairs on GFX9).
    Scalar (SGPR) extra_reg must go as src0 in VOP2 to satisfy constant bus restrictions.
    Carry propagation uses a VMovB32 before VAddCCOU32 to avoid SGPR+VCC in one instruction.
    Callers must broadcast byte 0 to all 4 byte positions after the wait, because
    v_mfma_scale uses bytes 0-3 for the 4 sub-K-blocks of the 128-element K-block.
    """
    # Flat load requires an even-aligned VGPR pair for the address.
    addr_pair = writer.vgprPool.checkOutAligned(2, 2, tag="dml_addrPair")
    tmp_lo = addr_pair
    tmp_hi = addr_pair + 1
    # addr_lo = base_lo + offset; sgpr(base_lo) as src0 satisfies the constant bus rule.
    module.add(VAddCOU32(vgpr(tmp_lo), VCC(), sgpr(base_lo_sgpr), vgpr(offset_vgpr),
                         comment=f"{comment}: addr_lo = base + offset."))
    # addr_hi = base_hi + carry. Move base_hi to VGPR first: reading an SGPR and VCC
    # in the same VAddCCOU32 instruction violates the constant bus restriction.
    module.add(VMovB32(vgpr(tmp_hi), sgpr(base_hi_sgpr),
                       comment=f"{comment}: addr_hi = base_hi."))
    module.add(VAddCCOU32(vgpr(tmp_hi), VCC(), vgpr(tmp_hi), 0, VCC(),
                          comment=f"{comment}: addr_hi += carry."))
    if extra_reg is not None:
        # extra_reg is src0 so it can be sgpr(...) or vgpr(...); tmp_lo is vsrc1.
        module.add(VAddCOU32(vgpr(tmp_lo), VCC(), extra_reg, vgpr(tmp_lo),
                             comment=f"{comment}: addr_lo += extra."))
        module.add(VAddCCOU32(vgpr(tmp_hi), VCC(), vgpr(tmp_hi), 0, VCC(),
                              comment=f"{comment}: addr_hi += carry."))
    module.add(FlatLoadD16U8(dst=vgpr(dst_vgpr), vaddr=vgpr(tmp_lo, 2), comment=comment))
    writer.vgprPool.checkIn(addr_pair)


def emitDeepseekScaleGR(writer, kernel):
    """Emit flat_load_d16_u8 for E8M0 scale bytes + drain wait + kb offset advance.

    Loads one E8M0 byte per scaleA row and one E8M0 byte per scaleB N-block.
    Uses flat_load to avoid requiring 4-aligned SGPR SRDs (which are scarce in
    StreamK kernels). Called once per main loop iteration for PGR=0 Deepseek kernels.

    The post-loop store-SGPR symbols (sgprScaleABuf, sgprScaleBBuf) are only defined
    in the post-loop section and cannot be referenced in the mainloop. Instead, a
    shared 2-SGPR pair (sharedBufPtrSgpr) is loaded each iteration from KernArgAddress
    immediately before the corresponding flat_loads. ScaleA and ScaleB loads are
    issued sequentially so the pair can be reused for both without SGPR overhead.
    kbBOffset and nNBlocksStride are VGPRs to keep SGPR usage within pool limits.
    kbOffset advances by 1 byte per K-block (E8M0: 1 byte per scale entry).
    """
    dml    = kernel["_deepseekML"]
    use_a  = kernel.get("UseDeepseekScaleA", False)
    use_b  = kernel.get("UseDeepseekScaleB", False)
    module = Module("DeepseekML scaleGR")
    module.addComment1("Deepseek mainloop: load fp32 scaleA/B for current K-block.")

    shared_ptr      = dml["sharedBufPtrSgpr"]
    off_a, off_b    = dml["scaleBufOffsets"]

    if use_a:
        mma_m   = (kernel["MacroTile0"] // kernel["MatrixInstM"]) // kernel["MIWaveGroup"][0]
        vaddrs  = dml["scaleAVaddrs"]
        vgprs_a = dml["scaleAVgprs"]
        kb_off  = dml["kbOffset"]
        # Reload ScaleABuf pointer into the shared pair before issuing flat_loads.
        module.add(SLoadB64(dst=sgpr(shared_ptr, 2), base=sgpr("KernArgAddress", 2),
                            soffset=hex(off_a), comment="load ScaleABuf ptr from kernel args."))
        module.add(SWaitCnt(kmcnt=0, comment="wait for ScaleABuf ptr."))
        for i in range(mma_m):
            _emitFlatLoad(module, writer, shared_ptr, shared_ptr + 1,
                          vaddrs + i, sgpr(kb_off), vgprs_a + i,
                          f"scaleA[mma0={i}, kb].")

    if use_b:
        # Reload ScaleBBuf pointer into the shared pair (overwrites ScaleABuf ptr).
        module.add(SLoadB64(dst=sgpr(shared_ptr, 2), base=sgpr("KernArgAddress", 2),
                            soffset=hex(off_b), comment="load ScaleBBuf ptr from kernel args."))
        module.add(SWaitCnt(kmcnt=0, comment="wait for ScaleBBuf ptr."))
        _emitFlatLoad(module, writer, shared_ptr, shared_ptr + 1,
                      dml["scaleBVaddr"], vgpr(dml["kbBOffset"]), dml["scaleBVgpr"],
                      "scaleB[kb, WG1_block].")

    module.add(SWaitCnt(vlcnt=0, comment="wait for Deepseek scale flat_loads."))

    # Broadcast the loaded E8M0 byte to all 4 byte positions of each scale VGPR.
    # v_mfma_scale_f32_16x16x128_f8f6f4 uses bytes 0-3 for the 4 sub-K-blocks
    # of the 128-element K-block; flat_load_d16_u8 fills only byte 0, so we
    # replicate it via AND+MUL to give every sub-K-block the same scale.
    # VMulLOU32 does not accept 32-bit literal immediates, so 0x01010101 is
    # moved to a temp VGPR once and reused for all scale VGPRs.
    if use_a or use_b:
        mul_const = writer.vgprPool.checkOut(1, tag="dml_bcastMul")
        module.add(VMovB32(dst=vgpr(mul_const), src=hex(0x01010101),
                           comment="broadcast multiplier: replicate byte to all 4 positions."))
        if use_a:
            for i in range(mma_m):
                dst = vgprs_a + i
                module.add(VAndB32(dst=vgpr(dst), src0=hex(0xFF), src1=vgpr(dst),
                                   comment=f"isolate scaleA[mma0={i}] E8M0 byte to bits [7:0]."))
                module.add(VMulLOU32(dst=vgpr(dst), src0=vgpr(dst), src1=vgpr(mul_const),
                                     comment=f"broadcast scaleA[mma0={i}] to all 4 byte positions."))
        if use_b:
            dst_b = dml["scaleBVgpr"]
            module.add(VAndB32(dst=vgpr(dst_b), src0=hex(0xFF), src1=vgpr(dst_b),
                               comment="isolate scaleB E8M0 byte to bits [7:0]."))
            module.add(VMulLOU32(dst=vgpr(dst_b), src0=vgpr(dst_b), src1=vgpr(mul_const),
                                 comment="broadcast scaleB to all 4 byte positions."))
        writer.vgprPool.checkIn(mul_const)

    if use_a:
        module.add(SAddU32(dst=sgpr(dml["kbOffset"]), src0=sgpr(dml["kbOffset"]), src1=1,
                           comment="kbOffset += 1 (next K-block E8M0 byte offset for scaleA)."))
    if use_b:
        module.add(VAddU32(dst=vgpr(dml["kbBOffset"]), src0=vgpr(dml["kbBOffset"]),
                           src1=vgpr(dml["nNBlocksStride"]),
                           comment="kbBOffset += nNBlocksStride (next K-block scaleB offset)."))
    return module


class SubtileDeepseekScaleEmitter:
    """Emit the DeepseekScale epilogue for the Subtile gfx950 fp8 kernel.

    Supports UseDeepseekScaleA only, UseDeepseekScaleB only, or both.
    """

    def __init__(self, writer, kernel):
        self.writer   = writer
        self.kernel   = kernel

        self.use_a = kernel.get("UseDeepseekScaleA", False)
        self.use_b = kernel.get("UseDeepseekScaleB", False)

        self.mfma_m        = kernel["MatrixInstM"]
        self.mfma_n        = kernel["MatrixInstN"]
        self.wave_size     = kernel["WavefrontSize"]
        self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.wave_size

        wg = kernel["MIWaveGroup"]
        self.wg_m, self.wg_n = wg[0], wg[1]

        self.mma_m       = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
        self.mma_n       = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
        self.macro_tile0 = kernel["MacroTile0"]
        self.macro_tile1 = kernel["MacroTile1"]
        self.num_rows    = self.mma_m * self.rows_per_lane

    # ------------------------------------------------------------------ #
    # Accumulator read/write helpers.                                      #
    # ------------------------------------------------------------------ #

    def _readAccInto(self, module, dst: int, vgprTiles, m: int, n: int, k: int,
                     comment: str) -> None:
        """Copy accumulator element (m, n, k) into VGPR dst."""
        tile = vgprTiles[n * self.mma_m + m]
        reg  = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            module.add(VMovB32(dst=vgpr(dst), src=vgpr(reg), comment=comment))
            return
        module.add(VAccvgprReadB32(vgpr(dst), accvgpr(reg), comment=comment))

    def _writeAccFrom(self, module, src: int, vgprTiles, m: int, n: int, k: int,
                      comment: str) -> None:
        """Write VGPR src back into accumulator element (m, n, k)."""
        tile = vgprTiles[n * self.mma_m + m]
        reg  = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            module.add(VMovB32(dst=vgpr(reg), src=vgpr(src), comment=comment))
            return
        module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(src), comment=comment))

    def _partial_idx(self, m: int, k: int) -> int:
        """Index into scale_a_vgprs for M-tile m, row-offset k."""
        return m * self.rows_per_lane + k

    # ------------------------------------------------------------------ #
    # ScaleA helpers (per-row).                                            #
    # ------------------------------------------------------------------ #

    def _computeWaveM(self, module, dst: int) -> None:
        """Compute waveM = waveId % wg_m into VGPR dst."""
        waveId  = self.writer.vgprPool.checkOut(1, tag="dsa_waveId")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="dsa_waveMDiv")
        tmpRes  = ContinuousRegister(tmpVgpr, 2)
        module.add(vectorStaticDivide(waveId, "Serial", self.wave_size, tmpRes,
                                      comment="waveId = Serial / WavefrontSize."))
        # wg_m is always a power of 2 (format9 constraint), so AND with wg_m-1 gives the modulo.
        module.add(VAndB32(dst=vgpr(dst), src0=vgpr(waveId), src1=self.wg_m - 1,
                           comment=f"waveM = waveId % {self.wg_m}."))
        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(waveId)

    def _buildScaleASrd(self, module, srd: int) -> None:
        """Load a 128-bit buffer descriptor for ScaleABuf into SGPRs [srd, srd+3]."""
        module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr("ScaleABuf", 2),
                           comment="scaleABuf SRD base."))
        module.add(SMovB32(dst=sgpr(srd + 2), src="BufferOOB",
                           comment="scaleABuf SRD limit."))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96",
                           comment="scaleABuf SRD flags."))

    def _computeRowBase(self, module, dst: int) -> None:
        """Compute row_base = WorkGroup0*MT0 + waveM*mma_m*mfma_m into VGPR dst.

        The wave M offset is added when wg_m > 1 so each wave addresses its own
        row region in scaleA.
        """
        mt0V = self.writer.vgprPool.checkOut(1, tag="dsa_rbMT0")
        module.add(VMovB32(dst=vgpr(mt0V), src=self.macro_tile0,
                           comment=f"MT0={self.macro_tile0}."))
        module.add(VMulLOU32(dst=vgpr(dst), src0=vgpr(mt0V), src1=sgpr("WorkGroup0"),
                             comment="rowBase = WorkGroup0 * MT0."))
        self.writer.vgprPool.checkIn(mt0V)
        if self.wg_m == 1:
            return
        waveM    = self.writer.vgprPool.checkOut(1, tag="dsa_rbWaveM")
        strideV  = self.writer.vgprPool.checkOut(1, tag="dsa_rbStride")
        self._computeWaveM(module, waveM)
        waveStride = self.mma_m * self.mfma_m
        module.add(VMovB32(dst=vgpr(strideV), src=waveStride,
                           comment=f"waveStride = mma_m * mfma_m = {waveStride}."))
        module.add(VMulLOU32(dst=vgpr(waveM), src0=vgpr(strideV), src1=vgpr(waveM),
                             comment="waveMOff = waveM * waveStride."))
        module.add(VAddU32(vgpr(dst), vgpr(dst), vgpr(waveM),
                           comment="rowBase += waveMOff."))
        self.writer.vgprPool.checkIn(strideV)
        self.writer.vgprPool.checkIn(waveM)

    def _setupScaleA(self, scale_srd: int, row_base_vgpr: int, lane_id: int,
                     row_group: int, row_group_byte: int) -> Module:
        """Build ScaleABuf SRD, compute row_base, derive lane_id and row_group."""
        module = Module("DeepseekScaleA setup")
        self._buildScaleASrd(module, scale_srd)
        self._computeRowBase(module, row_base_vgpr)

        # lane_id = Serial & (wave_size - 1).
        module.add(VAndB32(dst=vgpr(lane_id), src0=vgpr("Serial"), src1=self.wave_size - 1,
                           comment="lane_id = Serial & (wave_size-1)."))

        # row_group = lane_id >> log2(mfma_n).
        log2_mfma_n = int(_math.log2(self.mfma_n))
        module.add(VLShiftRightB32(
            dst=vgpr(row_group),
            shiftHex=hex(log2_mfma_n),
            src=vgpr(lane_id),
            comment=f"row_group = lane_id >> {log2_mfma_n} (= lane_id // {self.mfma_n}).",
        ))

        # row_group_byte = row_group * rows_per_lane (E8M0: 1 byte per row).
        module.add(VMulLOU32(
            dst=vgpr(row_group_byte),
            src0=self.rows_per_lane,
            src1=vgpr(row_group),
            comment=f"row_group_byte = row_group * {self.rows_per_lane} (E8M0 byte stride).",
        ))

        return module

    def _computeScaleAAddrBytes(self, module, global_addr: int, offset_tmp: int,
                                 row_base_vgpr: int, row_group_byte: int,
                                 m: int, k: int) -> None:
        """Compute the byte address for E8M0 scaleA element (m, k) into global_addr.

        Byte address = row_base + m*mfma_m + k + row_group * rows_per_lane.
        E8M0 has 1 byte per row, so no shift is needed.
        """
        m_k_offset = m * self.mfma_m + k
        module.add(VMovB32(dst=vgpr(global_addr), src=vgpr(row_base_vgpr),
                           comment=f"global_addr = row_base for scaleA[m={m},k={k}]."))
        if m_k_offset > 64:
            module.add(VMovB32(dst=vgpr(offset_tmp), src=m_k_offset,
                               comment=f"offset_tmp = {m_k_offset}."))
            module.add(VAddU32(vgpr(global_addr), vgpr(global_addr), vgpr(offset_tmp),
                               comment=f"global_addr += {m_k_offset} (m*mfmaM + k)."))
        elif m_k_offset > 0:
            module.add(VAddU32(vgpr(global_addr), vgpr(global_addr), m_k_offset,
                               comment=f"global_addr += {m_k_offset} (m*mfmaM + k)."))
        module.add(VAddU32(vgpr(global_addr), vgpr(global_addr), vgpr(row_group_byte),
                           comment="global_addr += row_group * rows_per_lane (E8M0 byte offset)."))

    def _loadScaleA(self, scale_srd: int, row_base_vgpr: int,
                    row_group_byte: int, scale_vgprs: int) -> Module:
        """Load E8M0 scaleA bytes from ScaleABuf and decode to fp32 in scale_vgprs."""
        module = Module("DeepseekScaleA loadScaleA")
        module.addComment1("DeepseekScaleA: load E8M0 scaleA bytes from ScaleABuf.")

        global_addr = self.writer.vgprPool.checkOut(1, tag="dsa_globalAddr")
        offset_tmp  = self.writer.vgprPool.checkOut(1, tag="dsa_offsetTmp")

        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                i = self._partial_idx(m, k)
                self._computeScaleAAddrBytes(module, global_addr, offset_tmp,
                                              row_base_vgpr, row_group_byte, m, k)
                module.add(BufferLoadU8(
                    dst=vgpr(scale_vgprs + i),
                    vaddr=vgpr(global_addr),
                    saddr=sgpr(scale_srd, 4),
                    soffset=0,
                    mubuf=MUBUFModifiers(offen=True),
                    comment=f"scaleA[m={m},k={k}]: load E8M0 byte from ScaleABuf.",
                ))

        module.add(SWaitCnt(vlcnt=0, comment="wait for all scaleA byte loads."))

        # Decode E8M0 to fp32: fp32 = byte << 23 (byte becomes the exponent field).
        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                i = self._partial_idx(m, k)
                module.add(VLShiftLeftB32(
                    dst=vgpr(scale_vgprs + i), shiftHex=hex(23), src=vgpr(scale_vgprs + i),
                    comment=f"decode E8M0 to fp32 for scaleA[m={m},k={k}].",
                ))

        self.writer.vgprPool.checkIn(offset_tmp)
        self.writer.vgprPool.checkIn(global_addr)
        return module

    def _applyScaleA(self, vgprTiles, scale_a_vgprs: int) -> Module:
        """Multiply every accumulator element by the corresponding scaleA[row]."""
        module = Module("DeepseekScaleA applyScaleA")
        module.addComment1("DeepseekScaleA: multiply each acc element by scaleA[row].")

        acc_tmp = self.writer.vgprPool.checkOut(1, tag="dsa_accTmp")

        for m in range(self.mma_m):
            for n in range(self.mma_n):
                for k in range(self.rows_per_lane):
                    ridx = scale_a_vgprs + self._partial_idx(m, k)
                    self._readAccInto(module, acc_tmp, vgprTiles, m, n, k,
                                      f"read acc[m={m},n={n},k={k}].")
                    module.add(VMulF32(dst=vgpr(acc_tmp), src0=vgpr(acc_tmp),
                                       src1=vgpr(ridx),
                                       comment=f"acc *= scaleA[m={m},k={k}]."))
                    self._writeAccFrom(module, acc_tmp, vgprTiles, m, n, k,
                                       f"write acc[m={m},n={n},k={k}].")

        self.writer.vgprPool.checkIn(acc_tmp)
        return module

    # ------------------------------------------------------------------ #
    # ScaleB helpers (per-128col N-block).                                 #
    # ------------------------------------------------------------------ #

    def _computeNBlock(self, n: int) -> int:
        """Return the compile-time local N-block index for accumulator tile position n.

        For configs where the wave's N-span fits within 128 columns, this is
        always 0. The caller adds the runtime WG1 base block to get the global index.
        """
        return (n * self.mfma_n) // 128

    def _distinctNBlocks(self) -> list:
        """Sorted list of distinct compile-time local N-block offsets in this wave."""
        return sorted({self._computeNBlock(n) for n in range(self.mma_n)})

    def _buildScaleBSrd(self, module, srd: int) -> None:
        """Load a 128-bit buffer descriptor for ScaleBBuf into SGPRs [srd, srd+3]."""
        module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr("ScaleBBuf", 2),
                           comment="scaleBBuf SRD base."))
        module.add(SMovB32(dst=sgpr(srd + 2), src="BufferOOB",
                           comment="scaleBBuf SRD limit."))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96",
                           comment="scaleBBuf SRD flags."))

    def _loadScaleB(self, scale_b_srd: int, scaleB_vgprs: int) -> Module:
        """Load E8M0 scaleB bytes from ScaleBBuf and decode to fp32 in scaleB_vgprs.

        Each distinct local N-block offset maps to one VGPR slot. For all
        existing configs (MT1=128, wave span <= 128), only slot 0 is populated
        and the byte address is simply WorkGroup1 (E8M0: 1 byte per N-block).
        """
        module = Module("DeepseekScaleB loadScaleB")
        module.addComment1("DeepseekScaleB: load E8M0 scaleB bytes from ScaleBBuf.")
        self._buildScaleBSrd(module, scale_b_srd)

        # mt1_blocks: number of 128-column scaleB blocks per WG tile (compile-time).
        mt1_blocks = self.macro_tile1 // 128
        distinct   = self._distinctNBlocks()
        addr       = self.writer.vgprPool.checkOut(1, tag="dsb_addr")

        for i, block_off in enumerate(distinct):
            # Byte offset = WG1 * mt1_blocks + block_off (E8M0: 1 byte per N-block).
            module.add(VMovB32(dst=vgpr(addr), src=sgpr("WorkGroup1"),
                               comment=f"addr = WorkGroup1 (scaleB block off={block_off})."))
            if mt1_blocks != 1:
                module.add(VMulLOU32(dst=vgpr(addr), src0=vgpr(addr), src1=mt1_blocks,
                                     comment=f"addr *= {mt1_blocks} (N-block index)."))
            if block_off > 0:
                module.add(VAddU32(vgpr(addr), vgpr(addr), block_off,
                                   comment=f"addr += block_off={block_off}."))
            module.add(BufferLoadU8(
                dst=vgpr(scaleB_vgprs + i),
                vaddr=vgpr(addr),
                saddr=sgpr(scale_b_srd, 4),
                soffset=0,
                mubuf=MUBUFModifiers(offen=True),
                comment=f"scaleB[WG1*{mt1_blocks}+{block_off}]: load E8M0 byte from ScaleBBuf.",
            ))

        module.add(SWaitCnt(vlcnt=0, comment="wait for all scaleB byte loads."))

        # Decode E8M0 to fp32: fp32 = byte << 23 (byte becomes the exponent field).
        for i in range(len(distinct)):
            module.add(VLShiftLeftB32(
                dst=vgpr(scaleB_vgprs + i), shiftHex=hex(23), src=vgpr(scaleB_vgprs + i),
                comment=f"decode E8M0 to fp32 for scaleB[{i}].",
            ))

        self.writer.vgprPool.checkIn(addr)
        return module

    def _applyScaleB(self, vgprTiles, scaleB_vgprs: int) -> Module:
        """Multiply every accumulator element by scaleB[n-block]."""
        module = Module("DeepseekScaleB applyScaleB")
        module.addComment1("DeepseekScaleB: multiply each acc element by scaleB[n-block].")

        distinct = self._distinctNBlocks()
        acc_tmp  = self.writer.vgprPool.checkOut(1, tag="dsb_accTmp")

        for m in range(self.mma_m):
            for b_idx, block_off in enumerate(distinct):
                ns = [n for n in range(self.mma_n) if self._computeNBlock(n) == block_off]
                for n in ns:
                    for k in range(self.rows_per_lane):
                        self._readAccInto(module, acc_tmp, vgprTiles, m, n, k,
                                          f"read acc[m={m},n={n},k={k}].")
                        module.add(VMulF32(dst=vgpr(acc_tmp), src0=vgpr(acc_tmp),
                                           src1=vgpr(scaleB_vgprs + b_idx),
                                           comment=f"acc *= scaleB[block={block_off}]."))
                        self._writeAccFrom(module, acc_tmp, vgprTiles, m, n, k,
                                           f"write acc[m={m},n={n},k={k}].")

        self.writer.vgprPool.checkIn(acc_tmp)
        return module

    def _applyScaleAB(self, vgprTiles, scale_a_vgprs: int, scaleB_vgprs: int) -> Module:
        """Multiply every acc element by scaleA[row] * scaleB[n-block].

        The product is precomputed per (m, k, block) triple and reused across
        all n tile positions mapping to the same block.
        """
        module = Module("DeepseekScaleAB applyScaleAB")
        module.addComment1("DeepseekScaleAB: acc *= scaleA[row] * scaleB[n-block].")

        distinct = self._distinctNBlocks()
        acc_tmp  = self.writer.vgprPool.checkOut(1, tag="dsab_accTmp")
        combined = self.writer.vgprPool.checkOut(1, tag="dsab_combined")

        for m in range(self.mma_m):
            for k in range(self.rows_per_lane):
                a_ridx = scale_a_vgprs + self._partial_idx(m, k)
                for b_idx, block_off in enumerate(distinct):
                    ns = [n for n in range(self.mma_n) if self._computeNBlock(n) == block_off]
                    # Precompute combined scale = scaleA[m,k] * scaleB[block].
                    module.add(VMulF32(dst=vgpr(combined), src0=vgpr(a_ridx),
                                       src1=vgpr(scaleB_vgprs + b_idx),
                                       comment=f"combined = scaleA[m={m},k={k}]*scaleB[{block_off}]."))
                    for n in ns:
                        self._readAccInto(module, acc_tmp, vgprTiles, m, n, k,
                                          f"read acc[m={m},n={n},k={k}].")
                        module.add(VMulF32(dst=vgpr(acc_tmp), src0=vgpr(acc_tmp),
                                           src1=vgpr(combined),
                                           comment=f"acc *= combined scale."))
                        self._writeAccFrom(module, acc_tmp, vgprTiles, m, n, k,
                                           f"write acc[m={m},n={n},k={k}].")

        self.writer.vgprPool.checkIn(combined)
        self.writer.vgprPool.checkIn(acc_tmp)
        return module

    # ------------------------------------------------------------------ #
    # Main emit entry point.                                               #
    # ------------------------------------------------------------------ #

    def _loadScales(self) -> tuple:
        """Load scaleA and/or scaleB from global memory into VGPRs.

        SRDs and address-computation temporaries are freed immediately after
        the loads complete to minimise register pressure during apply.
        Returns (module, scale_a_vgprs_or_None, scale_b_vgprs_or_None).
        """
        module = Module("DeepseekScale load scales")
        scale_a_vgprs = None
        if self.use_a:
            scale_a_vgprs  = self.writer.vgprPool.checkOut(self.num_rows, tag="dsa_scaleVgprs")
            lane_id        = self.writer.vgprPool.checkOut(1,             tag="dsa_laneId")
            row_group      = self.writer.vgprPool.checkOut(1,             tag="dsa_rowGroup")
            row_group_byte = self.writer.vgprPool.checkOut(1,             tag="dsa_rowGroupByte")
            row_base_vgpr  = self.writer.vgprPool.checkOut(1,             tag="dsa_rowBase")
            srd_a = self.writer.sgprPool.checkOutAligned(4, 4, tag="dsa_scaleASrd")
            module.add(self._setupScaleA(srd_a, row_base_vgpr, lane_id,
                                         row_group, row_group_byte))
            module.add(self._loadScaleA(srd_a, row_base_vgpr, row_group_byte, scale_a_vgprs))
            self.writer.sgprPool.checkIn(srd_a)
            self.writer.vgprPool.checkIn(row_base_vgpr)
            self.writer.vgprPool.checkIn(row_group_byte)
            self.writer.vgprPool.checkIn(row_group)
            self.writer.vgprPool.checkIn(lane_id)
        scale_b_vgprs = None
        if self.use_b:
            num_b = len(self._distinctNBlocks())
            scale_b_vgprs = self.writer.vgprPool.checkOut(num_b, tag="dsb_scaleVgprs")
            srd_b = self.writer.sgprPool.checkOutAligned(4, 4, tag="dsb_scaleBSrd")
            module.add(self._loadScaleB(srd_b, scale_b_vgprs))
            self.writer.sgprPool.checkIn(srd_b)
        return module, scale_a_vgprs, scale_b_vgprs

    def _applyAndFree(self, vgprTiles, scale_a_vgprs, scale_b_vgprs) -> Module:
        """Dispatch to the correct scale-apply path then release all scale VGPRs."""
        module = Module("DeepseekScale apply and free")
        if self.use_a and self.use_b:
            module.add(self._applyScaleAB(vgprTiles, scale_a_vgprs, scale_b_vgprs))
        elif self.use_a:
            module.add(self._applyScaleA(vgprTiles, scale_a_vgprs))
        else:
            module.add(self._applyScaleB(vgprTiles, scale_b_vgprs))
        if self.use_b:
            self.writer.vgprPool.checkIn(scale_b_vgprs)
        if self.use_a:
            self.writer.vgprPool.checkIn(scale_a_vgprs)
        return module

    def emit(self, vgprTiles) -> Module:
        """Return the full DeepseekScale epilogue module.

        Handles UseDeepseekScaleA only, UseDeepseekScaleB only, and A+B.
        vgprTiles: dtileInfo.vgprTiles, the per-tile allocator records for D.
        When the mainloop handles scaling (PGR=0 path, kernel has _deepseekML),
        the epilogue is a no-op — the WMMA instruction already applied the scales.
        """
        if self.kernel.get("_deepseekML"):
            return Module("DeepseekScale epilogue (skipped: mainloop path active)")
        module = Module("DeepseekScale epilogue")
        module.addComment1("DeepseekScale: fp8 dequantization scale epilogue.")
        module.add(SWaitCnt(waitAll=True, comment="flush MFMA pipeline before DeepseekScale."))
        load_module, scale_a_vgprs, scale_b_vgprs = self._loadScales()
        module.add(load_module)
        module.add(self._applyAndFree(vgprTiles, scale_a_vgprs, scale_b_vgprs))
        return module
