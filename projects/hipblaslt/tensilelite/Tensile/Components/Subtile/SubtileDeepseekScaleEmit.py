# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DeepseekScale mainloop scale emitter for the Subtile kernel (gfx950, fp8 in / f32 out).

Applies per-row E8M0 scaleA and/or per-128col-block E8M0 scaleB to the fp32
GEMM accumulator (AGPRs). Scale buffers hold one UE8M0 byte per element; the
kernel decodes each byte to fp32 via left-shift by 23 (same as the hardware
v_mfma_scale operand path). This implements the Deepseek V3 dequantization
pattern.

Full formula:
  D[m,n] = alpha * decode(scaleA[m]) * decode(scaleB[n//128]) * acc[m,n] + beta*C

If only scaleA: scaleB factor is omitted (unit scale applied via 0x7f byte).
If only scaleB: scaleA factor is omitted (unit scale applied via 0x7f byte).

Scale buffer layouts (E8M0, 1 byte per element):
  scaleA: shape [M, nKBlocks].
  scaleB: shape [nKBlocks, ceil(N/128)].

Application (mainloop, PGR=0): scaleA and scaleB are passed as mxsa/mxsb
  operands to v_mfma_scale_f32_16x16x128_f8f6f4, applied inline per K-block
  iteration.

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

N-block assignment (compile-time): each accumulator tile position n maps to
  local_n_block = (n * mfma_n) // 128.
For all existing configs (MT1=128, mma_n*mfma_n<=128), local_n_block=0 always,
so every wave uses a single scaleB value determined by WorkGroup1.
"""

import math as _math

from rocisa.code import Module
from rocisa.container import (
    VCC,
    sgpr,
    vgpr,
)
from rocisa.instruction import (
    FlatLoadD16U8,
    SAddU32,
    SLShiftRightB32,
    SLoadB64,
    SMulI32,
    SMovB32,
    SWaitCnt,
    VAddCCOU32,
    VAddCOU32,
    VAddU32,
    VAndB32,
    VLShiftRightB32,
    VMovB32,
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
        tmp_sgpr = writer.sgprPool.checkOut(1)
        module.add(SAddU32(dst=sgpr(tmp_sgpr), src0=sgpr("SizesFree+1"), src1=127,
                           comment="tmp = N + 127."))
        module.add(SLShiftRightB32(dst=sgpr(tmp_sgpr), shiftHex=hex(7), src=sgpr(tmp_sgpr),
                                   comment="tmp = ceil(N/128) (1 byte per N-block)."))
        module.add(VMovB32(dst=vgpr(nn_stride), src=sgpr(tmp_sgpr),
                           comment="nNBlocksStride = ceil(N/128)."))
        writer.sgprPool.checkIn(tmp_sgpr)
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

    # vlcnt=0 drains only the scale flat_loads issued above. This is a no-op at
    # PGR=0 because no data buffer_loads are in flight yet. Do not reorder data
    # GRs to precede this block — that would drain them here too, serialising
    # scale and data loads and eliminating load-MFMA overlap.
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

