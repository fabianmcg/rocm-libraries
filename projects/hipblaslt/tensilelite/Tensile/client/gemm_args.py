# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build raw kernel argument bytes for TensileLite GEMM kernels.

This module ports the argument-layout logic from
ContractionSolution.cpp:singleCallArgs (lines 548-1130) and kernelArgs
(lines 1557-1714) to Python.

Supported configuration subset (M1-M6-partial; extended per milestone):
  - stridedBatched in {True, False}
  - streamK in {0, 3, 4, 5}  (4=dynamic SK, 5=hybrid SK added in M6)
  - streamKAtomic = 0 only (atomic=1 skips workspace+flags block)
  - groupedGemm = False
  - GSU = 1, globalAccumulation in {0, 1, 2}
  - useInitialStrides = False
  - KernArgsVersion <= 2, useSFC = False
  - expertSchedulingMode = 0, debugKernel = False
  - MX block-scaled A and/or B (MXBlockA/B != 0), stridedBatched only (M4)
  - Epilogue slots: bias, ScaleAB, ScaleCD, ScaleAlphaVec, E tensor,
    activation args, AmaxD (M5)

Unsupported combinations raise NotImplementedError, including:
  - GSU > 1 && streamK == 0
  - globalAccumulation = 3 (MBSK)
  - useSFC = True
  - KernArgsVersion > 2
  - expertSchedulingMode != 0
  - debugKernel = True
  - streamKAtomic = 1
"""

from __future__ import annotations

import math
import struct

# Integer codes from rocisa::DataType enum (enum.hpp).
_dtypeHalf: int = 4          # rocisa::DataType::Half
_dtypeInt32: int = 6         # rocisa::DataType::Int32
_dtypeInt8: int = 8          # rocisa::DataType::Int8
_dtypeXf32: int = 10         # rocisa::DataType::XFloat32
_dtypeFp8e4m3fnuz: int = 11  # rocisa::DataType::Float8_fnuz  (E4M3 fnuz)
_dtypeBf8e5m2fnuz: int = 12  # rocisa::DataType::BFloat8_fnuz (E5M2 fnuz)
_dtypeFp8e4m3fn: int = 15    # rocisa::DataType::Float8       (E4M3 OCP)
_dtypeBf8e5m2: int = 16      # rocisa::DataType::BFloat8      (E5M2 OCP)


def _readMxBlock(solutionParams: dict, axis: str) -> int:
    """Read MXBlockA or MXBlockB from a solution dict (flat or nested ProblemType).

    Returns 0 when the key is absent, indicating no MX scaling on that operand.
    """
    key = f"MXBlock{axis}"
    if key in solutionParams:
        return int(solutionParams[key])
    pt = solutionParams.get("ProblemType") or {}
    return int(pt.get(key, 0)) if pt else 0


def _validateConfig(solutionParams: dict, problemParams: dict) -> None:
    """Raise NotImplementedError for unsupported configurations."""
    if solutionParams.get("debugKernel", False):
        raise NotImplementedError("debugKernel=True not supported")
    streamK = solutionParams.get("StreamK", 0)
    streamKAtomic = solutionParams.get("StreamKAtomic", 0)
    if streamKAtomic != 0:
        raise NotImplementedError("streamKAtomic != 0 not supported (skips workspace+flags block)")
    if streamK not in (0, 3, 4, 5):
        raise NotImplementedError(
            f"streamK={streamK} not supported; supported: "
            "0 (standard), 3 (two-tile SK), 4 (dynamic SK), 5 (hybrid SK)"
        )
    if solutionParams.get("UseSFC", False):
        raise NotImplementedError("useSFC=True not supported (different internalArg1 packing)")
    version = solutionParams.get("KernArgsVersion", 0)
    if version > 2:
        raise NotImplementedError(
            f"KernArgsVersion={version} > 2 not supported "
            "(different GSU bit-field: bits 0-11 for GSU, bits 12-13 for ntaBit/ntbBit)"
        )
    expertMode = solutionParams.get("ExpertSchedulingMode", 0)
    if expertMode != 0:
        raise NotImplementedError(
            f"expertSchedulingMode={expertMode} != 0 not supported (adds ESMRuntimeSupported slot)"
        )
    gsu = problemParams.get("gsu", 1)
    if gsu > 1 and streamK == 0:
        raise NotImplementedError(
            f"GSU={gsu} > 1 with streamK=0 not supported (emits workspace D/C pointers)"
        )
    globalAccum = solutionParams.get("GlobalAccumulation", 0)
    if globalAccum == 3:
        raise NotImplementedError(
            "globalAccumulation=3 (MBSK) not supported (adds dstD/Synchronizer/GSUSync trailing slots)"
        )


def _magicNumberAlg2(d: int) -> tuple[int, int]:
    """Return (magic, shift) for 32-bit unsigned fast division by d (algorithm 2).

    Mirrors the C++ magicNumber() call in ContractionSolution.cpp for StreamK args.
    """
    if d == 0:
        return 0, 0
    d = d & 0xFFFFFFFF
    a = 0
    nc = (-1 - (-d) % d) & 0xFFFFFFFF
    p = 31
    q1 = 0x80000000 // nc
    r1 = 0x80000000 - q1 * nc
    q2 = 0x7FFFFFFF // d
    r2 = 0x7FFFFFFF - q2 * d
    while p < 64:
        p += 1
        if r1 >= nc - r1:
            q1 = 2 * q1 + 1
            r1 = 2 * r1 - nc
        else:
            q1 = 2 * q1
            r1 = 2 * r1
        if r2 + 1 >= d - r2:
            if q2 >= 0x7FFFFFFF:
                a = 1
            q2 = 2 * q2 + 1
            r2 = 2 * r2 + 1 - d
        else:
            if q2 >= 0x80000000:
                a = 1
            q2 = 2 * q2
            r2 = 2 * r2 + 1
        delta = d - 1 - r2
        if not (p < 64 and (q1 < delta or (q1 == delta and r1 == 0))):
            break
    magic = (q2 + 1) & 0xFFFFFFFF
    shift = p - 32
    if a:
        shift |= 0x80000000
    return magic, shift & 0xFFFFFFFF


def _computeNumWorkGroups(solutionParams: dict, problemParams: dict) -> int:
    """Compute total flattened workgroup count for version >= 1.

    For version=0 this is only the X dimension (ceil(M/MT0)), but version=0
    is uncommon enough that M1 only encounters version >= 1 kernels. For
    version >= 1 the kernel receives the fully flattened count.
    """
    mt0 = solutionParams["MacroTile0"]
    mt1 = solutionParams["MacroTile1"]
    sizes = problemParams["sizes"]
    M = sizes[0]
    N = sizes[1]
    ngx = math.ceil(M / mt0)
    ngy = math.ceil(N / mt1)
    # batch index is at position 2 for 4-element sizes (batched GEMM).
    if len(sizes) == 4:
        batch = sizes[2]
    else:
        batch = 1
    version = solutionParams.get("KernArgsVersion", 0)
    if version >= 1:
        return ngx * ngy * batch
    # Version 0: only X dimension is passed (Y/Z handled separately by grid).
    return ngx


def _computeInternalArg0(
    solutionParams: dict,
    gsu: int,
) -> int:
    """Compute internalArg0 from GSU, WGM (v0 only), StaggerU, gsuc, gsuwgmrr.

    Mirrors ContractionSolution.cpp:kernelArgs internalArg0 assembly.
    """
    version = solutionParams.get("KernArgsVersion", 0)
    supportWgm = solutionParams.get("SupportCustomWGM", False)
    supportStaggerU = solutionParams.get("SupportCustomStaggerU", False)

    mask8 = 0xFF
    mask14 = 0x3FFF
    # Version >= 3 narrows GSU to 12 bits; M1 never exceeds version 2.
    gsuMask = mask14

    gsuc = 0
    gsuwgmrr = 0
    if version >= 2:
        gsuc = 1 if solutionParams.get("GlobalSplitUCoalesced", False) else 0
        gsuwgmrr = 1 if solutionParams.get("GlobalSplitUWorkGroupMappingRoundRobin", False) else 0

    arg0 = 0

    # Version-0 WGM packing: WGM in bits 8..15, both clamped to 255.
    if supportWgm and version == 0:
        wgm = int(solutionParams.get("WorkGroupMapping", 1))
        wgm = min(wgm, 255)
        gsu = min(gsu, 255)
        arg0 |= (mask8 & wgm) << 8

    # StaggerU packing into bits 16..29 (when SupportCustomStaggerU=True).
    if supportStaggerU:
        staggerU = int(solutionParams.get("StaggerU", 0))
        staggerUMapping = int(solutionParams.get("StaggerUMapping", 0))
        staggerUStrideShift = int(solutionParams.get("_staggerStrideShift", 0))
        staggerMask1 = 0x1F00
        su_word = (mask8 & staggerU) | (staggerMask1 & (staggerUStrideShift << 8))
        su_word |= staggerUMapping << 13
        arg0 |= su_word << 16

    arg0 |= (gsuc << 15) | (gsuwgmrr << 14) | (gsuMask & gsu)
    return arg0 & 0xFFFFFFFF


def _computeInternalArg1(solutionParams: dict, cu_count: int = 0) -> int:
    """Compute internalArg1 for version >= 1.

    Returns a value that fits in int32 (signed). Mirrors the C++ kernelArgs
    internalArg1 assembly for version 1 and version 2 (useSFC=False).

    When WorkGroupMappingXCCGroup == -1 and WorkGroupMappingXCC >= 1, the C++
    code substitutes pAMDGPU->computeUnitCount (the device CU count).  Pass
    that value as cu_count so the Python path matches.  Callers that cannot
    provide cu_count should filter such solutions out before calling this
    function.
    """
    version = solutionParams.get("KernArgsVersion", 0)
    supportWgm = solutionParams.get("SupportCustomWGM", False)

    if not supportWgm:
        return 0

    wgm = int(solutionParams.get("WorkGroupMapping", 1))
    if version == 1:
        return wgm

    # Version >= 2, useSFC=False (useSFC=True is rejected in _validateConfig).
    wgmxcc = int(solutionParams.get("WorkGroupMappingXCC", 0))
    wgmxccg = int(solutionParams.get("WorkGroupMappingXCCGroup", 0))

    # When wgmxccg == -1 and wgmxcc >= 1 the C++ code reads pAMDGPU->computeUnitCount.
    # Use the caller-supplied cu_count to replicate that; raise if unavailable.
    if wgmxcc >= 1 and wgmxccg == -1:
        if cu_count <= 0:
            raise NotImplementedError(
                "wgmxccg=-1 with wgmxcc>=1 requires hardware CU count; "
                "pass cu_count=<device multiprocessor_count> to _computeInternalArg1"
            )
        wgmxccg = cu_count

    # When workGroupMappingXCC == -1 the C++ code uses wgmxccchunk. We reject
    # this case the same way.
    if wgmxcc == -1:
        raise NotImplementedError(
            "WorkGroupMappingXCC=-1 (WGMXCCn1 mode) requires autoWGMXCCCHUNK from calculateAutoWGM; "
            "filter solutions with WorkGroupMappingXCC=-1"
        )

    mask16 = 0xFFFF
    return int((wgmxccg << 22) | (wgmxcc << 16) | (mask16 & wgm))


def _buildKernelArgsHeader(solutionParams: dict, problemParams: dict,
                           cu_count: int = 0) -> bytes:
    """Build the kernelArgs header (gemm_count + internalArg0 + [internalArg1 + numWG]).

    For useUniversalArgs=True (the only supported M1 path), kernelArgs is called with
    Legacy=False before singleCallArgs, prepending this header to the arg buffer.
    Mirrors ContractionSolution.cpp:kernelArgs<T_Debug=false, Legacy=false>().
    """
    version = solutionParams.get("KernArgsVersion", 0)
    stridedBatched = solutionParams.get("StridedBatched", True)
    gsu = int(problemParams.get("gsu", 1))

    # argType=0 for strided-batched, argType=3 for pointer-array batch.
    arg_type = 0 if stridedBatched else 3
    gemm_count = (1 & 0x3FFFFFFF) | (arg_type << 30)

    arg0 = _computeInternalArg0(solutionParams, gsu)
    num_wg = _computeNumWorkGroups(solutionParams, problemParams)

    buf = struct.pack("<I", gemm_count)
    buf += struct.pack("<I", arg0)
    if version >= 1:
        arg1 = _computeInternalArg1(solutionParams, cu_count=cu_count)
        buf += struct.pack("<i", arg1)   # int32
        buf += struct.pack("<I", num_wg)
    return buf


def _buildProblemSizes(problemParams: dict) -> bytes:
    """Pack problem sizes as uint32. Sizes: [M, N, batch, K] or [M, N, K]."""
    return b"".join(struct.pack("<I", s) for s in problemParams["sizes"])


def _packPtr(addr: int) -> bytes:
    """Pack a device pointer (uint64, little-endian)."""
    return struct.pack("<Q", addr & 0xFFFFFFFFFFFFFFFF)


def _buildPointers(solutionParams: dict, tensors: dict) -> bytes:
    """Build the D/C and A/B pointer slots from singleCallArgs.

    For stridedBatched: uses d/c/a/b direct device pointers.
    For non-stridedBatched: uses batchD/batchC, batchA/batchB pointer arrays.
    MX scale pointers (mxsa after A, mxsb after B) are injected when
    MXBlockA or MXBlockB are non-zero; only supported for stridedBatched.
    StreamK parallel-reduction replaces d/c with ws_d/ws_c; tree-reduction
    uses d/c directly (workspace is appended separately by _buildStreamKWorkspace).
    """
    stridedBatched = solutionParams.get("StridedBatched", True)
    streamK = solutionParams.get("StreamK", 0)
    mxBlockA = _readMxBlock(solutionParams, "A")
    mxBlockB = _readMxBlock(solutionParams, "B")
    # Parallel reduction is not exposed in M1; tree reduction uses d/c directly.

    buf = b""
    if stridedBatched:
        buf += _packPtr(tensors["D"])
        buf += _packPtr(tensors["C"])
    else:
        buf += _packPtr(tensors["batchD"])
        buf += _packPtr(tensors["batchC"])

    if stridedBatched:
        buf += _packPtr(tensors["A"])
        if mxBlockA:
            buf += _packPtr(tensors.get("mxsa", 0))
        buf += _packPtr(tensors["B"])
        if mxBlockB:
            buf += _packPtr(tensors.get("mxsb", 0))
    else:
        buf += _packPtr(tensors["batchA"])
        buf += _packPtr(tensors["batchB"])

    # StreamK workspace + flags (tree reduction: flags = Synchronizer pointer).
    if streamK > 0:
        buf += _packPtr(tensors.get("workspace", 0))
        buf += _packPtr(tensors.get("flags", 0))

    return buf


def _buildStrides(solutionParams: dict, problemParams: dict) -> bytes:
    """Build stride uint32 slots for D, C, A, [mxsa,] B, [mxsb].

    With useInitialStridesCD=False and useInitialStridesAB=False, strides start
    from dimension index 1 (the C++ startStrideCD/startStrideAB = 1 case).
    For batched GEMM (4-element sizes), each tensor has 3 dimensions → 2 strides.
    For non-batched GEMM (3-element sizes), each tensor has 2 dimensions → 1 stride.

    MX scale strides (strideMXSA after A, strideMXSB after B) are appended when
    MXBlockA or MXBlockB are non-zero.  Use problemParams keys:
      ld_mxsa, stride_mxsa  — leading dim and batch stride for scale A
      ld_mxsb, stride_mxsb  — leading dim and batch stride for scale B
    """
    sizes = problemParams["sizes"]
    # Batched GEMM: 4 problem sizes [M, N, batch, K] → 3D tensors → 2 strides per tensor.
    batched = len(sizes) == 4
    mxBlockA = _readMxBlock(solutionParams, "A")
    mxBlockB = _readMxBlock(solutionParams, "B")
    buf = b""

    def packStrides(ld: int, batchStride: int) -> bytes:
        out = struct.pack("<I", ld)
        if batched:
            out += struct.pack("<I", batchStride)
        return out

    buf += packStrides(problemParams["ldd"], problemParams.get("stride_d", 0))
    buf += packStrides(problemParams["ldc"], problemParams.get("stride_c", 0))
    buf += packStrides(problemParams["lda"], problemParams.get("stride_a", 0))
    if mxBlockA:
        buf += packStrides(problemParams.get("ld_mxsa", 0), problemParams.get("stride_mxsa", 0))
    buf += packStrides(problemParams["ldb"], problemParams.get("stride_b", 0))
    if mxBlockB:
        buf += packStrides(problemParams.get("ld_mxsb", 0), problemParams.get("stride_mxsb", 0))
    return buf


def _readComputeTypeCode(solutionParams: dict) -> int:
    """Read ComputeDataType integer code from a solution dict.

    Handles flat dicts (top-level ComputeDataType key, used in manually
    constructed test dicts and tuning YAMLs) and compiled Solution dicts
    (where the value lives in a nested ProblemType Mapping and may be a
    DataType object with a .value attribute).
    """
    val = solutionParams.get("ComputeDataType")
    if val is not None:
        return int(val.value) if hasattr(val, "value") else int(val)
    pt = solutionParams.get("ProblemType") or {}
    val = pt.get("ComputeDataType") if pt else None
    if val is None:
        return 0
    return int(val.value) if hasattr(val, "value") else int(val)


def _readHPA(solutionParams: dict) -> bool:
    """Read HighPrecisionAccumulate from a solution dict.

    Handles both flat dicts and compiled Solution dicts (nested ProblemType).
    """
    if "HighPrecisionAccumulate" in solutionParams:
        return bool(solutionParams["HighPrecisionAccumulate"])
    pt = solutionParams.get("ProblemType") or {}
    return bool(pt.get("HighPrecisionAccumulate", False)) if pt else False


def _alphaTypeIsHalf(solutionParams: dict) -> bool:
    """Return True when alpha/beta must be packed as float16 (2-byte + alpha_2 slot).

    From ContractionSolution.cpp: the alpha_2/beta_2 slots appear only when
    alphaType == rocisa::DataType::Half. This equals the compute type, which is
    fp32 for HPA GEMM (the typical case) and fp16 only for non-HPA fp16 GEMM.
    """
    if _readHPA(solutionParams):
        return False
    return _readComputeTypeCode(solutionParams) == _dtypeHalf


def _alphaTypeIsInt32(solutionParams: dict) -> bool:
    """Return True when alpha/beta must be packed as int32 (non-HPA int8 GEMM).

    For int8 GEMM without HighPrecisionAccumulate, the compute type is Int32
    and alpha/beta are packed as 4-byte signed integers, not floats.
    """
    if _readHPA(solutionParams):
        return False
    return _readComputeTypeCode(solutionParams) == _dtypeInt32


def _packScalar(value: float, isHalf: bool, isInt32: bool = False) -> bytes:
    """Pack a scalar as float32, float16+duplicate, or int32 (4 bytes in all cases)."""
    if isInt32:
        # Non-HPA int8 GEMM: alpha/beta are 4-byte signed integers.
        return struct.pack("<i", int(round(value)))
    if not isHalf:
        return struct.pack("<f", float(value))
    # float16: 2-byte primary slot + 2-byte alpha_2/beta_2 duplicate slot.
    import numpy as np
    raw = np.float16(value).view(np.uint16)
    return struct.pack("<H", int(raw)) + struct.pack("<H", int(raw))


def _buildAlphaBeta(solutionParams: dict, problemParams: dict) -> bytes:
    """Pack alpha and (when useBeta=True) beta scalar slots."""
    isHalf = _alphaTypeIsHalf(solutionParams)
    isInt32 = _alphaTypeIsInt32(solutionParams)
    alpha = float(problemParams.get("alpha", 1.0))
    buf = _packScalar(alpha, isHalf, isInt32)

    if solutionParams.get("UseBeta", True):
        beta = float(problemParams.get("beta", 0.0))
        buf += _packScalar(beta, isHalf, isInt32)
    return buf


def _buildStreamK3Args(solutionParams: dict, problemParams: dict) -> bytes:
    """Build the six StreamK=3 (tree-reduction) argument slots.

    Appended after alpha/beta. sk_args must be provided in problemParams under
    the "sk" key with fields: iters_per_tile, sk_iters_per_wg, sk_grid, sk_tiles.
    The magic-number pair for iters_per_tile is computed here.
    """
    sk = problemParams.get("sk", {})
    iters_per_tile = int(sk.get("iters_per_tile", 1))
    sk_iters_per_wg = int(sk.get("sk_iters_per_wg", 0))
    sk_grid = int(sk.get("sk_grid", 1))
    sk_tiles = int(sk.get("sk_tiles", 0))

    magic, shift = _magicNumberAlg2(iters_per_tile)

    buf = struct.pack("<I", iters_per_tile)
    buf += struct.pack("<I", magic)
    buf += struct.pack("<I", shift)
    buf += struct.pack("<I", sk_iters_per_wg)
    buf += struct.pack("<I", sk_grid)
    buf += struct.pack("<I", sk_tiles)
    return buf


def _buildStreamK4Args(solutionParams: dict, problemParams: dict) -> bytes:
    """Build the six StreamK=4 (dynamic SK) argument slots.

    Appended after alpha/beta. sk4_args must be provided in problemParams under
    the "sk4" key with fields: iters_per_tile, tiles, sk_tiles, sk_split, sk_grid.
    sk_iters_per_wi and total_items are computed here via the CeilDivide pattern
    from ContractionSolution.cpp:778-806.
    """
    sk4 = problemParams.get("sk4", {})
    iters_per_tile = int(sk4.get("iters_per_tile", 1))
    tiles = int(sk4.get("tiles", 1))
    sk_tiles = int(sk4.get("sk_tiles", 0))
    sk_split = int(sk4.get("sk_split", 2))
    sk_grid = int(sk4.get("sk_grid", 1))

    # Mirrors CeilDivide(itersPerTile, skSplit) then recalculate skSplit.
    sk_iters_per_wi = math.ceil(iters_per_tile / max(1, sk_split))
    sk_split = math.ceil(iters_per_tile / max(1, sk_iters_per_wi))
    total_items = (tiles - sk_tiles) + sk_tiles * sk_split

    buf = struct.pack("<I", iters_per_tile)
    buf += struct.pack("<I", total_items)
    buf += struct.pack("<I", sk_tiles)
    buf += struct.pack("<I", sk_split)
    buf += struct.pack("<I", sk_iters_per_wi)
    buf += struct.pack("<I", sk_grid)
    return buf


def _buildStreamK5DynamicArgs(sk5: dict) -> bytes:
    """Build six SK5-dynamic slots (same layout as SK4 plus the mode bit).

    Mode bit 30 of the SKTiles slot signals the dynamic sub-mode to the kernel.
    """
    iters_per_tile = int(sk5.get("iters_per_tile", 1))
    tiles = int(sk5.get("tiles", 1))
    sk_tiles = int(sk5.get("sk_tiles", 0))
    sk_split = int(sk5.get("sk_split", 2))
    sk_grid = int(sk5.get("sk_grid", 1))

    sk_iters_per_wi = math.ceil(iters_per_tile / max(1, sk_split))
    sk_split = math.ceil(iters_per_tile / max(1, sk_iters_per_wi))
    total_items = (tiles - sk_tiles) + sk_tiles * sk_split
    # Bit 30 marks the dynamic (SK4-like) sub-mode; bit 31 is reserved for magic-div.
    packed_sk_tiles = sk_tiles | 0x40000000

    buf = struct.pack("<I", iters_per_tile)
    buf += struct.pack("<I", total_items)
    buf += struct.pack("<I", packed_sk_tiles)
    buf += struct.pack("<I", sk_split)
    buf += struct.pack("<I", sk_iters_per_wi)
    buf += struct.pack("<I", sk_grid)
    return buf


def _buildStreamK5StaticArgs(sk5: dict) -> bytes:
    """Build six SK5-static slots (mirrors standalone SK3 arg packing).

    Accepts pre-computed sk_iters_per_wg and sk_tiles; magic numbers are
    derived here from iters_per_tile using the same algorithm as SK3.
    """
    iters_per_tile = int(sk5.get("iters_per_tile", 1))
    sk_iters_per_wg = int(sk5.get("sk_iters_per_wg", 0))
    sk_grid = int(sk5.get("sk_grid", 1))
    sk_tiles = int(sk5.get("sk_tiles", 0))

    magic, shift = _magicNumberAlg2(iters_per_tile)

    buf = struct.pack("<I", iters_per_tile)
    buf += struct.pack("<I", magic)
    buf += struct.pack("<I", shift)
    buf += struct.pack("<I", sk_iters_per_wg)
    buf += struct.pack("<I", sk_grid)
    buf += struct.pack("<I", sk_tiles)
    return buf


def _buildStreamK5Args(solutionParams: dict, problemParams: dict) -> bytes:
    """Build six StreamK=5 (hybrid SK) argument slots.

    Dispatches to the dynamic (SK4-like) or static (SK3-like) sub-path based
    on problemParams["sk5"]["effective_dynamic"]. The caller must determine
    effectiveDynamic from ContractionSolution::streamK5EffectiveDynamic.
    """
    sk5 = problemParams.get("sk5", {})
    if sk5.get("effective_dynamic", True):
        return _buildStreamK5DynamicArgs(sk5)
    return _buildStreamK5StaticArgs(sk5)


def _readPTFlag(solutionParams: dict, key: str, default=None):
    """Read an epilogue flag from solutionParams, trying top-level then ProblemType.

    Handles both flat dicts (top-level key) and compiled Solution dicts
    (where epilogue flags live in a nested ProblemType sub-dict).
    """
    if key in solutionParams:
        return solutionParams[key]
    pt = solutionParams.get("ProblemType") or {}
    return pt.get(key, default)


def _buildEpilogueArgs(solutionParams: dict, problemParams: dict, tensors: dict) -> bytes:
    """Build epilogue argument slots after alpha/beta.

    Ports the epilogue section of ContractionSolution.cpp:singleCallArgs
    (lines 997-1113) covering ScaleAB, ScaleCD, ScaleAlphaVec, Bias,
    factorDim, E tensor, activation args, and AmaxD.
    """
    useScaleAB = _readPTFlag(solutionParams, "UseScaleAB", "")
    useScaleCD = bool(_readPTFlag(solutionParams, "UseScaleCD", False))
    useScaleAlphaVec = int(_readPTFlag(solutionParams, "UseScaleAlphaVec", 0))
    useBias = int(_readPTFlag(solutionParams, "UseBias", 0))
    useE = bool(_readPTFlag(solutionParams, "UseE", False))
    outputAmaxD = bool(_readPTFlag(solutionParams, "OutputAmaxD", False))
    useGradient = bool(_readPTFlag(solutionParams, "Gradient", False))
    activationType = _readPTFlag(solutionParams, "ActivationType", "none")
    activationFused = bool(solutionParams.get("ActivationFused",
                           (_readPTFlag(solutionParams, "ActivationFused", True))))
    stridedBatched = solutionParams.get("StridedBatched", True)
    actStr = str(activationType).lower() if activationType else "none"
    runActivation = (actStr not in ("none", "0")) and activationFused

    buf = _buildScaleAbSlots(useScaleAB, tensors)
    buf += _buildScaleCdSlots(useScaleCD, tensors)
    buf += _buildScaleAlphaVecSlot(useScaleAlphaVec, tensors)
    buf += _buildBiasSlots(useBias, useGradient, stridedBatched, problemParams, tensors)
    buf += _buildFactorDimSlot(useScaleAlphaVec, useBias, problemParams)
    buf += _buildESlots(useE, problemParams, tensors)
    buf += _buildActivationSlots(runActivation, actStr, solutionParams, problemParams)
    buf += _buildAmaxDSlots(outputAmaxD, tensors)
    return buf


def _buildScaleAbSlots(useScaleAB: str, tensors: dict) -> bytes:
    """Append scaleA + scaleB pointers when UseScaleAB is non-empty."""
    if not useScaleAB:
        return b""
    return _packPtr(tensors.get("scaleA", 0)) + _packPtr(tensors.get("scaleB", 0))


def _buildScaleCdSlots(useScaleCD: bool, tensors: dict) -> bytes:
    """Append scaleC + scaleD pointers when UseScaleCD is True."""
    if not useScaleCD:
        return b""
    return _packPtr(tensors.get("scaleC", 0)) + _packPtr(tensors.get("scaleD", 0))


def _buildScaleAlphaVecSlot(useScaleAlphaVec: int, tensors: dict) -> bytes:
    """Append scaleAlphaVec pointer when UseScaleAlphaVec is non-zero."""
    if not useScaleAlphaVec:
        return b""
    return _packPtr(tensors.get("scaleAlphaVec", 0))


def _buildBiasSlots(
    useBias: int,
    useGradient: bool,
    stridedBatched: bool,
    problemParams: dict,
    tensors: dict,
) -> bytes:
    """Append bias pointer + bias_type + strideBias when UseBias is non-zero."""
    if not useBias:
        return b""
    if stridedBatched:
        buf = _packPtr(tensors.get("bias", 0))
    else:
        buf = _packPtr(tensors.get("batchBias", 0))
    # bias_type and strideBias: appended when not gradient, or gradient with A/B biasSrc.
    # M5 only supports useGradient=False (the common case).
    if not useGradient:
        biasType = int(problemParams.get("biasType", 0))
        # strideBias=0 signals non-batched bias; the kernel uses SizeI as the SRD bound.
        strideBias = int(problemParams.get("strideBias", 0))
        buf += struct.pack("<I", biasType)
        buf += struct.pack("<I", strideBias)
    return buf


def _buildFactorDimSlot(useScaleAlphaVec: int, useBias: int, problemParams: dict) -> bytes:
    """Append factorDim when UseScaleAlphaVec==3 or UseBias==3."""
    if useScaleAlphaVec != 3 and useBias != 3:
        return b""
    factorDim = int(problemParams.get("factorDim", 0))
    return struct.pack("<I", factorDim)


def _buildESlots(useE: bool, problemParams: dict, tensors: dict) -> bytes:
    """Append E tensor pointer and strides when UseE is True."""
    if not useE:
        return b""
    buf = _packPtr(tensors.get("e", 0))
    sizes = problemParams["sizes"]
    batched = len(sizes) == 4
    buf += struct.pack("<I", int(problemParams.get("lde", 0)))
    if batched:
        buf += struct.pack("<I", int(problemParams.get("stride_e", 0)))
    return buf


def _buildActivationSlots(
    runActivation: bool,
    actStr: str,
    solutionParams: dict,
    problemParams: dict,
) -> bytes:
    """Append activation scalar args and optional activationType enum."""
    if not runActivation:
        return b""
    from .reference import _ACT_ARG_COUNT
    argCount = int(solutionParams.get("ActivationArgLength",
                   _ACT_ARG_COUNT.get(actStr, 0)))
    actArgs = problemParams.get("activationArgs", [])
    buf = b""
    for i in range(argCount):
        val = float(actArgs[i]) if i < len(actArgs) else 0.0
        buf += struct.pack("<f", val)
    if actStr in ("all", "hipblaslt_all"):
        actEnum = int(problemParams.get("activationEnum", 0))
        buf += struct.pack("<I", actEnum)
    return buf


def _buildAmaxDSlots(outputAmaxD: bool, tensors: dict) -> bytes:
    """Append AddrAmaxOut + AmaxWS + AmaxSync pointers when OutputAmaxD is True."""
    if not outputAmaxD:
        return b""
    buf = _packPtr(tensors.get("amaxD", 0))
    buf += _packPtr(tensors.get("amaxWS", 0))
    buf += _packPtr(tensors.get("amaxSync", 0))
    return buf


def buildKernelArgs(
    solutionParams: dict,
    problemParams: dict,
    tensors: dict,
    cu_count: int = 0,
) -> bytes:
    """Build the raw argument buffer for a TensileLite GEMM kernel.

    Ports ContractionSolution.cpp:singleCallArgs (lines 548-1135) and
    kernelArgs (lines 1557-1714) to Python, including epilogue arg slots
    from lines 997-1113 (ScaleAB, ScaleCD, ScaleAlphaVec, Bias, E, Activation,
    AmaxD).

    solutionParams: solution dict from enumerateAllSolutions (includes
                    InternalArgsSupport fields injected by task 0.8).
                    Required fields: KernArgsVersion, UseUniversalArgs,
                    UseSFC, SupportCustomWGM, SupportCustomStaggerU,
                    MacroTile0, MacroTile1, WorkGroupMapping,
                    StaggerU, StaggerUMapping, _staggerStrideShift,
                    StreamK (0/3/4/5), StreamKAtomic, GlobalSplitU,
                    GlobalSplitUCoalesced, GlobalSplitUWorkGroupMappingRoundRobin,
                    StridedBatched, UseBeta.
                    Epilogue flags are read from the nested ProblemType sub-dict
                    or from the top-level dict directly.

    problemParams:  problem dimensions with keys: sizes ([M, N, batch, K] or [M, N, K]),
                    ldd, ldc, lda, ldb, stride_d, stride_c, stride_a, stride_b,
                    alpha, beta, gsu (default 1).
                    MX scale tensors (when MXBlockA/B non-zero in solutionParams):
                      ld_mxsa, stride_mxsa  — leading dim and batch stride for scale A
                      ld_mxsb, stride_mxsb  — leading dim and batch stride for scale B
                    Epilogue params (optional, default to zero/empty):
                      biasType, strideBias, factorDim,
                      lde, stride_e, activationArgs, activationEnum.

    tensors:        device pointers (int) keyed by 'D', 'C', 'A', 'B'
                    (for stridedBatched=True), or 'batchD', 'batchC', 'batchA',
                    'batchB' (for stridedBatched=False), and optionally
                    'workspace', 'flags' (for streamK > 0),
                    'mxsa', 'mxsb' (for MX-scaled kernels),
                    'bias', 'scaleA', 'scaleB', 'scaleC', 'scaleD',
                    'scaleAlphaVec', 'e', 'amaxD', 'amaxWS', 'amaxSync'.

    cu_count:       device CU count (multiprocessor_count) needed when
                    WorkGroupMappingXCCGroup == -1. Pass 0 when not using
                    XCC-aware WGM; defaults to 0.

    Returns raw bytes suitable for use as the kernel argument buffer.
    """
    _validateConfig(solutionParams, problemParams)

    # kernelArgs header comes first when useUniversalArgs=True (always for M1).
    buf = _buildKernelArgsHeader(solutionParams, problemParams, cu_count=cu_count)
    buf += _buildProblemSizes(problemParams)
    buf += _buildPointers(solutionParams, tensors)
    buf += _buildStrides(solutionParams, problemParams)
    buf += _buildAlphaBeta(solutionParams, problemParams)

    streamK = solutionParams.get("StreamK", 0)
    if streamK == 3:
        buf += _buildStreamK3Args(solutionParams, problemParams)
    elif streamK == 4:
        buf += _buildStreamK4Args(solutionParams, problemParams)
    elif streamK == 5:
        buf += _buildStreamK5Args(solutionParams, problemParams)

    buf += _buildEpilogueArgs(solutionParams, problemParams, tensors)
    return buf
