# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build raw kernel argument bytes for TensileLite GEMM kernels.

This module ports the argument-layout logic from
ContractionSolution.cpp:singleCallArgs (lines 548-1130) and kernelArgs
(lines 1557-1714) to Python.

Supported configuration subset (M1-M5; extended per milestone):
  - stridedBatched in {True, False}
  - streamK in {0, 3}  (M6+ adds 4, 5)
  - streamKAtomic = 0 only (atomic=1 skips workspace+flags block)
  - groupedGemm = False
  - GSU = 1, globalAccumulation in {0, 1, 2}
  - useInitialStrides = False
  - KernArgsVersion <= 2, useSFC = False
  - expertSchedulingMode = 0, debugKernel = False

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
_DTYPE_HALF: int = 4     # rocisa::DataType::Half
_DTYPE_INT32: int = 6    # rocisa::DataType::Int32
_DTYPE_INT8: int = 8     # rocisa::DataType::Int8
_DTYPE_XF32: int = 10    # rocisa::DataType::XFloat32


def _validateConfig(solutionParams: dict, problemParams: dict) -> None:
    """Raise NotImplementedError for unsupported configurations."""
    if solutionParams.get("debugKernel", False):
        raise NotImplementedError("debugKernel=True not supported")
    streamK = solutionParams.get("StreamK", 0)
    streamKAtomic = solutionParams.get("StreamKAtomic", 0)
    if streamKAtomic != 0:
        raise NotImplementedError("streamKAtomic != 0 not supported (skips workspace+flags block)")
    if streamK not in (0, 3):
        raise NotImplementedError(
            f"streamK={streamK} not supported; supported: 0 (standard), 3 (two-tile SK)"
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
    StreamK parallel-reduction replaces d/c with ws_d/ws_c; tree-reduction
    uses d/c directly (workspace is appended separately by _buildStreamKWorkspace).
    """
    stridedBatched = solutionParams.get("StridedBatched", True)
    streamK = solutionParams.get("StreamK", 0)
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
        buf += _packPtr(tensors["B"])
    else:
        buf += _packPtr(tensors["batchA"])
        buf += _packPtr(tensors["batchB"])

    # StreamK workspace + flags (tree reduction: flags = Synchronizer pointer).
    if streamK > 0:
        buf += _packPtr(tensors.get("workspace", 0))
        buf += _packPtr(tensors.get("flags", 0))

    return buf


def _buildStrides(solutionParams: dict, problemParams: dict) -> bytes:
    """Build stride uint32 slots for D, C, A, B.

    With useInitialStridesCD=False and useInitialStridesAB=False, strides start
    from dimension index 1 (the C++ startStrideCD/startStrideAB = 1 case).
    For batched GEMM (4-element sizes), each tensor has 3 dimensions → 2 strides.
    For non-batched GEMM (3-element sizes), each tensor has 2 dimensions → 1 stride.
    """
    sizes = problemParams["sizes"]
    # Batched GEMM: 4 problem sizes [M, N, batch, K] → 3D tensors → 2 strides per tensor.
    batched = len(sizes) == 4
    buf = b""

    def packStrides(ld: int, batchStride: int) -> bytes:
        out = struct.pack("<I", ld)
        if batched:
            out += struct.pack("<I", batchStride)
        return out

    buf += packStrides(problemParams["ldd"], problemParams.get("stride_d", 0))
    buf += packStrides(problemParams["ldc"], problemParams.get("stride_c", 0))
    buf += packStrides(problemParams["lda"], problemParams.get("stride_a", 0))
    buf += packStrides(problemParams["ldb"], problemParams.get("stride_b", 0))
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
    return _readComputeTypeCode(solutionParams) == _DTYPE_HALF


def _alphaTypeIsInt32(solutionParams: dict) -> bool:
    """Return True when alpha/beta must be packed as int32 (non-HPA int8 GEMM).

    For int8 GEMM without HighPrecisionAccumulate, the compute type is Int32
    and alpha/beta are packed as 4-byte signed integers, not floats.
    """
    if _readHPA(solutionParams):
        return False
    return _readComputeTypeCode(solutionParams) == _DTYPE_INT32


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


def buildKernelArgs(
    solutionParams: dict,
    problemParams: dict,
    tensors: dict,
    cu_count: int = 0,
) -> bytes:
    """Build the raw argument buffer for a TensileLite GEMM kernel.

    Ports ContractionSolution.cpp:singleCallArgs (lines 548-1135) and
    kernelArgs (lines 1557-1714) to Python.

    solutionParams: solution dict from enumerateAllSolutions (includes
                    InternalArgsSupport fields injected by task 0.8).
                    Required fields: KernArgsVersion, UseUniversalArgs,
                    UseSFC, SupportCustomWGM, SupportCustomStaggerU,
                    MacroTile0, MacroTile1, WorkGroupMapping,
                    StaggerU, StaggerUMapping, _staggerStrideShift,
                    StreamK, StreamKAtomic, GlobalSplitU,
                    GlobalSplitUCoalesced, GlobalSplitUWorkGroupMappingRoundRobin,
                    StridedBatched, UseBeta.

    problemParams:  problem dimensions with keys: sizes ([M, N, batch, K] or [M, N, K]),
                    ldd, ldc, lda, ldb, stride_d, stride_c, stride_a, stride_b,
                    alpha, beta, gsu (default 1).

    tensors:        device pointers (int) keyed by 'D', 'C', 'A', 'B'
                    (for stridedBatched=True), or 'batchD', 'batchC', 'batchA',
                    'batchB' (for stridedBatched=False), and optionally
                    'workspace', 'flags' (for streamK > 0).

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

    if solutionParams.get("StreamK", 0) == 3:
        buf += _buildStreamK3Args(solutionParams, problemParams)

    return buf
