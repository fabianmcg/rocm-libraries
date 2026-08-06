# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SweepRunner: enumerate, compile, and benchmark solutions from a YAML.

Two operating modes are supported:

Compile mode (libraryPath=None): enumerate ALL fork solutions from the YAML,
compile each to HSACO, and benchmark every (problem_size, solution) pair.
The results CSV has one GFLOPS column per solution, one row per size.

Library mode (libraryPath set): load a pre-built TensileLibrary.yaml.  For
each problem size, LibraryRunner.find_best selects the single best solution;
only that winner is benchmarked via the same KernelRunner path as compile mode.
The results CSV has a single Winner column, one row per size.  No compilation.
The .co file is loaded from the library directory (the same pre-built code
object the C++ client uses).
"""

from __future__ import annotations

import configparser
import ctypes
import functools
import logging
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from Tensile.client.harness import BenchmarkResult, BufferPool, KernelRunner
from Tensile.client.gemm_args import _computeInternalArg0, _computeInternalArg1, _readMxBlock
from Tensile.client.reporters import LibraryUpdateReporter, ResultsCSVReporter

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SweepResult — wraps BenchmarkResult with problem/solution metadata.
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Benchmark result for one (problem_size, solution) pair."""

    solutionIdx: int
    solutionName: str
    problemSize: tuple
    benchmark: BenchmarkResult
    gflops: float
    validation: str = "SKIPPED"


# ---------------------------------------------------------------------------
# Tensile setup helpers (mirrors test_gemm_standard.py pattern).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _cachedIsaInfoMap(isa, cxx: str):
    """Return ISA capability map for one ISA version, cached to avoid repeated amdclang++ invocations."""
    from Tensile.Common.Capabilities import makeIsaInfoMap
    return makeIsaInfoMap([isa], cxx)


def _setupTensile(chip: str):
    """Initialize Tensile assembler + ISA map for kernel compilation."""
    from pathlib import Path
    from Tensile.Toolchain.Validators import validateToolchain
    from Tensile.Toolchain.Component import Assembler
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.GlobalParameters import assignGlobalParameters
    from Tensile.Common.Types import DebugConfig

    gfx = chip.split(":")[0]
    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa(gfx)
    isaInfoMap = _cachedIsaInfoMap(isa, cxx)
    assignGlobalParameters({}, isaInfoMap)
    assembler = Assembler(Path(cxx), co_version="6")
    return assembler, isaInfoMap, DebugConfig()


def _generateAsm(solution, assembler, debugConfig):
    """Return (asm_string, kernel_name) for a Solution object."""
    import rocisa
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelNameMin

    kwa = KernelWriterAssembly(assembler, debugConfig)
    ti = rocisa.rocIsa.getInstance()
    kwa.setRocIsa(ti.getData(), ti.getOutputOptions())
    kernel = solution.getKernels()[0]
    kernel.duplicate = False
    err, asmStr = kwa.getSourceFileString(kernel)
    if err:
        raise RuntimeError(f"assembly generation failed: {err}")
    return asmStr, getKernelNameMin(kernel, splitGSU=False)


def _deviceCuCount() -> int:
    """Return the device compute-unit count for device 0."""
    try:
        import amdgpu_exec
        props = amdgpu_exec._runtime_module.hip_get_device_props(0)
        return int(props.get("multiprocessor_count", 0))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Element-size helpers.
# ---------------------------------------------------------------------------


def _elemSize(dtField) -> int:
    """Return element size in bytes from a Tensile DataType field.

    Uses DataType.numBytes() when available; falls back to a numeric code table.
    """
    if dtField is not None and hasattr(dtField, "numBytes"):
        n = dtField.numBytes()
        return n if n > 0 else 4
    # Numeric fallback: Float=0→4B, Double=1→8B, Half=4→2B, BFloat16=7→2B.
    _TABLE = {0: 4, 1: 8, 4: 2, 7: 2, 8: 2, 9: 2}
    try:
        return _TABLE.get(int(dtField), 4)
    except (TypeError, ValueError):
        return 4


def _aElemSize(solDict: dict) -> int:
    """Return element size in bytes for A/B operands."""
    return _elemSize(solDict.get("DataType", None))


def _dElemSize(solDict: dict) -> int:
    """Return element size in bytes for D/C operands."""
    return _elemSize(solDict.get("DestDataType", solDict.get("DataType", None)))


def _dtypeValue(dtField) -> int:
    """Return the integer DataTypeEnum value from a DataType object or int."""
    if dtField is None:
        return -1
    return int(dtField.value) if hasattr(dtField, "value") else int(dtField)


def _readDtypeInt(solDict: dict, key: str) -> int:
    """Read a DataType integer from solDict, checking top-level then ProblemType.

    Compiled Solution dicts store per-type fields inside a nested ProblemType
    mapping (as DataType enum objects); manually constructed dicts may store
    integer codes at the top level.  This mirrors the pattern in gemm_args.py.
    """
    val = solDict.get(key)
    if val is not None:
        return _dtypeValue(val)
    pt = solDict.get("ProblemType") or {}
    val = pt.get(key) if pt else None
    return _dtypeValue(val)


# DataTypeEnum.XFloat32 integer code — fp32 storage with truncated-mantissa math.
_DTYPE_XF32 = 10


def _getFp8MlDtype(typeCode: int):
    """Return the ml_dtypes numpy type for a fp8/bf8 dtype code, or None.

    Returns None if ml_dtypes is not installed or the code is unrecognised.
    """
    try:
        import ml_dtypes
    except ImportError:
        return None
    _fp8Map = {
        11: ml_dtypes.float8_e4m3fnuz,
        12: ml_dtypes.float8_e5m2fnuz,
        15: ml_dtypes.float8_e4m3fn,
        16: ml_dtypes.float8_e5m2,
    }
    return _fp8Map.get(typeCode)


def _selectRefStd(inType: int):
    """Return 5-tuple for fp32 (0), fp16 (4), or bf16 (7) standard GEMM, or None."""
    from Tensile.client.reference import (
        gemm, gemmFp16, gemmBf16,
        RTOL_FP32, ATOL_FP32, RTOL_FP16, ATOL_FP16, RTOL_BF16, ATOL_BF16,
    )
    if inType == 0:
        return np.float32, np.float32, gemm, RTOL_FP32, ATOL_FP32
    if inType == 4:
        return np.float16, np.float16, gemmFp16, RTOL_FP16, ATOL_FP16
    if inType == 7:
        try:
            import ml_dtypes
        except ImportError:
            return None
        return ml_dtypes.bfloat16, ml_dtypes.bfloat16, gemmBf16, RTOL_BF16, ATOL_BF16
    return None


def _selectRefXf32():
    """Return 5-tuple for XFloat32 math GEMM (fp32 storage, 10-bit mantissa compute)."""
    from Tensile.client.reference import gemmXf32, RTOL_XF32, ATOL_XF32
    return np.float32, np.float32, gemmXf32, RTOL_XF32, ATOL_XF32


def _selectRefInt8(outType: int):
    """Return 5-tuple for int8 input GEMM (int32 or int8 output), or None.

    Wraps gemmInt8 in a closure so the caller can treat it as a standard
    (A, B, alpha, beta, C) → D function regardless of the outputInt8 flag.
    """
    from Tensile.client.reference import gemmInt8, RTOL_INT8, ATOL_INT8
    if outType not in (6, 8):
        return None
    outputInt8 = (outType == 8)
    npOutDtype = np.int8 if outputInt8 else np.int32

    def refFn(A, B, alpha, beta, C):
        return gemmInt8(A, B, alpha, beta, C, outputInt8=outputInt8)

    return np.int8, npOutDtype, refFn, RTOL_INT8, ATOL_INT8


def _selectRefFp8(inType: int, outType: int):
    """Return 5-tuple for fp8/bf8 input GEMM (HPA, fp32 or fp8 output), or None.

    Wraps gemmFp8 in a closure capturing the dtype arguments so the caller
    can use the standard (A, B, alpha, beta, C) → D signature.
    """
    from Tensile.client.reference import gemmFp8, RTOL_FP8, ATOL_FP8
    mlDtypeIn = _getFp8MlDtype(inType)
    if mlDtypeIn is None:
        return None
    if outType in (0, -1):
        npOutDtype = np.float32
        mlDtypeOut = np.float32
    else:
        mlDtypeOut = _getFp8MlDtype(outType)
        if mlDtypeOut is None:
            return None
        npOutDtype = mlDtypeOut

    def refFn(A, B, alpha, beta, C):
        return gemmFp8(A, B, mlDtypeIn, mlDtypeIn, mlDtypeOut, alpha, beta, C)

    return mlDtypeIn, npOutDtype, refFn, RTOL_FP8, ATOL_FP8


def _selectReference(solDict: dict):
    """Return (npDtype, npOutDtype, refFn, rtol, atol) for a supported GEMM, or None.

    npDtype is the numpy dtype for A and B inputs; npOutDtype is the dtype for
    the D output buffer (may differ, e.g. int8→int32 or fp8→fp32). refFn always
    has the signature (A, B, alpha, beta, C) → D. Returns None so the caller
    records SKIPPED for StreamK!=0, MX-scaled kernels, mismatched types, and
    dtypes without a reference implementation.
    """
    if solDict.get("StreamK", 0) != 0:
        return None
    # MX block-scaled GEMM requires scale tensors not available in basic verification.
    if _readMxBlock(solDict, "A") or _readMxBlock(solDict, "B"):
        return None
    inType = _readDtypeInt(solDict, "DataType")
    outType = _readDtypeInt(solDict, "DestDataType")
    if outType == -1:
        outType = inType
    if inType in (0, 4, 7):
        if inType != outType:
            return None
        if inType == 0 and _readDtypeInt(solDict, "F32XdlMathOp") == _DTYPE_XF32:
            return _selectRefXf32()
        return _selectRefStd(inType)
    if inType == 8:
        return _selectRefInt8(outType)
    if inType in (11, 12, 15, 16):
        return _selectRefFp8(inType, outType)
    return None


# ---------------------------------------------------------------------------
# Library-YAML solDict builder — constructs a solDict from a TensileLibrary
# solution entry without invoking any part of the Tensile parameter pipeline.
# ---------------------------------------------------------------------------

# tensilelite_runtime dtype name -> DataTypeEnum integer code.
_rtNameToDtypeInt = {v: k for k, v in {
    0: "Float", 1: "Double", 4: "Half", 7: "BFloat16", 8: "Int8", 10: "XFloat32",
}.items()}

# tensilelite_runtime / library YAML dtype name -> DataTypeEnum integer code.
# Library YAML uses the full CamelCase names used by Tensile internally.
_libDtypeNameToInt = {
    "Float": 0, "Double": 1, "Half": 4, "BFloat16": 7, "Int8": 8,
    "XFloat32": 10, "Int32": 6, "Int16": 5,
    "Float8": 11, "BFloat8": 12,
    "Float8_fnuz": 11, "BFloat8_fnuz": 12,
    "Float8_ocp": 15, "BFloat8_ocp": 16,
}


def _libDtypeToInt(name: str) -> int:
    """Convert a library-YAML type name string to a DataTypeEnum integer."""
    return _libDtypeNameToInt.get(name, -1)


def _solDictFromLibSol(libSol: dict, chip: str) -> dict:
    """Build a minimal solDict from a TensileLibrary YAML solution entry.

    This extracts exactly the fields that _computeInternalArg0/1, _buildSweepArgs,
    _computeNumWg, and _selectReference need, reading them directly from the
    sizeMapping, problemType, and internalArgsSupport sub-dicts of the library
    YAML solution object.  No Tensile solution pipeline is invoked.
    """
    sm = libSol.get("sizeMapping", {})
    pt = libSol.get("problemType", {})
    ias = libSol.get("internalArgsSupport", {})

    macroTile = sm.get("macroTile", [16, 16, 1])
    wg = sm.get("workGroup", [16, 16, 1])
    waveNum = sm.get("waveNum", 1)
    # SubtileImpl kernels (useSubtileImpl=True) use workGroup as the HIP block
    # directly — each element is a hardware thread, not a wavefront.  Standard
    # kernels count workGroup * waveNum threads per block.
    useSubtileImpl = bool(sm.get("useSubtileImpl", False))

    aTypeName = pt.get("aType", "BFloat16")
    dTypeName = pt.get("dType", aTypeName)
    aTypeInt = _libDtypeToInt(aTypeName)
    dTypeInt = _libDtypeToInt(dTypeName)

    # F32XdlMathOp: XFloat32=10 when f32XdlMathOp='XFloat32', else Float=0.
    f32XdlMathOp = 10 if pt.get("f32XdlMathOp") == "XFloat32" else 0

    staggerStrideShift = sm.get("staggerStrideShift", 0)

    solDict = {
        # KernArgs version / internal-args flags.
        "KernArgsVersion": ias.get("version", 0),
        "SupportCustomWGM": ias.get("wgm", False),
        "SupportCustomStaggerU": ias.get("staggerU", False),
        "SupportUserGSU": ias.get("gsu", False),
        "UseSFC": ias.get("useSFC", False),
        "UseUniversalArgs": ias.get("useUniversalArgs", False),
        # Tile geometry.
        "MacroTile0": macroTile[0],
        "MacroTile1": macroTile[1],
        "NumThreads": wg[0] * wg[1] * wg[2] if useSubtileImpl else wg[0] * wg[1] * wg[2] * waveNum,
        # Work-group mapping and stagger.
        "WorkGroupMapping": sm.get("workGroupMapping", 8),
        "WorkGroupMappingXCC": sm.get("workGroupMappingXCC", 0),
        "WorkGroupMappingXCCGroup": sm.get("workGroupMappingXCCGroup", -1),
        "StaggerU": sm.get("staggerU", 0),
        "StaggerUMapping": sm.get("staggerUMapping", 0),
        "_staggerStrideShift": staggerStrideShift,
        # GSU.
        "GlobalSplitU": sm.get("globalSplitU", 1),
        "GlobalSplitUCoalesced": sm.get("globalSplitUCoalesced", False),
        "GlobalSplitUWorkGroupMappingRoundRobin": sm.get("globalSplitUWorkGroupMappingRoundRobin", False),
        # Stream-K.
        "StreamK": sm.get("streamK", 0),
        "StreamKAtomic": sm.get("streamKAtomic", 0),
        # Misc kernel behaviour.
        "StridedBatched": pt.get("stridedBatched", True),
        "UseBeta": pt.get("useBeta", True),
        "GlobalAccumulation": sm.get("globalAccumulation", 0),
        "ExpertSchedulingMode": sm.get("expertSchedulingMode", 0),
        "ActivationFused": sm.get("activationFused", True),
        # GlobalAccumulation / MBSK fields needed for buffer sizing and arg layout.
        "AdaptiveGemmGSUA": sm.get("adaptiveGemmGSUA", 0),
        "WorkspaceSizePerElemC": sm.get("workspaceSizePerElemC", 4),
        "SynchronizerSizePerWG": sm.get("synchronizerSizePerWG", 0),
        # Types (integer codes for _selectReference / _aElemSize / _dElemSize).
        "DataType": aTypeInt,
        "DestDataType": dTypeInt,
        "F32XdlMathOp": f32XdlMathOp,
        # ProblemType sub-dict for stride / epilogue / reference selection.
        "ProblemType": {
            "TransposeA": pt.get("transA", False),
            "TransposeB": pt.get("transB", False),
            "HighPrecisionAccumulate": pt.get("highPrecisionAccumulate", False),
            "GroupedGemm": pt.get("groupedGemm", False),
            "DataType": aTypeInt,
            "DestDataType": dTypeInt,
            "F32XdlMathOp": f32XdlMathOp,
            # Epilogue flags — forwarded directly from problemType.
            "UseBias": int(pt.get("useBias", 0)),
            "UseScaleAB": pt.get("useScaleAB", ""),
            "UseScaleCD": bool(pt.get("useScaleCD", False)),
            "UseScaleAlphaVec": int(pt.get("useScaleAlphaVec", 0)),
            "UseE": bool(pt.get("useE", False)),
            "OutputAmaxD": bool(pt.get("outputAmaxD", False)),
            "Gradient": bool(pt.get("useGradient", False)),
            "ActivationType": pt.get("activationType", "none"),
        },
    }
    # Duplicate epilogue flags at top level so _readPTFlag finds them either way.
    for key in ("UseBias", "UseScaleAB", "UseScaleCD", "UseScaleAlphaVec",
                "UseE", "OutputAmaxD", "Gradient", "ActivationType"):
        solDict[key] = solDict["ProblemType"][key]
    return solDict


# ---------------------------------------------------------------------------
# GlobalAccumulation mode helpers.
#
# Library YAML sizeMapping.globalAccumulation numeric values (library-specific
# — may differ from current Contractions.py):
#   0  standard store to D
#   2  interleaved fp32 partials (KernelWriter bpeCexternal = fp32 at gsu>0)
#   4  MultipleBufferSingleKernel (MBSK): single kernel, trailing kernarg slots
# ---------------------------------------------------------------------------


def _gaMode(solDict: dict) -> int:
    """Return the GlobalAccumulation integer from solDict."""
    return int(solDict.get("GlobalAccumulation", 0))


def _isMbsk(solDict: dict) -> bool:
    """Return True for MultipleBufferSingleKernel kernels (GA=4 or AdaptiveGemmGSUA=1).

    These kernels require three extra trailing kernarg slots: dstD, Synchronizer,
    GSUSync; and a pre-zeroed synchronizer GPU buffer.
    """
    return _gaMode(solDict) == 4 or int(solDict.get("AdaptiveGemmGSUA", 0)) == 1


def _usesFp32External(solDict: dict, gsu: int) -> bool:
    """Return True when the kernel writes fp32 into its D-pointer target.

    Mirrors KernelWriter.py:7573-7578: bpeCexternal = fp32 whenever
    gsu > 0 and GlobalAccumulation is set and is not PartialsBuffer.
    In this library GA=2 triggers this at gsu=1.
    """
    ga = _gaMode(solDict)
    return gsu > 0 and ga not in (0,)


def _storeElemSize(solDict: dict, gsu: int) -> int:
    """Return the element size in bytes the kernel uses when writing to D.

    For _usesFp32External kernels this is WorkspaceSizePerElemC (fp32 = 4),
    not _dElemSize (bf16 = 2).  Determines minimum safe D/C buffer allocation.
    """
    if _usesFp32External(solDict, gsu):
        return int(solDict.get("WorkspaceSizePerElemC", 4))
    return _dElemSize(solDict)


def _synchronizerBytes(solDict: dict, M: int, N: int, batch: int) -> int:
    """Return bytes needed for the MBSK synchronizer buffer.

    Mirrors ContractionSolution.cpp:3969-3979: tiles * SynchronizerSizePerWG * 4.
    Falls back to a conservative 409,600-element allocation (matching the C++
    ClientProblemFactory.cpp fixed workspace) when SynchronizerSizePerWG is 0.
    """
    if not _isMbsk(solDict):
        return 0
    mt0 = solDict["MacroTile0"]
    mt1 = solDict["MacroTile1"]
    tiles = math.ceil(M / mt0) * math.ceil(N / mt1) * batch
    perWg = int(solDict.get("SynchronizerSizePerWG", 0))
    if perWg > 0:
        return tiles * perWg * 4
    # Conservative fallback: match C++ ClientProblemFactory fixed buffer.
    return 409_600 * 4


# ---------------------------------------------------------------------------
# Kernel argument builder for stridedBatched NT GEMM.
# ---------------------------------------------------------------------------


def _computeNumWg(solDict: dict, M: int, N: int, batch: int) -> int:
    """Compute work-group count for stridedBatched GEMM."""
    mt0 = solDict["MacroTile0"]
    mt1 = solDict["MacroTile1"]
    return math.ceil(M / mt0) * math.ceil(N / mt1) * batch


def _computeStrides(solDict: dict, M: int, N: int, K: int):
    """Return (lda, strideA, ldb, strideB, ldd, strideD, ldc, strideC).

    Strides depend on transpose flags stored in ProblemType:
      TN (TransposeA=True,  TransposeB=False): lda=K, ldb=K
      NT (TransposeA=False, TransposeB=True):  lda=M, ldb=N
    Column-major layout: leading dim is the number of rows in the stored matrix.
    """
    pt = solDict.get("ProblemType") or {}
    transA = bool(pt.get("TransposeA", False))
    transB = bool(pt.get("TransposeB", False))
    lda = K if transA else M
    ldb = N if transB else K
    ldd = ldc = M
    strideA = lda * (M if transA else K)
    strideB = ldb * (K if transB else N)
    strideD = strideC = M * N
    return lda, strideA, ldb, strideB, ldd, strideD, ldc, strideC


def _hasEpilogue(solDict: dict) -> bool:
    """Return True when the solution requires epilogue kernel args beyond alpha/beta."""
    from Tensile.client.gemm_args import _readPTFlag
    return bool(
        _readPTFlag(solDict, "UseScaleAB", "")
        or _readPTFlag(solDict, "UseScaleCD", False)
        or _readPTFlag(solDict, "UseScaleAlphaVec", 0)
        or _readPTFlag(solDict, "UseBias", 0)
        or _readPTFlag(solDict, "UseE", False)
        or _readPTFlag(solDict, "OutputAmaxD", False)
        or _readPTFlag(solDict, "ActivationType", "none") not in ("none", "0", None, 0)
    )


def _buildEpilogueTypedArgs(solDict: dict, M: int, epilogueBufs: dict) -> list:
    """Return typed arg list items for epilogue slots (mirrors _buildEpilogueArgs byte layout).

    epilogueBufs maps tensor name -> GpuBuffer (zeroed). Missing names default to
    null pointer (0). Callers must keep the buffers alive until after kernel launch.
    """
    from Tensile.client.gemm_args import _readPTFlag
    pt = solDict.get("ProblemType") or {}

    useScaleAB = _readPTFlag(solDict, "UseScaleAB", "")
    useScaleCD = bool(_readPTFlag(solDict, "UseScaleCD", False))
    useScaleAlphaVec = int(_readPTFlag(solDict, "UseScaleAlphaVec", 0))
    useBias = int(_readPTFlag(solDict, "UseBias", 0))
    useE = bool(_readPTFlag(solDict, "UseE", False))
    outputAmaxD = bool(_readPTFlag(solDict, "OutputAmaxD", False))
    activationType = _readPTFlag(solDict, "ActivationType", "none")
    activationFused = bool(solDict.get("ActivationFused", True))
    actStr = str(activationType).lower() if activationType else "none"
    runActivation = actStr not in ("none", "0") and activationFused

    def _ptr(name):
        buf = epilogueBufs.get(name)
        return ctypes.c_void_p(buf.ptr_value if buf is not None else 0)

    args = []
    if useScaleAB:
        args.extend([_ptr("scaleA"), _ptr("scaleB")])
    if useScaleCD:
        args.extend([_ptr("scaleC"), _ptr("scaleD")])
    if useScaleAlphaVec:
        args.append(_ptr("scaleAlphaVec"))
    if useBias:
        args.append(_ptr("bias"))
        args.extend([np.uint32(0), np.uint32(0)])  # biasType=0, strideBias=0
    if useScaleAlphaVec == 3 or useBias == 3:
        args.append(np.uint32(0))  # factorDim
    if useE:
        args.append(_ptr("e"))
        args.extend([np.uint32(M), np.uint32(M * M)])  # lde, stride_e (placeholder)
    if runActivation:
        from Tensile.client.reference import _ACT_ARG_COUNT
        argCount = int(solDict.get("ActivationArgLength", _ACT_ARG_COUNT.get(actStr, 0)))
        for _ in range(argCount):
            args.append(np.float32(0.0))
        if actStr in ("all", "hipblaslt_all"):
            args.append(np.uint32(0))  # activationEnum
    if outputAmaxD:
        args.extend([_ptr("amaxD"), _ptr("amaxWS"), _ptr("amaxSync")])
    return args


def _buildSweepArgs(solDict: dict, M: int, N: int, batch: int, K: int,
                    dBuf, cBuf, aBuf, bBuf, cuCount: int = 0,
                    alpha: float = 1.0, beta: float = 0.0,
                    epilogueBufs: dict = None) -> list:
    """Build typed kernel arg list for stridedBatched GEMM.

    Uses argType=0 (stridedBatched=True) in gemm_count bits 30-31.
    Strides are computed from TransposeA/TransposeB in ProblemType.
    epilogueBufs maps epilogue tensor name -> GpuBuffer for kernels that need them.
    """
    version = solDict.get("KernArgsVersion", 0)
    numWg = _computeNumWg(solDict, M, N, batch)
    arg0 = _computeInternalArg0(solDict, gsu=1)
    gemmCount = (1 & 0x3FFFFFFF) | (0 << 30)

    args = [np.uint32(gemmCount), np.uint32(arg0)]
    if version >= 1:
        arg1 = _computeInternalArg1(solDict, cu_count=cuCount)
        args.append(np.int32(arg1))
        args.append(np.uint32(numWg))

    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])
    for ptr in [dBuf.ptr_value, cBuf.ptr_value, aBuf.ptr_value, bBuf.ptr_value]:
        args.append(ctypes.c_void_p(ptr))

    lda, strideA, ldb, strideB, ldd, strideD, ldc, strideC = _computeStrides(solDict, M, N, K)
    args.extend([
        np.uint32(ldd), np.uint32(strideD),
        np.uint32(ldc), np.uint32(strideC),
        np.uint32(lda), np.uint32(strideA),
        np.uint32(ldb), np.uint32(strideB),
        np.float32(alpha), np.float32(beta),
    ])
    if epilogueBufs is not None:
        args.extend(_buildEpilogueTypedArgs(solDict, M, epilogueBufs))
    # Batch offset args added by feat(hipblaslt): 64-bit offset support (#7585).
    # Placed at the tail of non-grouped kernarg buffers (Signature.py, line 338).
    args.extend([np.int64(0), np.int64(0), np.int64(0), np.int64(0)])
    return args


# ---------------------------------------------------------------------------
def _buildSweepArgsGA(solDict: dict, M: int, N: int, batch: int, K: int,
                      dBuf, cBuf, aBuf, bBuf, cuCount: int,
                      gsu: int, syncBuf=None,
                      alpha: float = 1.0, beta: float = 0.0,
                      epilogueBufs: dict = None) -> list:
    """Build typed kernel arg list for GlobalAccumulation != 0 kernels.

    Differences from _buildSweepArgs:
    - Real gsu is packed into internalArg0 (not hardcoded 1).
    - For MBSK (GA=4): three trailing slots (dstD, Synchronizer, GSUSync) are
      appended between the epilogue block and the batch-offset tail, and
      alpha/beta are passed as 1.0/0.0 (the kernel handles finalization).
    - For GA=2/gsu=1: the header, pointer, and stride layout are identical to
      _buildSweepArgs; only the caller's buffer sizes differ (fp32-sized D/C).
    """
    version = solDict.get("KernArgsVersion", 0)
    numWg = _computeNumWg(solDict, M, N, batch)
    arg0 = _computeInternalArg0(solDict, gsu=gsu)
    gemmCount = (1 & 0x3FFFFFFF) | (0 << 30)

    args = [np.uint32(gemmCount), np.uint32(arg0)]
    if version >= 1:
        arg1 = _computeInternalArg1(solDict, cu_count=cuCount)
        args.append(np.int32(arg1))
        args.append(np.uint32(numWg))

    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])
    for ptr in [dBuf.ptr_value, cBuf.ptr_value, aBuf.ptr_value, bBuf.ptr_value]:
        args.append(ctypes.c_void_p(ptr))

    lda, strideA, ldb, strideB, ldd, strideD, ldc, strideC = _computeStrides(solDict, M, N, K)
    args.extend([
        np.uint32(ldd), np.uint32(strideD),
        np.uint32(ldc), np.uint32(strideC),
        np.uint32(lda), np.uint32(strideA),
        np.uint32(ldb), np.uint32(strideB),
    ])

    # MBSK finalizes in the same kernel; alpha/beta are absorbed into the
    # internal accumulation, so the host passes 1/0 here.
    if _isMbsk(solDict):
        args.extend([np.float32(1.0), np.float32(0.0)])
    else:
        args.extend([np.float32(alpha), np.float32(beta)])

    if epilogueBufs is not None:
        args.extend(_buildEpilogueTypedArgs(solDict, M, epilogueBufs))

    # MBSK trailing slots: dstD, Synchronizer, GSUSync.
    # Placed after the epilogue block and before the batch-offset tail so the
    # four int64 batch offsets land at the correct byte offset (Signature.py:328-348).
    if _isMbsk(solDict):
        # dstD = final bf16 output (same as the dBuf we already passed as D).
        args.append(ctypes.c_void_p(dBuf.ptr_value))
        # Synchronizer = zeroed sync buffer for cross-WG atomics.
        syncPtr = syncBuf.ptr_value if syncBuf is not None else 0
        args.append(ctypes.c_void_p(syncPtr))
        args.append(np.uint32(0))  # GSUSync

    # Batch offset args (Signature.py:338-348, always last).
    args.extend([np.int64(0), np.int64(0), np.int64(0), np.int64(0)])
    return args


# KernelRunner factory with icache-copy resolution.
# ---------------------------------------------------------------------------


def _makeRunner(hsacoBytes: bytes, kernelName: str, icacheCopies) -> KernelRunner:
    """Create a KernelRunner, resolving icacheCopies='auto' via ELF analysis.

    When icacheCopies='auto', writes the HSACO bytes to a temp file so that
    get_icache_module_copies can inspect the ELF symbol table.  Falls back to
    1 copy if tensilelite_runtime is unavailable.
    """
    if icacheCopies != "auto":
        return KernelRunner.fromHsaco(hsacoBytes, kernelName,
                                      nModuleCopies=int(icacheCopies))

    coPath = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".co", delete=False) as f:
            f.write(hsacoBytes)
            coPath = f.name
        return KernelRunner.fromHsaco(hsacoBytes, kernelName,
                                      nModuleCopies="auto", coPath=coPath)
    finally:
        if coPath is not None:
            try:
                os.unlink(coPath)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Solution filtering (mirrors M1–M7 test filter).
# ---------------------------------------------------------------------------


def _shouldSkip(solDict: dict) -> bool:
    """Return True if this solution must be excluded from a compile-mode sweep.

    Excludes MX block-scaled kernels, whose scale-tensor arguments are not
    built by _buildSweepArgs, and WorkGroupMapping=0 kernels, whose auto-WGM
    value can only be resolved from a loaded C++ library (library mode).
    StaggerU=0 is no longer excluded: for the streamK=0 path the kernel arg
    packs StaggerU verbatim, so StaggerU=0 already matches the C++ client.
    """
    if _readMxBlock(solDict, "A") or _readMxBlock(solDict, "B"):
        return True
    return solDict.get("WorkGroupMapping", 0) == 0


# DataTypeEnum integer code -> tensilelite_runtime dtype name.
_rtDtypeNames = {0: "Float", 1: "Double", 4: "Half", 7: "BFloat16",
                 8: "Int8", 10: "XFloat32"}


def _dtypeIntToRtName(code: int) -> str:
    """Map a DataTypeEnum integer code to a tensilelite_runtime dtype name."""
    name = _rtDtypeNames.get(code)
    if name is None:
        raise NotImplementedError(f"unsupported dtype code for library mode: {code}")
    return name


# ---------------------------------------------------------------------------
# SweepRunner.
# ---------------------------------------------------------------------------


class SweepRunner:
    """Enumerate, compile, and benchmark all solutions from a benchmark YAML.

    Parameters
    ----------
    yamlPath:
        Path to a benchmark YAML file (BenchmarkProblems format).
    libraryPath:
        When set, enables library mode: load this TensileLibrary.yaml and
        benchmark only the single winner selected by LibraryRunner.find_best
        for each problem size.  The .co file is read from the same directory
        as the YAML.  When None (default), compile mode is used.
    nWarmup:
        Number of warmup iterations before timing begins.
    nIters:
        Number of timed iterations per benchmark.
    rotatingBuffers:
        Number of rotating D output buffers to prevent cache aliasing.
    icacheCopies:
        'auto' to detect from ELF, or an integer number of module copies.
    problemIdx:
        Index into BenchmarkProblems[] to select the problem group.
    groupIdx:
        Sub-group index within the selected BenchmarkProblems group.
    numElementsToValidate:
        0 disables validation (benchmark-only), -1 validates all output elements,
        N>0 validates the first N flattened elements.
    saveCoPath:
        Directory to save compiled .co files (one per kernel). When None,
        compiled HSACO bytes are used only for in-process benchmarking.
    problemSizes:
        Explicit list of problem-size tuples to sweep, e.g. [(M, N, batch, K), ...].
        When provided these are used directly instead of parsing the YAML, which
        is required in library mode for YAMLs that lack a BenchmarkProblems block.
        When None (default) sizes are parsed from the BenchmarkProblems YAML.
    """

    def _resolveYamlFromIni(self, iniPath: str) -> str:
        """Extract the benchmark YAML path from a ClientParameters .ini file.

        Parses the flat key=value .ini using configparser with a synthetic section
        header and returns the value of the benchmark-yaml key.
        Raises KeyError when the key is absent.
        """
        config = configparser.RawConfigParser(strict=False)
        with open(iniPath) as f:
            content = f.read()
        config.read_string('[default]\n' + content)
        section = config['default']
        if 'benchmark-yaml' not in section:
            raise KeyError(
                f"benchmark-yaml key not found in .ini file: {iniPath}; "
                "re-generate the .ini with a version of ClientWriter that "
                "writes this key"
            )
        return section['benchmark-yaml']

    def __init__(self, yamlPath: str, libraryPath: Optional[str] = None,
                 nWarmup: int = 2, nIters: int = 10,
                 rotatingBuffers: int = 8, icacheCopies="auto",
                 problemIdx: int = 0, groupIdx: int = 0,
                 numElementsToValidate: int = 0,
                 saveCoPath: Optional[str] = None,
                 problemSizes: Optional[list] = None,
                 pinClocks: bool = False,
                 amdSmiPath: Optional[str] = None,
                 _finalYaml: bool = False) -> None:
        if yamlPath.endswith('.ini'):
            self._yamlPath = self._resolveYamlFromIni(yamlPath)
        else:
            self._yamlPath = yamlPath
        self._libraryPath = libraryPath
        self._nWarmup = nWarmup
        self._nIters = nIters
        self._rotatingBuffers = rotatingBuffers
        self._icacheCopies = icacheCopies
        self._problemIdx = problemIdx
        self._groupIdx = groupIdx
        self._numElementsToValidate = numElementsToValidate
        self._saveCoPath = saveCoPath
        self._problemSizes = (
            [tuple(int(x) for x in s) for s in problemSizes]
            if problemSizes else None
        )
        self._pinClocks = pinClocks
        self._amdSmiPath = amdSmiPath
        self._finalYaml = _finalYaml

    def _compileOneSolution(self, sol, sid, chip: str, assembler,
                            debugConfig) -> Optional[dict]:
        """Compile one solution; return a compiled entry dict or None on failure."""
        import amdgpu_exec
        from Tensile.client.yaml_solution_builder import (
            _injectInternalArgsSupport,
        )

        rawDict = dict(sol)
        solDict = _injectInternalArgsSupport(rawDict, chip)
        if _shouldSkip(solDict):
            return None
        try:
            asmStr, kernelName = _generateAsm(sol, assembler, debugConfig)
            hsaco = amdgpu_exec.compile_asm_to_hsaco(asmStr, chip)
        except Exception as exc:
            _log.warning("solution %s failed to compile: %s", sid, exc)
            return None
        if self._saveCoPath is not None:
            coPath = os.path.join(self._saveCoPath, f"{kernelName}.co")
            with open(coPath, "wb") as f:
                f.write(hsaco)
            _log.info("saved .co: %s", coPath)
        return {
            "solDict": solDict,
            "rawDict": rawDict,
            "kernelName": kernelName,
            "hsaco": hsaco,
            "sid": sid,
            "solution": sol,
        }

    def _compileAll(self, chip: str, assembler, isaInfoMap,
                    debugConfig) -> list:
        """Compile all solutions from the YAML group; return compiled entries.

        Each entry is a dict with keys: solDict, rawDict, kernelName, hsaco,
        sid, solution. When saveCoPath is set, each compiled HSACO is also
        written to {saveCoPath}/{kernelName}.co for use by the C++ client.
        Solutions that fail to compile or are filtered are skipped with a warning.
        """
        from Tensile.client.yaml_solution_builder import (
            solutionsFromYaml, solutionsFromFinalYaml,
        )

        if self._saveCoPath is not None:
            os.makedirs(self._saveCoPath, exist_ok=True)

        try:
            if self._finalYaml:
                sols, problemSizes = solutionsFromFinalYaml(
                    self._yamlPath, assembler, isaInfoMap, debugConfig, chip)
                # Store problem sizes for _runCompile to use.
                self._finalYamlProblemSizes = problemSizes
            else:
                self._finalYamlProblemSizes = None
                sols = solutionsFromYaml(
                    self._yamlPath, assembler, isaInfoMap, debugConfig,
                    problemIdx=self._problemIdx, groupIdx=self._groupIdx,
                )
        except Exception as exc:
            _log.warning("solution enumeration failed: %s", exc)
            return []

        compiled = []
        for sol, sid in sols:
            entry = self._compileOneSolution(sol, sid, chip, assembler, debugConfig)
            if entry is not None:
                compiled.append(entry)
        return compiled

    def _allocBufs(self, solDict, M, N, batch, K, gsu: int = 1):
        """Allocate device buffers for one benchmark run; caller must free.

        For GlobalAccumulation != 0 kernels the kernel writes fp32 into D, so
        dSize uses _storeElemSize (4 B) rather than _dElemSize (2 B for bf16).
        Returns (aBuf, bBuf, cBuf, dPool, syncBuf) where syncBuf is None unless
        the solution is MBSK (GA=4), in which case it is a pre-zeroed synchronizer.
        """
        from amdgpu_exec import GpuBuffer
        aSize = M * K * batch * _aElemSize(solDict)
        bSize = N * K * batch * _aElemSize(solDict)
        dSize = M * N * batch * _storeElemSize(solDict, gsu)
        syncBuf = None
        if _isMbsk(solDict):
            syncBytes = _synchronizerBytes(solDict, M, N, batch)
            syncBuf = GpuBuffer(max(syncBytes, 4))
            syncBuf.memset(0)
        return (GpuBuffer(aSize), GpuBuffer(bSize), GpuBuffer(dSize),
                BufferPool(nSlots=self._rotatingBuffers, sizeBytes=dSize,
                           gpuBufferCls=GpuBuffer), syncBuf)

    def _allocEpilogueBufs(self, solDict, M, N, batch) -> dict:
        """Allocate zeroed GPU buffers for epilogue tensors; caller must free values."""
        from amdgpu_exec import GpuBuffer
        from Tensile.client.gemm_args import _readPTFlag
        bufs = {}
        scalar = 4  # 4-byte scalar epilogue tensors
        if _readPTFlag(solDict, "UseScaleAB", ""):
            bufs["scaleA"] = GpuBuffer(scalar)
            bufs["scaleB"] = GpuBuffer(scalar)
        if _readPTFlag(solDict, "UseScaleCD", False):
            bufs["scaleC"] = GpuBuffer(scalar)
            bufs["scaleD"] = GpuBuffer(scalar)
        if _readPTFlag(solDict, "UseScaleAlphaVec", 0):
            bufs["scaleAlphaVec"] = GpuBuffer(M * scalar)
        if _readPTFlag(solDict, "UseBias", 0):
            bufs["bias"] = GpuBuffer(M * scalar)
        if _readPTFlag(solDict, "UseE", False):
            dSize = M * N * batch * _dElemSize(solDict)
            bufs["e"] = GpuBuffer(dSize)
        if _readPTFlag(solDict, "OutputAmaxD", False):
            bufs["amaxD"] = GpuBuffer(scalar)
            bufs["amaxWS"] = GpuBuffer(scalar)
            bufs["amaxSync"] = GpuBuffer(scalar)
        for buf in bufs.values():
            buf.memset(0)
        return bufs

    @staticmethod
    def _freeEpilogueBufs(epilogueBufs: dict) -> None:
        """Free all GPU buffers in an epilogue buffer dict."""
        for buf in epilogueBufs.values():
            buf.free()

    def _benchmarkOne(self, entry: dict, M: int, N: int, batch: int,
                      K: int, cuCount: int):
        """Benchmark one (solution, problem) pair.

        Returns (gflops, BenchmarkResult) on success, or (-1.0, None) on error.
        """
        solDict = entry["solDict"]
        gsu = int(solDict.get("_resolvedGsu", 1))
        aBuf, bBuf, cBuf, dPool, syncBuf = self._allocBufs(solDict, M, N, batch, K, gsu)
        epilogueBufs = self._allocEpilogueBufs(solDict, M, N, batch) if _hasEpilogue(solDict) else {}
        try:
            runner = _makeRunner(entry["hsaco"], entry["kernelName"], self._icacheCopies)
            ga = _gaMode(solDict)

            def argsFn(_ignored):
                dBuf = dPool.next()
                if ga != 0:
                    return _buildSweepArgsGA(solDict, M, N, batch, K,
                                             dBuf, cBuf, aBuf, bBuf, cuCount, gsu,
                                             syncBuf=syncBuf,
                                             epilogueBufs=epilogueBufs or None)
                return _buildSweepArgs(solDict, M, N, batch, K,
                                       dBuf, cBuf, aBuf, bBuf, cuCount,
                                       epilogueBufs=epilogueBufs or None)

            benchResult = runner.run(
                argsFn=argsFn,
                grid=(_computeNumWg(solDict, M, N, batch), 1, 1),
                block=(solDict["NumThreads"], 1, 1),
                nWarmup=self._nWarmup,
                nIters=self._nIters,
            )
            # Use min (best iteration) to match C++ WinnerGFlops (= minimum time).
            gflops = 2 * M * N * K * batch / (benchResult.minUs * 1e-6) / 1e9
            return gflops, benchResult
        except Exception as exc:
            _log.warning("benchmark failed for %s size=(%d,%d,%d,%d): %s",
                         entry["sid"], M, N, batch, K, exc)
            return -1.0, None
        finally:
            aBuf.free(); bBuf.free(); cBuf.free(); dPool.freeAll()
            if syncBuf is not None:
                syncBuf.free()
            self._freeEpilogueBufs(epilogueBufs)

    def _makeVerifyInputs(self, solDict, npDtype, M, N, batch, K):
        """Return deterministic host A/B flat buffers in the kernel's strided layout.

        Layout depends on TransposeA/TransposeB from ProblemType:
          TN: A stored as K×M (lda=K), B stored as K×N (ldb=K)
          NT: A stored as M×K (lda=M), B stored as N×K (ldb=N)
        Buffers are column-major (Fortran order).
        """
        pt = solDict.get("ProblemType") or {}
        transA = bool(pt.get("TransposeA", False))
        transB = bool(pt.get("TransposeB", False))
        rng = np.random.default_rng(seed=M * 1000 + N + K)
        if npDtype == np.int8:
            aShape = (K, M) if transA else (M, K)
            bShape = (K, N) if not transB else (N, K)
            aNp = np.asfortranarray(rng.integers(-50, 50, size=aShape).astype(npDtype))
            bNp = np.asfortranarray(rng.integers(-50, 50, size=bShape).astype(npDtype))
        else:
            aShape = (K, M) if transA else (M, K)
            bShape = (K, N) if not transB else (N, K)
            aNp = np.asfortranarray(rng.random(aShape).astype(npDtype))
            bNp = np.asfortranarray(rng.random(bShape).astype(npDtype))
        aHost = np.tile(aNp.ravel(order="F"), batch)
        bHost = np.tile(bNp.ravel(order="F"), batch)
        return aHost, bHost

    def _computeVerifyRef(self, solDict, aHost, bHost, refFn, M, N, batch, K):
        """Compute the flat column-major reference D for the strided-batched GEMM."""
        pt = solDict.get("ProblemType") or {}
        transA = bool(pt.get("TransposeA", False))
        transB = bool(pt.get("TransposeB", False))
        # Reconstruct the 2-D slice from the flat buffer using the stored shape.
        if transA:
            aSlice = aHost[:K * M].reshape(K, M, order="F")  # K×M stored, op(A)=A^T=M×K
        else:
            aSlice = aHost[:M * K].reshape(M, K, order="F")  # M×K stored, op(A)=A
        if transB:
            bSlice = bHost[:N * K].reshape(N, K, order="F")  # N×K stored, op(B)=B^T=K×N
        else:
            bSlice = bHost[:K * N].reshape(K, N, order="F")  # K×N stored, op(B)=B
        # refFn expects (A_mxk, B_kxn, alpha, beta, C) where A_mxk = op(A_stored).
        opA = aSlice.T if transA else aSlice        # M×K
        opB = bSlice.T if not transB else bSlice    # K×N
        dOne = refFn(opA, opB, 1.0, 0.0, None)
        return np.tile(np.asfortranarray(dOne).ravel(order="F"), batch)

    def _compareVerify(self, dHost, dRef, rtol, atol, label) -> str:
        """Compare readback vs reference; return 'PASS' or 'FAIL:<message>'.

        For numElementsToValidate>0 only the first N flattened elements are
        checked (a deliberate simplification of the C++ NextPrime sampling).
        """
        from Tensile.client.reference import assertClose
        gpuCmp, refCmp = dHost, dRef
        if self._numElementsToValidate > 0:
            n = min(self._numElementsToValidate, dHost.size)
            gpuCmp, refCmp = dHost[:n], dRef[:n]
        try:
            assertClose(gpuCmp, refCmp, rtol=rtol, atol=atol, label=label)
        except AssertionError as exc:
            return f"FAIL:{exc}"
        return "PASS"

    def _freeVerifyBuffers(self, buffers) -> None:
        """Free any allocated verification buffers, ignoring None slots."""
        for buf in buffers:
            if buf is not None:
                buf.free()

    def _verifyOne(self, entry: dict, M: int, N: int, batch: int,
                   K: int, cuCount: int) -> str:
        """Validate one (solution, problem) pair; return PASS / FAIL:<msg> / SKIPPED."""
        solDict = entry["solDict"]
        ga = _gaMode(solDict)
        # GA=2 (interleaved fp32 partials): final bf16 readback layout not yet
        # confirmed — skip rather than risk a false PASS/FAIL.
        if ga == 2:
            return "SKIPPED"
        selected = _selectReference(solDict)
        if selected is None:
            return "SKIPPED"
        npDtype, npOutDtype, refFn, rtol, atol = selected
        from amdgpu_exec import GpuBuffer, GpuEvent
        aBuf = bBuf = cBuf = dBuf = syncBuf = None
        epilogueBufs = {}
        gsu = int(solDict.get("_resolvedGsu", 1))
        try:
            aHost, bHost = self._makeVerifyInputs(solDict, npDtype, M, N, batch, K)
            dHost = np.zeros(M * N * batch, dtype=npOutDtype)
            storeSize = M * N * batch * _storeElemSize(solDict, gsu)
            aBuf = GpuBuffer(aHost.nbytes); aBuf.copy_from_host(aHost)
            bBuf = GpuBuffer(bHost.nbytes); bBuf.copy_from_host(bHost)
            cBuf = GpuBuffer(storeSize); cBuf.memset(0)
            dBuf = GpuBuffer(storeSize); dBuf.memset(0)
            if _isMbsk(solDict):
                syncBytes = _synchronizerBytes(solDict, M, N, batch)
                syncBuf = GpuBuffer(max(syncBytes, 4))
                syncBuf.memset(0)
            epilogueBufs = self._allocEpilogueBufs(solDict, M, N, batch) if _hasEpilogue(solDict) else {}
            runner = _makeRunner(entry["hsaco"], entry["kernelName"], self._icacheCopies)
            def argsFn(_ignored):
                if ga != 0:
                    return _buildSweepArgsGA(solDict, M, N, batch, K,
                                             dBuf, cBuf, aBuf, bBuf, cuCount, gsu,
                                             syncBuf=syncBuf,
                                             epilogueBufs=epilogueBufs or None)
                return _buildSweepArgs(
                    solDict, M, N, batch, K, dBuf, cBuf, aBuf, bBuf, cuCount,
                    epilogueBufs=epilogueBufs or None)
            runner.run(argsFn=argsFn,
                       grid=(_computeNumWg(solDict, M, N, batch), 1, 1),
                       block=(solDict["NumThreads"], 1, 1),
                       nWarmup=0, nIters=1)
            # For MBSK (GA=4) the final output is in dstD = dBuf (same pointer).
            dBuf.copy_to_host(dHost)
            ev = GpuEvent(); ev.record(); ev.synchronize()
            dRef = self._computeVerifyRef(solDict, aHost, bHost, refFn, M, N, batch, K)
            return self._compareVerify(dHost, dRef, rtol, atol, entry["sid"])
        except Exception as exc:
            _log.warning("verify failed for %s: %s", entry["sid"], exc)
            return f"FAIL:{exc}"
        finally:
            self._freeVerifyBuffers((aBuf, bBuf, cBuf, dBuf))
            if syncBuf is not None:
                syncBuf.free()
            self._freeEpilogueBufs(epilogueBufs)

    def _benchmarkProblem(self, probSize: tuple, compiled: list,
                          cuCount: int) -> list:
        """Benchmark all solutions for one problem size.

        Returns a list of SweepResult (one per solution), with gflops=-1.0
        for any solution that errored.
        """
        M, N, batch, K = probSize[0], probSize[1], probSize[2], probSize[3]
        results = []
        for i, entry in enumerate(compiled):
            gflops, br = self._benchmarkOne(entry, M, N, batch, K, cuCount)
            if br is None:
                br = BenchmarkResult(timesNs=[], warmupN=self._nWarmup)
            validation = "SKIPPED"
            if self._numElementsToValidate != 0:
                validation = self._verifyOne(entry, M, N, batch, K, cuCount)
            results.append(SweepResult(
                solutionIdx=i,
                solutionName=entry["sid"],
                problemSize=probSize,
                benchmark=br,
                gflops=gflops,
                validation=validation,
            ))
        return results

    def _aggregateValidation(self, sizeResults: list) -> str:
        """Summarize per-solution validation into one row-level status."""
        if self._numElementsToValidate == 0:
            return "SKIPPED"
        fails = [r.validation for r in sizeResults if r.validation.startswith("FAIL")]
        if fails:
            return fails[0]
        if any(r.validation == "PASS" for r in sizeResults):
            return "PASS"
        return "SKIPPED"

    def _reportProblem(self, probSize, sizeResults, csvRep, luRep):
        """Write CSV and library-update rows for one problem size."""
        M, N, batch, K = probSize[0], probSize[1], probSize[2], probSize[3]
        if len(probSize) >= 8:
            ldd, ldc, lda, ldb = probSize[4], probSize[5], probSize[6], probSize[7]
        else:
            lda, ldb, ldd, ldc = M, N, M, M
        if csvRep:
            solResults = [(r.solutionName, r.gflops) for r in sizeResults]
            csvRep.writeRow(
                sizeParams={"sizes": list(probSize), "ldd": ldd, "ldc": ldc,
                            "lda": lda, "ldb": ldb, "totalFlops": 2 * M * N * K * batch},
                solutionResults=solResults,
                validation=self._aggregateValidation(sizeResults),
            )
        if luRep:
            valid = [r for r in sizeResults if r.gflops > 0]
            if valid:
                winner = max(valid, key=lambda r: r.gflops)
                luRep.writeRow(list(probSize), winner.solutionIdx, winner.gflops)

    def _problemTypeFields(self, solDict: dict) -> dict:
        """Extract dtype/transpose fields for building a runtime Problem."""
        pt = solDict.get("ProblemType") or {}
        aCode = _readDtypeInt(solDict, "DataType")
        dCode = _readDtypeInt(solDict, "DestDataType")
        if dCode == -1:
            dCode = aCode
        return {
            "aName": _dtypeIntToRtName(aCode),
            "dName": _dtypeIntToRtName(dCode),
            "transA": bool(pt.get("TransposeA", False)),
            "transB": bool(pt.get("TransposeB", False)),
            "hpa": bool(pt.get("HighPrecisionAccumulate", False)),
        }

    def _loadSolutionMetadataFromLibrary(self, chip: str):
        """Return (byName, ptFields) by reading the library YAML directly.

        Builds solDicts from the library's sizeMapping/problemType/internalArgsSupport
        without invoking any part of the Tensile solution pipeline.  This is O(n_solutions)
        YAML parsing instead of O(n_solutions) Tensile parameter validation, which is
        the dominant cost in _enumerateSolutionMetadata for large YAMLs.
        """
        from Tensile import LibraryIO
        data = LibraryIO.readYAML(self._libraryPath)
        libSols = data.get("solutions", [])
        byName = {}
        ptFields = None
        for idx, libSol in enumerate(libSols):
            name = libSol.get("kernelName") or libSol.get("name", "")
            if not name or name in byName:
                continue
            try:
                solDict = _solDictFromLibSol(libSol, chip)
            except Exception as exc:
                _log.warning("could not build solDict for %s: %s", name, exc)
                continue
            byName[name] = {"solDict": solDict, "rawDict": libSol,
                            "sid": name, "index": idx}
            if ptFields is None:
                ptFields = self._problemTypeFields(solDict)
        if ptFields is None and byName:
            ptFields = self._problemTypeFields(next(iter(byName.values()))["solDict"])
        return byName, ptFields

    def _enumerateSolutionMetadata(self, chip: str, assembler, isaInfoMap,
                                   debugConfig):
        """Return (solDictByName, ptFields) from the YAML without compiling.

        solDictByName maps getKernelNameMin(kernel) -> a metadata dict with keys
        solDict, rawDict, sid, index. ptFields captures the shared problem-type
        dtype/transpose info used to build runtime Problem objects.
        """
        from Tensile.client.yaml_solution_builder import (
            solutionsFromYaml, _injectInternalArgsSupport,
        )
        from Tensile.SolutionStructs.Naming import getKernelNameMin
        from Tensile import LibraryIO

        byName = {}
        ptFields = None
        globalIdx = 0
        # Enumerate all problem groups so library winners from any group can be found.
        # ptFields is scoped to self._problemIdx so library queries use the correct
        # problem type when problemIdx > 0 (e.g. cross-running F16 groups).
        try:
            data = LibraryIO.readYAML(self._yamlPath)
            numProblems = len(data.get("BenchmarkProblems", []))
        except Exception:
            numProblems = 1
        for pIdx in range(numProblems):
            gIdx = 0
            while True:
                try:
                    sols = solutionsFromYaml(
                        self._yamlPath, assembler, isaInfoMap, debugConfig,
                        problemIdx=pIdx, groupIdx=gIdx,
                    )
                except (IndexError, KeyError):
                    break
                if not sols:
                    gIdx += 1
                    if gIdx > 32:
                        break
                    continue
                for sol, sid in sols:
                    solDict = _injectInternalArgsSupport(dict(sol), chip)
                    name = getKernelNameMin(sol.getKernels()[0], splitGSU=False)
                    if name not in byName:
                        byName[name] = {"solDict": solDict, "rawDict": dict(sol),
                                        "sid": sid, "index": globalIdx}
                        globalIdx += 1
                    # Capture ptFields only from the target problemIdx group.
                    if ptFields is None and pIdx == self._problemIdx:
                        ptFields = self._problemTypeFields(solDict)
                gIdx += 1
                if gIdx > 32:
                    break
        # Fallback: if target problemIdx produced no solutions, use first available.
        if ptFields is None:
            for meta in byName.values():
                ptFields = self._problemTypeFields(meta["solDict"])
                break
        return byName, ptFields

    def _discoverCodeObjects(self) -> list:
        """Return sorted .co file paths in the library directory."""
        import glob
        libDir = os.path.dirname(os.path.abspath(self._libraryPath))
        coPaths = sorted(glob.glob(os.path.join(libDir, "*.co")))
        if not coPaths:
            raise FileNotFoundError(f"no .co files found next to library: {libDir}")
        return coPaths

    def _coBytesForKernel(self, kernelName: str, coPaths: list,
                          cache: dict):
        """Return the .co bytes exporting kernelName, or None; caches by name."""
        if kernelName in cache:
            return cache[kernelName]
        from amdgpu_exec import GpuModule
        for coPath in coPaths:
            with open(coPath, "rb") as f:
                data = f.read()
            # one-shot probe for the exported symbol; the temporary module is discarded.
            try:
                GpuModule(data).get_function(kernelName)
            except Exception:
                continue
            cache[kernelName] = data
            return data
        cache[kernelName] = None
        return None

    def _makeRtProblem(self, probSize: tuple, ptFields: dict):
        """Build a tensilelite_runtime Problem for one problem size."""
        import tensilelite_runtime as rt
        M, N, batch, K = probSize[0], probSize[1], probSize[2], probSize[3]
        return rt.Problem(
            M=M, N=N, K=K,
            dtype_a=ptFields["aName"], dtype_b=ptFields["aName"],  # standard GEMM: B input dtype matches A.
            dtype_c=ptFields["dName"], dtype_d=ptFields["dName"],
            trans_a=ptFields["transA"], trans_b=ptFields["transB"],
            high_precision_accumulate=ptFields["hpa"],
            batch_size=batch,
        )

    def _resolveWinnerEntry(self, probSize, libRunner, ptFields, byName,
                            coPaths, coCache):
        """Find the library winner for probSize; build a benchmark entry or None."""
        prob = self._makeRtProblem(probSize, ptFields)
        sol = libRunner.find_best(prob)
        if sol is None:
            _log.warning("library found no solution for size=%s", probSize)
            return None
        name = sol.kernel_name
        meta = byName.get(name)
        if meta is None:
            _log.warning("winner %s not in YAML enumeration; skip size=%s",
                         name, probSize)
            return None
        coBytes = self._coBytesForKernel(name, coPaths, coCache)
        if coBytes is None:
            _log.warning("no .co exports %s; skip size=%s", name, probSize)
            return None
        solDict = dict(meta["solDict"])
        autoWgm, autoGsu, autoStaggerU = libRunner.autoParams(sol, prob)
        # Inject the C++-resolved auto values so _computeInternalArg0/1 pack the
        # same WorkGroupMapping and StaggerU the C++ client would use (handles
        # WorkGroupMapping=0 and GlobalSplitU=0 kernels correctly).
        solDict["WorkGroupMapping"] = autoWgm
        solDict["StaggerU"] = autoStaggerU
        solDict["_resolvedGsu"] = max(1, int(autoGsu))
        return {"solDict": solDict, "rawDict": meta["rawDict"],
                "kernelName": name, "hsaco": coBytes, "sid": meta["sid"],
                "solution": None, "solutionIdx": meta["index"]}

    def _benchmarkLibrary(self, probSize, libRunner, ptFields, byName,
                          coPaths, coCache, cuCount):
        """Benchmark the library winner for one size; return a SweepResult or None."""
        M, N, batch, K = probSize[0], probSize[1], probSize[2], probSize[3]
        entry = self._resolveWinnerEntry(probSize, libRunner, ptFields, byName,
                                         coPaths, coCache)
        if entry is None:
            return None
        gflops, br = self._benchmarkOne(entry, M, N, batch, K, cuCount)
        if br is None:
            br = BenchmarkResult(timesNs=[], warmupN=self._nWarmup)
        validation = "SKIPPED"
        if self._numElementsToValidate != 0:
            validation = self._verifyOne(entry, M, N, batch, K, cuCount)
        return SweepResult(solutionIdx=entry["solutionIdx"], solutionName=entry["sid"],
                           problemSize=probSize, benchmark=br, gflops=gflops,
                           validation=validation)

    def _reportLibraryProblem(self, probSize, result, csvRep, luRep):
        """Write one winner row (CSV + library-update) for a library-mode size."""
        M, N, batch, K = probSize[0], probSize[1], probSize[2], probSize[3]
        if len(probSize) >= 8:
            ldd, ldc, lda, ldb = probSize[4], probSize[5], probSize[6], probSize[7]
        else:
            lda, ldb, ldd, ldc = M, N, M, M
        gflops = result.gflops if result else -1.0
        if csvRep:
            validation = result.validation if result else "SKIPPED"
            csvRep.writeRow(
                sizeParams={"sizes": list(probSize), "ldd": ldd, "ldc": ldc,
                            "lda": lda, "ldb": ldb,
                            "totalFlops": 2 * M * N * K * batch},
                solutionResults=[("Winner", gflops)],
                validation=validation)
        if luRep and result and result.gflops > 0:
            luRep.writeRow(list(probSize), result.solutionIdx, result.gflops)

    def _resolveProblemSizes(self):
        """Return the problem sizes to sweep: explicit override or parsed YAML.

        A problemSizes list passed to __init__ takes precedence (required for
        non-BenchmarkProblems YAMLs).  Otherwise sizes are parsed from the
        BenchmarkProblems YAML; an empty parse is logged so the caller knows to
        pass problemSizes explicitly.
        """
        from Tensile.client.yaml_solution_builder import problemSizesFromYaml
        if self._problemSizes is not None:
            return self._problemSizes
        probSizes = problemSizesFromYaml(
            self._yamlPath, problemIdx=self._problemIdx, groupIdx=self._groupIdx)
        if not probSizes:
            _log.error(
                "no problem sizes parsed from %s; pass problemSizes= to "
                "SweepRunner for non-BenchmarkProblems YAMLs", self._yamlPath)
        return probSizes

    def _runLibrary(self, resultsCsv, libraryUpdateFile):
        """Library-mode sweep: benchmark the per-size winner from a pre-built library."""
        from Tensile.client.library_runner import LibraryRunner
        import amdgpu_exec

        chip = amdgpu_exec.get_chip()
        byName, ptFields = self._loadSolutionMetadataFromLibrary(chip)
        if not byName:
            _log.warning("no solutions enumerated; library sweep returns empty")
            return []
        coPaths = self._discoverCodeObjects()
        libRunner = LibraryRunner(self._libraryPath)
        probSizes = self._resolveProblemSizes()
        cuCount = _deviceCuCount()
        numSizeDims = len(probSizes[0]) if probSizes else 4
        csvRep, luRep = self._openReporters(
            resultsCsv, ["Winner"], numSizeDims, libraryUpdateFile)
        coCache = {}
        try:
            if self._pinClocks and self._amdSmiPath:
                self._applyClockPin()
            allResults = []
            for probSize in probSizes:
                result = self._benchmarkLibrary(
                    probSize, libRunner, ptFields, byName, coPaths, coCache, cuCount)
                if result is not None:
                    allResults.append(result)
                self._reportLibraryProblem(probSize, result, csvRep, luRep)
        finally:
            if self._pinClocks and self._amdSmiPath:
                self._resetClockPin()
            if csvRep:
                csvRep.close()
            if luRep:
                luRep.close()
        return allResults

    def _runSudoCmd(self, args: list, desc: str):
        """Run a sudo command; raise PermissionError on failure or missing sudo."""
        try:
            result = subprocess.run(["sudo"] + args, capture_output=True)
        except FileNotFoundError:
            raise PermissionError("sudo not found; clock pinning requires sudo")
        if result.returncode != 0:
            raise PermissionError(f"clock {desc} failed with returncode {result.returncode}")

    def _applyClockPin(self):
        """Pin GPU clocks and fan speed; sleep 1 second to allow stabilization."""
        self._runSudoCmd([self._amdSmiPath, "set", "-g", "0", "--fan", "255"], "fan set")
        self._runSudoCmd([self._amdSmiPath, "set", "-g", "0", "--perf-level", "HIGH"], "perf-level set")
        time.sleep(1)

    def _resetClockPin(self):
        """Reset GPU clocks and fan speed to driver defaults after benchmarking."""
        self._runSudoCmd([self._amdSmiPath, "reset", "-g", "0", "--clocks", "--fans"], "reset")

    def _compile(self):
        """Detect chip, set up Tensile, compile all solutions; return (compiled, chip)."""
        import amdgpu_exec
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        compiled = self._compileAll(chip, assembler, isaInfoMap, debugConfig)
        return compiled, chip

    def _openReporters(self, resultsCsv: Optional[str], solNames: list,
                       numSizeDims: int,
                       libraryUpdateFile: Optional[str]):
        """Open CSV and library-update reporters; write the CSV header."""
        csvRep = (ResultsCSVReporter(resultsCsv, solNames, numSizeDims=numSizeDims,
                                      includeValidation=(self._numElementsToValidate != 0))
                  if resultsCsv else None)
        luRep = LibraryUpdateReporter(libraryUpdateFile) if libraryUpdateFile else None
        if csvRep:
            csvRep.writeHeader()
        return csvRep, luRep

    def _runCompile(self, resultsCsv, libraryUpdateFile):
        """Compile-mode sweep: compile all YAML solutions and benchmark every size."""
        from Tensile.client.yaml_solution_builder import problemSizesFromYaml

        compiled, _ = self._compile()
        if not compiled:
            _log.warning("no solutions compiled; sweep returns empty")
            return []
        if self._problemSizes is not None:
            probSizes = self._problemSizes
        elif self._finalYaml and self._finalYamlProblemSizes:
            probSizes = self._finalYamlProblemSizes
        else:
            probSizes = problemSizesFromYaml(self._yamlPath,
                                             problemIdx=self._problemIdx,
                                             groupIdx=self._groupIdx)
        cuCount = _deviceCuCount()
        solNames = [e["sid"] for e in compiled]
        # numSizeDims from actual tuple length — batched GEMM returns 8-dim tuples.
        numSizeDims = len(probSizes[0]) if probSizes else 4
        csvRep, luRep = self._openReporters(resultsCsv, solNames, numSizeDims,
                                            libraryUpdateFile)
        try:
            if self._pinClocks and self._amdSmiPath:
                self._applyClockPin()
            allResults = []
            for probSize in probSizes:
                sizeResults = self._benchmarkProblem(probSize, compiled, cuCount)
                allResults.extend(sizeResults)
                self._reportProblem(probSize, sizeResults, csvRep, luRep)
        finally:
            if self._pinClocks and self._amdSmiPath:
                self._resetClockPin()
            if csvRep:
                csvRep.close()
            if luRep:
                luRep.close()
        return allResults

    def run(self, resultsCsv: Optional[str] = None,
            libraryUpdateFile: Optional[str] = None,
            hwMonitor: bool = False,
            boundsCheck: bool = False,
            rocprofCounters=None) -> list:
        """Run the sweep and return a list of SweepResult.

        resultsCsv:        path to write results.csv (omit to skip).
        libraryUpdateFile: path to write the library-update YAML (omit to skip).
        hwMonitor, boundsCheck, rocprofCounters: reserved for future use.
        Returns a flat list of SweepResult (one per problem_size × solution).
        """
        if self._libraryPath is not None:
            return self._runLibrary(resultsCsv, libraryUpdateFile)
        return self._runCompile(resultsCsv, libraryUpdateFile)
