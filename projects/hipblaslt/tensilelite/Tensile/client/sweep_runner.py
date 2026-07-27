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


def _setupTensile(chip: str):
    """Initialize Tensile assembler + ISA map for kernel compilation."""
    from pathlib import Path
    from Tensile.Toolchain.Validators import validateToolchain
    from Tensile.Toolchain.Component import Assembler
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.GlobalParameters import assignGlobalParameters
    from Tensile.Common.Types import DebugConfig

    gfx = chip.split(":")[0]
    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa(gfx)
    isaInfoMap = makeIsaInfoMap([isa], cxx)
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
# Kernel argument builder for stridedBatched NT GEMM.
# ---------------------------------------------------------------------------


def _computeNumWg(solDict: dict, M: int, N: int, batch: int) -> int:
    """Compute work-group count for NT stridedBatched GEMM."""
    mt0 = solDict["MacroTile0"]
    mt1 = solDict["MacroTile1"]
    return math.ceil(M / mt0) * math.ceil(N / mt1) * batch


def _buildSweepArgs(solDict: dict, M: int, N: int, batch: int, K: int,
                    dBuf, cBuf, aBuf, bBuf, cuCount: int = 0,
                    alpha: float = 1.0, beta: float = 0.0) -> list:
    """Build typed kernel arg list for stridedBatched NT GEMM.

    Uses argType=0 (stridedBatched=True) in gemm_count bits 30-31.
    NT column-major strides: lda=M, ldb=N, ldd=ldc=M.
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

    lda, ldb, ldd, ldc = M, N, M, M
    strideA, strideB, strideD, strideC = M * K, N * K, M * N, M * N
    args.extend([
        np.uint32(ldd), np.uint32(strideD),
        np.uint32(ldc), np.uint32(strideC),
        np.uint32(lda), np.uint32(strideA),
        np.uint32(ldb), np.uint32(strideB),
        np.float32(alpha), np.float32(beta),
    ])
    return args


# ---------------------------------------------------------------------------
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


def _shouldSkip(rawDict: dict, solDict: dict) -> bool:
    """Return True if this solution should be excluded from the sweep.

    Skips solutions where WorkGroupMapping=0 (requires calculateAutoWGM) or
    where StaggerU=0 with SupportCustomStaggerU=True (requires calculateAutoStaggerU).
    Both auto-computation methods are not available until M10.
    """
    if solDict.get("WorkGroupMapping", 0) == 0:
        return True
    isp = rawDict.get("InternalSupportParams", {}) or {}
    return solDict.get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False)


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
                 pinClocks: bool = False,
                 amdSmiPath: Optional[str] = None) -> None:
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
        self._pinClocks = pinClocks
        self._amdSmiPath = amdSmiPath

    def _compileOneSolution(self, sol, sid, chip: str, assembler,
                            debugConfig) -> Optional[dict]:
        """Compile one solution; return a compiled entry dict or None on failure."""
        import amdgpu_exec
        from Tensile.client.yaml_solution_builder import (
            _injectInternalArgsSupport,
        )

        rawDict = dict(sol)
        solDict = _injectInternalArgsSupport(rawDict, chip)
        if _shouldSkip(rawDict, solDict):
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
        from Tensile.client.yaml_solution_builder import solutionsFromYaml

        if self._saveCoPath is not None:
            os.makedirs(self._saveCoPath, exist_ok=True)

        try:
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

    def _allocBufs(self, solDict, M, N, batch, K):
        """Allocate device buffers for one benchmark run; caller must free."""
        from amdgpu_exec import GpuBuffer
        aSize = M * K * batch * _aElemSize(solDict)
        bSize = N * K * batch * _aElemSize(solDict)
        dSize = M * N * batch * _dElemSize(solDict)
        return (GpuBuffer(aSize), GpuBuffer(bSize), GpuBuffer(dSize),
                BufferPool(nSlots=self._rotatingBuffers, sizeBytes=dSize,
                           gpuBufferCls=GpuBuffer))

    def _benchmarkOne(self, entry: dict, M: int, N: int, batch: int,
                      K: int, cuCount: int):
        """Benchmark one (solution, problem) pair.

        Returns (gflops, BenchmarkResult) on success, or (-1.0, None) on error.
        """
        solDict = entry["solDict"]
        aBuf, bBuf, cBuf, dPool = self._allocBufs(solDict, M, N, batch, K)
        try:
            runner = _makeRunner(entry["hsaco"], entry["kernelName"], self._icacheCopies)

            def argsFn(_ignored):
                return _buildSweepArgs(solDict, M, N, batch, K,
                                       dPool.next(), cBuf, aBuf, bBuf, cuCount)

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

    def _makeVerifyInputs(self, npDtype, M, N, batch, K):
        """Return deterministic host A/B flat buffers matching the NT strided layout.

        For int8 inputs, generates integer values in [-50, 50) to avoid
        accumulation overflow into the int32 range. For all other dtypes,
        uniform random [0, 1) values are cast to npDtype.
        """
        rng = np.random.default_rng(seed=M * 1000 + N + K)
        if npDtype == np.int8:
            aNp = np.asfortranarray(rng.integers(-50, 50, size=(M, K)).astype(npDtype))
            bNp = np.asfortranarray(rng.integers(-50, 50, size=(N, K)).astype(npDtype))
        else:
            aNp = np.asfortranarray(rng.random((M, K)).astype(npDtype))
            bNp = np.asfortranarray(rng.random((N, K)).astype(npDtype))
        aHost = np.tile(aNp.ravel(order="F"), batch)
        bHost = np.tile(bNp.ravel(order="F"), batch)
        return aHost, bHost

    def _computeVerifyRef(self, aHost, bHost, refFn, M, N, batch, K):
        """Compute the flat column-major reference D for the strided-batched GEMM."""
        aSlice = aHost[:M * K].reshape(M, K, order="F")
        bSlice = bHost[:N * K].reshape(N, K, order="F")
        dOne = refFn(aSlice, bSlice.T, 1.0, 0.0, None)
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
        selected = _selectReference(solDict)
        if selected is None:
            return "SKIPPED"
        npDtype, npOutDtype, refFn, rtol, atol = selected
        from amdgpu_exec import GpuBuffer, GpuEvent
        aBuf = bBuf = cBuf = dBuf = None
        try:
            aHost, bHost = self._makeVerifyInputs(npDtype, M, N, batch, K)
            dHost = np.zeros(M * N * batch, dtype=npOutDtype)
            aBuf = GpuBuffer(aHost.nbytes); aBuf.copy_from_host(aHost)
            bBuf = GpuBuffer(bHost.nbytes); bBuf.copy_from_host(bHost)
            cBuf = GpuBuffer(dHost.nbytes); cBuf.memset(0)
            dBuf = GpuBuffer(dHost.nbytes); dBuf.memset(0)
            runner = _makeRunner(entry["hsaco"], entry["kernelName"], self._icacheCopies)
            def argsFn(_ignored):
                return _buildSweepArgs(
                    solDict, M, N, batch, K, dBuf, cBuf, aBuf, bBuf, cuCount)
            runner.run(argsFn=argsFn,
                       grid=(_computeNumWg(solDict, M, N, batch), 1, 1),
                       block=(solDict["NumThreads"], 1, 1),
                       nWarmup=0, nIters=1)
            dBuf.copy_to_host(dHost)
            ev = GpuEvent(); ev.record(); ev.synchronize()
            dRef = self._computeVerifyRef(aHost, bHost, refFn, M, N, batch, K)
            return self._compareVerify(dHost, dRef, rtol, atol, entry["sid"])
        except Exception as exc:
            # Any failure during setup or run must not abort the sweep.
            _log.warning("verify failed for %s: %s", entry["sid"], exc)
            return f"FAIL:{exc}"
        finally:
            self._freeVerifyBuffers((aBuf, bBuf, cBuf, dBuf))

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

        sols = solutionsFromYaml(
            self._yamlPath, assembler, isaInfoMap, debugConfig,
            problemIdx=self._problemIdx, groupIdx=self._groupIdx,
        )
        byName = {}
        ptFields = None
        for idx, (sol, sid) in enumerate(sols):
            solDict = _injectInternalArgsSupport(dict(sol), chip)
            name = getKernelNameMin(sol.getKernels()[0], splitGSU=False)
            byName[name] = {"solDict": solDict, "rawDict": dict(sol),
                            "sid": sid, "index": idx}
            if ptFields is None:
                ptFields = self._problemTypeFields(solDict)
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
        return {"solDict": meta["solDict"], "rawDict": meta["rawDict"],
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

    def _runLibrary(self, resultsCsv, libraryUpdateFile):
        """Library-mode sweep: benchmark the per-size winner from a pre-built library."""
        from Tensile.client.library_runner import LibraryRunner
        from Tensile.client.yaml_solution_builder import problemSizesFromYaml
        import amdgpu_exec

        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        byName, ptFields = self._enumerateSolutionMetadata(
            chip, assembler, isaInfoMap, debugConfig)
        if not byName:
            _log.warning("no solutions enumerated; library sweep returns empty")
            return []
        coPaths = self._discoverCodeObjects()
        libRunner = LibraryRunner(self._libraryPath)
        probSizes = problemSizesFromYaml(
            self._yamlPath, problemIdx=self._problemIdx, groupIdx=self._groupIdx)
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
