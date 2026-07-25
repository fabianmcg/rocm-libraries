# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SweepRunner: enumerate, compile, and benchmark all solutions in a YAML.

Orchestrates: enumerate solutions via the Tensile fork pipeline → compile to
HSACO → benchmark each (problem_size, solution) pair with KernelRunner using
rotating output buffers and I-cache module rotation → write results.csv and
library-update YAML.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from Tensile.client.harness import BenchmarkResult, BufferPool, KernelRunner
from Tensile.client.gemm_args import _computeInternalArg0, _computeInternalArg1
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
    # Numeric fallback: Float=0→4B, Double=1→8B, Half=4→2B, BFloat16=9→2B.
    _TABLE = {0: 4, 1: 8, 4: 2, 8: 2, 9: 2}
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
        Unused at present; reserved for future LibraryRunner integration.
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
    """

    def __init__(self, yamlPath: str, libraryPath: Optional[str] = None,
                 nWarmup: int = 2, nIters: int = 10,
                 rotatingBuffers: int = 8, icacheCopies="auto",
                 problemIdx: int = 0, groupIdx: int = 0) -> None:
        self._yamlPath = yamlPath
        self._libraryPath = libraryPath
        self._nWarmup = nWarmup
        self._nIters = nIters
        self._rotatingBuffers = rotatingBuffers
        self._icacheCopies = icacheCopies
        self._problemIdx = problemIdx
        self._groupIdx = groupIdx

    def _compileAll(self, chip: str, assembler, isaInfoMap,
                    debugConfig) -> list:
        """Compile all solutions from the YAML group; return compiled entries.

        Each entry is a dict with keys: solDict, rawDict, kernelName, hsaco, sid.
        Solutions that fail to compile or are filtered are skipped with a warning.
        """
        import amdgpu_exec
        from epilogues.epilogue_harness.yaml_solution_builder import (
            solutionsFromYaml, _injectInternalArgsSupport,
        )

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
            rawDict = dict(sol)
            solDict = _injectInternalArgsSupport(rawDict, chip)
            if _shouldSkip(rawDict, solDict):
                continue
            try:
                asmStr, kernelName = _generateAsm(sol, assembler, debugConfig)
                hsaco = amdgpu_exec.compile_asm_to_hsaco(asmStr, chip)
            except Exception as exc:
                _log.warning("solution %s failed to compile: %s", sid, exc)
                continue
            compiled.append({
                "solDict": solDict,
                "rawDict": rawDict,
                "kernelName": kernelName,
                "hsaco": hsaco,
                "sid": sid,
            })
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
            gflops = 2 * M * N * K * batch / (benchResult.meanUs * 1e-6) / 1e9
            return gflops, benchResult
        except Exception as exc:
            _log.warning("benchmark failed for %s size=(%d,%d,%d,%d): %s",
                         entry["sid"], M, N, batch, K, exc)
            return -1.0, None
        finally:
            aBuf.free(); bBuf.free(); cBuf.free(); dPool.freeAll()

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
            results.append(SweepResult(
                solutionIdx=i,
                solutionName=entry["sid"],
                problemSize=probSize,
                benchmark=br,
                gflops=gflops,
            ))
        return results

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
            )
        if luRep:
            valid = [r for r in sizeResults if r.gflops > 0]
            if valid:
                winner = max(valid, key=lambda r: r.gflops)
                luRep.writeRow(list(probSize), winner.solutionIdx, winner.gflops)

    def _compile(self):
        """Detect chip, set up Tensile, compile all solutions; return (compiled, chip)."""
        import amdgpu_exec
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        compiled = self._compileAll(chip, assembler, isaInfoMap, debugConfig)
        return compiled, chip

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
        from epilogues.epilogue_harness.yaml_solution_builder import problemSizesFromYaml

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
        csvRep = (ResultsCSVReporter(resultsCsv, solNames, numSizeDims=numSizeDims)
                  if resultsCsv else None)
        luRep = LibraryUpdateReporter(libraryUpdateFile) if libraryUpdateFile else None
        if csvRep:
            csvRep.writeHeader()
        allResults = []
        for probSize in probSizes:
            sizeResults = self._benchmarkProblem(probSize, compiled, cuCount)
            allResults.extend(sizeResults)
            self._reportProblem(probSize, sizeResults, csvRep, luRep)
        if csvRep:
            csvRep.close()
        if luRep:
            luRep.close()
        return allResults
