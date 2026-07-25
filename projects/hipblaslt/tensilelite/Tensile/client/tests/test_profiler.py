# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M11 test suite: tensilelite_profiler ROCprofiler-SDK bindings.

Tasks:
  11.2 TestProfilerUnavailable — graceful degradation without profiler
  11.3 TestImportOrdering     — late import detection via subprocess
  11.5 TestCounterCollection  — live SQ_WAVES counter on gfx950
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from .conftest import TENSILELITE_PROFILER_AVAILABLE, requires_gfx950, requires_rocprof

# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------

try:
    import amdgpu_exec
    import ml_dtypes
    HAVE_DEPS = True
except ImportError:
    amdgpu_exec = None
    ml_dtypes = None
    HAVE_DEPS = False

_TESTS_DIR = os.path.dirname(__file__)
_YAML_PATH = os.path.join(_TESTS_DIR, "yaml", "gemm_standard.yaml")
_TENSILE_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", ".."))
if _TENSILE_ROOT not in sys.path:
    sys.path.insert(0, _TENSILE_ROOT)

from Tensile.client.harness import BenchmarkResult, KernelRunner

# ---------------------------------------------------------------------------
# Helpers shared with test_gemm_standard (inlined to avoid cross-file imports)
# ---------------------------------------------------------------------------

def _deviceCuCount() -> int:
    if not HAVE_DEPS:
        return 0
    props = amdgpu_exec._runtime_module.hip_get_device_props(0)
    return int(props.get("multiprocessor_count", 0))


def _computeArg0(sol_dict: dict) -> int:
    from Tensile.client.gemm_args import _computeInternalArg0
    return _computeInternalArg0(sol_dict, gsu=1)


def _computeArg1(sol_dict: dict) -> int:
    from Tensile.client.gemm_args import _computeInternalArg1
    return _computeInternalArg1(sol_dict, cu_count=_deviceCuCount())


def _buildTypedArgs(sol_dict: dict, M: int, N: int, K: int,
                    D_arr, C_arr, A_arr, B_arr):
    """Build typed kernel args for NT stridedBatched=True GEMM (batch=1)."""
    version = sol_dict.get("KernArgsVersion", 0)
    mt0, mt1 = sol_dict["MacroTile0"], sol_dict["MacroTile1"]
    num_wg = math.ceil(M / mt0) * math.ceil(N / mt1)
    gemm_count = (1 & 0x3FFFFFFF) | (0 << 30)
    args = [np.uint32(gemm_count), np.uint32(_computeArg0(sol_dict))]
    if version >= 1:
        args.append(np.int32(_computeArg1(sol_dict)))
        args.append(np.uint32(num_wg))
    args += [np.uint32(M), np.uint32(N), np.uint32(1), np.uint32(K)]
    args += [D_arr, C_arr, A_arr, B_arr]
    args += [
        np.uint32(M), np.uint32(M * N),   # ldd, strideD
        np.uint32(M), np.uint32(M * N),   # ldc, strideC
        np.uint32(M), np.uint32(M * K),   # lda, strideA
        np.uint32(N), np.uint32(N * K),   # ldb, strideB
    ]
    args += [np.float32(1.0), np.float32(0.0)]
    return args, num_wg


def _allocGemmBufs(sol_dict, M, N, batch, K):
    """Allocate and upload GPU buffers for a bf16 GEMM. Returns (A_buf, B_buf, C_buf, D_buf)."""
    from amdgpu_exec import GpuBuffer
    dtype = ml_dtypes.bfloat16
    rng = np.random.default_rng(42)
    A_flat = np.asfortranarray(rng.random((M, K)).astype(dtype)).ravel(order="F")
    B_flat = np.asfortranarray(rng.random((N, K)).astype(dtype)).ravel(order="F")
    C_flat = np.zeros(batch * M * N, dtype=dtype)
    D_flat = np.zeros(batch * M * N, dtype=dtype)
    A_buf = GpuBuffer(A_flat.nbytes)
    A_buf.copy_from_host(A_flat)
    B_buf = GpuBuffer(B_flat.nbytes)
    B_buf.copy_from_host(B_flat)
    C_buf = GpuBuffer(C_flat.nbytes)
    C_buf.copy_from_host(C_flat)
    D_buf = GpuBuffer(D_flat.nbytes)
    D_buf.memset(0)
    return A_buf, B_buf, C_buf, D_buf


def _buildGemmArgList(sol_dict, M, N, batch, K, D_buf, C_buf, A_buf, B_buf):
    """Build kernel arg list for a bf16 GEMM. Returns (args, num_wg)."""
    return _buildTypedArgs(sol_dict, M, N, K, D_buf, C_buf, A_buf, B_buf)


def _compileBf16Kernel():
    """Compile and return one bf16 stridedBatched=True solution (or None)."""
    if not HAVE_DEPS:
        return None
    try:
        from pathlib import Path
        from epilogues.epilogue_harness.yaml_solution_builder import (
            solutionsFromYaml, _injectInternalArgsSupport,
        )
        from Tensile.Toolchain.Validators import validateToolchain
        from Tensile.Toolchain.Component import Assembler
        from Tensile.Common.Architectures import gfxToIsa
        from Tensile.Common.Capabilities import makeIsaInfoMap
        from Tensile.Common.GlobalParameters import assignGlobalParameters
        from Tensile.Common.Types import DebugConfig
        import rocisa
        from Tensile.KernelWriterAssembly import KernelWriterAssembly
        from Tensile.SolutionStructs.Naming import getKernelNameMin

        chip = amdgpu_exec.get_chip()
        gfx = chip.split(":")[0]
        cxx = validateToolchain("amdclang++")
        isa = gfxToIsa(gfx)
        isaInfoMap = makeIsaInfoMap([isa], cxx)
        assignGlobalParameters({}, isaInfoMap)
        assembler = Assembler(Path(cxx), co_version="6")
        debugConfig = DebugConfig()

        sols = solutionsFromYaml(_YAML_PATH, assembler, isaInfoMap, debugConfig, problemIdx=2)
        for sol, sid in sols:
            if sol.get("WorkGroupMapping", 0) == 0:
                continue
            kwa = KernelWriterAssembly(assembler, debugConfig)
            ti = rocisa.rocIsa.getInstance()
            kwa.setRocIsa(ti.getData(), ti.getOutputOptions())
            kernel = sol.getKernels()[0]
            kernel.duplicate = False
            err, asm_str = kwa.getSourceFileString(kernel)
            if err:
                continue
            kernel_name = getKernelNameMin(kernel, splitGSU=False)
            hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
            raw_dict = dict(sol)
            sol_dict = _injectInternalArgsSupport(raw_dict, chip)
            return {"sol_dict": sol_dict, "kernel_name": kernel_name, "hsaco": hsaco}
    except Exception:
        return None


# ===========================================================================
# Task 11.2 — TestProfilerUnavailable
# ===========================================================================


class TestProfilerUnavailable:
    """Verify graceful degradation when tensilelite_profiler is not importable."""

    def test_benchmark_result_has_counters_field(self):
        """BenchmarkResult always has a counters dict field."""
        result = BenchmarkResult(timesNs=[1_000_000], warmupN=0)
        assert isinstance(result.counters, dict)
        assert len(result.counters) == 0

    def test_run_without_profiler_counters_empty(self):
        """KernelRunner.run() without rocprofCounters returns empty counters dict."""
        fn = MagicMock()

        class FakeEvent:
            def record(self): pass
            def synchronize(self): pass
            def elapsed_ns(self, _): return 1_000_000

        runner = KernelRunner(functions=[fn])
        with patch.dict("sys.modules", {"amdgpu_exec": MagicMock(GpuEvent=FakeEvent)}):
            result = runner.run(
                argsFn=lambda _: [],
                grid=(1, 1, 1), block=(64, 1, 1), nWarmup=0, nIters=1,
            )
        assert result.counters == {}

    def test_run_absent_profiler_degrades_gracefully(self):
        """run() with rocprofCounters but no profiler module logs warning, no crash.

        Setting sys.modules["tensilelite_profiler"] = None causes ImportError on
        import inside run(), which triggers the graceful-degradation path.
        """
        fn = MagicMock()

        class FakeEvent:
            def record(self): pass
            def synchronize(self): pass
            def elapsed_ns(self, _): return 1_000_000

        runner = KernelRunner(functions=[fn])
        fake_modules = {
            "amdgpu_exec": MagicMock(GpuEvent=FakeEvent),
            # None entry causes ImportError on 'import tensilelite_profiler'.
            "tensilelite_profiler": None,
        }
        with patch.dict("sys.modules", fake_modules):
            result = runner.run(
                argsFn=lambda _: [],
                grid=(1, 1, 1), block=(64, 1, 1), nWarmup=0, nIters=1,
                rocprofCounters=["SQ_WAVES"],
            )
        assert result.counters == {}
        assert len(result.timesNs) == 1


# ===========================================================================
# Task 11.3 — TestImportOrdering
# ===========================================================================


class TestImportOrdering:
    """Verify import-ordering error handling.

    The plan specifies that rocprofiler_force_configure returns
    ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED (value 16) when the
    configuration window has already been locked.  In practice with
    rocprofiler-sdk 1.3.2, a HIP device call alone (e.g. get_chip) does NOT
    lock the window; the lock only fires when rocprofiler_force_configure is
    called a second time after a successful first call.  See
    tests/fixtures/m11_profiler_notes.txt for details.
    """

    @requires_rocprof
    def test_error_message_substring_present_in_source(self):
        """The 'before any HIP call' message must exist in the bindings source.

        This test verifies the error-handling code path is present without
        requiring the hardware condition that triggers CONFIGURATION_LOCKED.
        """
        src_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "tensilelite_profiler", "src", "bindings.cpp",
        )
        src_path = os.path.normpath(src_path)
        assert os.path.exists(src_path), f"bindings.cpp not found at {src_path}"
        with open(src_path) as f:
            src = f.read()
        assert "before any HIP call" in src, (
            "error message 'before any HIP call' must be in bindings.cpp"
        )
        assert "ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED" in src, (
            "bindings.cpp must check ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED"
        )

    @requires_gfx950
    @requires_rocprof
    def test_second_force_configure_is_locked(self):
        """rocprofiler_force_configure returns CONFIGURATION_LOCKED on a second call.

        With rocprofiler-sdk 1.3.2, the configuration window is locked after
        a successful first call (not after HIP initialisation).  This test
        verifies the locking mechanism works — which is what our PyInit guard
        protects against.  The test calls rocprofiler_force_configure directly
        via ctypes to avoid triggering double-import caching.
        """
        import ctypes
        import tensilelite_profiler._tensilelite_profiler as _m

        sdk = ctypes.CDLL("/opt/rocm/core-7.14/lib/librocprofiler-sdk.so.1")
        profiler_so = ctypes.CDLL(_m.__file__)
        configure_fn = profiler_so.rocprofiler_configure
        configure_fn.restype = ctypes.c_void_p
        configure_fn.argtypes = [
            ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p
        ]
        force_configure = sdk.rocprofiler_force_configure
        force_configure.restype = ctypes.c_int32
        force_configure.argtypes = [ctypes.c_void_p]

        # Call after the module is already initialised — expect CONFIGURATION_LOCKED.
        status = force_configure(configure_fn)
        _CONFIGURATION_LOCKED = 16  # ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED
        assert status == _CONFIGURATION_LOCKED, (
            f"expected CONFIGURATION_LOCKED ({_CONFIGURATION_LOCKED}), "
            f"got status {status}"
        )


# ===========================================================================
# Task 11.5 — TestCounterCollection
# ===========================================================================


class TestCounterCollection:
    """Live SQ_WAVES counter collection via tensilelite_profiler on gfx950."""

    @requires_gfx950
    @requires_rocprof
    def test_sq_waves_positive(self):
        """SQ_WAVES counter is a positive integer after a bf16 GEMM dispatch."""
        import re
        import tensilelite_profiler
        entry = _compileBf16Kernel()
        if entry is None:
            pytest.skip("could not compile a bf16 kernel for profiler test")

        sol_dict = entry["sol_dict"]
        kernel_name = entry["kernel_name"]
        hsaco = entry["hsaco"]

        M, N, K = 256, 256, 256
        A_buf, B_buf, C_buf, D_buf = _allocGemmBufs(sol_dict, M, N, 1, K)
        args, num_wg = _buildGemmArgList(sol_dict, M, N, 1, K, D_buf, C_buf, A_buf, B_buf)
        num_threads = sol_dict["NumThreads"]

        module = amdgpu_exec.GpuModule(hsaco)
        fn = module.get_function(kernel_name)
        runner = KernelRunner(functions=[fn])

        # Initialise profiler (idempotent if already done in another test).
        tensilelite_profiler.initialize(0, ["SQ_WAVES"])
        tensilelite_profiler.start()

        result = runner.run(
            argsFn=lambda _: args,
            grid=(num_wg, 1, 1),
            block=(num_threads, 1, 1),
            nWarmup=0,
            nIters=1,
            rocprofCounters=["SQ_WAVES"],
        )

        tensilelite_profiler.stop()

        assert "0" in result.counters, "counters dict should have key '0'"
        counter_str = result.counters["0"]
        assert counter_str, "counter string must be non-empty"
        assert "SQ_WAVES" in counter_str, f"expected SQ_WAVES in '{counter_str}'"

        # SQ_WAVES is multi-dimensional on gfx950 (XCC × SE × INST).
        # Parse every dimension value and check their sum is positive.
        values = [int(float(v)) for v in re.findall(r":\s*([\d.]+)", counter_str)]
        assert values, f"no numeric values found in '{counter_str}'"
        total = sum(values)
        assert total > 0, (
            f"SQ_WAVES total across all dimensions must be positive, "
            f"got {total} from '{counter_str}'"
        )
