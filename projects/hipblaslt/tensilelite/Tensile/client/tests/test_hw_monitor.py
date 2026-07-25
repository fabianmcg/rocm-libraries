# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M9 test suite: HardwareMonitor (pyamdsmi/amdsmi GPU polling).

Tasks 9.1–9.3 of the TensileLite Python client plan:
  9.1  HardwareMonitor implementation (hw_monitor.py)
  9.2  KernelRunner.run(hwMonitor=True) integration
  9.3  Tests: with amdsmi (GPU) and without amdsmi (pure Python)

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec + ml_dtypes.
Pure-Python tests run under plain tox -e unit.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import amdgpu_exec
    import ml_dtypes
    haveDeps = True
except ImportError:
    amdgpu_exec = None
    ml_dtypes = None
    haveDeps = False

from .conftest import requires_gfx950

_testsDir = os.path.dirname(__file__)
_yamlPath = os.path.join(_testsDir, "yaml", "gemm_standard.yaml")
_tensileRoot = os.path.abspath(os.path.join(_testsDir, "..", "..", "..", ".."))

if _tensileRoot not in sys.path:
    sys.path.insert(0, _tensileRoot)

from Tensile.client.harness import BenchmarkResult, KernelRunner
from Tensile.client.hw_monitor import HardwareMonitor
import Tensile.client.hw_monitor as _hw_mod


# ---------------------------------------------------------------------------
# Compilation helpers (mirrors test_harness_rotation.py pattern).
# ---------------------------------------------------------------------------


def _setupTensile(chip: str):
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
    import rocisa
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelNameMin

    kwa = KernelWriterAssembly(assembler, debugConfig)
    ti = rocisa.rocIsa.getInstance()
    kwa.setRocIsa(ti.getData(), ti.getOutputOptions())
    kernel = solution.getKernels()[0]
    kernel.duplicate = False
    err, asm_str = kwa.getSourceFileString(kernel)
    if err:
        raise RuntimeError(f"assembly generation failed: {err}")
    return asm_str, getKernelNameMin(kernel, splitGSU=False)


def _compileBf16Solutions():
    """Compile bf16 HPA solutions from YAML group 2."""
    if not haveDeps:
        return []
    try:
        from epilogues.epilogue_harness.yaml_solution_builder import solutionsFromYaml
        from epilogues.epilogue_harness.yaml_solution_builder import _injectInternalArgsSupport
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        sols = solutionsFromYaml(_yamlPath, assembler, isaInfoMap, debugConfig, problemIdx=2)
    except Exception as exc:
        import warnings
        warnings.warn(f"could not compile bf16 solutions: {exc}")
        return []

    compiled = []
    for sol, sid in sols:
        try:
            asm_str, kernel_name = _generateAsm(sol, assembler, debugConfig)
            hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
        except Exception as exc:
            import warnings
            warnings.warn(f"solution {sid} failed to compile: {exc}")
            continue
        raw_dict = dict(sol)
        sol_dict = _injectInternalArgsSupport(raw_dict, chip)
        if sol_dict.get("WorkGroupMapping", 0) == 0:
            continue
        isp = raw_dict.get("InternalSupportParams", {})
        if sol_dict.get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
            continue
        compiled.append({
            "sol_dict": sol_dict,
            "kernel_name": kernel_name,
            "hsaco": hsaco,
            "chip": chip,
            "sid": sid,
        })
    return compiled


def _deviceCuCount() -> int:
    if not haveDeps:
        return 0
    props = amdgpu_exec._runtime_module.hip_get_device_props(0)
    return int(props.get("multiprocessor_count", 0))


def _buildArgs(sol_dict, M, N, batch, K, D_buf, C_buf, A_buf, B_buf, alpha=1.0, beta=0.0):
    """Build kernel argument list for stridedBatched=True NT GEMM."""
    from Tensile.client.gemm_args import _computeInternalArg0, _computeInternalArg1

    version = sol_dict.get("KernArgsVersion", 0)
    mt0 = sol_dict["MacroTile0"]
    mt1 = sol_dict["MacroTile1"]
    num_wg = math.ceil(M / mt0) * math.ceil(N / mt1) * batch
    arg0 = _computeInternalArg0(sol_dict, gsu=1)
    gemm_count = (1 & 0x3FFFFFFF) | (0 << 30)
    args = [np.uint32(gemm_count), np.uint32(arg0)]
    if version >= 1:
        arg1 = _computeInternalArg1(sol_dict, cu_count=_deviceCuCount())
        args.append(np.int32(arg1))
        args.append(np.uint32(num_wg))
    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])
    args.extend([
        ctypes.c_void_p(D_buf.ptr_value),
        ctypes.c_void_p(C_buf.ptr_value),
        ctypes.c_void_p(A_buf.ptr_value),
        ctypes.c_void_p(B_buf.ptr_value),
    ])
    lda, ldb, ldd, ldc = M, N, M, M
    args.extend([
        np.uint32(ldd), np.uint32(M * N),
        np.uint32(ldc), np.uint32(M * N),
        np.uint32(lda), np.uint32(M * K),
        np.uint32(ldb), np.uint32(N * K),
    ])
    args.extend([np.float32(alpha), np.float32(beta)])
    return args, num_wg


# ---------------------------------------------------------------------------
# Session fixture.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bf16Entry():
    """Return the first usable bf16 solution entry, or None if unavailable."""
    entries = _compileBf16Solutions()
    return entries[0] if entries else None


# ---------------------------------------------------------------------------
# TestHwMonitorNoAmdsmi — pure Python, no GPU required.
# ---------------------------------------------------------------------------


class TestHwMonitorNoAmdsmi:
    """HardwareMonitor is a no-op when amdsmi is unavailable."""

    def test_no_amdsmi_no_crash(self):
        """__enter__ logs a warning and returns without crashing when amdsmi is absent."""
        with patch.object(_hw_mod, "_amdsmiMod", None), \
             patch.dict("sys.modules", {"pyamdsmi": None, "amdsmi": None}), \
             patch("Tensile.client.hw_monitor.os.path.isdir", return_value=False):
            mon = HardwareMonitor(deviceId=0, intervalMs=10)
            with mon:
                pass

        assert mon.avgGpuClockMhz == 0.0
        assert mon.avgTempEdge == 0.0
        assert mon.avgSocClockMhz == 0.0
        assert mon.avgMemClockMhz == 0.0

    def test_no_amdsmi_no_thread(self):
        """No daemon thread is started when amdsmi is absent."""
        with patch.object(_hw_mod, "_amdsmiMod", None), \
             patch.dict("sys.modules", {"pyamdsmi": None, "amdsmi": None}), \
             patch("Tensile.client.hw_monitor.os.path.isdir", return_value=False):
            mon = HardwareMonitor()
            with mon:
                pass

        assert mon._thread is None

    def test_no_amdsmi_kernel_runner_integration(self):
        """KernelRunner.run(hwMonitor=True) returns result.hw=None when amdsmi absent."""

        class FakeEvent:
            def record(self): pass
            def synchronize(self): pass
            def elapsed_ns(self, other): return 1_000_000

        fn = MagicMock()
        runner = KernelRunner(functions=[fn])

        with patch.object(_hw_mod, "_amdsmiMod", None), \
             patch.dict("sys.modules", {"pyamdsmi": None, "amdsmi": None}), \
             patch("Tensile.client.hw_monitor.os.path.isdir", return_value=False), \
             patch.dict("sys.modules", {"amdgpu_exec": MagicMock(GpuEvent=FakeEvent)}):
            result = runner.run(
                argsFn=lambda _: [],
                grid=(1, 1, 1),
                block=(64, 1, 1),
                nWarmup=0,
                nIters=2,
                hwMonitor=True,
            )

        # monitor is created but falls back to no-op; hw is still attached.
        assert result.hw is not None
        assert result.hw.avgGpuClockMhz == 0.0


# ---------------------------------------------------------------------------
# TestHwMonitorWithAmdsmi — GPU required.
# ---------------------------------------------------------------------------


class TestHwMonitorWithAmdsmi:
    """HardwareMonitor produces plausible readings on a real gfx950 GPU."""

    @requires_gfx950
    def test_monitor_values_plausible(self):
        """Clock and temperature are non-zero after a brief polling window."""
        if not haveDeps:
            pytest.skip("amdgpu_exec not installed")

        mon = HardwareMonitor(deviceId=0, intervalMs=10)
        with mon:
            time.sleep(0.1)  # collect roughly 10 samples

        assert mon.avgGpuClockMhz > 0, (
            f"expected avgGpuClockMhz > 0, got {mon.avgGpuClockMhz}"
        )
        assert mon.avgTempEdge > 0, (
            f"expected avgTempEdge > 0, got {mon.avgTempEdge} "
            "(falls back to hotspot temperature when edge sensor is N/A)"
        )

    def _allocBf16Bufs(self, M, N, batch, K):
        """Allocate and upload bf16 A/B/C/D GpuBuffers; return (A_buf, B_buf, C_buf, D_buf)."""
        from amdgpu_exec import GpuBuffer
        rng = np.random.default_rng(seed=9)
        A_np = np.asfortranarray(rng.random((M, K)).astype(ml_dtypes.bfloat16))
        B_np = np.asfortranarray(rng.random((N, K)).astype(ml_dtypes.bfloat16))
        C_np = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)
        A_buf = GpuBuffer(A_np.nbytes)
        B_buf = GpuBuffer(B_np.nbytes)
        C_buf = GpuBuffer(C_np.nbytes)
        D_buf = GpuBuffer(M * N * batch * 2)
        A_buf.copy_from_host(A_np)
        B_buf.copy_from_host(B_np)
        C_buf.copy_from_host(C_np)
        return A_buf, B_buf, C_buf, D_buf

    def test_kernel_runner_with_hw_monitor(self, bf16Entry):
        """run(hwMonitor=True) attaches a HardwareMonitor with plausible readings."""
        if not haveDeps or ml_dtypes is None:
            pytest.skip("amdgpu_exec or ml_dtypes not installed")
        if bf16Entry is None:
            pytest.skip("no bf16 solution compiled")
        sol_dict = bf16Entry["sol_dict"]
        M, N, batch, K = 512, 512, 1, 512
        A_buf, B_buf, C_buf, D_buf = self._allocBf16Bufs(M, N, batch, K)
        args, num_wg = _buildArgs(sol_dict, M, N, batch, K, D_buf, C_buf, A_buf, B_buf)
        runner = KernelRunner.fromHsaco(bf16Entry["hsaco"], bf16Entry["kernel_name"], nModuleCopies=1)
        result = runner.run(
            argsFn=lambda _: args,
            grid=(num_wg, 1, 1),
            block=(sol_dict["NumThreads"], 1, 1),
            nWarmup=2,
            nIters=10,
            hwMonitor=True,
        )
        assert result.hw is not None, "expected hw field to be set"
        assert result.hw.avgGpuClockMhz > 0, (
            f"expected avgGpuClockMhz > 0, got {result.hw.avgGpuClockMhz}"
        )
        assert result.hw.avgTempEdge > 0, (
            f"expected avgTempEdge > 0, got {result.hw.avgTempEdge} "
            "(falls back to hotspot temperature when edge sensor is N/A)"
        )
        for buf in [A_buf, B_buf, C_buf, D_buf]:
            buf.free()
