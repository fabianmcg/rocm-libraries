# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M7 test suite: rotating buffers, I-cache module rotation, timing helpers.

Tasks 7.2–7.5 of the TensileLite Python client plan.
GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec + ml_dtypes.
Pure-Python tests run under plain tox -e unit.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import tempfile

import numpy as np
import pytest
from unittest.mock import MagicMock

try:
    import amdgpu_exec
    import ml_dtypes
    HAVE_DEPS = True
except ImportError:
    amdgpu_exec = None
    ml_dtypes = None
    HAVE_DEPS = False

try:
    from tensilelite_runtime import get_icache_module_copies
    HAVE_RUNTIME = True
except ImportError:
    get_icache_module_copies = None
    HAVE_RUNTIME = False

from .conftest import requires_gfx950

_TESTS_DIR = os.path.dirname(__file__)
_YAML_PATH = os.path.join(_TESTS_DIR, "yaml", "gemm_standard.yaml")
_TENSILE_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", ".."))

if _TENSILE_ROOT not in sys.path:
    sys.path.insert(0, _TENSILE_ROOT)

from Tensile.client.harness import BenchmarkResult, BufferPool, KernelRunner, autoScaleIters
from Tensile.client.gemm_args import (
    _computeInternalArg0,
    _computeInternalArg1,
)
from epilogues.epilogue_harness.yaml_solution_builder import _injectInternalArgsSupport


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_gemm_standard.py pattern)
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
    """Compile bf16 HPA solutions from YAML group 2 (same as test_gemm_standard)."""
    if not HAVE_DEPS:
        return []
    try:
        from epilogues.epilogue_harness.yaml_solution_builder import solutionsFromYaml
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        sols = solutionsFromYaml(_YAML_PATH, assembler, isaInfoMap, debugConfig, problemIdx=2)
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
    if not HAVE_DEPS:
        return 0
    props = amdgpu_exec._runtime_module.hip_get_device_props(0)
    return int(props.get("multiprocessor_count", 0))


def _buildArgs(sol_dict: dict, M: int, N: int, batch: int, K: int,
               D_buf, C_buf, A_buf, B_buf, alpha: float = 1.0, beta: float = 0.0):
    """Build a kernel argument list for stridedBatched=True NT GEMM."""
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
    stride_a, stride_b, stride_d, stride_c = M * K, N * K, M * N, M * N
    args.extend([
        np.uint32(ldd), np.uint32(stride_d),
        np.uint32(ldc), np.uint32(stride_c),
        np.uint32(lda), np.uint32(stride_a),
        np.uint32(ldb), np.uint32(stride_b),
    ])
    args.extend([np.float32(alpha), np.float32(beta)])
    return args, num_wg


# ---------------------------------------------------------------------------
# Session fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bf16Entry():
    """Return the first usable bf16 solution entry, or None if unavailable."""
    entries = _compileBf16Solutions()
    return entries[0] if entries else None


# ---------------------------------------------------------------------------
# Task 7.3 — TestBufferPool (pure Python)
# ---------------------------------------------------------------------------


class TestBufferPool:
    """Slot cycling in pure Python (no GPU)."""

    def test_cycles_three_slots(self):
        """pool.next() cycles through exactly nSlots buffers."""
        fake_cls = MagicMock(side_effect=lambda sz: MagicMock())
        pool = BufferPool(nSlots=3, sizeBytes=64, gpuBufferCls=fake_cls)
        bufs = [pool.next() for _ in range(6)]
        assert bufs[0] is bufs[3]
        assert bufs[1] is bufs[4]
        assert bufs[2] is bufs[5]

    def test_slot_advances_every_iteration(self):
        """Each call to next() advances the slot, including during warmup."""
        slots = [MagicMock() for _ in range(4)]
        idx = [0]

        def fake_cls(sz):
            m = slots[idx[0] % 4]
            idx[0] += 1
            return m

        pool = BufferPool(nSlots=4, sizeBytes=128, gpuBufferCls=fake_cls)
        seen = [pool.next() for _ in range(8)]
        # Every 4 calls should return the same slot.
        assert seen[0] is seen[4]
        assert seen[1] is seen[5]
        assert seen[2] is seen[6]
        assert seen[3] is seen[7]


# ---------------------------------------------------------------------------
# Task 7.4 — autoScaleIters (pure Python)
# ---------------------------------------------------------------------------


class TestTimingHelpers:
    """autoScaleIters replicates BenchmarkTimer::numEnqueuesPerSync."""

    def test_no_flop_budget_returns_base(self):
        assert autoScaleIters(flops=1_000_000, minFlopsPerSync=0) == 1

    def test_flop_budget_scales_up(self):
        # 10 GFLOPS per iteration, 100 GFLOPS budget → 10 iterations.
        flops = 10_000_000_000
        result = autoScaleIters(flops=flops, minFlopsPerSync=100_000_000_000)
        assert result == 10

    def test_max_enqueues_clamps(self):
        # Budget demands 10 but max is 5.
        flops = 10_000_000_000
        result = autoScaleIters(
            flops=flops,
            minFlopsPerSync=100_000_000_000,
            maxEnqueuesPerSync=5,
        )
        assert result == 5

    def test_negative_max_means_no_limit(self):
        # maxEnqueuesPerSync=-1 → no upper bound.
        flops = 1_000
        result = autoScaleIters(
            flops=flops,
            minFlopsPerSync=1_000_000,
            numEnqueuesPerSync=1,
            maxEnqueuesPerSync=-1,
        )
        assert result == 1000

    def test_zero_flops_does_not_divide_by_zero(self):
        # flops=0 is treated as max(0, 1) = 1.
        result = autoScaleIters(flops=0, minFlopsPerSync=100, numEnqueuesPerSync=1)
        assert result == 100

    def test_base_always_respected(self):
        # Even when flops budget needs fewer enqueues, numEnqueuesPerSync is the floor.
        result = autoScaleIters(
            flops=1_000_000_000_000,
            minFlopsPerSync=100,
            numEnqueuesPerSync=7,
        )
        assert result == 7


# ---------------------------------------------------------------------------
# Task 7.4 — TestTimingStats (GPU)
# ---------------------------------------------------------------------------


class TestTimingStats:
    """Verify BenchmarkResult timing invariants on real GPU data."""

    @requires_gfx950
    def test_ordering_invariants(self, bf16Entry):
        """p50_us <= p95_us and min_us <= p50_us and gflops > 0."""
        if not HAVE_DEPS or ml_dtypes is None:
            pytest.skip("amdgpu_exec or ml_dtypes not installed")
        if bf16Entry is None:
            pytest.skip("no bf16 solution compiled")

        from amdgpu_exec import GpuBuffer

        sol_dict = bf16Entry["sol_dict"]
        kernel_name = bf16Entry["kernel_name"]
        hsaco = bf16Entry["hsaco"]
        M, N, batch, K = 512, 512, 1, 512
        num_threads = sol_dict["NumThreads"]

        rng = np.random.default_rng(seed=42)
        A_np = np.asfortranarray(rng.random((M, K)).astype(ml_dtypes.bfloat16))
        B_np = np.asfortranarray(rng.random((N, K)).astype(ml_dtypes.bfloat16))
        D_np = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)
        C_np = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)
        A_buf = GpuBuffer(A_np.nbytes)
        B_buf = GpuBuffer(B_np.nbytes)
        C_buf = GpuBuffer(C_np.nbytes)
        D_buf = GpuBuffer(D_np.nbytes)
        A_buf.copy_from_host(A_np)
        B_buf.copy_from_host(B_np)
        C_buf.copy_from_host(C_np)

        args, num_wg = _buildArgs(sol_dict, M, N, batch, K, D_buf, C_buf, A_buf, B_buf)
        runner = KernelRunner.fromHsaco(hsaco, kernel_name, nModuleCopies=1)
        result = runner.run(
            argsFn=lambda _: args,
            grid=(num_wg, 1, 1),
            block=(num_threads, 1, 1),
            nWarmup=2,
            nIters=10,
        )

        assert result.minUs <= result.p50Us
        assert result.p50Us <= result.p95Us
        assert result.gflops(M, N, K) > 0

        for buf in [A_buf, B_buf, C_buf, D_buf]:
            buf.free()


# ---------------------------------------------------------------------------
# Task 7.2 — TestModuleRotation (GPU)
# ---------------------------------------------------------------------------


class TestModuleRotation:
    """n_module_copies=4 produces the same output as n_module_copies=1."""

    @requires_gfx950
    def test_output_identical_to_single_module(self, bf16Entry):
        if not HAVE_DEPS or ml_dtypes is None:
            pytest.skip("amdgpu_exec or ml_dtypes not installed")
        if bf16Entry is None:
            pytest.skip("no bf16 solution compiled")

        from amdgpu_exec import GpuBuffer

        sol_dict = bf16Entry["sol_dict"]
        kernel_name = bf16Entry["kernel_name"]
        hsaco = bf16Entry["hsaco"]
        M, N, batch, K = 256, 256, 1, 256
        num_threads = sol_dict["NumThreads"]

        rng = np.random.default_rng(seed=7)
        A_np = np.asfortranarray(rng.random((M, K)).astype(ml_dtypes.bfloat16))
        B_np = np.asfortranarray(rng.random((N, K)).astype(ml_dtypes.bfloat16))
        C_np = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)
        D_out = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)

        A_buf = GpuBuffer(A_np.nbytes)
        B_buf = GpuBuffer(B_np.nbytes)
        C_buf = GpuBuffer(C_np.nbytes)
        D_buf = GpuBuffer(D_out.nbytes)
        A_buf.copy_from_host(A_np)
        B_buf.copy_from_host(B_np)
        C_buf.copy_from_host(C_np)

        args, num_wg = _buildArgs(sol_dict, M, N, batch, K, D_buf, C_buf, A_buf, B_buf)

        # Run with 1 module copy, read reference output.
        D_buf.memset(0)
        runner1 = KernelRunner.fromHsaco(hsaco, kernel_name, nModuleCopies=1)
        runner1.run(
            argsFn=lambda _: args,
            grid=(num_wg, 1, 1),
            block=(num_threads, 1, 1),
            nWarmup=0,
            nIters=1,
        )
        D_ref = np.empty_like(D_out)
        D_buf.copy_to_host(D_ref)

        # Run with 4 module copies, compare output.
        D_buf.memset(0)
        runner4 = KernelRunner.fromHsaco(hsaco, kernel_name, nModuleCopies=4)
        runner4.run(
            argsFn=lambda _: args,
            grid=(num_wg, 1, 1),
            block=(num_threads, 1, 1),
            nWarmup=0,
            nIters=1,
        )
        D_rot = np.empty_like(D_out)
        D_buf.copy_to_host(D_rot)

        np.testing.assert_array_equal(D_ref, D_rot)

        for buf in [A_buf, B_buf, C_buf, D_buf]:
            buf.free()


# ---------------------------------------------------------------------------
# Task 7.1 — TestIcacheCopyCount
# ---------------------------------------------------------------------------


class TestIcacheCopyCount:
    """get_icache_module_copies on a compiled .co returns a positive integer."""

    @requires_gfx950
    def test_positive_integer_from_co_file(self, bf16Entry):
        if not HAVE_DEPS:
            pytest.skip("amdgpu_exec not installed")
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        if bf16Entry is None:
            pytest.skip("no bf16 solution compiled")

        hsaco = bf16Entry["hsaco"]
        # Write hsaco bytes to a temp file so get_icache_module_copies can parse it.
        with tempfile.NamedTemporaryFile(suffix=".co", delete=False) as tmp:
            tmp.write(hsaco)
            tmp_path = tmp.name

        try:
            n = get_icache_module_copies(tmp_path)
            assert isinstance(n, int)
            assert n > 0, f"expected positive int, got {n}"
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Task 7.2/7.5 — TestGflopsPlausibility (GPU)
# ---------------------------------------------------------------------------


class TestGflopsPlausibility:
    """GFLOPS for a 1024x1024x1024 bf16 GEMM is in [100, 1_000_000]."""

    @requires_gfx950
    def test_gflops_in_range(self, bf16Entry):
        if not HAVE_DEPS or ml_dtypes is None:
            pytest.skip("amdgpu_exec or ml_dtypes not installed")
        if bf16Entry is None:
            pytest.skip("no bf16 solution compiled")

        from amdgpu_exec import GpuBuffer

        sol_dict = bf16Entry["sol_dict"]
        kernel_name = bf16Entry["kernel_name"]
        hsaco = bf16Entry["hsaco"]
        M, N, batch, K = 1024, 1024, 1, 1024
        num_threads = sol_dict["NumThreads"]

        rng = np.random.default_rng(seed=100)
        A_np = np.asfortranarray(rng.random((M, K)).astype(ml_dtypes.bfloat16))
        B_np = np.asfortranarray(rng.random((N, K)).astype(ml_dtypes.bfloat16))
        C_np = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)
        D_np = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)

        A_buf = GpuBuffer(A_np.nbytes)
        B_buf = GpuBuffer(B_np.nbytes)
        C_buf = GpuBuffer(C_np.nbytes)
        D_buf = GpuBuffer(D_np.nbytes)
        A_buf.copy_from_host(A_np)
        B_buf.copy_from_host(B_np)
        C_buf.copy_from_host(C_np)

        args, num_wg = _buildArgs(sol_dict, M, N, batch, K, D_buf, C_buf, A_buf, B_buf)
        runner = KernelRunner.fromHsaco(hsaco, kernel_name, nModuleCopies=1)
        result = runner.run(
            argsFn=lambda _: args,
            grid=(num_wg, 1, 1),
            block=(num_threads, 1, 1),
            nWarmup=3,
            nIters=10,
        )

        gflops = result.gflops(M, N, K)
        # Lower bound (100 GFLOPS) catches ns-treated-as-µs bugs.
        # Upper bound (1_000_000 GFLOPS = 1 Exa-FLOPS) catches µs-treated-as-ms bugs
        # while remaining safely above any real hardware peak (gfx950 ~383 TFLOPS bf16).
        assert 100 <= gflops <= 1_000_000, (
            f"GFLOPS {gflops:.1f} is outside [100, 1000000] — "
            "possible unit-conversion bug or kernel not running"
        )

        for buf in [A_buf, B_buf, C_buf, D_buf]:
            buf.free()
