# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M8 test suite: sentinel-based bounds checking via BoundedBuffer.

Task 8.1 — TestSentinelIntegrity: allocation + checkSentinel without a kernel launch.
Task 8.2 — TestCorrectKernel:     bf16 GEMM with bounds_check=True; sentinel stays intact.
Task 8.3 — TestOverrunDetection:  kernel deliberately writes past a 4-byte valid region;
                                  checkSentinel() must return False.

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec + ml_dtypes.
TestSentinelIntegrity requires only tensilelite_bounds (which calls hipMalloc).
"""

from __future__ import annotations

import ctypes
import math
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guards — checked at collection time, not at runtime.
# ---------------------------------------------------------------------------

try:
    import amdgpu_exec
    import ml_dtypes
    haveDeps = True
except ImportError:
    amdgpu_exec = None
    ml_dtypes = None
    haveDeps = False

try:
    from tensilelite_bounds import BoundedBuffer
    haveBounds = True
except ImportError:
    BoundedBuffer = None
    haveBounds = False

from .conftest import requires_gfx950

# Skip markers — require both tensilelite_bounds and amdgpu_exec.
requiresBounds = pytest.mark.skipif(
    not haveBounds,
    reason="tensilelite_bounds not installed",
)
requiresDepsAndBounds = pytest.mark.skipif(
    not (haveBounds and haveDeps),
    reason="tensilelite_bounds or amdgpu_exec/ml_dtypes not installed",
)

# ---------------------------------------------------------------------------
# Module-level paths
# ---------------------------------------------------------------------------

_testsDir = os.path.dirname(__file__)
_yamlPath = os.path.join(_testsDir, "yaml", "gemm_standard.yaml")
_tensileRoot = os.path.abspath(os.path.join(_testsDir, "..", "..", "..", ".."))

if _tensileRoot not in sys.path:
    sys.path.insert(0, _tensileRoot)

from Tensile.client.harness import BufferPool, KernelRunner
from Tensile.client.gemm_args import (
    _computeInternalArg0,
    _computeInternalArg1,
)
from Tensile.client.reference import gemmBf16, assertClose, RTOL_BF16, ATOL_BF16
from Tensile.client.yaml_solution_builder import _injectInternalArgsSupport

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_harness_rotation.py pattern)
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
    if not haveDeps:
        return []
    try:
        from Tensile.client.yaml_solution_builder import solutionsFromYaml
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
        ctypes.c_void_p(D_buf.ptrValue),
        ctypes.c_void_p(int(C_buf.ptr)),
        ctypes.c_void_p(int(A_buf.ptr)),
        ctypes.c_void_p(int(B_buf.ptr)),
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
# Bounds-check test helpers
# ---------------------------------------------------------------------------


def _allocBf16Bufs(solDict, M, N, batch, K):
    """Allocate and upload A, B, C as GpuBuffer and D as a 1-slot BoundedBuffer pool."""
    from amdgpu_exec import GpuBuffer

    rng = np.random.default_rng(seed=42)
    A_np = np.asfortranarray(rng.random((M, K)).astype(ml_dtypes.bfloat16))
    B_np = np.asfortranarray(rng.random((N, K)).astype(ml_dtypes.bfloat16))
    C_np = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)

    A_buf = GpuBuffer(A_np.nbytes)
    B_buf = GpuBuffer(B_np.nbytes)
    C_buf = GpuBuffer(C_np.nbytes)
    A_buf.copy_from_host(A_np)
    B_buf.copy_from_host(B_np)
    C_buf.copy_from_host(C_np)

    D_size = M * N * batch * 2  # bf16 = 2 bytes per element.
    D_pool = BufferPool(nSlots=1, sizeBytes=D_size, gpuBufferCls=BoundedBuffer)
    return A_buf, B_buf, C_buf, D_pool, A_np, B_np


def _verifyBf16Result(D_pool, A_np, B_np, M, N, batch):
    """Copy D from device to host and compare to the numpy bf16 GEMM reference."""
    from amdgpu_exec._runtime_module import Ptr
    import amdgpu_exec._runtime_module as _rt

    D_result = np.empty(M * N * batch, dtype=ml_dtypes.bfloat16)
    _rt.hip_memcpy_device_to_host_async(
        Ptr(D_result.ctypes.data),
        Ptr(D_pool._slots[0].ptrValue),
        D_result.nbytes,
        None,
    )
    _rt.hip_device_synchronize()
    D_ref = gemmBf16(A_np, B_np.T)
    D_result_2d = D_result.reshape(M, N, order="F")
    assertClose(D_result_2d.astype(np.float32), D_ref.astype(np.float32),
                rtol=RTOL_BF16, atol=ATOL_BF16)


def _runAndCheckSentinel(entry, M, N, batch, K, boundedBuf):
    """Launch the kernel with boundedBuf as D and return boundedBuf.checkSentinel()."""
    from amdgpu_exec import GpuBuffer, GpuEvent, GpuModule

    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_threads = sol_dict["NumThreads"]

    rng = np.random.default_rng(seed=7)
    A_np = np.asfortranarray(rng.random((M, K)).astype(ml_dtypes.bfloat16))
    B_np = np.asfortranarray(rng.random((N, K)).astype(ml_dtypes.bfloat16))
    C_np = np.zeros(M * N * batch, dtype=ml_dtypes.bfloat16)

    A_buf = GpuBuffer(A_np.nbytes)
    B_buf = GpuBuffer(B_np.nbytes)
    C_buf = GpuBuffer(C_np.nbytes)
    A_buf.copy_from_host(A_np)
    B_buf.copy_from_host(B_np)
    C_buf.copy_from_host(C_np)

    args, num_wg = _buildArgs(sol_dict, M, N, batch, K, boundedBuf, C_buf, A_buf, B_buf)

    module = GpuModule(hsaco)
    fn = module.get_function(kernel_name)
    start = GpuEvent()
    stop = GpuEvent()
    start.record()
    fn.launch((num_wg, 1, 1), (num_threads, 1, 1), args)
    stop.record()
    stop.synchronize()

    result = boundedBuf.checkSentinel()
    for buf in [A_buf, B_buf, C_buf]:
        buf.free()
    return result


# ---------------------------------------------------------------------------
# Session fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bf16Entry():
    """Return the first usable bf16 solution entry, or None if unavailable."""
    entries = _compileBf16Solutions()
    return entries[0] if entries else None


# ---------------------------------------------------------------------------
# Task 8.1 — TestSentinelIntegrity (requires hipMalloc but no kernel launch)
# ---------------------------------------------------------------------------


class TestSentinelIntegrity:
    """Sentinel slots are intact immediately after BoundedBuffer construction."""

    @requiresBounds
    def test_sentinel_intact_after_alloc(self):
        """checkSentinel() returns True before any kernel touches the buffer."""
        buf = BoundedBuffer(size_bytes=64, sentinel_slots=4)
        try:
            assert buf.checkSentinel(), "sentinel must be intact after allocation"
        finally:
            buf.free()

    @requiresBounds
    def test_sentinelPtr_differs_from_ptrValue_by_size(self):
        """sentinelPtr equals ptrValue + size_bytes."""
        size = 128
        buf = BoundedBuffer(size_bytes=size, sentinel_slots=1)
        try:
            assert buf.sentinelPtr == buf.ptrValue + size
        finally:
            buf.free()

    @requiresBounds
    def test_dataPtr_aliases_ptrValue(self):
        """dataPtr is an alias for ptrValue."""
        buf = BoundedBuffer(size_bytes=32, sentinel_slots=2)
        try:
            assert buf.dataPtr == buf.ptrValue
        finally:
            buf.free()


# ---------------------------------------------------------------------------
# Task 8.3 — TestCorrectKernel (GPU, @requires_gfx950)
# ---------------------------------------------------------------------------


class TestCorrectKernel:
    """Correct bf16 GEMM leaves sentinel intact and output matches numpy reference."""

    @requires_gfx950
    def test_bf16_gemm_no_overrun(self, bf16Entry):
        if not haveBounds or not haveDeps or ml_dtypes is None:
            pytest.skip("tensilelite_bounds, amdgpu_exec, or ml_dtypes not installed")
        if bf16Entry is None:
            pytest.skip("no bf16 solution compiled")

        from amdgpu_exec import GpuModule

        sol_dict = bf16Entry["sol_dict"]
        kernel_name = bf16Entry["kernel_name"]
        hsaco = bf16Entry["hsaco"]
        M, N, batch, K = 256, 256, 1, 256
        num_threads = sol_dict["NumThreads"]

        A_buf, B_buf, C_buf, D_pool, A_np, B_np = _allocBf16Bufs(sol_dict, M, N, batch, K)
        def make_args(out_buf):
            return _buildArgs(sol_dict, M, N, batch, K, out_buf, C_buf, A_buf, B_buf)[0]

        module = GpuModule(hsaco)
        fn = module.get_function(kernel_name)
        runner = KernelRunner(functions=[fn], outputPool=D_pool)
        _, num_wg = _buildArgs(sol_dict, M, N, batch, K, D_pool._slots[0], C_buf, A_buf, B_buf)

        # boundsCheck=True: runner calls checkSentinel() on each pool slot after run.
        result = runner.run(
            argsFn=make_args,
            grid=(num_wg, 1, 1),
            block=(num_threads, 1, 1),
            nWarmup=0,
            nIters=1,
            boundsCheck=True,
        )
        assert result.timesNs, "expected at least one timing sample"

        # GPU uses NT layout: D[m,n] = sum_k A[m,k]*B[n,k], stored column-major.
        _verifyBf16Result(D_pool, A_np, B_np, M, N, batch)

        for buf in [A_buf, B_buf, C_buf]:
            buf.free()
        D_pool.freeAll()


# ---------------------------------------------------------------------------
# Task 8.3 — TestOverrunDetection (GPU, @requires_gfx950)
# ---------------------------------------------------------------------------


class TestOverrunDetection:
    """Kernel that writes past a 4-byte valid region must overwrite the sentinel."""

    @requires_gfx950
    def test_sentinel_overwritten_by_overrun(self, bf16Entry):
        """BoundedBuffer(size_bytes=4) used as D; kernel writes 64*64*2 bytes → overrun."""
        if not haveBounds or not haveDeps or ml_dtypes is None:
            pytest.skip("tensilelite_bounds, amdgpu_exec, or ml_dtypes not installed")
        if bf16Entry is None:
            pytest.skip("no bf16 solution compiled")

        # Output will be M*N*2 = 64*64*2 = 8192 bytes, well past the 4-byte region.
        M, N, batch, K = 64, 64, 1, 64
        # Only 4 bytes of valid region; kernel writes 8192 bytes → guaranteed overrun.
        D_buf = BoundedBuffer(size_bytes=4, sentinel_slots=4)
        try:
            sentinel_ok = _runAndCheckSentinel(bf16Entry, M, N, batch, K, D_buf)
            assert not sentinel_ok, (
                "expected sentinel to be overwritten by an 8192-byte write into a 4-byte region"
            )
        finally:
            D_buf.free()
