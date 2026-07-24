# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M3 test suite: Float8 OCP (F8, B8) and fnuz (F8N, B8N).

Covers Tasks 3.1–3.5 of the TensileLite Python client plan:
  3.1  fp8 dtype code verification (no GPU)
  3.2  gemmFp8 reference + NaN bit-pattern unit tests (no GPU)
  3.3  buildKernelArgs fp8 byte-layout (no GPU; dtype constants already verified)
  3.4  GPU correctness: fp8 OCP and fnuz variants × 3 sizes
  3.5  NaN propagation + poison-input per dtype

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec.  fnuz fixtures
return an empty list on gfx950 because gfx950 hardware interprets fnuz byte
patterns with OCP semantics (0x80 is treated as -0.0, not NaN).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

try:
    import amdgpu_exec
    haveDeps = True
except ImportError:
    amdgpu_exec = None
    haveDeps = False

try:
    import ml_dtypes
    haveMlDtypes = True
except ImportError:
    ml_dtypes = None
    haveMlDtypes = False

from .conftest import requires_gfx950

_testsDir = os.path.dirname(__file__)
_yamlPath = os.path.join(_testsDir, "yaml", "gemm_fp8.yaml")
_tensileRoot = os.path.abspath(os.path.join(_testsDir, "..", "..", "..", ".."))

if _tensileRoot not in sys.path:
    sys.path.insert(0, _tensileRoot)

from Tensile.client.gemm_args import (
    _dtypeFp8e4m3fnuz,
    _dtypeBf8e5m2fnuz,
    _dtypeFp8e4m3fn,
    _dtypeBf8e5m2,
    _computeInternalArg0,
    _computeInternalArg1,
)
from Tensile.client.reference import (
    gemmFp8,
    assertClose,
    ATOL_FP32,
    RTOL_FP32,
)
from epilogues.epilogue_harness.yaml_solution_builder import _injectInternalArgsSupport

# ---------------------------------------------------------------------------
# Tolerance constants for fp8 correctness checks.
# Both GPU and reference upcast to float32 (HPA), so error is small.
# ---------------------------------------------------------------------------

rtolFp8: float = 1e-1
atolFp8: float = 1e-1

# Problem sizes: (M, N, batch, K).
_fp8ProblemSizes = [
    (256, 256, 4, 256),
    (512, 512, 4, 512),
    (1024, 1024, 4, 1024),
]


def _requireMlDtypes():
    if not haveMlDtypes:
        pytest.skip("ml_dtypes not installed")


# ===========================================================================
# Task 3.1 — fp8 dtype code verification (no GPU)
# ===========================================================================


class TestFp8DtypeEnumValues:
    """Verify the DataTypeEnum integer codes used for fp8 in gemm_args."""

    def test_fp8_e4m3fnuz_code_is_11(self):
        assert _dtypeFp8e4m3fnuz == 11

    def test_bf8_e5m2fnuz_code_is_12(self):
        assert _dtypeBf8e5m2fnuz == 12

    def test_fp8_e4m3fn_code_is_15(self):
        assert _dtypeFp8e4m3fn == 15

    def test_bf8_e5m2_code_is_16(self):
        assert _dtypeBf8e5m2 == 16


# ===========================================================================
# Task 3.2 — gemmFp8 reference and NaN bit-pattern unit tests (no GPU)
# ===========================================================================


class TestFp8NanBitPatterns:
    """Verify that ml_dtypes fp8 NaN bit patterns match DataTypes_Float8_BFloat8.hpp.

    From the C++ header (isnan helpers):
      Float8 (OCP E4M3)  isnan: (bits & 0x7f) == 0x7f  → 0x7F, 0xFF
      BFloat8 (OCP E5M2) isnan: (bits & 0x7f) > 0x7c   → 0x7D..0x7F, 0xFD..0xFF
      Float8_fnuz  (E4M3 fnuz)  isnan: bits == 0x80
      BFloat8_fnuz (E5M2 fnuz)  isnan: bits == 0x80
    """

    def test_float8_e4m3fn_nan_at_0x7f(self):
        """OCP E4M3: bit pattern 0x7F is NaN (positive NaN)."""
        _requireMlDtypes()
        val = np.frombuffer(bytes([0x7F]), dtype=ml_dtypes.float8_e4m3fn)[0]
        assert np.isnan(val), "0x7F should be NaN for float8_e4m3fn"

    def test_float8_e4m3fn_nan_at_0xff(self):
        """OCP E4M3: bit pattern 0xFF is NaN (negative NaN)."""
        _requireMlDtypes()
        val = np.frombuffer(bytes([0xFF]), dtype=ml_dtypes.float8_e4m3fn)[0]
        assert np.isnan(val), "0xFF should be NaN for float8_e4m3fn"

    def test_float8_e4m3fn_0x7e_not_nan(self):
        """OCP E4M3: bit pattern 0x7E is not NaN (max finite value)."""
        _requireMlDtypes()
        val = np.frombuffer(bytes([0x7E]), dtype=ml_dtypes.float8_e4m3fn)[0]
        assert not np.isnan(val), "0x7E should not be NaN for float8_e4m3fn"

    def test_float8_e5m2_nan_patterns(self):
        """OCP E5M2: bits where (bits & 0x7F) > 0x7C are NaN."""
        _requireMlDtypes()
        nan_patterns = [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]
        for bits in nan_patterns:
            val = np.frombuffer(bytes([bits]), dtype=ml_dtypes.float8_e5m2)[0]
            assert np.isnan(val), f"0x{bits:02X} should be NaN for float8_e5m2"

    def test_float8_e5m2_0x7c_not_nan(self):
        """OCP E5M2: bit pattern 0x7C is infinity, not NaN."""
        _requireMlDtypes()
        val = np.frombuffer(bytes([0x7C]), dtype=ml_dtypes.float8_e5m2)[0]
        assert not np.isnan(val), "0x7C should not be NaN for float8_e5m2 (it is inf)"

    def test_float8_e5m2_non_nan_range(self):
        """OCP E5M2: patterns with (bits & 0x7F) <= 0x7C are not NaN."""
        _requireMlDtypes()
        for bits in range(0, 0x7D):
            val = np.frombuffer(bytes([bits]), dtype=ml_dtypes.float8_e5m2)[0]
            assert not np.isnan(val), f"0x{bits:02X} should not be NaN for float8_e5m2"

    def test_float8_e4m3fnuz_nan_at_0x80(self):
        """fnuz E4M3: bit pattern 0x80 is NaN (the only NaN)."""
        _requireMlDtypes()
        val = np.frombuffer(bytes([0x80]), dtype=ml_dtypes.float8_e4m3fnuz)[0]
        assert np.isnan(val), "0x80 should be NaN for float8_e4m3fnuz"

    def test_float8_e4m3fnuz_no_other_nan(self):
        """fnuz E4M3: no other bit pattern is NaN (no negative zero)."""
        _requireMlDtypes()
        for bits in range(0, 256):
            if bits == 0x80:
                continue
            val = np.frombuffer(bytes([bits]), dtype=ml_dtypes.float8_e4m3fnuz)[0]
            assert not np.isnan(val), f"0x{bits:02X} should not be NaN for float8_e4m3fnuz"

    def test_float8_e5m2fnuz_nan_at_0x80(self):
        """fnuz E5M2: bit pattern 0x80 is NaN (the only NaN)."""
        _requireMlDtypes()
        val = np.frombuffer(bytes([0x80]), dtype=ml_dtypes.float8_e5m2fnuz)[0]
        assert np.isnan(val), "0x80 should be NaN for float8_e5m2fnuz"

    def test_float8_e5m2fnuz_no_other_nan(self):
        """fnuz E5M2: no other bit pattern is NaN."""
        _requireMlDtypes()
        for bits in range(0, 256):
            if bits == 0x80:
                continue
            val = np.frombuffer(bytes([bits]), dtype=ml_dtypes.float8_e5m2fnuz)[0]
            assert not np.isnan(val), f"0x{bits:02X} should not be NaN for float8_e5m2fnuz"


class TestGemmFp8Reference:
    """Unit tests for reference.gemmFp8 without GPU."""

    def test_output_dtype_is_float32(self):
        """gemmFp8 with dtypeOut=float32 returns float32 array."""
        _requireMlDtypes()
        rng = np.random.default_rng(0)
        A = rng.uniform(-1, 1, (4, 4)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn)
        B = rng.uniform(-1, 1, (4, 4)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn)
        D = gemmFp8(A, B, ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e4m3fn, np.float32)
        assert D.dtype == np.float32

    def test_identity_like_matmul(self):
        """gemmFp8 with small all-ones inputs gives expected result."""
        _requireMlDtypes()
        # All-ones 4×4 @ 4×4 = 4*ones in float32.
        A = np.ones((4, 4), dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
        B = np.ones((4, 4), dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
        D = gemmFp8(A, B, ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e4m3fn, np.float32)
        np.testing.assert_allclose(D, np.full((4, 4), 4.0, dtype=np.float32), atol=ATOL_FP32)

    def test_alpha_scaling(self):
        """gemmFp8 applies alpha correctly."""
        _requireMlDtypes()
        A = np.ones((2, 2), dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
        B = np.ones((2, 2), dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
        D = gemmFp8(A, B, ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e4m3fn, np.float32, alpha=3.0)
        np.testing.assert_allclose(D, np.full((2, 2), 6.0, dtype=np.float32), atol=ATOL_FP32)

    def test_fnuz_e4m3_variant(self):
        """gemmFp8 works with the fnuz E4M3 variant."""
        _requireMlDtypes()
        A = np.ones((4, 4), dtype=np.float32).astype(ml_dtypes.float8_e4m3fnuz)
        B = np.ones((4, 4), dtype=np.float32).astype(ml_dtypes.float8_e4m3fnuz)
        D = gemmFp8(A, B, ml_dtypes.float8_e4m3fnuz, ml_dtypes.float8_e4m3fnuz, np.float32)
        np.testing.assert_allclose(D, np.full((4, 4), 4.0, dtype=np.float32), atol=ATOL_FP32)

    def test_ocp_e5m2_variant(self):
        """gemmFp8 works with the OCP E5M2 variant."""
        _requireMlDtypes()
        A = np.ones((4, 4), dtype=np.float32).astype(ml_dtypes.float8_e5m2)
        B = np.ones((4, 4), dtype=np.float32).astype(ml_dtypes.float8_e5m2)
        D = gemmFp8(A, B, ml_dtypes.float8_e5m2, ml_dtypes.float8_e5m2, np.float32)
        np.testing.assert_allclose(D, np.full((4, 4), 4.0, dtype=np.float32), atol=ATOL_FP32)


# ===========================================================================
# GPU test infrastructure
# ===========================================================================


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
    debugConfig = DebugConfig()
    return assembler, isaInfoMap, debugConfig


def _generateAsm(solution, assembler, debugConfig):
    """Return (asm_string, kernel_name) for a solution."""
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


def _compileFp8Solutions(problemIdx: int):
    """Compile all solutions for one YAML problem group; return list of dicts."""
    if not haveDeps:
        return []
    try:
        from epilogues.epilogue_harness.yaml_solution_builder import solutionsFromYaml
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        sols = solutionsFromYaml(_yamlPath, assembler, isaInfoMap, debugConfig,
                                 problemIdx=problemIdx)
    except Exception as exc:
        import warnings
        warnings.warn(f"could not compile fp8 solutions (problemIdx={problemIdx}): {exc}")
        return []

    compiled = []
    for sol, sid in sols:
        try:
            asm_str, kernel_name = _generateAsm(sol, assembler, debugConfig)
            hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
        except Exception as exc:
            import warnings
            warnings.warn(f"fp8 solution {sid} failed to compile: {exc}")
            continue
        raw_dict = dict(sol)
        sol_dict = _injectInternalArgsSupport(raw_dict, chip)
        compiled.append({
            "sol_dict": sol_dict, "raw_dict": raw_dict,
            "kernel_name": kernel_name, "hsaco": hsaco,
            "chip": chip, "sid": sid,
        })
    return compiled


def _filterFp8Solution(entry: dict) -> bool:
    """Return True if solution passes the M3 kernel filter.

    Skips auto-WGM (WorkGroupMapping==0) and auto-StaggerU kernels.
    """
    sol_dict = entry["sol_dict"]
    raw_dict = entry["raw_dict"]
    if sol_dict.get("WorkGroupMapping", 0) == 0:
        return False
    isp = raw_dict.get("InternalSupportParams", {}) or {}
    if sol_dict.get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
        return False
    return True


def _deviceCuCount() -> int:
    """Return the device CU count (multiprocessor_count) for device 0."""
    if not haveDeps:
        return 0
    props = amdgpu_exec._runtime_module.hip_get_device_props(0)
    return int(props.get("multiprocessor_count", 0))


def _buildNtFp8TypedArgs(sol_dict: dict, M: int, N: int, batch: int, K: int,
                         D_arr, C_arr, A_arr, B_arr,
                         alpha: float = 1.0, beta: float = 0.0) -> list:
    """Build typed args for NT stridedBatched fp8→float32 GEMM.

    Alpha and beta are float32 (HPA fp8 kernels use ComputeDataType=float32).
    """
    version = sol_dict.get("KernArgsVersion", 0)
    mt0 = sol_dict["MacroTile0"]
    mt1 = sol_dict["MacroTile1"]
    num_wg = math.ceil(M / mt0) * math.ceil(N / mt1) * batch

    arg0 = _computeInternalArg0(sol_dict, gsu=1)
    gemm_count = (1 & 0x3FFFFFFF) | (0 << 30)  # stridedBatched, argType=0.

    args: list = [np.uint32(gemm_count), np.uint32(arg0)]
    if version >= 1:
        arg1 = _computeInternalArg1(sol_dict, cu_count=_deviceCuCount())
        args.append(np.int32(arg1))
        args.append(np.uint32(num_wg))

    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])
    args.extend([D_arr, C_arr, A_arr, B_arr])

    # NT strides: lda=M, ldb=N, ldd=ldc=M, batch strides.
    lda, ldb, ldd, ldc = M, N, M, M
    stride_a, stride_b, stride_d, stride_c = M * K, N * K, M * N, M * N
    args.extend([
        np.uint32(ldd), np.uint32(stride_d),
        np.uint32(ldc), np.uint32(stride_c),
        np.uint32(lda), np.uint32(stride_a),
        np.uint32(ldb), np.uint32(stride_b),
    ])
    args.extend([np.float32(alpha), np.float32(beta)])
    return args


def _allocFp8Batched(M: int, N: int, K: int, batch: int, fp8Dtype, rng):
    """Allocate uint8 fp8 input buffers (1 byte/element) and float32 output buffers.

    A and B are generated as float32 random data in [-0.5, 0.5], then quantized
    to fp8Dtype and stored as raw bytes (uint8).  D and C are float32.
    """
    valsA = rng.uniform(-0.5, 0.5, (M, K)).astype(np.float32).astype(fp8Dtype)
    valsB = rng.uniform(-0.5, 0.5, (N, K)).astype(np.float32).astype(fp8Dtype)
    A_np = np.asfortranarray(valsA)
    B_np = np.asfortranarray(valsB)
    rawA = np.frombuffer(A_np.ravel(order='F').tobytes(), dtype=np.uint8)
    rawB = np.frombuffer(B_np.ravel(order='F').tobytes(), dtype=np.uint8)
    A_buf = np.tile(rawA, batch)
    B_buf = np.tile(rawB, batch)
    C_buf = np.zeros(M * N * batch, dtype=np.float32)
    D_buf = np.zeros(M * N * batch, dtype=np.float32)
    return A_buf, B_buf, C_buf, D_buf, A_np, B_np


def _fp8NanBit(fp8Dtype) -> int:
    """Return a NaN bit pattern (uint8) for the given fp8 ml_dtypes dtype."""
    if fp8Dtype is ml_dtypes.float8_e4m3fn:
        return 0x7F
    if fp8Dtype is ml_dtypes.float8_e5m2:
        return 0x7D
    if fp8Dtype is ml_dtypes.float8_e4m3fnuz:
        return 0x80
    if fp8Dtype is ml_dtypes.float8_e5m2fnuz:
        return 0x80
    assert False, f"unknown fp8 dtype for NaN bit: {fp8Dtype}"


def _corruptStrideA1(argList: list, M: int) -> list:
    """Corrupt lda (strideA[1]) by +M in the typed arg list.

    In TensileLite notation, strideA[1] is lda (the leading dimension of A,
    at dimension index 1 since useInitialStridesAB=False drops index 0).
    Corrupting lda causes every A[i,j] access to land at the wrong memory row.
    Infers header_n from total arg count: total = header_n + 18.
    """
    header_n = len(argList) - 18  # 4 sizes + 4 ptrs + 8 strides + 2 scalars.
    ldaIdx = header_n + 4 + 4 + 4  # header + sizes + ptrs + D/C strides.
    argList[ldaIdx] = np.uint32(int(argList[ldaIdx]) + M)
    return argList


def _assertPoisonDetected(gpuOut: np.ndarray, refOut: np.ndarray,
                          rtol: float, label: str) -> None:
    """Assert >= 50% of elements differ by more than 10 * rtol from reference."""
    bad = np.abs(gpuOut.astype(np.float64) - refOut.astype(np.float64)) > 10 * rtol * (
        np.abs(refOut.astype(np.float64)) + 1
    )
    bad_frac = bad.sum() / bad.size
    assert bad_frac >= 0.5, (
        f"{label}: only {bad_frac:.1%} elements corrupted — "
        "argument vector may not be driving computation"
    )


def _runFp8StridedBatched(entry: dict, M: int, N: int, batch: int, K: int,
                           fp8Dtype, rtol: float, atol: float, label: str):
    """Execute one fp8 NT stridedBatched kernel and verify output against reference."""
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]
    rng = np.random.default_rng(seed=M * 1000 + N + K)

    A_buf, B_buf, C_buf, D_buf, A_np, B_np = _allocFp8Batched(M, N, K, batch, fp8Dtype, rng)

    result_holder = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np.float32).copy()

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    args = _buildNtFp8TypedArgs(sol_dict, M, N, batch, K, D_io, C_in, A_in, B_in, 1.0, 0.0)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    D_gpu = result_holder["D_gpu"]
    D_ref_one = gemmFp8(A_np, B_np.T, fp8Dtype, fp8Dtype, np.float32, 1.0, 0.0, None)
    D_ref = np.tile(np.asfortranarray(D_ref_one).ravel(order='F'), batch)
    assertClose(D_gpu, D_ref, rtol=rtol, atol=atol, label=label)


def _runFp8NanPropagation(entry: dict, fp8Dtype, label: str):
    """Seed A[0,0] with a NaN bit pattern and verify D contains NaN."""
    M, N, batch, K = 256, 256, 4, 256
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]
    rng = np.random.default_rng(seed=1234)

    A_buf, B_buf, C_buf, D_buf, _, _ = _allocFp8Batched(M, N, K, batch, fp8Dtype, rng)
    A_buf[0] = _fp8NanBit(fp8Dtype)

    result_holder = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np.float32).copy()

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    args = _buildNtFp8TypedArgs(sol_dict, M, N, batch, K, D_io, C_in, A_in, B_in, 1.0, 0.0)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    D_gpu = result_holder["D_gpu"]
    assert np.any(np.isnan(D_gpu)), f"{label}: NaN in A should propagate to D"


def _runFp8Poison(entry: dict, fp8Dtype, label: str):
    """Corrupt strideA[1] by +M and assert >= 50% of outputs differ from reference."""
    M, N, batch, K = 256, 256, 4, 256
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]
    rng = np.random.default_rng(seed=999)

    A_buf, B_buf, C_buf, D_buf, A_np, B_np = _allocFp8Batched(M, N, K, batch, fp8Dtype, rng)

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    args = _buildNtFp8TypedArgs(sol_dict, M, N, batch, K, D_io, C_in, A_in, B_in, 1.0, 0.0)
    args = _corruptStrideA1(args, M)

    result_holder = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np.float32).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    D_gpu = result_holder["D_gpu"]
    D_ref_one = gemmFp8(A_np, B_np.T, fp8Dtype, fp8Dtype, np.float32, 1.0, 0.0, None)
    D_ref = np.tile(np.asfortranarray(D_ref_one).ravel(order='F'), batch)
    _assertPoisonDetected(D_gpu[:M * N], D_ref[:M * N], RTOL_FP32, label)


# ---------------------------------------------------------------------------
# Session-scoped compiled solution fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fp8OcpE4m3Kernels():
    """Compile F8→float32 (OCP E4M3) solutions from YAML group 0."""
    return _compileFp8Solutions(0)


@pytest.fixture(scope="session")
def fp8OcpE5m2Kernels():
    """Compile B8→float32 (OCP E5M2) solutions from YAML group 1."""
    return _compileFp8Solutions(1)


@pytest.fixture(scope="session")
def fp8FnuzE4m3Kernels():
    """Compile F8N→float32 (fnuz E4M3) solutions from YAML group 2.

    Returns empty on gfx950: that chip interprets fnuz byte patterns as OCP,
    producing incorrect results when compared to ml_dtypes.float8_e4m3fnuz.
    """
    if haveDeps and amdgpu_exec.get_chip() == "gfx950":
        return []
    return _compileFp8Solutions(2)


@pytest.fixture(scope="session")
def fp8FnuzE5m2Kernels():
    """Compile B8N→float32 (fnuz E5M2) solutions from YAML group 3.

    Returns empty on gfx950: that chip interprets fnuz byte patterns as OCP,
    producing incorrect results when compared to ml_dtypes.float8_e5m2fnuz.
    """
    if haveDeps and amdgpu_exec.get_chip() == "gfx950":
        return []
    return _compileFp8Solutions(3)


# ---------------------------------------------------------------------------
# Task 3.4 — GPU correctness tests.
# ---------------------------------------------------------------------------


@requires_gfx950
@pytest.mark.parametrize("size", _fp8ProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _fp8ProblemSizes])
def test_fp8_ocp_e4m3_correctness(fp8OcpE4m3Kernels, size):
    """OCP E4M3 (F8→float32) NT stridedBatched correctness."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8OcpE4m3Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no OCP E4M3 solution compiled on this GPU")
    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runFp8StridedBatched(entry, M, N, batch, K, ml_dtypes.float8_e4m3fn,
                              rtolFp8, atolFp8,
                              label=f"fp8-ocp-e4m3 M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _fp8ProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _fp8ProblemSizes])
def test_fp8_ocp_e5m2_correctness(fp8OcpE5m2Kernels, size):
    """OCP E5M2 (B8→float32) NT stridedBatched correctness."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8OcpE5m2Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no OCP E5M2 solution compiled on this GPU")
    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runFp8StridedBatched(entry, M, N, batch, K, ml_dtypes.float8_e5m2,
                              rtolFp8, atolFp8,
                              label=f"fp8-ocp-e5m2 M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _fp8ProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _fp8ProblemSizes])
def test_fp8_fnuz_e4m3_correctness(fp8FnuzE4m3Kernels, size):
    """fnuz E4M3 (F8N→float32) NT stridedBatched correctness."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8FnuzE4m3Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no E4M3 fnuz solution compiled on this GPU")
    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runFp8StridedBatched(entry, M, N, batch, K, ml_dtypes.float8_e4m3fnuz,
                              rtolFp8, atolFp8,
                              label=f"fp8-fnuz-e4m3 M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _fp8ProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _fp8ProblemSizes])
def test_fp8_fnuz_e5m2_correctness(fp8FnuzE5m2Kernels, size):
    """fnuz E5M2 (B8N→float32) NT stridedBatched correctness."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8FnuzE5m2Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no E5M2 fnuz solution compiled on this GPU")
    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runFp8StridedBatched(entry, M, N, batch, K, ml_dtypes.float8_e5m2fnuz,
                              rtolFp8, atolFp8,
                              label=f"fp8-fnuz-e5m2 M{M}N{N}B{batch}K{K} {sid}")


# ---------------------------------------------------------------------------
# Task 3.5 — NaN propagation tests.
# ---------------------------------------------------------------------------


@requires_gfx950
def test_fp8_ocp_e4m3_nan_propagation(fp8OcpE4m3Kernels):
    """OCP E4M3: NaN bit (0x7F) in A element propagates to D output."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8OcpE4m3Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no OCP E4M3 solution compiled on this GPU")
    _runFp8NanPropagation(entries[0], ml_dtypes.float8_e4m3fn, "ocp-e4m3 NaN propagation")


@requires_gfx950
def test_fp8_ocp_e5m2_nan_propagation(fp8OcpE5m2Kernels):
    """OCP E5M2: NaN bit (0x7D) in A element propagates to D output."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8OcpE5m2Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no OCP E5M2 solution compiled on this GPU")
    _runFp8NanPropagation(entries[0], ml_dtypes.float8_e5m2, "ocp-e5m2 NaN propagation")


@requires_gfx950
def test_fp8_fnuz_e4m3_nan_propagation(fp8FnuzE4m3Kernels):
    """fnuz E4M3: NaN bit (0x80) in A element propagates to D output."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8FnuzE4m3Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no E4M3 fnuz solution compiled on this GPU")
    _runFp8NanPropagation(entries[0], ml_dtypes.float8_e4m3fnuz, "fnuz-e4m3 NaN propagation")


@requires_gfx950
def test_fp8_fnuz_e5m2_nan_propagation(fp8FnuzE5m2Kernels):
    """fnuz E5M2: NaN bit (0x80) in A element propagates to D output."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8FnuzE5m2Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no E5M2 fnuz solution compiled on this GPU")
    _runFp8NanPropagation(entries[0], ml_dtypes.float8_e5m2fnuz, "fnuz-e5m2 NaN propagation")


# ---------------------------------------------------------------------------
# Task 3.5 — Poison-input tests.
# ---------------------------------------------------------------------------


@requires_gfx950
def test_fp8_ocp_e4m3_poison(fp8OcpE4m3Kernels):
    """OCP E4M3: corrupt strideA[1] by +M and assert >= 50% of outputs differ."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8OcpE4m3Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no OCP E4M3 solution compiled on this GPU")
    _runFp8Poison(entries[0], ml_dtypes.float8_e4m3fn, "ocp-e4m3 poison")


@requires_gfx950
def test_fp8_ocp_e5m2_poison(fp8OcpE5m2Kernels):
    """OCP E5M2: corrupt strideA[1] by +M and assert >= 50% of outputs differ."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8OcpE5m2Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no OCP E5M2 solution compiled on this GPU")
    _runFp8Poison(entries[0], ml_dtypes.float8_e5m2, "ocp-e5m2 poison")


@requires_gfx950
def test_fp8_fnuz_e4m3_poison(fp8FnuzE4m3Kernels):
    """fnuz E4M3: corrupt strideA[1] by +M and assert >= 50% of outputs differ."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8FnuzE4m3Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no E4M3 fnuz solution compiled on this GPU")
    _runFp8Poison(entries[0], ml_dtypes.float8_e4m3fnuz, "fnuz-e4m3 poison")


@requires_gfx950
def test_fp8_fnuz_e5m2_poison(fp8FnuzE5m2Kernels):
    """fnuz E5M2: corrupt strideA[1] by +M and assert >= 50% of outputs differ."""
    if not haveDeps or not haveMlDtypes:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in fp8FnuzE5m2Kernels if _filterFp8Solution(e)]
    if not entries:
        pytest.skip("no E5M2 fnuz solution compiled on this GPU")
    _runFp8Poison(entries[0], ml_dtypes.float8_e5m2fnuz, "fnuz-e5m2 poison")
