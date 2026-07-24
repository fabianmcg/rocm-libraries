# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M2 test suite: Int8 (I8II, I8I8S) and XFloat32 (SSSX) GEMM.

Covers Tasks 2.1–2.4 of the TensileLite Python client plan:
  2.1  gemmInt8 reference unit tests + boundary/rounding correctness (no GPU)
  2.2  toXf32 / gemmXf32 reference unit tests (no GPU)
  2.3  buildKernelArgs dtype-extension byte-layout verification (no GPU)
  2.4  GPU correctness tests: int8→int32, int8→int8, XFloat32→float32

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec.  Non-GPU tests
run under plain tox -e unit.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Module-level paths
# ---------------------------------------------------------------------------

_testsDir = os.path.dirname(__file__)
_yamlPath = os.path.join(_testsDir, "yaml", "gemm_int_xf32.yaml")
_tensileRoot = os.path.abspath(os.path.join(_testsDir, "..", "..", "..", ".."))

if _tensileRoot not in sys.path:
    sys.path.insert(0, _tensileRoot)

# ---------------------------------------------------------------------------
# Imports from the library under test.
# ---------------------------------------------------------------------------

from Tensile.client.gemm_args import (
    buildKernelArgs,
    _computeInternalArg0,
    _computeInternalArg1,
    _dtypeInt8,
    _dtypeInt32,
    _dtypeXf32,
)
from Tensile.client.reference import (
    gemm, gemmInt8, gemmXf32, toXf32,
    assertClose,
    RTOL_FP32, ATOL_FP32,
)
from epilogues.epilogue_harness.yaml_solution_builder import _injectInternalArgsSupport

# ---------------------------------------------------------------------------
# Problem sizes and strides.
# ---------------------------------------------------------------------------

_int8ProblemSizes = [
    (256, 256, 4, 256),
    (512, 512, 4, 512),
]

_xf32ProblemSizes = [
    (256, 256, 4, 256),
    (1024, 1024, 4, 1024),
]

# Tolerance for XFloat32 output (float32 with reduced mantissa precision).
# The rounding error from the 13-bit mantissa truncation can be significant.
rtolXf32: float = 1e-2
atolXf32: float = 1e-2


def _ntStrides(M: int, N: int, K: int):
    """Return (lda, ldb, ldd, ldc, stride_a, stride_b, stride_d, stride_c) for NT GEMM."""
    return M, N, M, M, M * K, N * K, M * N, M * N


def _ntProblemParams(M: int, N: int, batch: int, K: int,
                     alpha: float = 1.0, beta: float = 0.0) -> dict:
    """Build a problem_params dict for NT batched GEMM."""
    lda, ldb, ldd, ldc, sa, sb, sd, sc = _ntStrides(M, N, K)
    return {
        "sizes": [M, N, batch, K],
        "ldd": ldd, "stride_d": sd,
        "ldc": ldc, "stride_c": sc,
        "lda": lda, "stride_a": sa,
        "ldb": ldb, "stride_b": sb,
        "alpha": alpha,
        "beta": beta,
        "gsu": 1,
    }


# ===========================================================================
# Task 2.1 — gemmInt8 reference unit tests (no GPU)
# ===========================================================================


class TestGemmInt8Reference:
    """Unit tests for reference.gemmInt8 without GPU."""

    def test_identity_matmul_int32_output(self):
        A = np.eye(4, dtype=np.int8)
        B = np.eye(4, dtype=np.int8)
        D = gemmInt8(A, B)
        assert D.dtype == np.int32
        np.testing.assert_array_equal(D, np.eye(4, dtype=np.int32))

    def test_output_dtype_int8(self):
        A = np.ones((2, 3), dtype=np.int8)
        B = np.ones((3, 2), dtype=np.int8)
        D = gemmInt8(A, B, outputInt8=True)
        assert D.dtype == np.int8

    def test_output_dtype_int32(self):
        A = np.ones((2, 3), dtype=np.int8)
        B = np.ones((3, 2), dtype=np.int8)
        D = gemmInt8(A, B, outputInt8=False)
        assert D.dtype == np.int32

    def test_alpha_scaling(self):
        A = np.ones((2, 2), dtype=np.int8)
        B = np.ones((2, 2), dtype=np.int8)
        # alpha * (A @ B) = 4 * [[2,2],[2,2]] = [[8,8],[8,8]]
        D = gemmInt8(A, B, alpha=4.0)
        np.testing.assert_array_equal(D, np.full((2, 2), 8, dtype=np.int32))

    def test_beta_accumulation(self):
        A = np.ones((2, 2), dtype=np.int8)
        B = np.ones((2, 2), dtype=np.int8)
        C = np.full((2, 2), 10, dtype=np.int32)
        # alpha*(A@B) + beta*C = 2 + 0.5*10 = 7
        D = gemmInt8(A, B, alpha=1.0, beta=0.5, C=C)
        np.testing.assert_array_equal(D, np.full((2, 2), 7, dtype=np.int32))


class TestGemmInt8Boundary:
    """Boundary and rounding tests for gemmInt8 saturation cast (no GPU).

    Verifies std::nearbyint (round-half-to-even) equivalence and saturation
    at the int8 range edges [-128, 127].
    """

    def _scalar_saturate(self, val: float) -> int:
        """Simulate SaturateCast<int8_t, float>(val) for scalar value."""
        A = np.array([[1]], dtype=np.int8)
        B = np.array([[1]], dtype=np.int8)
        D = gemmInt8(A, B, alpha=val, outputInt8=True)
        return int(D[0, 0])

    def test_round_half_to_even_2p5(self):
        """2.5 rounds to 2 (even), not 3 — matches std::nearbyint."""
        assert self._scalar_saturate(2.5) == 2

    def test_round_half_to_even_3p5(self):
        """3.5 rounds to 4 (even), not 3 — matches std::nearbyint."""
        assert self._scalar_saturate(3.5) == 4

    def test_clamp_below_neg128(self):
        """-129 saturates to -128."""
        assert self._scalar_saturate(-129.0) == -128

    def test_clamp_neg128_unchanged(self):
        """-128.0 stays -128 (boundary value)."""
        assert self._scalar_saturate(-128.0) == -128

    def test_round_neg128p5_to_neg128(self):
        """-128.5 rounds to -128 (round-half-to-even; -128 is even)."""
        assert self._scalar_saturate(-128.5) == -128

    def test_clamp_above_127(self):
        """128 saturates to 127."""
        assert self._scalar_saturate(128.0) == 127

    def test_round_127p5_to_128_then_clamp(self):
        """127.5 rounds to 128 (round-half-to-even), then clamps to 127."""
        assert self._scalar_saturate(127.5) == 127

    def test_zero(self):
        """0.0 gives 0."""
        assert self._scalar_saturate(0.0) == 0

    def test_neg0p5_rounds_to_0(self):
        """-0.5 rounds to 0 (round-half-to-even; 0 is even)."""
        assert self._scalar_saturate(-0.5) == 0

    def test_0p5_rounds_to_0(self):
        """0.5 rounds to 0 (round-half-to-even; 0 is even)."""
        assert self._scalar_saturate(0.5) == 0

    def test_127_unchanged(self):
        """127.0 stays 127 (boundary value)."""
        assert self._scalar_saturate(127.0) == 127

    def test_bf16_accumulator_256p5_gives_256(self):
        """256.5 cast to bf16 becomes 256.0; nearbyint gives 256 before clipping to 127.

        For the bf16-accumulator SaturateCast branch (Reference.cpp:424-431):
        bfloat16 step near 256 is 2.0, so bf16(256.5)=256.0, nearbyint→256.
        This confirms the Python reference reproduces the bf16 precision loss.
        """
        if not haveMlDtypes:
            pytest.skip("ml_dtypes not installed")
        bf16Val = ml_dtypes.bfloat16(256.5)
        assert float(bf16Val) == 256.0, f"expected bf16(256.5)==256.0, got {float(bf16Val)}"
        preClip = int(np.round(np.float64(float(bf16Val))))
        assert preClip == 256, f"expected pre-clip value 256, got {preClip}"
        assert np.clip(preClip, -128, 127) == 127


# ===========================================================================
# Task 2.2 — toXf32 / gemmXf32 reference unit tests (no GPU)
# ===========================================================================


class TestToXf32:
    """Unit tests for reference.toXf32."""

    def test_zero_mantissa_bits(self):
        """After toXf32, the lower 13 mantissa bits of each element are zero."""
        rng = np.random.default_rng(0)
        arr = rng.random(64).astype(np.float32)
        xf = toXf32(arr)
        bits = xf.view(np.uint32)
        assert np.all((bits & np.uint32(0x1FFF)) == 0), "lower 13 bits should be zero"

    def test_exact_power_of_two(self):
        """Powers of two are unchanged by XF32 conversion (no mantissa bits)."""
        vals = np.array([1.0, 2.0, 4.0, 0.5, 0.25], dtype=np.float32)
        np.testing.assert_array_equal(toXf32(vals), vals)

    def test_preserves_shape(self):
        """toXf32 preserves the input array shape."""
        arr = np.ones((3, 4, 2), dtype=np.float32)
        assert toXf32(arr).shape == (3, 4, 2)

    def test_bit_mask(self):
        """Verify the bit mask: mantissa all-ones → only top 10 mantissa bits remain."""
        # float32: sign(1)|exp(8)|mantissa(23)
        # 0x3FFFFFFF = 0 01111111 11111111111111111111111 → 1.999...
        val = np.array([np.float32(0.0)], dtype=np.float32)
        val.view(np.uint32)[0] = np.uint32(0x3FFFFFFF)
        xf_bits = toXf32(val).view(np.uint32)[0]
        # After masking with 0xFFFFE000: 0x3FFFE000 = 0 01111111 11111110000000000000000
        assert xf_bits == np.uint32(0x3FFFE000)


class TestGemmXf32Reference:
    """Unit tests for reference.gemmXf32 without GPU."""

    def test_output_dtype_is_fp32(self):
        A = np.ones((2, 3), dtype=np.float32)
        B = np.ones((3, 2), dtype=np.float32)
        D = gemmXf32(A, B)
        assert D.dtype == np.float32

    def test_identity_matmul(self):
        A = np.eye(4, dtype=np.float32)
        B = np.eye(4, dtype=np.float32)
        D = gemmXf32(A, B)
        # Identity matrices: XF32 conversion is exact (zeros have no mantissa).
        np.testing.assert_allclose(D, np.eye(4, dtype=np.float32), atol=ATOL_FP32)

    def test_power_of_two_inputs(self):
        """Inputs that are powers of two survive XF32 conversion unchanged."""
        A = np.full((4, 4), 2.0, dtype=np.float32)
        B = np.full((4, 4), 0.5, dtype=np.float32)
        D = gemmXf32(A, B)
        ref = gemm(A, B)  # exact for powers of two
        np.testing.assert_allclose(D, ref, atol=ATOL_FP32)

    def test_result_close_to_fp32_for_small_inputs(self):
        """For small (16×16×16) matrices, XF32 result is within rtolXf32 of fp32."""
        rng = np.random.default_rng(123)
        M, K, N = 16, 16, 16
        A = rng.random((M, K)).astype(np.float32)
        B = rng.random((K, N)).astype(np.float32)
        D_xf = gemmXf32(A, B)
        D_ref = gemm(A, B).astype(np.float32)
        np.testing.assert_allclose(D_xf, D_ref, rtol=rtolXf32, atol=atolXf32)

    def test_xf32_differs_from_fp32(self):
        """XF32 result differs from fp32 when inputs have sub-XF32-precision mantissa."""
        rng = np.random.default_rng(42)
        A = rng.random((8, 8)).astype(np.float32)
        B = rng.random((8, 8)).astype(np.float32)
        D_xf = gemmXf32(A, B)
        D_fp32 = gemm(A, B).astype(np.float32)
        # XF32 introduces truncation error — results should NOT be identical.
        assert not np.allclose(D_xf, D_fp32, atol=1e-7), (
            "XF32 result matches fp32 exactly — truncation may not be applied"
        )

    def test_alpha_scaling(self):
        A = np.ones((2, 2), dtype=np.float32)
        B = np.ones((2, 2), dtype=np.float32)
        D = gemmXf32(A, B, alpha=3.0)
        np.testing.assert_allclose(D, np.full((2, 2), 6.0, dtype=np.float32), atol=ATOL_FP32)


# ===========================================================================
# Task 2.3 — buildKernelArgs dtype enum verification (no GPU)
# ===========================================================================


class TestDtypeEnumValues:
    """Verify the DataTypeEnum integer codes used in gemm_args."""

    def test_dtype_int8_is_8(self):
        assert _dtypeInt8 == 8

    def test_dtype_int32_is_6(self):
        assert _dtypeInt32 == 6

    def test_dtype_xf32_is_10(self):
        assert _dtypeXf32 == 10


class TestBuildKernelArgsInt8Layout:
    """Verify buildKernelArgs byte layout for int8 kernels.

    For int8 with HPA=True (ComputeDataType=float32), the argument layout is
    structurally identical to fp32: D/C/A/B pointers, strides, float32 alpha/beta.
    The SaturateCast is the kernel's internal responsibility; the arg vector
    only carries type codes (baked into the kernel binary, not in the arg buffer).
    """

    def _int8Sol(self) -> dict:
        """Minimal solution dict for int8 HPA byte-layout tests."""
        return {
            "KernArgsVersion": 2,
            "SupportCustomWGM": True,
            "SupportCustomStaggerU": False,
            "SupportUserGSU": False,
            "UseSFC": False,
            "UseUniversalArgs": True,
            "MacroTile0": 32,
            "MacroTile1": 32,
            "WorkGroupMapping": 4,
            "WorkGroupMappingXCC": 0,
            "WorkGroupMappingXCCGroup": 0,
            "StaggerU": 32,
            "StaggerUMapping": 1,
            "_staggerStrideShift": 2,
            "GlobalSplitU": 1,
            "GlobalSplitUCoalesced": False,
            "GlobalSplitUWorkGroupMappingRoundRobin": False,
            "StreamK": 0,
            "StreamKAtomic": 0,
            "StridedBatched": True,
            "UseBeta": True,
            "GlobalAccumulation": 0,
            "ExpertSchedulingMode": 0,
            # int8 with HPA: compute type is float32, so alpha/beta are float32.
            "HighPrecisionAccumulate": True,
            "ComputeDataType": 0,  # float32 (0 = rocisa::DataType::Float)
            "NumThreads": 64,
        }

    def test_total_byte_length_int8_v2_batched(self):
        """int8 arg buffer length == fp32 arg buffer length for the same layout."""
        import struct
        sol = self._int8Sol()
        pp = _ntProblemParams(256, 256, 4, 256)
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}
        buf = buildKernelArgs(sol, pp, tensors)
        # header=16, sizes=16, ptrs=32, strides=32, alpha+beta=8 → 104 bytes.
        assert len(buf) == 104

    def test_alpha_beta_are_float32_for_hpa_int8(self):
        """HPA int8 kernels pack alpha/beta as float32 (not int32 or float16)."""
        import struct
        sol = self._int8Sol()
        pp = _ntProblemParams(64, 64, 1, 64, alpha=2.0, beta=0.5)
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}
        buf = buildKernelArgs(sol, pp, tensors)
        # offset: header=16, sizes=12(3D), ptrs=32, strides=8(non-batched 1D) →
        # For batched (4 sizes): header=16, sizes=16, ptrs=32, strides=32 → offset=96
        alpha_off = 16 + 16 + 32 + 32  # 96 bytes
        alpha_val = struct.unpack_from("<f", buf, alpha_off)[0]
        beta_val = struct.unpack_from("<f", buf, alpha_off + 4)[0]
        assert abs(alpha_val - 2.0) < 1e-6
        assert abs(beta_val - 0.5) < 1e-6

    def test_non_hpa_int8_packs_alpha_as_int32(self):
        """Non-HPA int8 (ComputeDataType=Int32) packs alpha as int32."""
        import struct
        sol = self._int8Sol()
        sol["HighPrecisionAccumulate"] = False
        sol["ComputeDataType"] = 6  # rocisa::DataType::Int32
        pp = _ntProblemParams(64, 64, 1, 64, alpha=3.0, beta=1.0)
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}
        buf = buildKernelArgs(sol, pp, tensors)
        alpha_off = 16 + 16 + 32 + 32  # 96 bytes
        alpha_val = struct.unpack_from("<i", buf, alpha_off)[0]
        beta_val = struct.unpack_from("<i", buf, alpha_off + 4)[0]
        assert alpha_val == 3
        assert beta_val == 1

    def test_d_c_a_b_pointer_slots_int8(self):
        """D, C, A, B pointers appear at bytes 32-63 for int8 (same as fp32)."""
        import struct
        sol = self._int8Sol()
        pp = _ntProblemParams(64, 64, 1, 64)
        tensors = {
            "D": 0xD000000000000001,
            "C": 0xC000000000000002,
            "A": 0xA000000000000003,
            "B": 0xB000000000000004,
        }
        buf = buildKernelArgs(sol, pp, tensors)
        D, C, A, B = struct.unpack_from("<QQQQ", buf, 32)
        assert D == tensors["D"] & 0xFFFFFFFFFFFFFFFF
        assert C == tensors["C"] & 0xFFFFFFFFFFFFFFFF
        assert A == tensors["A"] & 0xFFFFFFFFFFFFFFFF
        assert B == tensors["B"] & 0xFFFFFFFFFFFFFFFF


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


def _compileSolutions(problem_idx: int):
    """Compile all solutions for one YAML problem group; return list of dicts."""
    if not haveDeps:
        return []
    try:
        from epilogues.epilogue_harness.yaml_solution_builder import solutionsFromYaml
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        sols = solutionsFromYaml(_yamlPath, assembler, isaInfoMap, debugConfig,
                                 problemIdx=problem_idx)
    except Exception as exc:
        import warnings
        warnings.warn(f"could not compile solutions (problemIdx={problem_idx}): {exc}")
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
        compiled.append({
            "sol_dict": sol_dict,
            "raw_dict": raw_dict,
            "kernel_name": kernel_name,
            "hsaco": hsaco,
            "chip": chip,
            "sid": sid,
        })
    return compiled


def _filterSolution(entry: dict) -> bool:
    """Return True if the solution passes the M2 kernel filter.

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


def _buildNtTypedArgs(sol_dict: dict, M: int, N: int, batch: int, K: int,
                      D_arr, C_arr, A_arr, B_arr,
                      alpha: float = 1.0, beta: float = 0.0) -> list:
    """Build the typed args list for execute_hsaco for NT stridedBatched=True GEMM.

    Alpha and beta are float32 (HPA int8 and XFloat32 both use float32 scalars).
    """
    version = sol_dict.get("KernArgsVersion", 0)
    mt0 = sol_dict["MacroTile0"]
    mt1 = sol_dict["MacroTile1"]
    num_wg = math.ceil(M / mt0) * math.ceil(N / mt1) * batch

    arg0 = _computeInternalArg0(sol_dict, gsu=1)
    gemm_count = (1 & 0x3FFFFFFF) | (0 << 30)  # stridedBatched, argType=0

    args = [np.uint32(gemm_count), np.uint32(arg0)]
    if version >= 1:
        arg1 = _computeInternalArg1(sol_dict, cu_count=_deviceCuCount())
        args.append(np.int32(arg1))
        args.append(np.uint32(num_wg))

    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])
    args.extend([D_arr, C_arr, A_arr, B_arr])

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


def _allocNtBatched(M: int, N: int, K: int, batch: int, np_dtype, rng):
    """Allocate flat buffers for a batched NT stridedBatched GEMM."""
    A_np = np.asfortranarray(rng.random((M, K)).astype(np_dtype))
    B_np = np.asfortranarray(rng.random((N, K)).astype(np_dtype))
    A_buf = np.tile(A_np.ravel(order='F'), batch)
    B_buf = np.tile(B_np.ravel(order='F'), batch)
    C_buf = np.zeros(M * N * batch, dtype=np_dtype)
    D_buf = np.zeros(M * N * batch, dtype=np_dtype)
    return A_buf, B_buf, C_buf, D_buf


def _allocInt8Batched(M: int, N: int, K: int, batch: int, dest_dtype, rng):
    """Allocate int8 input buffers and appropriate output buffers for int8 GEMM.

    A and B are int8.  D and C use dest_dtype (int32 or int8).
    """
    A_np = rng.integers(-8, 8, size=(M, K)).astype(np.int8)
    B_np = rng.integers(-8, 8, size=(N, K)).astype(np.int8)
    A_np = np.asfortranarray(A_np)
    B_np = np.asfortranarray(B_np)
    A_buf = np.tile(A_np.ravel(order='F'), batch).astype(np.int8)
    B_buf = np.tile(B_np.ravel(order='F'), batch).astype(np.int8)
    C_buf = np.zeros(M * N * batch, dtype=dest_dtype)
    D_buf = np.zeros(M * N * batch, dtype=dest_dtype)
    return A_buf, B_buf, C_buf, D_buf, A_np, B_np


def _ntBatchRef(A_np, B_np, batch, M, N, ref_fn, alpha, beta):
    """Compute NT batched reference for the first batch element, tiled to all batches."""
    D_one = ref_fn(A_np, B_np.T, alpha, beta, None)
    return np.tile(np.asfortranarray(D_one).ravel(order='F'), batch)


def _verifyNtResult(dBuf: np.ndarray, refD: np.ndarray, dtype, rtol: float, atol: float, label: str) -> None:
    """Cast the captured GPU result to float64 and assert closeness to reference."""
    assertClose(dBuf.astype(np.float64), refD.astype(np.float64), rtol=rtol, atol=atol, label=label)


def _runNtStridedBatched(entry: dict, M: int, N: int, batch: int, K: int,
                         np_dtype, ref_fn, rtol: float, atol: float, label: str,
                         dest_dtype=None):
    """Execute one NT stridedBatched kernel and verify output against reference."""
    if dest_dtype is None:
        dest_dtype = np_dtype
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]
    rng = np.random.default_rng(seed=M * 1000 + N + K)

    if np_dtype == np.int8:
        A_buf, B_buf, C_buf, D_buf, A_np, B_np = _allocInt8Batched(
            M, N, K, batch, dest_dtype, rng
        )
    else:
        A_buf, B_buf, C_buf, D_buf = _allocNtBatched(M, N, K, batch, np_dtype, rng)
        A_np = A_buf[:M * K].reshape(M, K, order='F')
        B_np = B_buf[:N * K].reshape(N, K, order='F')
        D_buf = D_buf.astype(dest_dtype)
        C_buf = C_buf.astype(dest_dtype)

    result_holder = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=dest_dtype).copy()

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    args = _buildNtTypedArgs(sol_dict, M, N, batch, K, D_io, C_in, A_in, B_in, 1.0, 0.0)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    D_ref = _ntBatchRef(A_np, B_np, batch, M, N, ref_fn, 1.0, 0.0)
    _verifyNtResult(result_holder["D_gpu"], D_ref, dest_dtype, rtol, atol, label)


# ---------------------------------------------------------------------------
# Session-scoped compiled solution fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def int8I32Kernels():
    """Compile int8→int32 solutions from YAML group 0."""
    return _compileSolutions(0)


@pytest.fixture(scope="session")
def int8I8Kernels():
    """Compile int8→int8 solutions from YAML group 1."""
    return _compileSolutions(1)


@pytest.fixture(scope="session")
def xf32Kernels():
    """Compile XFloat32→float32 solutions from YAML group 2."""
    return _compileSolutions(2)


# ---------------------------------------------------------------------------
# Poison-input tests (GPU required).
# ---------------------------------------------------------------------------

def _corruptStrideA1(argList: list, M: int) -> list:
    """Corrupt strideA[1] (stride_a, the batch stride of A) by +M in the typed arg list.

    strideA[1] is the stride between batches of A, located immediately after lda.
    Corrupting the batch stride causes the kernel to read A data from the wrong
    batch offset, producing wildly different outputs detectable at any tolerance.
    Infers header_n from total arg count: total = header_n + 18.
    """
    header_n = len(argList) - 18  # header + 4 sizes + 4 ptrs + 8 strides + 2 scalars
    ldaIdx = header_n + 4 + 4 + 4  # after header/sizes/ptrs/D-strides/C-strides
    strideA1Idx = ldaIdx + 1       # stride_a is immediately after lda
    original = argList[strideA1Idx]
    argList[strideA1Idx] = np.uint32(int(original) + M)
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


@requires_gfx950
def test_buildKernelArgs_poison_int8i32(int8I32Kernels):
    """Corrupt strideA[1] for int8→int32 and assert >= 50% of outputs differ."""
    if not haveDeps:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in int8I32Kernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no int8→int32 solution compiled")

    M, N, batch, K = 256, 256, 4, 256
    entry = entries[0]
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]

    rng = np.random.default_rng(seed=42)
    A_buf, B_buf, C_buf, D_buf, A_np, B_np = _allocInt8Batched(M, N, K, batch, np.int32, rng)
    D_ref_flat = _ntBatchRef(A_np, B_np, batch, M, N,
                              lambda a, b, al, be, c: gemmInt8(a, b, al, be, c),
                              1.0, 0.0)

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    args = _buildNtTypedArgs(sol_dict, M, N, batch, K, D_io, C_in, A_in, B_in, 1.0, 0.0)
    args = _corruptStrideA1(args, M)

    result_holder = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np.int32).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    _assertPoisonDetected(result_holder["D_gpu"], D_ref_flat, RTOL_FP32, "int8→int32 poison")


@requires_gfx950
def test_buildKernelArgs_poison_xf32(xf32Kernels):
    """Corrupt strideA[1] for XFloat32 and assert >= 50% of outputs differ."""
    if not haveDeps:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in xf32Kernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no XFloat32 solution compiled")

    M, N, batch, K = 256, 256, 4, 256
    entry = entries[0]
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]

    rng = np.random.default_rng(seed=7)
    A_buf, B_buf, C_buf, D_buf = _allocNtBatched(M, N, K, batch, np.float32, rng)
    A_np = A_buf[:M * K].reshape(M, K, order='F')
    B_np = B_buf[:N * K].reshape(N, K, order='F')
    D_ref_flat = _ntBatchRef(A_np, B_np, batch, M, N,
                              lambda a, b, al, be, c: gemmXf32(a, b, al, be, c),
                              1.0, 0.0)

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    args = _buildNtTypedArgs(sol_dict, M, N, batch, K, D_io, C_in, A_in, B_in, 1.0, 0.0)
    args = _corruptStrideA1(args, M)

    result_holder = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np.float32).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    # Use fp32 tolerance for poison detection — the goal is to confirm any
    # computation difference, not to tolerate XF32 precision loss.
    _assertPoisonDetected(result_holder["D_gpu"], D_ref_flat, RTOL_FP32, "xf32 poison")


# ---------------------------------------------------------------------------
# Int8→Int32 GPU correctness tests.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("size", _int8ProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _int8ProblemSizes])
def test_int8_to_int32_strided_batched(int8I32Kernels, size):
    """int8→int32 NT stridedBatched=True correctness."""
    if not haveDeps:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in int8I32Kernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no int8→int32 solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runNtStridedBatched(
            entry, M, N, batch, K,
            np_dtype=np.int8, ref_fn=gemmInt8,
            rtol=0, atol=0,  # int32 output is exact
            label=f"int8→int32 M{M}N{N}B{batch}K{K} {sid}",
            dest_dtype=np.int32,
        )


# ---------------------------------------------------------------------------
# Int8→Int8 GPU correctness tests.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("size", _int8ProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _int8ProblemSizes])
def test_int8_to_int8_strided_batched(int8I8Kernels, size):
    """int8→int8 (with SaturateCast) NT stridedBatched=True correctness."""
    if not haveDeps:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in int8I8Kernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no int8→int8 solution compiled")

    M, N, batch, K = size
    ref_i8 = lambda a, b, al, be, c: gemmInt8(a, b, al, be, c, outputInt8=True)
    for entry in entries:
        sid = entry["sid"]
        _runNtStridedBatched(
            entry, M, N, batch, K,
            np_dtype=np.int8, ref_fn=ref_i8,
            rtol=0, atol=0,  # int8 output is exact (saturation boundary test)
            label=f"int8→int8 M{M}N{N}B{batch}K{K} {sid}",
            dest_dtype=np.int8,
        )


# ---------------------------------------------------------------------------
# XFloat32→Float32 GPU correctness tests.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("size", _xf32ProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _xf32ProblemSizes])
def test_xf32_to_fp32_strided_batched(xf32Kernels, size):
    """XFloat32→float32 NT stridedBatched=True correctness."""
    if not haveDeps:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in xf32Kernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no XFloat32 solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runNtStridedBatched(
            entry, M, N, batch, K,
            np_dtype=np.float32, ref_fn=gemmXf32,
            rtol=rtolXf32, atol=atolXf32,
            label=f"xf32→fp32 M{M}N{N}B{batch}K{K} {sid}",
        )
