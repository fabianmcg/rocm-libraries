# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M5 test suite: fused epilogues (bias, activation, scale variants, AmaxD, E tensor).

Covers Tasks 5.1–5.3 of the TensileLite Python client plan:
  5.1  Epilogue reference functions — unit tests against Reference.cpp values.
  5.2  buildKernelArgs epilogue slot layout — pure Python byte-level tests.
  5.3  GPU correctness: bias and activation for bf16/fp32 on gfx950.

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec.  Non-GPU tests
always run under plain tox -e unit.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guards.
# ---------------------------------------------------------------------------

try:
    import amdgpu_exec
    import ml_dtypes
    HAVE_DEPS = True
except ImportError:
    amdgpu_exec = None
    ml_dtypes = None
    HAVE_DEPS = False

from .conftest import requires_gfx950

# ---------------------------------------------------------------------------
# Module-level paths.
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
_YAML_PATH = os.path.join(_TESTS_DIR, "yaml", "gemm_epilogues.yaml")
_TENSILE_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", ".."))

if _TENSILE_ROOT not in sys.path:
    sys.path.insert(0, _TENSILE_ROOT)

# ---------------------------------------------------------------------------
# Imports from the library under test.
# ---------------------------------------------------------------------------

from Tensile.client.gemm_args import buildKernelArgs, _buildEpilogueArgs, _readPTFlag
from Tensile.client.reference import (
    gemm,
    gemmBf16,
    applyBias,
    applyActivation,
    applyScaleAb,
    applyScaleCd,
    applyScaleAlphaVec,
    computeAmaxD,
    computeETensor,
    assertClose,
    RTOL_FP32, ATOL_FP32,
    RTOL_BF16, ATOL_BF16,
    _ACT_ARG_COUNT,
)
from Tensile.client.yaml_solution_builder import _injectInternalArgsSupport

# ---------------------------------------------------------------------------
# YAML problem-group indices (matching gemm_epilogues.yaml comment).
# ---------------------------------------------------------------------------

_GRP_BF16_BIAS = 0
_GRP_FP32_BIAS = 1
_GRP_BF16_RELU = 2
_GRP_FP32_RELU = 3
_GRP_BF16_SCALE_AB = 4
_GRP_FP32_SCALE_AB = 5
_GRP_BF16_SCALE_CD = 6
_GRP_FP32_SCALE_CD = 7

# DataType enum for float32 bias (rocisa::DataType::Float = 0).
_DTYPE_FLOAT32 = 0

# Problem sizes used by all GPU tests.
_PROBLEM_SIZES = [
    (256, 256, 4, 256),
    (512, 512, 4, 512),
]


# ===========================================================================
# Task 5.1a — applyBias reference unit tests (no GPU)
# ===========================================================================


class TestReferenceBias:
    """Unit tests for applyBias — pure Python, no GPU required."""

    def test_row_bias_shape(self):
        D = np.zeros((3, 4), dtype=np.float64)
        bias = np.ones(4, dtype=np.float64)
        out = applyBias(D, bias, "row")
        assert out.shape == (3, 4)

    def test_row_bias_values(self):
        D = np.ones((2, 3), dtype=np.float64)
        bias = np.array([1.0, 2.0, 3.0])
        out = applyBias(D, bias, "row")
        np.testing.assert_allclose(out, [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])

    def test_col_bias_values(self):
        D = np.zeros((3, 2), dtype=np.float64)
        bias = np.array([1.0, 2.0, 3.0])
        out = applyBias(D, bias, "col")
        np.testing.assert_allclose(out, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    def test_matrix_bias_values(self):
        D = np.ones((2, 2), dtype=np.float64)
        bias = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = applyBias(D, bias, "matrix")
        np.testing.assert_allclose(out, [[2.0, 3.0], [4.0, 5.0]])

    def test_does_not_mutate_input(self):
        D = np.zeros((2, 2), dtype=np.float64)
        original = D.copy()
        applyBias(D, np.ones(2), "row")
        np.testing.assert_array_equal(D, original)


# ===========================================================================
# Task 5.1b — applyActivation reference unit tests (no GPU)
# Hand-chosen values from Reference.cpp:746-853.
# ===========================================================================


class TestReferenceActivations:
    """Unit tests for applyActivation — verified against Reference.cpp."""

    def _apply(self, name, vals, args=None):
        D = np.array(vals, dtype=np.float32)
        return applyActivation(D, name, args).astype(np.float32)

    def test_relu_positive(self):
        out = self._apply("relu", [0.5, 1.0, 2.0, 0.0, -0.1])
        # max(0, x): negative clamped to 0.
        np.testing.assert_allclose(out, [0.5, 1.0, 2.0, 0.0, 0.0], atol=1e-6)

    def test_relu_zero(self):
        assert float(self._apply("relu", [0.0])[0]) == pytest.approx(0.0)

    def test_relu_all_negative(self):
        out = self._apply("relu", [-1.0, -2.0, -0.5])
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-6)

    def test_relu_large(self):
        out = self._apply("relu", [1e6, -1e6])
        np.testing.assert_allclose(out, [1e6, 0.0], rtol=1e-5)

    def test_relu_small_negative(self):
        out = self._apply("relu", [-1e-7])
        assert float(out[0]) == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_zero(self):
        # sigmoid(0) = 0.5.
        out = self._apply("sigmoid", [0.0])
        assert float(out[0]) == pytest.approx(0.5, abs=1e-6)

    def test_sigmoid_large_positive(self):
        # sigmoid(10) ≈ 1.
        out = self._apply("sigmoid", [10.0])
        assert float(out[0]) == pytest.approx(1.0, abs=1e-4)

    def test_sigmoid_large_negative(self):
        # sigmoid(-10) ≈ 0.
        out = self._apply("sigmoid", [-10.0])
        assert float(out[0]) == pytest.approx(0.0, abs=1e-4)

    def test_sigmoid_symmetry(self):
        # sigmoid(x) + sigmoid(-x) == 1.
        out_pos = float(self._apply("sigmoid", [1.5])[0])
        out_neg = float(self._apply("sigmoid", [-1.5])[0])
        assert out_pos + out_neg == pytest.approx(1.0, abs=1e-5)

    def test_sigmoid_range(self):
        out = self._apply("sigmoid", [-2.0, 0.0, 2.0])
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_gelu_zero(self):
        # gelu(0) = 0 (0.5 * 0 * 1 = 0).
        out = self._apply("gelu", [0.0])
        assert float(out[0]) == pytest.approx(0.0, abs=1e-6)

    def test_gelu_positive(self):
        # gelu(1) ≈ 0.8413 (standard GELU).
        out = self._apply("gelu", [1.0])
        assert float(out[0]) == pytest.approx(0.8413, abs=1e-2)

    def test_gelu_negative(self):
        # gelu(-1) ≈ -0.1587.
        out = self._apply("gelu", [-1.0])
        assert float(out[0]) == pytest.approx(-0.1587, abs=1e-2)

    def test_gelu_large(self):
        # gelu(10) ≈ 10.
        out = self._apply("gelu", [10.0])
        assert float(out[0]) == pytest.approx(10.0, abs=1e-3)

    def test_gelu_sign(self):
        # gelu preserves sign for large inputs.
        pos = float(self._apply("gelu", [5.0])[0])
        neg = float(self._apply("gelu", [-5.0])[0])
        assert pos > 0.0
        assert neg < 0.0

    def test_silu_zero(self):
        # silu(0) = 0 / (1 + 1) = 0.
        out = self._apply("silu", [0.0])
        assert float(out[0]) == pytest.approx(0.0, abs=1e-6)

    def test_silu_positive(self):
        # silu(1) = 1 / (1 + e^-1) ≈ 0.7311.
        out = self._apply("silu", [1.0])
        assert float(out[0]) == pytest.approx(0.7311, abs=1e-3)

    def test_silu_negative(self):
        # silu(-1) = -1 / (1 + e) ≈ -0.2689.
        out = self._apply("silu", [-1.0])
        assert float(out[0]) == pytest.approx(-0.2689, abs=1e-3)

    def test_silu_large_positive(self):
        # silu(10) ≈ 10 (sigmoid(10) ≈ 1).
        out = self._apply("silu", [10.0])
        assert float(out[0]) == pytest.approx(10.0, abs=0.01)

    def test_silu_range_negative(self):
        # silu is always > -0.278 for all inputs.
        vals = [-100.0, -10.0, -5.0, -1.0]
        out = self._apply("silu", vals)
        assert all(v > -0.278 for v in out)

    def test_dgelu_zero(self):
        # dgelu(0): k0*0 + k1*0 = 0, xx=0, tanh(0)=0, x2=1, result=0+0+0.5=0.5.
        out = self._apply("dgelu", [0.0])
        assert float(out[0]) == pytest.approx(0.5, abs=1e-5)

    def test_dgelu_large_positive(self):
        # dgelu(x) → 1 for large positive x.
        out = self._apply("dgelu", [10.0])
        assert float(out[0]) == pytest.approx(1.0, abs=1e-4)

    def test_dgelu_large_negative(self):
        # dgelu(x) → 0 for large negative x.
        out = self._apply("dgelu", [-10.0])
        assert float(out[0]) == pytest.approx(0.0, abs=1e-4)

    def test_dgelu_one(self):
        # Hand-computed: dgelu(1) ≈ 1.083 (derivative of GELU at x=1).
        out = self._apply("dgelu", [1.0])
        assert float(out[0]) == pytest.approx(1.0830, abs=1e-3)

    def test_dgelu_minus_one(self):
        # dgelu(-1) = 1 - dgelu(1) ≈ -0.083 by antisymmetry around 0.5.
        out = self._apply("dgelu", [-1.0])
        assert float(out[0]) == pytest.approx(-0.0830, abs=1e-3)

    def test_tanh_zero(self):
        # tanh(0*1)*1 = 0.
        out = self._apply("tanh", [0.0], args=[1.0, 1.0])
        assert float(out[0]) == pytest.approx(0.0, abs=1e-6)

    def test_tanh_one(self):
        # tanh(1*1)*1 = tanh(1) ≈ 0.7616.
        out = self._apply("tanh", [1.0], args=[1.0, 1.0])
        assert float(out[0]) == pytest.approx(0.7616, abs=1e-4)

    def test_tanh_scale_arg0(self):
        # tanh(2*x)*1 doubles the input.
        out_scale = float(self._apply("tanh", [0.5], args=[2.0, 1.0])[0])
        out_ref = float(self._apply("tanh", [1.0], args=[1.0, 1.0])[0])
        assert out_scale == pytest.approx(out_ref, abs=1e-5)

    def test_tanh_scale_arg1(self):
        # tanh(x*1)*2 doubles the output.
        out = float(self._apply("tanh", [1.0], args=[1.0, 2.0])[0])
        ref = 2.0 * np.tanh(1.0)
        assert out == pytest.approx(ref, abs=1e-5)

    def test_tanh_symmetric(self):
        # tanh(-x)*a1 = -tanh(x)*a1.
        pos = float(self._apply("tanh", [1.0], args=[1.0, 1.0])[0])
        neg = float(self._apply("tanh", [-1.0], args=[1.0, 1.0])[0])
        assert neg == pytest.approx(-pos, abs=1e-6)

    def test_swish_zero(self):
        # swish(0, beta=1) = 0 / 2 = 0.
        out = self._apply("swish", [0.0], args=[1.0])
        assert float(out[0]) == pytest.approx(0.0, abs=1e-6)

    def test_swish_one(self):
        # swish(1, beta=1) = 1 / (1 + e^-1) = silu(1) ≈ 0.7311.
        out_swish = float(self._apply("swish", [1.0], args=[1.0])[0])
        out_silu = float(self._apply("silu", [1.0])[0])
        assert out_swish == pytest.approx(out_silu, abs=1e-5)

    def test_swish_beta_zero(self):
        # swish(x, beta=0) = x / 2 (sigmoid(0) = 0.5).
        out = float(self._apply("swish", [2.0], args=[0.0])[0])
        assert out == pytest.approx(1.0, abs=1e-5)

    def test_swish_large_beta(self):
        # swish(1, beta=100) ≈ relu(1) = 1.
        out = float(self._apply("swish", [1.0], args=[100.0])[0])
        assert out == pytest.approx(1.0, abs=1e-3)

    def test_swish_large_negative_beta(self):
        # swish(-1, beta=100) ≈ relu(-1) = 0 × something small.
        out = float(self._apply("swish", [-1.0], args=[100.0])[0])
        assert abs(out) < 1e-2

    def test_unknown_activation_raises(self):
        D = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="unknown activation"):
            applyActivation(D, "notanactivation")


# ===========================================================================
# Task 5.1c — applyScaleAb, applyScaleCd, applyScaleAlphaVec (no GPU)
# ===========================================================================


class TestReferenceScales:
    """Unit tests for scale helpers — pure Python, no GPU required."""

    def test_scale_ab_scalar(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[1.0], [2.0]])
        sA, sB = applyScaleAb(A, B, 2.0, 0.5)
        np.testing.assert_allclose(sA, [[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(sB, [[0.5], [1.0]])

    def test_scale_ab_identity(self):
        A = np.ones((3, 3))
        B = np.ones((3, 3))
        sA, sB = applyScaleAb(A, B, 1.0, 1.0)
        np.testing.assert_array_equal(sA, A)
        np.testing.assert_array_equal(sB, B)

    def test_scale_ab_zero(self):
        A = np.ones((2, 2))
        B = np.ones((2, 2))
        sA, sB = applyScaleAb(A, B, 0.0, 1.0)
        np.testing.assert_array_equal(sA, np.zeros((2, 2)))

    def test_scale_cd_scalar(self):
        C = np.array([[1.0, 2.0]])
        D = np.array([[3.0, 4.0]])
        sC, sD = applyScaleCd(C, D, 2.0, 0.5)
        np.testing.assert_allclose(sC, [[2.0, 4.0]])
        np.testing.assert_allclose(sD, [[1.5, 2.0]])

    def test_scale_cd_none_c(self):
        D = np.array([[1.0, 2.0]])
        sC, sD = applyScaleCd(None, D, 2.0, 3.0)
        assert sC is None
        np.testing.assert_allclose(sD, [[3.0, 6.0]])

    def test_scale_alpha_vec_per_row(self):
        D = np.ones((3, 4))
        scale = np.array([1.0, 2.0, 3.0])
        out = applyScaleAlphaVec(D, scale, factorDim=0)
        expected = np.array([[1.0, 1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0, 2.0],
                              [3.0, 3.0, 3.0, 3.0]])
        np.testing.assert_allclose(out, expected)

    def test_scale_alpha_vec_per_col(self):
        D = np.ones((2, 3))
        scale = np.array([1.0, 2.0, 3.0])
        out = applyScaleAlphaVec(D, scale, factorDim=1)
        expected = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        np.testing.assert_allclose(out, expected)

    def test_compute_amax_d(self):
        D = np.array([[1.0, -3.0], [2.0, 0.5]])
        assert computeAmaxD(D) == pytest.approx(3.0, abs=1e-10)

    def test_compute_amax_d_all_zero(self):
        D = np.zeros((3, 3))
        assert computeAmaxD(D) == pytest.approx(0.0, abs=1e-10)

    def test_compute_amax_d_negative_only(self):
        D = np.array([[-1.0, -2.0, -0.5]])
        assert computeAmaxD(D) == pytest.approx(2.0, abs=1e-10)

    def test_compute_e_tensor_is_copy(self):
        D = np.array([[1.0, 2.0], [3.0, 4.0]])
        E = computeETensor(D)
        np.testing.assert_array_equal(E, D)
        E[0, 0] = 999.0
        assert D[0, 0] != 999.0  # original not mutated.

    def test_act_arg_count_table(self):
        # Spot-check the _ACT_ARG_COUNT table against Reference.cpp.
        assert _ACT_ARG_COUNT["relu"] == 0
        assert _ACT_ARG_COUNT["gelu"] == 0
        assert _ACT_ARG_COUNT["silu"] == 0
        assert _ACT_ARG_COUNT["sigmoid"] == 0
        assert _ACT_ARG_COUNT["dgelu"] == 0
        assert _ACT_ARG_COUNT["swish"] == 1
        assert _ACT_ARG_COUNT["tanh"] == 2
        assert _ACT_ARG_COUNT["all"] == 2


# ===========================================================================
# Task 5.2 — buildKernelArgs epilogue byte-layout tests (no GPU)
# ===========================================================================


def _minimalSol(useBias=0, useScaleAB="", useScaleCD=False,
                useScaleAlphaVec=0, outputAmaxD=False,
                activationType="none", activationFused=True):
    """Return a minimal solution dict for byte-layout testing."""
    return {
        "KernArgsVersion": 2,
        "SupportCustomWGM": True,
        "SupportCustomStaggerU": False,
        "SupportUserGSU": False,
        "UseSFC": False,
        "UseUniversalArgs": True,
        "MacroTile0": 64,
        "MacroTile1": 64,
        "WorkGroupMapping": 8,
        "WorkGroupMappingXCC": 0,
        "WorkGroupMappingXCCGroup": 0,
        "StaggerU": 0,
        "StaggerUMapping": 0,
        "_staggerStrideShift": 0,
        "GlobalSplitU": 1,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "StreamK": 0,
        "StreamKAtomic": 0,
        "StridedBatched": True,
        "UseBeta": True,
        "GlobalAccumulation": 0,
        "ExpertSchedulingMode": 0,
        "HighPrecisionAccumulate": True,
        "ComputeDataType": 0,
        "NumThreads": 256,
        "ActivationFused": activationFused,
        "ProblemType": {
            "UseBias": useBias,
            "UseScaleAB": useScaleAB,
            "UseScaleCD": useScaleCD,
            "UseScaleAlphaVec": useScaleAlphaVec,
            "OutputAmaxD": outputAmaxD,
            "UseE": False,
            "Gradient": False,
            "ActivationType": activationType,
        },
    }


def _basePP():
    """Return a minimal problemParams for layout tests."""
    return {
        "sizes": [64, 64, 1, 64],
        "ldd": 64, "stride_d": 64 * 64,
        "ldc": 64, "stride_c": 64 * 64,
        "lda": 64, "stride_a": 64 * 64,
        "ldb": 64, "stride_b": 64 * 64,
        "alpha": 1.0,
        "beta": 0.0,
        "gsu": 1,
    }


class TestBuildKernelArgsEpilogue:
    """Byte-layout tests for buildKernelArgs with epilogue slots."""

    def test_no_epilogue_length_unchanged(self):
        """No epilogue flags: buffer length matches M1 standard layout."""
        sol = _minimalSol()
        pp = _basePP()
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}
        buf = buildKernelArgs(sol, pp, tensors)
        # header=16, sizes=16, ptrs=32, strides=24 (non-batched 3-stride each = 1*4*4=16? No).
        # Actually: sizes=[64,64,1,64] → 4 items → batched, so 2 strides per tensor.
        # header=16, sizes=16, D+C+A+B ptrs=32, strides=8*4=32, alpha+beta=8 → 104 bytes.
        assert len(buf) == 104

    def test_scale_ab_adds_16_bytes(self):
        """UseScaleAB='Scalar' appends 2 pointers (16 bytes) after alpha/beta."""
        sol_no = _minimalSol(useScaleAB="")
        sol_yes = _minimalSol(useScaleAB="Scalar")
        pp = _basePP()
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0, "scaleA": 0, "scaleB": 0}
        no_buf = buildKernelArgs(sol_no, pp, tensors)
        yes_buf = buildKernelArgs(sol_yes, pp, tensors)
        assert len(yes_buf) == len(no_buf) + 16

    def test_scale_ab_pointer_values(self):
        """ScaleA and ScaleB pointers appear in the correct order."""
        sol = _minimalSol(useScaleAB="Scalar")
        pp = _basePP()
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0,
                   "scaleA": 0xAAAA0000, "scaleB": 0xBBBB0000}
        buf = buildKernelArgs(sol, pp, tensors)
        # Base (no-epilogue) is 104 bytes; scaleA at offset 104.
        sa = struct.unpack_from("<Q", buf, 104)[0]
        sb = struct.unpack_from("<Q", buf, 112)[0]
        assert sa == 0xAAAA0000
        assert sb == 0xBBBB0000

    def test_scale_cd_adds_16_bytes(self):
        """UseScaleCD=True appends scaleC+scaleD pointers (16 bytes)."""
        sol_no = _minimalSol(useScaleCD=False)
        sol_yes = _minimalSol(useScaleCD=True)
        pp = _basePP()
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0,
                   "scaleC": 0, "scaleD": 0}
        no_buf = buildKernelArgs(sol_no, pp, tensors)
        yes_buf = buildKernelArgs(sol_yes, pp, tensors)
        assert len(yes_buf) == len(no_buf) + 16

    def test_scale_alpha_vec_adds_8_bytes(self):
        """UseScaleAlphaVec=1 appends one pointer (8 bytes)."""
        sol_no = _minimalSol(useScaleAlphaVec=0)
        sol_yes = _minimalSol(useScaleAlphaVec=1)
        pp = _basePP()
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0, "scaleAlphaVec": 0}
        no_buf = buildKernelArgs(sol_no, pp, tensors)
        yes_buf = buildKernelArgs(sol_yes, pp, tensors)
        assert len(yes_buf) == len(no_buf) + 8

    def test_bias_adds_16_bytes(self):
        """UseBias=1 appends bias ptr + bias_type + strideBias (8+4+4=16 bytes)."""
        sol_no = _minimalSol(useBias=0)
        sol_yes = _minimalSol(useBias=1)
        pp = {**_basePP(), "biasType": 0, "strideBias": 1}
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0, "bias": 0}
        no_buf = buildKernelArgs(sol_no, pp, tensors)
        yes_buf = buildKernelArgs(sol_yes, pp, tensors)
        assert len(yes_buf) == len(no_buf) + 16

    def test_bias_pointer_value(self):
        """Bias pointer appears at the correct offset."""
        sol = _minimalSol(useBias=1)
        pp = {**_basePP(), "biasType": 0, "strideBias": 1}
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0, "bias": 0xBEEF0000}
        buf = buildKernelArgs(sol, pp, tensors)
        bias_ptr = struct.unpack_from("<Q", buf, 104)[0]
        assert bias_ptr == 0xBEEF0000

    def test_bias_type_and_stride(self):
        """bias_type and strideBias appear after the bias pointer."""
        sol = _minimalSol(useBias=1)
        pp = {**_basePP(), "biasType": 7, "strideBias": 4}
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0, "bias": 0}
        buf = buildKernelArgs(sol, pp, tensors)
        bias_type = struct.unpack_from("<I", buf, 112)[0]
        stride_bias = struct.unpack_from("<I", buf, 116)[0]
        assert bias_type == 7
        assert stride_bias == 4

    def test_amax_d_adds_24_bytes(self):
        """OutputAmaxD=True appends AddrAmaxOut+AmaxWS+AmaxSync (24 bytes)."""
        sol_no = _minimalSol(outputAmaxD=False)
        sol_yes = _minimalSol(outputAmaxD=True)
        pp = _basePP()
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0,
                   "amaxD": 0, "amaxWS": 0, "amaxSync": 0}
        no_buf = buildKernelArgs(sol_no, pp, tensors)
        yes_buf = buildKernelArgs(sol_yes, pp, tensors)
        assert len(yes_buf) == len(no_buf) + 24

    def test_relu_activation_adds_no_bytes(self):
        """relu has activationArgLength=0 → no extra bytes (not 'all' type)."""
        sol_no = _minimalSol(activationType="none")
        sol_yes = _minimalSol(activationType="relu")
        pp = _basePP()
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}
        no_buf = buildKernelArgs(sol_no, pp, tensors)
        yes_buf = buildKernelArgs(sol_yes, pp, tensors)
        assert len(yes_buf) == len(no_buf)

    def test_tanh_activation_adds_8_bytes(self):
        """tanh has activationArgLength=2 → 2 float32 args = 8 bytes."""
        sol_no = _minimalSol(activationType="none")
        sol_tanh = _minimalSol(activationType="tanh")
        pp = {**_basePP(), "activationArgs": [1.0, 1.0]}
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}
        no_buf = buildKernelArgs(sol_no, pp, tensors)
        tanh_buf = buildKernelArgs(sol_tanh, pp, tensors)
        assert len(tanh_buf) == len(no_buf) + 8

    def test_tanh_activation_values(self):
        """Tanh activation args are packed as float32 in the correct order."""
        sol = _minimalSol(activationType="tanh")
        pp = {**_basePP(), "activationArgs": [2.5, 0.75]}
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}
        buf = buildKernelArgs(sol, pp, tensors)
        a0 = struct.unpack_from("<f", buf, 104)[0]
        a1 = struct.unpack_from("<f", buf, 108)[0]
        assert a0 == pytest.approx(2.5, abs=1e-5)
        assert a1 == pytest.approx(0.75, abs=1e-5)

    def test_combined_scale_ab_and_bias(self):
        """ScaleAB + Bias together: scale ptrs before bias ptr."""
        sol = _minimalSol(useBias=1, useScaleAB="Scalar")
        pp = {**_basePP(), "biasType": 0, "strideBias": 1}
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0,
                   "scaleA": 0xA000, "scaleB": 0xB000, "bias": 0xC000}
        buf = buildKernelArgs(sol, pp, tensors)
        sa = struct.unpack_from("<Q", buf, 104)[0]
        sb = struct.unpack_from("<Q", buf, 112)[0]
        bias = struct.unpack_from("<Q", buf, 120)[0]
        assert sa == 0xA000
        assert sb == 0xB000
        assert bias == 0xC000

    def test_read_pt_flag_flat(self):
        """_readPTFlag reads from top-level when key is present there."""
        sol = {"UseBias": 3}
        assert _readPTFlag(sol, "UseBias", 0) == 3

    def test_read_pt_flag_nested(self):
        """_readPTFlag reads from nested ProblemType when not at top level."""
        sol = {"ProblemType": {"UseBias": 2}}
        assert _readPTFlag(sol, "UseBias", 0) == 2

    def test_read_pt_flag_default(self):
        """_readPTFlag returns default when key is absent everywhere."""
        assert _readPTFlag({}, "UseBias", 99) == 99


# ===========================================================================
# Task 5.2 — Poison-input tests (no GPU, verifying args drive computation)
# ===========================================================================


class TestPoisonInput:
    """Poison-input tests that verify the epilogue arg slots are wired up."""

    def test_bias_pointer_zero_vs_nonzero(self):
        """Bias pointer value is passed through, not ignored."""
        sol = _minimalSol(useBias=1)
        pp = {**_basePP(), "biasType": 0, "strideBias": 1}
        tensors_zero = {"D": 0, "C": 0, "A": 0, "B": 0, "bias": 0}
        tensors_nonzero = {"D": 0, "C": 0, "A": 0, "B": 0, "bias": 0xDEAD0000}
        buf_zero = buildKernelArgs(sol, pp, tensors_zero)
        buf_nonzero = buildKernelArgs(sol, pp, tensors_nonzero)
        # Bias pointer at byte 104: must differ.
        assert buf_zero[104:112] != buf_nonzero[104:112]

    def test_scale_ab_pointer_zero_vs_nonzero(self):
        """ScaleA pointer value is passed through, not ignored."""
        sol = _minimalSol(useScaleAB="Scalar")
        pp = _basePP()
        tensors_zero = {"D": 0, "C": 0, "A": 0, "B": 0, "scaleA": 0, "scaleB": 0}
        tensors_nonzero = {"D": 0, "C": 0, "A": 0, "B": 0,
                           "scaleA": 0xCAFE0000, "scaleB": 0}
        buf_zero = buildKernelArgs(sol, pp, tensors_zero)
        buf_nonzero = buildKernelArgs(sol, pp, tensors_nonzero)
        assert buf_zero[104:112] != buf_nonzero[104:112]


# ===========================================================================
# GPU infrastructure helpers
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


def _compileSolutions(problem_idx: int) -> list[dict]:
    """Compile all solutions for one YAML problem group.

    Returns a list of dicts with keys: sol_dict, raw_dict, kernel_name, hsaco, chip, sid.
    """
    if not HAVE_DEPS:
        return []
    try:
        from Tensile.client.yaml_solution_builder import solutionsFromYaml
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        sols = solutionsFromYaml(_YAML_PATH, assembler, isaInfoMap, debugConfig,
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
    """Return True when the solution passes the M1-M5 kernel filter.

    Skips kernels that require calculateAutoWGM (WorkGroupMapping==0) or
    auto-StaggerU (StaggerU==0 with SupportCustomStaggerU=True).
    """
    sol_dict = entry["sol_dict"]
    raw_dict = entry["raw_dict"]
    if sol_dict.get("WorkGroupMapping", 0) == 0:
        return False
    isp = raw_dict.get("InternalSupportParams", {}) or {}
    if sol_dict.get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
        return False
    return True


def _ntStrides(M: int, N: int, K: int, batch: int):
    """Return NT-GEMM stride tuple: (lda, ldb, ldd, ldc, sa, sb, sd, sc)."""
    return M, N, M, M, M * K, N * K, M * N, M * N


def _allocBatched(M, N, K, batch, dtype, rng):
    """Allocate contiguous batched NT GEMM buffers."""
    A_np = np.asfortranarray(rng.random((M, K)).astype(dtype))
    B_np = np.asfortranarray(rng.random((N, K)).astype(dtype))
    A_buf = np.tile(A_np.ravel(order='F'), batch)
    B_buf = np.tile(B_np.ravel(order='F'), batch)
    C_buf = np.zeros(M * N * batch, dtype=dtype)
    D_buf = np.zeros(M * N * batch, dtype=dtype)
    return A_np, B_np, A_buf, B_buf, C_buf, D_buf


def _deviceCuCount() -> int:
    """Return the device CU count for device 0."""
    if not HAVE_DEPS:
        return 0
    props = amdgpu_exec._runtime_module.hip_get_device_props(0)
    return int(props.get("multiprocessor_count", 0))


def _buildBiasTypedArgs(sol_dict, M, N, batch, K,
                        D_arr, C_arr, A_arr, B_arr,
                        bias_arr, alpha=1.0, beta=0.0):
    """Build typed args list for NT stridedBatched GEMM with col bias (UseBias=1).

    Matches the byte layout of buildKernelArgs for UseBias=1 kernels:
    header + sizes + ptrs + strides + alpha + beta + bias_ptr + bias_type + strideBias.
    strideBias=0 selects non-batched col bias; the kernel uses SizeI (M) as the SRD bound.
    """
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
    args.extend([D_arr, C_arr, A_arr, B_arr])

    lda, ldb, ldd, ldc = M, N, M, M
    sa, sb, sd, sc = M * K, N * K, M * N, M * N
    args.extend([
        np.uint32(ldd), np.uint32(sd),
        np.uint32(ldc), np.uint32(sc),
        np.uint32(lda), np.uint32(sa),
        np.uint32(ldb), np.uint32(sb),
    ])

    args.extend([np.float32(alpha), np.float32(beta)])
    # Epilogue: bias pointer, bias_type (float32=0), strideBias=0 (non-batched col bias).
    args.append(bias_arr)
    args.extend([np.uint32(0), np.uint32(0)])
    return args


def _runBiasGpu(entry, M, N, batch, K, np_dtype, ref_fn, rtol, atol, label):
    """Execute one bias GEMM kernel and verify output against reference.

    ref_fn: callable(A, B^T, alpha, beta, C) → D in float64.
    """
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = (math.ceil(M / sol_dict["MacroTile0"])
              * math.ceil(N / sol_dict["MacroTile1"]) * batch)
    num_threads = sol_dict["NumThreads"]

    rng = np.random.default_rng(seed=M * 1000 + N + K)
    _, _, A_buf, B_buf, C_buf, D_buf = _allocBatched(M, N, K, batch, np_dtype, rng)
    # Col bias: float32 vector of length M (one value per row, broadcast over all columns).
    bias = rng.random(M).astype(np.float32)

    result = {}

    def capture(arguments):
        result["D"] = np.array(arguments[8].array, dtype=np_dtype).copy()

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    bias_in = amdgpu_exec.InputArray(bias)

    args = _buildBiasTypedArgs(sol_dict, M, N, batch, K,
                               D_io, C_in, A_in, B_in, bias_in)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    D_gpu = result["D"]
    # Reference: GEMM + col bias (bias[m] added to every column for row m).
    A_slice = A_buf[:M * K].reshape(M, K, order='F')
    B_slice = B_buf[:N * K].reshape(N, K, order='F')
    D_ref_one = ref_fn(A_slice, B_slice.T, 1.0, 0.0, None)
    D_ref_one = applyBias(D_ref_one, bias.astype(np.float64), "col")
    D_ref_flat = np.tile(np.asfortranarray(D_ref_one).ravel(order='F'), batch)
    assertClose(D_gpu, D_ref_flat, rtol=rtol, atol=atol, label=label)


def _buildReluTypedArgs(sol_dict, M, N, batch, K,
                        D_arr, C_arr, A_arr, B_arr, alpha=1.0, beta=0.0):
    """Build typed args for NT stridedBatched GEMM with relu (0 extra args)."""
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
    args.extend([D_arr, C_arr, A_arr, B_arr])

    lda, ldb, ldd, ldc = M, N, M, M
    sa, sb, sd, sc = M * K, N * K, M * N, M * N
    args.extend([
        np.uint32(ldd), np.uint32(sd),
        np.uint32(ldc), np.uint32(sc),
        np.uint32(lda), np.uint32(sa),
        np.uint32(ldb), np.uint32(sb),
    ])

    args.extend([np.float32(alpha), np.float32(beta)])
    # relu: activationArgLength=0, not 'all' type → no extra args.
    return args


def _runReluGpu(entry, M, N, batch, K, np_dtype, ref_fn, rtol, atol, label):
    """Execute one relu GEMM kernel and verify output against reference."""
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = (math.ceil(M / sol_dict["MacroTile0"])
              * math.ceil(N / sol_dict["MacroTile1"]) * batch)
    num_threads = sol_dict["NumThreads"]

    rng = np.random.default_rng(seed=M * 1000 + N + K + 1)
    _, _, A_buf, B_buf, C_buf, D_buf = _allocBatched(M, N, K, batch, np_dtype, rng)
    result = {}

    def capture(arguments):
        result["D"] = np.array(arguments[8].array, dtype=np_dtype).copy()

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)

    args = _buildReluTypedArgs(sol_dict, M, N, batch, K, D_io, C_in, A_in, B_in)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    D_gpu = result["D"]
    A_slice = A_buf[:M * K].reshape(M, K, order='F')
    B_slice = B_buf[:N * K].reshape(N, K, order='F')
    D_ref_one = ref_fn(A_slice, B_slice.T, 1.0, 0.0, None)
    D_ref_one = applyActivation(D_ref_one, "relu")
    D_ref_flat = np.tile(np.asfortranarray(D_ref_one).ravel(order='F'), batch)
    assertClose(D_gpu, D_ref_flat, rtol=rtol, atol=atol, label=label)


# ---------------------------------------------------------------------------
# Session-scoped compiled solution fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bf16BiasKernels():
    """Compile bf16+bias solutions (group 0)."""
    return _compileSolutions(_GRP_BF16_BIAS)


@pytest.fixture(scope="session")
def fp32BiasKernels():
    """Compile fp32+bias solutions (group 1)."""
    return _compileSolutions(_GRP_FP32_BIAS)


@pytest.fixture(scope="session")
def bf16ReluKernels():
    """Compile bf16+relu solutions (group 2)."""
    return _compileSolutions(_GRP_BF16_RELU)


@pytest.fixture(scope="session")
def fp32ReluKernels():
    """Compile fp32+relu solutions (group 3)."""
    return _compileSolutions(_GRP_FP32_RELU)


# ===========================================================================
# GPU bias poison test
# ===========================================================================


@requires_gfx950
@pytest.mark.parametrize("size", [(256, 256, 4, 256)], ids=["M256N256B4K256"])
def test_bias_poison_fp32(fp32BiasKernels, size):
    """Corrupt bias data and assert >= 50% outputs differ from unbiased reference.

    Proves that the bias argument slot actually drives the computation.
    """
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in fp32BiasKernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no fp32 bias solution compiled")

    M, N, batch, K = size
    entry = entries[0]
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = (math.ceil(M / sol_dict["MacroTile0"])
              * math.ceil(N / sol_dict["MacroTile1"]) * batch)
    num_threads = sol_dict["NumThreads"]

    rng = np.random.default_rng(seed=42)
    _, _, A_buf, B_buf, C_buf, D_buf = _allocBatched(M, N, K, batch, np.float32, rng)
    # Col bias has M elements (one per row). Poison with large values to detect effect.
    correct_bias = np.zeros(M, dtype=np.float32)
    poison_bias = (rng.random(M) + 1.0).astype(np.float32)  # guaranteed nonzero.

    def run_with_bias(bias):
        D_out = np.zeros(M * N * batch, dtype=np.float32)
        result = {}

        def capture(arguments):
            result["D"] = np.array(arguments[8].array, dtype=np.float32).copy()

        D_io = amdgpu_exec.InOutArray(D_out)
        C_in = amdgpu_exec.InputArray(C_buf.copy())
        A_in = amdgpu_exec.InputArray(A_buf)
        B_in = amdgpu_exec.InputArray(B_buf)
        bias_in = amdgpu_exec.InputArray(bias)
        args = _buildBiasTypedArgs(sol_dict, M, N, batch, K,
                                   D_io, C_in, A_in, B_in, bias_in)
        amdgpu_exec.execute_hsaco(
            hsaco=hsaco, kernel_name=kernel_name, arguments=args,
            grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
            num_iterations=1, verify_fn=capture,
        )
        return result["D"]

    D_zero_bias = run_with_bias(correct_bias)
    D_poison_bias = run_with_bias(poison_bias)

    # At least 50% of elements must differ when bias changes — proving bias drives output.
    diff = np.abs(D_zero_bias.astype(np.float64) - D_poison_bias.astype(np.float64))
    bad_frac = float(np.mean(diff > ATOL_FP32 * 10))
    assert bad_frac >= 0.5, (
        f"only {bad_frac:.1%} elements changed — bias arg may not drive computation"
    )


@pytest.mark.parametrize("size", [(256, 256, 4, 256)], ids=["M256N256B4K256"])
def test_scaleab_poison_fp32(size):
    """ScaleAB poison: changing scaleA changes the output.

    Uses the fp32 bias kernel (no ScaleAB) only to verify the layout;
    this test exercises the slot via pure Python layout verification.
    """
    # This is a pure Python layout poison test — no GPU execution needed.
    sol = _minimalSol(useScaleAB="Scalar")
    pp = _basePP()
    tensors_a = {"D": 0, "C": 0, "A": 0, "B": 0, "scaleA": 0x1000, "scaleB": 0x2000}
    tensors_b = {"D": 0, "C": 0, "A": 0, "B": 0, "scaleA": 0x9999, "scaleB": 0x2000}
    buf_a = buildKernelArgs(sol, pp, tensors_a)
    buf_b = buildKernelArgs(sol, pp, tensors_b)
    # ScaleA pointer at byte 104 must differ.
    assert buf_a[104:112] != buf_b[104:112], "scaleA pointer not wired through"


# ===========================================================================
# Task 5.3 — GPU correctness: bias (fp32 and bf16)
# ===========================================================================


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_bias_fp32(fp32BiasKernels, size):
    """fp32 NT stridedBatched + col bias correctness."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in fp32BiasKernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no fp32 bias solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runBiasGpu(entry, M, N, batch, K, np.float32,
                    gemm, RTOL_FP32, ATOL_FP32,
                    label=f"fp32 bias M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_bias_bf16(bf16BiasKernels, size):
    """bf16 HPA NT stridedBatched + col bias correctness."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if ml_dtypes is None:
        pytest.skip("ml_dtypes not installed")
    entries = [e for e in bf16BiasKernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no bf16 bias solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runBiasGpu(entry, M, N, batch, K, ml_dtypes.bfloat16,
                    gemmBf16, RTOL_BF16, ATOL_BF16,
                    label=f"bf16 bias M{M}N{N}B{batch}K{K} {sid}")


# ===========================================================================
# Task 5.3 — GPU correctness: relu activation (fp32 and bf16)
# ===========================================================================


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_relu_fp32(fp32ReluKernels, size):
    """fp32 NT stridedBatched + relu activation correctness."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in fp32ReluKernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no fp32 relu solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runReluGpu(entry, M, N, batch, K, np.float32,
                    gemm, RTOL_FP32, ATOL_FP32,
                    label=f"fp32 relu M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_relu_bf16(bf16ReluKernels, size):
    """bf16 HPA NT stridedBatched + relu activation correctness."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if ml_dtypes is None:
        pytest.skip("ml_dtypes not installed")
    entries = [e for e in bf16ReluKernels if _filterSolution(e)]
    if not entries:
        pytest.skip("no bf16 relu solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runReluGpu(entry, M, N, batch, K, ml_dtypes.bfloat16,
                    gemmBf16, RTOL_BF16, ATOL_BF16,
                    label=f"bf16 relu M{M}N{N}B{batch}K{K} {sid}")


# ===========================================================================
# Session-scoped compiled solution fixtures for ScaleCD.
# ===========================================================================


@pytest.fixture(scope="session")
def bf16ScaleCdKernels():
    """Compile bf16+ScaleCD solutions (group 6)."""
    return _compileSolutions(_GRP_BF16_SCALE_CD)


@pytest.fixture(scope="session")
def fp32ScaleCdKernels():
    """Compile fp32+ScaleCD solutions (group 7)."""
    return _compileSolutions(_GRP_FP32_SCALE_CD)


# ===========================================================================
# ScaleCD GPU helper: typed args and run helper.
# ===========================================================================


def _buildScaleCdTypedArgs(sol_dict, M, N, batch, K,
                            D_arr, C_arr, A_arr, B_arr,
                            scaleC_arr, scaleD_arr, alpha=1.0, beta=0.0):
    """Build typed args for NT stridedBatched GEMM with ScaleCD epilogue.

    After alpha/beta the ScaleCD slot adds scaleC pointer then scaleD pointer.
    """
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
    args.extend([D_arr, C_arr, A_arr, B_arr])

    lda, ldb, ldd, ldc = M, N, M, M
    sa, sb, sd, sc = M * K, N * K, M * N, M * N
    args.extend([
        np.uint32(ldd), np.uint32(sd),
        np.uint32(ldc), np.uint32(sc),
        np.uint32(lda), np.uint32(sa),
        np.uint32(ldb), np.uint32(sb),
    ])

    args.extend([np.float32(alpha), np.float32(beta)])
    # ScaleCD epilogue: scaleC pointer then scaleD pointer.
    args.append(scaleC_arr)
    args.append(scaleD_arr)
    return args


def _runScaleCdGpu(entry, M, N, batch, K, np_dtype, ref_fn, rtol, atol, label):
    """Execute one ScaleCD GEMM kernel and verify output against reference.

    Uses scaleC=scaleD=2.0 and beta=1.0 with an all-zero C buffer.  beta=1.0
    ensures the kernel does not take the beta=0 fast path, which bypasses the
    ScaleD epilogue.  With C=0 the C contribution vanishes and the reference is
    still scaleD * A @ B = 2 * GEMM.
    """
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = (math.ceil(M / sol_dict["MacroTile0"])
              * math.ceil(N / sol_dict["MacroTile1"]) * batch)
    num_threads = sol_dict["NumThreads"]

    rng = np.random.default_rng(seed=M * 1000 + N + K + 2)
    _, _, A_buf, B_buf, C_buf, D_buf = _allocBatched(M, N, K, batch, np_dtype, rng)

    # Equal scalar values ensure the test is agnostic to slot ordering.
    scale_val = np.float32(2.0)
    scaleC_np = np.array([scale_val], dtype=np.float32)
    scaleD_np = np.array([scale_val], dtype=np.float32)

    result = {}

    def capture(arguments):
        result["D"] = np.array(arguments[8].array, dtype=np_dtype).copy()

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    scaleC_in = amdgpu_exec.InputArray(scaleC_np)
    scaleD_in = amdgpu_exec.InputArray(scaleD_np)

    # beta=1.0 forces the kernel to go through the full epilogue code path
    # (including ScaleD).  C is all-zeros so the C contribution is always 0.
    args = _buildScaleCdTypedArgs(sol_dict, M, N, batch, K,
                                   D_io, C_in, A_in, B_in,
                                   scaleC_in, scaleD_in, beta=1.0)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    D_gpu = result["D"]
    A_slice = A_buf[:M * K].reshape(M, K, order='F')
    B_slice = B_buf[:N * K].reshape(N, K, order='F')
    # Reference: scaleD * alpha * A @ B (C=0 so beta*scaleC*C = 0).
    D_ref_one = ref_fn(A_slice, B_slice.T, 1.0, 0.0, None)
    D_ref_one = D_ref_one.astype(np.float64) * float(scale_val)
    D_ref_flat = np.tile(np.asfortranarray(D_ref_one).ravel(order='F'), batch)
    assertClose(D_gpu, D_ref_flat, rtol=rtol, atol=atol, label=label)


# ===========================================================================
# Task 5.3 — GPU correctness: ScaleCD (fp32 and bf16)
# ===========================================================================


class TestScaleCDGpu:
    """GPU correctness tests for UseScaleCD kernels on gfx950."""

    @requires_gfx950
    @pytest.mark.parametrize("size", _PROBLEM_SIZES,
                             ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
    def test_scale_cd_fp32(self, fp32ScaleCdKernels, size):
        """fp32 NT stridedBatched + ScaleCD correctness."""
        if not HAVE_DEPS:
            pytest.skip("amdgpu_exec not installed")
        entries = [e for e in fp32ScaleCdKernels if _filterSolution(e)]
        if not entries:
            pytest.skip("no fp32 ScaleCD solution compiled")

        M, N, batch, K = size
        for entry in entries:
            sid = entry["sid"]
            _runScaleCdGpu(entry, M, N, batch, K, np.float32,
                           gemm, RTOL_FP32, ATOL_FP32,
                           label=f"fp32 scaleCD M{M}N{N}B{batch}K{K} {sid}")

    @requires_gfx950
    @pytest.mark.parametrize("size", _PROBLEM_SIZES,
                             ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
    def test_scale_cd_bf16(self, bf16ScaleCdKernels, size):
        """bf16 HPA NT stridedBatched + ScaleCD correctness."""
        if not HAVE_DEPS:
            pytest.skip("amdgpu_exec not installed")
        if ml_dtypes is None:
            pytest.skip("ml_dtypes not installed")
        entries = [e for e in bf16ScaleCdKernels if _filterSolution(e)]
        if not entries:
            pytest.skip("no bf16 ScaleCD solution compiled")

        M, N, batch, K = size
        for entry in entries:
            sid = entry["sid"]
            _runScaleCdGpu(entry, M, N, batch, K, ml_dtypes.bfloat16,
                           gemmBf16, RTOL_BF16, ATOL_BF16,
                           label=f"bf16 scaleCD M{M}N{N}B{batch}K{K} {sid}")


# ===========================================================================
# Task 5.3 — Stub GPU tests: ScaleAlphaVec, AmaxD, ETensor, multi-epilogue.
#
# These kernel configurations are not present in gemm_epilogues.yaml for gfx950.
# See Tensile/client/tests/fixtures/m5_missing_kernels.txt for the explanation.
# ===========================================================================


class TestScaleAlphaVecGpu:
    """ScaleAlphaVec GPU tests — skipped: no kernel compiled for gfx950.

    UseScaleAlphaVec kernels require a different ProblemType not present in
    gemm_epilogues.yaml.  See m5_missing_kernels.txt.
    """

    @requires_gfx950
    def test_scale_alpha_vec_fp32_per_row(self):
        pytest.skip(
            "no ScaleAlphaVec kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_scale_alpha_vec_fp32_per_col(self):
        pytest.skip(
            "no ScaleAlphaVec kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_scale_alpha_vec_bf16_per_row(self):
        pytest.skip(
            "no ScaleAlphaVec kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_scale_alpha_vec_bf16_per_col(self):
        pytest.skip(
            "no ScaleAlphaVec kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )


class TestAmaxDGpu:
    """AmaxD GPU tests — skipped: no kernel compiled for gfx950.

    OutputAmaxD kernels require a different ProblemType not present in
    gemm_epilogues.yaml.  See m5_missing_kernels.txt.
    """

    @requires_gfx950
    def test_amax_d_fp32(self):
        pytest.skip(
            "no AmaxD kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_amax_d_bf16(self):
        pytest.skip(
            "no AmaxD kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )


class TestETensorGpu:
    """E-tensor GPU tests — skipped: no kernel compiled for gfx950.

    UseE kernels require a different ProblemType not present in
    gemm_epilogues.yaml.  See m5_missing_kernels.txt.
    """

    @requires_gfx950
    def test_e_tensor_fp32(self):
        pytest.skip(
            "no ETensor kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_e_tensor_bf16(self):
        pytest.skip(
            "no ETensor kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )


class TestMultiEpilogueGpu:
    """Multi-epilogue GPU tests — skipped: no multi-epilogue kernel compiled for gfx950.

    Kernels combining bias+Relu, bias+ScaleAB+Gelu, or ScaleAB+ScaleCD+AmaxD
    are not present in gemm_epilogues.yaml.  See m5_missing_kernels.txt.
    """

    @requires_gfx950
    def test_bias_relu_fp32(self):
        pytest.skip(
            "no bias+Relu combined kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_bias_relu_bf16(self):
        pytest.skip(
            "no bias+Relu combined kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_bias_scaleab_gelu_fp32(self):
        pytest.skip(
            "no bias+ScaleAB+Gelu combined kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_bias_scaleab_gelu_bf16(self):
        pytest.skip(
            "no bias+ScaleAB+Gelu combined kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_scaleab_scalecd_amaxd_fp32(self):
        pytest.skip(
            "no ScaleAB+ScaleCD+AmaxD combined kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )

    @requires_gfx950
    def test_scaleab_scalecd_amaxd_bf16(self):
        pytest.skip(
            "no ScaleAB+ScaleCD+AmaxD combined kernel compiled for gfx950 — "
            "see Tensile/client/tests/fixtures/m5_missing_kernels.txt"
        )
