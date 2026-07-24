# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for Tensile/client/reference.py — no GPU required."""

import numpy as np
import pytest

from Tensile.client.reference import (
    gemm,
    assertClose,
    RTOL_BF16, ATOL_BF16,
    RTOL_FP16, ATOL_FP16,
    RTOL_FP32, ATOL_FP32,
)


# ---------------------------------------------------------------------------
# gemm() tests.
# ---------------------------------------------------------------------------


def test_gemm_basic():
    A = np.eye(4, dtype=np.float32)
    B = np.eye(4, dtype=np.float32)
    D = gemm(A, B)
    np.testing.assert_allclose(D, np.eye(4, dtype=np.float64), atol=1e-10)


def test_gemm_alpha_beta():
    A = np.ones((2, 3), dtype=np.float32)
    B = np.ones((3, 2), dtype=np.float32)
    C = np.ones((2, 2), dtype=np.float32)
    D = gemm(A, B, alpha=2.0, beta=0.5, C=C)
    # alpha*(A@B) + beta*C = 2*(3*ones) + 0.5*ones = 6.5
    np.testing.assert_allclose(D, np.full((2, 2), 6.5), atol=1e-10)


def test_gemm_no_c():
    A = np.ones((2, 3), dtype=np.float32)
    B = np.ones((3, 2), dtype=np.float32)
    D = gemm(A, B, beta=1.0, C=None)
    # beta*C is zero when C is None regardless of beta
    np.testing.assert_allclose(D, np.full((2, 2), 3.0), atol=1e-10)


def test_gemm_returns_float64():
    A = np.ones((2, 2), dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)
    D = gemm(A, B)
    assert D.dtype == np.float64


def test_gemm_rectangular():
    A = np.arange(6, dtype=np.float32).reshape(2, 3)
    B = np.arange(6, dtype=np.float32).reshape(3, 2)
    D = gemm(A, B)
    expected = A.astype(np.float64) @ B.astype(np.float64)
    np.testing.assert_allclose(D, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# assertClose() tests.
# ---------------------------------------------------------------------------


def test_assertClose_passes():
    gpu = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assertClose(gpu, ref, rtol=RTOL_FP32, atol=ATOL_FP32)


def test_assertClose_fails_with_detail():
    gpu = np.array([1.0, 2.0, 999.0], dtype=np.float32)
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(AssertionError, match="mismatch"):
        assertClose(gpu, ref, rtol=RTOL_FP32, atol=ATOL_FP32)


def test_assertClose_2d():
    gpu = np.zeros((3, 3), dtype=np.float32)
    ref = np.zeros((3, 3), dtype=np.float32)
    gpu[1, 2] = 1.0  # introduce one bad element
    with pytest.raises(AssertionError) as exc_info:
        assertClose(gpu, ref, rtol=0.0, atol=1e-8, label="D")
    assert "D" in str(exc_info.value)
    assert "row=" in str(exc_info.value)


def test_assertClose_label_in_error():
    gpu = np.array([999.0])
    ref = np.array([0.0])
    with pytest.raises(AssertionError, match="partialBuf"):
        assertClose(gpu, ref, rtol=0.0, atol=0.0, label="partialBuf")


# ---------------------------------------------------------------------------
# Tolerance constant sanity checks.
# ---------------------------------------------------------------------------


def test_tolerance_ordering():
    # Tighter types should have tighter (smaller) tolerances.
    assert ATOL_FP32 < ATOL_FP16 < ATOL_BF16
    assert RTOL_FP32 < RTOL_FP16 < RTOL_BF16
