# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M3 test suite: Float8 OCP (F8, B8) and fnuz (F8N, B8N) — pure-Python tests.

Covers Tasks 3.1–3.2 of the TensileLite Python client plan:
  3.1  fp8 dtype code verification (no GPU)
  3.2  gemmFp8 reference + NaN bit-pattern unit tests (no GPU)

GPU tests (Tasks 3.3-3.5) are added in the follow-up commit.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

try:
    import ml_dtypes
    haveMlDtypes = True
except ImportError:
    ml_dtypes = None
    haveMlDtypes = False

_tensileRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _tensileRoot not in sys.path:
    sys.path.insert(0, _tensileRoot)

from Tensile.client.gemm_args import (
    _dtypeFp8e4m3fnuz,
    _dtypeBf8e5m2fnuz,
    _dtypeFp8e4m3fn,
    _dtypeBf8e5m2,
)
from Tensile.client.reference import gemmFp8, ATOL_FP32


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
