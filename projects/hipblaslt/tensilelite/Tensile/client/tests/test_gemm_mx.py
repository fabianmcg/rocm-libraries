# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M4 test suite: MX block-scaled GEMM (F8 and F4 with E8 scales).

Covers Tasks 4.1-4.5 of the TensileLite Python client plan:
  4.1  E8 and E5M3 scale decoder unit tests (no GPU)
  4.2  Float4/Float6/BFloat6 unpacker unit tests (no GPU)
  4.3  gemmMx reference unit tests (no GPU)
  4.4  buildKernelArgs MX byte-layout unit test (no GPU)
  4.5  GPU correctness: MX F8 and MX F4 TN batched on gfx950

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec.
Float6/BFloat6 GPU tests are absent: those types require gfx1250.
"""

from __future__ import annotations

import math
import os
import struct
import sys

import numpy as np
import pytest

try:
    import amdgpu_exec
    haveDeps = True
except ImportError:
    amdgpu_exec = None
    haveDeps = False

from .conftest import requires_gfx950

_testsDir = os.path.dirname(__file__)
_yamlPath = os.path.join(_testsDir, "yaml", "gemm_mx.yaml")
_tensileRoot = os.path.abspath(os.path.join(_testsDir, "..", "..", "..", ".."))

if _tensileRoot not in sys.path:
    sys.path.insert(0, _tensileRoot)

from Tensile.client.mx_types import (
    decodeE8,
    decodeE5m3,
    unpackFloat4,
    unpackFloat6E2m3,
    unpackBfloat6E3m2,
)
from Tensile.client.reference import gemmMx, assertClose
from Tensile.client.gemm_args import (
    _computeInternalArg0,
    _computeInternalArg1,
    buildKernelArgs,
)
from epilogues.epilogue_harness.yaml_solution_builder import _injectInternalArgsSupport

# Tolerance for MX GEMM correctness checks.
rtolMx: float = 1e-1
atolMx: float = 1e-1

# Problem sizes: (M, N, batch, K).
_mxProblemSizes = [
    (256, 256, 4, 256),
    (512, 512, 4, 512),
]

# MX scale block size (32 elements on gfx950).
_mxBlockK = 32


# ===========================================================================
# Task 4.1 - E8 and E5M3 decoder unit tests (no GPU)
# ===========================================================================


class TestDecodeE8:
    """Hand-computed (byte -> float) pairs for the UE8M0 decoder."""

    # 10 reference pairs: (byte, expected_value).
    # value = 2^(byte - 127); 0xFF is NaN.
    _pairs = [
        (0x00, 2.0 ** -127),
        (0x01, 2.0 ** -126),
        (0x7E, 0.5),
        (0x7F, 1.0),
        (0x80, 2.0),
        (0x81, 4.0),
        (0x82, 8.0),
        (0x83, 16.0),
        (0xFE, 2.0 ** 127),
        (0xFF, np.nan),
    ]

    @pytest.mark.parametrize("byte_val,expected", _pairs,
                             ids=[f"0x{b:02X}" for b, _ in _pairs])
    def test_single_byte(self, byte_val, expected):
        """Each E8 byte decodes to the expected float32 value."""
        result = decodeE8(np.array([byte_val], dtype=np.uint8))
        assert result.dtype == np.float32
        if np.isnan(expected):
            assert np.isnan(result[0]), f"expected NaN for 0x{byte_val:02X}"
        else:
            np.testing.assert_allclose(result[0], expected, rtol=1e-6, atol=0,
                                       err_msg=f"byte 0x{byte_val:02X}")

    def test_batch_decode(self):
        """Vectorised decode of all non-NaN pairs in one call."""
        non_nan = [(b, v) for b, v in self._pairs if not np.isnan(v)]
        bytes_in = np.array([b for b, _ in non_nan], dtype=np.uint8)
        expected = np.array([v for _, v in non_nan], dtype=np.float64)
        np.testing.assert_allclose(decodeE8(bytes_in).astype(np.float64),
                                   expected, rtol=1e-6, atol=0)

    def test_output_dtype_is_float32(self):
        """decodeE8 always returns float32."""
        assert decodeE8(np.array([0x7F], dtype=np.uint8)).dtype == np.float32


class TestDecodeE5m3:
    """Hand-computed (byte -> float) pairs for the E5M3 unsigned-float decoder.

    Formula: exp = data >> 3, mant = data & 7.
    Normal (exp > 0): 2^(exp-15) * (1 + mant/8).
    Subnormal (exp == 0): 2^-14 * (mant/8).
    NaN: 0xFF only.
    """

    _pairs = [
        (0x00, 0.0),                      # subnormal, mant=0
        (0x01, 2.0 ** -17),              # subnormal, 2^-14*(1/8)=2^-17
        (0x07, 2.0 ** -14 * 7.0 / 8),   # subnormal, 2^-14*(7/8)
        (0x08, 2.0 ** -14),              # normal exp=1, mant=0
        (0x0F, 2.0 ** -14 * 1.875),     # normal exp=1, mant=7: 2^-14*(1+7/8)
        (0x78, 1.0),                     # normal exp=15, mant=0: 2^0*1.0
        (0x7F, 1.875),                   # normal exp=15, mant=7: 2^0*(1+7/8)
        (0x80, 2.0),                     # normal exp=16, mant=0: 2^1*1.0
        (0xFE, 2.0 ** 16 * 1.75),       # normal exp=31, mant=6: 2^16*(1+6/8)
        (0xFF, np.nan),                  # NaN sentinel
    ]

    @pytest.mark.parametrize("byte_val,expected", _pairs,
                             ids=[f"0x{b:02X}" for b, _ in _pairs])
    def test_single_byte(self, byte_val, expected):
        """Each E5M3 byte decodes to the expected float32 value."""
        result = decodeE5m3(np.array([byte_val], dtype=np.uint8))
        assert result.dtype == np.float32
        if np.isnan(expected):
            assert np.isnan(result[0]), f"expected NaN for 0x{byte_val:02X}"
        else:
            np.testing.assert_allclose(float(result[0]), expected, rtol=1e-6, atol=0,
                                       err_msg=f"byte 0x{byte_val:02X}")

    def test_batch_decode(self):
        """Vectorised decode of all non-NaN pairs in one call."""
        non_nan = [(b, v) for b, v in self._pairs if not np.isnan(v)]
        bytes_in = np.array([b for b, _ in non_nan], dtype=np.uint8)
        expected = np.array([v for _, v in non_nan], dtype=np.float64)
        np.testing.assert_allclose(decodeE5m3(bytes_in).astype(np.float64),
                                   expected, rtol=1e-6, atol=0)

    def test_output_dtype_is_float32(self):
        """decodeE5m3 always returns float32."""
        assert decodeE5m3(np.array([0x78], dtype=np.uint8)).dtype == np.float32


# ===========================================================================
# Task 4.2 - Unpacker unit tests (no GPU)
# ===========================================================================


class TestUnpackFloat4:
    """Hand-computed (packed_byte -> [elem0, elem1]) pairs for unpackFloat4.

    E2M1 nibble values ([s][e1][e0][m], bias=1):
      0x0=0.0  0x1=0.5  0x2=1.0  0x3=1.5
      0x4=2.0  0x5=3.0  0x6=4.0  0x7=6.0
      0x8=-0.0 0x9=-0.5 0xA=-1.0 0xB=-1.5
      0xC=-2.0 0xD=-3.0 0xE=-4.0 0xF=-6.0
    Low nibble (bits[3:0]) gives element 0; high nibble (bits[7:4]) gives element 1.
    """

    # 12 triples: (packed_byte, elem_even, elem_odd).
    _pairs = [
        (0x00, 0.0,  0.0),
        (0x01, 0.5,  0.0),   # low=0x1, high=0x0
        (0x10, 0.0,  0.5),   # low=0x0, high=0x1
        (0x22, 1.0,  1.0),
        (0x23, 1.5,  1.0),   # low=0x3, high=0x2
        (0x32, 1.0,  1.5),   # low=0x2, high=0x3
        (0x45, 3.0,  2.0),   # low=0x5=3.0, high=0x4=2.0
        (0x67, 6.0,  4.0),   # low=0x7=6.0, high=0x6=4.0
        (0x76, 4.0,  6.0),   # low=0x6=4.0, high=0x7=6.0
        (0xAA, -1.0, -1.0),  # 0xA=-1.0 both
        (0xCB, -1.5, -2.0),  # low=0xB=-1.5, high=0xC=-2.0
        (0xFF, -6.0, -6.0),
    ]

    @pytest.mark.parametrize("packed,e0,e1", _pairs,
                             ids=[f"0x{p:02X}" for p, _, _ in _pairs])
    def test_single_byte(self, packed, e0, e1):
        """Each packed byte decodes to the expected (elem0, elem1) pair."""
        result = unpackFloat4(np.array([packed], dtype=np.uint8))
        assert result.shape == (2,)
        assert result.dtype == np.float32
        np.testing.assert_allclose(float(result[0]), e0, atol=1e-7,
                                   err_msg=f"byte 0x{packed:02X} element 0")
        np.testing.assert_allclose(float(result[1]), e1, atol=1e-7,
                                   err_msg=f"byte 0x{packed:02X} element 1")

    def test_output_shape_doubles_last_dim(self):
        """unpackFloat4 output has shape (..., 2K) for input (..., K)."""
        result = unpackFloat4(np.zeros((4, 8), dtype=np.uint8))
        assert result.shape == (4, 16)

    def test_all_zero_bytes(self):
        """A buffer of 0x00 bytes decodes to all 0.0 values."""
        result = unpackFloat4(np.zeros(16, dtype=np.uint8))
        np.testing.assert_array_equal(result, np.zeros(32, dtype=np.float32))


def _pack6bitGroup(elem6_list: list) -> bytes:
    """Pack a list of 32 six-bit integers into 24 bytes (little-endian bit order).

    Every 4 elements map to 3 bytes:
      b0 = e0[5:0] | (e1[1:0] << 6)
      b1 = (e1[5:2]) | (e2[3:0] << 4)
      b2 = (e2[5:4]) | (e3[5:0] << 2)
    """
    assert len(elem6_list) == 32
    buf = bytearray(24)
    for g in range(8):
        e0, e1, e2, e3 = (elem6_list[4 * g + i] for i in range(4))
        buf[3 * g]     = (e0 & 0x3F) | ((e1 & 0x3) << 6)
        buf[3 * g + 1] = ((e1 >> 2) & 0xF) | ((e2 & 0xF) << 4)
        buf[3 * g + 2] = ((e2 >> 4) & 0x3) | ((e3 & 0x3F) << 2)
    return bytes(buf)


class TestUnpackFloat6E2m3:
    """Unit tests for Float6 E2M3 unpacker.

    E2M3 format: [s][e1][e0][m2][m1][m0], bias=1.
    Normal (exp > 0): (-1)^s * 2^(exp-1) * (1 + mant/8).
    Subnormal (exp == 0): (-1)^s * (mant/8).
    """

    def test_all_zeros_decode_to_zero(self):
        """All-zero packed bytes decode to 0.0."""
        result = unpackFloat6E2m3(np.zeros(24, dtype=np.uint8), 32)
        assert result.shape == (32,)
        np.testing.assert_array_equal(result, np.zeros(32, dtype=np.float32))

    def test_output_dtype_is_float32(self):
        """unpackFloat6E2m3 returns float32."""
        assert unpackFloat6E2m3(np.zeros(24, dtype=np.uint8), 32).dtype == np.float32

    def test_ten_known_values(self):
        """Ten hand-computed (raw6, expected_float) pairs for E2M3.

        Derivations (s=bit5, exp=bits[4:3], mant=bits[2:0]):
          0x00=000000: s=0, exp=0, mant=0 -> 0.0
          0x01=000001: s=0, exp=0, mant=1 -> 1/8=0.125
          0x07=000111: s=0, exp=0, mant=7 -> 7/8=0.875
          0x08=001000: s=0, exp=1, mant=0 -> 2^0*1.0=1.0
          0x0F=001111: s=0, exp=1, mant=7 -> 2^0*(1+7/8)=1.875
          0x10=010000: s=0, exp=2, mant=0 -> 2^1*1.0=2.0
          0x17=010111: s=0, exp=2, mant=7 -> 2^1*(1+7/8)=3.75
          0x18=011000: s=0, exp=3, mant=0 -> 2^2*1.0=4.0
          0x20=100000: s=1, exp=0, mant=0 -> -0.0=0.0
          0x28=101000: s=1, exp=1, mant=0 -> -2^0*1.0=-1.0
        """
        known = [
            (0x00, 0.0),
            (0x01, 0.125),
            (0x07, 0.875),
            (0x08, 1.0),
            (0x0F, 1.875),
            (0x10, 2.0),
            (0x17, 3.75),
            (0x18, 4.0),
            (0x20, 0.0),
            (0x28, -1.0),
        ]
        for raw6, expected in known:
            elems = [raw6] + [0] * 31
            packed = np.frombuffer(_pack6bitGroup(elems), dtype=np.uint8)
            result = unpackFloat6E2m3(packed, 32)
            np.testing.assert_allclose(
                float(result[0]), expected, atol=1e-7,
                err_msg=f"E2M3 raw 0x{raw6:02X}",
            )

    def test_first_and_last_element(self):
        """First and last elements of a packed buffer decode correctly."""
        # Element 0: 0x15=010101 -> s=0, exp=2, mant=5 -> 2^1*(1+5/8)=2*1.625=3.25
        # Element 31: 0x3F=111111 -> s=1, exp=3, mant=7 -> -2^2*(1+7/8)=-4*1.875=-7.5
        elems = [0x15] + [0] * 30 + [0x3F]
        packed = np.frombuffer(_pack6bitGroup(elems), dtype=np.uint8)
        result = unpackFloat6E2m3(packed, 32)
        np.testing.assert_allclose(float(result[0]), 3.25, rtol=1e-6)
        np.testing.assert_allclose(float(result[31]), -7.5, rtol=1e-6)


class TestUnpackBfloat6E3m2:
    """Unit tests for BFloat6 E3M2 unpacker.

    E3M2 format: [s][e2][e1][e0][m1][m0], bias=3.
    Normal (exp > 0): (-1)^s * 2^(exp-3) * (1 + mant/4).
    Subnormal (exp == 0): (-1)^s * 2^-2 * (mant/4).
    """

    def test_all_zeros_decode_to_zero(self):
        """All-zero packed bytes decode to 0.0."""
        result = unpackBfloat6E3m2(np.zeros(24, dtype=np.uint8), 32)
        assert result.shape == (32,)
        np.testing.assert_array_equal(result, np.zeros(32, dtype=np.float32))

    def test_output_dtype_is_float32(self):
        """unpackBfloat6E3m2 returns float32."""
        assert unpackBfloat6E3m2(np.zeros(24, dtype=np.uint8), 32).dtype == np.float32

    def test_ten_known_values(self):
        """Ten hand-computed (raw6, expected_float) pairs for E3M2.

        Derivations (s=bit5, exp=bits[4:2], mant=bits[1:0]):
          0x00=000000: s=0, exp=0, mant=0 -> 0.0
          0x01=000001: s=0, exp=0, mant=1 -> 2^-2*(1/4)=0.0625
          0x03=000011: s=0, exp=0, mant=3 -> 2^-2*(3/4)=0.1875
          0x04=000100: s=0, exp=1, mant=0 -> 2^(1-3)*1.0=0.25
          0x08=001000: s=0, exp=2, mant=0 -> 2^(2-3)*1.0=0.5
          0x0C=001100: s=0, exp=3, mant=0 -> 2^(3-3)*1.0=1.0
          0x0F=001111: s=0, exp=3, mant=3 -> 2^0*(1+3/4)=1.75
          0x10=010000: s=0, exp=4, mant=0 -> 2^(4-3)*1.0=2.0
          0x14=010100: s=0, exp=5, mant=0 -> 2^(5-3)*1.0=4.0
          0x2C=101100: s=1, exp=3, mant=0 -> -2^(3-3)*1.0=-1.0
        """
        known = [
            (0x00, 0.0),
            (0x01, 0.0625),
            (0x03, 0.1875),
            (0x04, 0.25),
            (0x08, 0.5),
            (0x0C, 1.0),
            (0x0F, 1.75),
            (0x10, 2.0),
            (0x14, 4.0),
            (0x2C, -1.0),
        ]
        for raw6, expected in known:
            elems = [raw6] + [0] * 31
            packed = np.frombuffer(_pack6bitGroup(elems), dtype=np.uint8)
            result = unpackBfloat6E3m2(packed, 32)
            np.testing.assert_allclose(
                float(result[0]), expected, atol=1e-7,
                err_msg=f"E3M2 raw 0x{raw6:02X}: expected {expected}",
            )


# ===========================================================================
# Task 4.3 - gemmMx reference unit tests (no GPU)
# ===========================================================================


class TestGemmMxReference:
    """Unit tests for reference.gemmMx without GPU."""

    def _scaleOnes(self, shape):
        """Return a uint8 E8 scale buffer encoding all-1.0 values (0x7F)."""
        return np.full(shape, 0x7F, dtype=np.uint8)

    def test_output_shape(self):
        """gemmMx returns (M, N) float32 output."""
        A = np.ones((4, 32), dtype=np.float32)
        B = np.ones((6, 32), dtype=np.float32)
        D = gemmMx(A, B, self._scaleOnes((4, 1)), self._scaleOnes((6, 1)), blockK=32)
        assert D.shape == (4, 6)
        assert D.dtype == np.float32

    def test_unit_scale_equals_plain_gemm(self):
        """gemmMx with all-1.0 scales equals a plain float32 GEMM."""
        rng = np.random.default_rng(0)
        M, N, K = 16, 12, 64
        kBlocks = K // 32
        A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
        B = rng.uniform(-1, 1, (N, K)).astype(np.float32)
        D_mx = gemmMx(A, B, self._scaleOnes((M, kBlocks)), self._scaleOnes((N, kBlocks)),
                      blockK=32)
        np.testing.assert_allclose(D_mx, (A @ B.T).astype(np.float32), rtol=1e-5, atol=1e-5)

    def test_zero_scale_gives_near_zero_output(self):
        """E8 scale 0x00 (= 2^-127) produces near-zero output."""
        A = np.ones((4, 32), dtype=np.float32)
        B = np.ones((4, 32), dtype=np.float32)
        sa = np.zeros((4, 1), dtype=np.uint8)   # 0x00 -> 2^-127
        D = gemmMx(A, B, sa, self._scaleOnes((4, 1)), blockK=32)
        assert np.all(np.abs(D) < 1e-30)

    def test_alpha_scaling(self):
        """gemmMx applies alpha correctly."""
        A = np.ones((2, 32), dtype=np.float32)
        B = np.ones((2, 32), dtype=np.float32)
        sa = self._scaleOnes((2, 1))
        sb = self._scaleOnes((2, 1))
        D2 = gemmMx(A, B, sa, sb, blockK=32, alpha=2.0)
        D1 = gemmMx(A, B, sa, sb, blockK=32, alpha=1.0)
        np.testing.assert_allclose(D2, 2.0 * D1, rtol=1e-6)

    def test_beta_accumulation(self):
        """gemmMx adds beta * C when beta is non-zero."""
        A = np.ones((4, 32), dtype=np.float32)
        B = np.ones((4, 32), dtype=np.float32)
        sa = self._scaleOnes((4, 1))
        sb = self._scaleOnes((4, 1))
        C = np.full((4, 4), 10.0, dtype=np.float32)
        D = gemmMx(A, B, sa, sb, blockK=32, beta=0.5, C=C)
        D_ref = gemmMx(A, B, sa, sb, blockK=32) + 0.5 * C
        np.testing.assert_allclose(D, D_ref, rtol=1e-6)


# ===========================================================================
# Task 4.4 - buildKernelArgs MX byte-layout unit test (no GPU)
# ===========================================================================


class TestBuildKernelArgsMx:
    """Verify that MX pointer and stride slots appear at the correct offsets."""

    def _makeSol(self, mxBlockA=0, mxBlockB=0):
        """Minimal solution dict for a stridedBatched GEMM with MX flags."""
        return {
            "KernArgsVersion": 2,
            "UseUniversalArgs": True,
            "UseSFC": False,
            "SupportCustomWGM": True,
            "SupportCustomStaggerU": False,
            "MacroTile0": 128,
            "MacroTile1": 128,
            "WorkGroupMapping": 4,
            "WorkGroupMappingXCC": 0,
            "WorkGroupMappingXCCGroup": 0,
            "StaggerU": 0,
            "StaggerUMapping": 0,
            "_staggerStrideShift": 0,
            "StreamK": 0,
            "StreamKAtomic": 0,
            "GlobalSplitU": 1,
            "GlobalAccumulation": 0,
            "GlobalSplitUCoalesced": False,
            "GlobalSplitUWorkGroupMappingRoundRobin": False,
            "StridedBatched": True,
            "UseBeta": True,
            "HighPrecisionAccumulate": True,
            "ComputeDataType": 0,
            "MXBlockA": mxBlockA,
            "MXBlockB": mxBlockB,
        }

    def _makeProb(self, kBlocks=8, M=256, N=256):
        """Minimal problem-param dict for a batched GEMM with MX strides."""
        K = kBlocks * 32
        return {
            "sizes": [M, N, 4, K],
            "ldd": M, "stride_d": M * N,
            "ldc": M, "stride_c": M * N,
            "lda": M, "stride_a": M * K,
            "ldb": N, "stride_b": N * K,
            "ld_mxsa": kBlocks, "stride_mxsa": M * kBlocks,
            "ld_mxsb": kBlocks, "stride_mxsb": N * kBlocks,
            "alpha": 1.0,
            "beta": 0.0,
        }

    def _makeTensors(self, useMx=False):
        t = {"D": 0x1000, "C": 0x2000, "A": 0x3000, "B": 0x4000}
        if useMx:
            t["mxsa"] = 0x5000
            t["mxsb"] = 0x6000
        return t

    def test_no_mx_base_length(self):
        """Without MX, buildKernelArgs produces 104 bytes for batched v2 GEMM.

        Layout: header(16) + sizes(16) + ptrs(32) + strides(32) + scalars(8).
        """
        raw = buildKernelArgs(self._makeSol(0, 0), self._makeProb(),
                              self._makeTensors(False))
        assert len(raw) == 104

    def test_mx_both_adds_pointers_and_strides(self):
        """With MXBlockA=32 and MXBlockB=32, buffer grows by 32 bytes.

        +2 ptr slots (mxsa, mxsb): +16 bytes.
        +4 stride u32s (ld_mxsa, stride_mxsa, ld_mxsb, stride_mxsb): +16 bytes.
        Total: 104 + 32 = 136 bytes.
        """
        raw = buildKernelArgs(self._makeSol(32, 32), self._makeProb(),
                              self._makeTensors(True))
        assert len(raw) == 136

    def test_mxsa_pointer_at_expected_offset(self):
        """mxsa pointer (0x5000) appears right after the A pointer.

        Pointer section starts at header(16) + sizes(16) = 32.
        D(8) + C(8) + A(8) = 24 bytes in -> mxsa at offset 32+24=56.
        """
        raw = buildKernelArgs(self._makeSol(32, 32), self._makeProb(),
                              self._makeTensors(True))
        mxsa_val = struct.unpack_from("<Q", raw, 56)[0]
        assert mxsa_val == 0x5000, f"mxsa: expected 0x5000, got 0x{mxsa_val:X}"

    def test_mxsb_pointer_at_expected_offset(self):
        """mxsb pointer (0x6000) appears right after the B pointer.

        After mxsa(8) comes B(8), then mxsb(8) -> offset 32+24+8+8=72.
        """
        raw = buildKernelArgs(self._makeSol(32, 32), self._makeProb(),
                              self._makeTensors(True))
        mxsb_val = struct.unpack_from("<Q", raw, 72)[0]
        assert mxsb_val == 0x6000, f"mxsb: expected 0x6000, got 0x{mxsb_val:X}"

    def test_ld_mxsa_at_expected_stride_offset(self):
        """ld_mxsa appears in the stride section right after A strides.

        With MX the pointer section is 48 bytes (D+C+A+mxsa+B+mxsb = 6*8).
        Strides start at header(16)+sizes(16)+ptrs(48)=80.
        D(8)+C(8)+A(8)=24 bytes in -> ld_mxsa at 80+24=104.
        """
        kBlocks = 8
        raw = buildKernelArgs(self._makeSol(32, 32), self._makeProb(kBlocks),
                              self._makeTensors(True))
        ld_mxsa = struct.unpack_from("<I", raw, 104)[0]
        assert ld_mxsa == kBlocks, f"ld_mxsa: expected {kBlocks}, got {ld_mxsa}"


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
    return assembler, isaInfoMap, DebugConfig()


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


def _compileMxSolutions(problemIdx: int):
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
        warnings.warn(f"could not compile MX solutions (problemIdx={problemIdx}): {exc}")
        return []

    compiled = []
    for sol, sid in sols:
        try:
            asm_str, kernel_name = _generateAsm(sol, assembler, debugConfig)
            hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
        except Exception as exc:
            import warnings
            warnings.warn(f"MX solution {sid} failed to compile: {exc}")
            continue
        raw_dict = dict(sol)
        sol_dict = _injectInternalArgsSupport(raw_dict, chip)
        compiled.append({
            "sol_dict": sol_dict, "raw_dict": raw_dict,
            "kernel_name": kernel_name, "hsaco": hsaco,
            "chip": chip, "sid": sid,
        })
    return compiled


def _filterMxSolution(entry: dict) -> bool:
    """Return True if solution passes the M4 MX kernel filter.

    Skips auto-WGM (WorkGroupMapping==0). MX kernels always have StaggerU=0,
    so we do not skip StaggerU=0 here (unlike the fp8 filter).
    """
    return entry["sol_dict"].get("WorkGroupMapping", 0) != 0


def _deviceCuCount() -> int:
    """Return device CU count (multiprocessor_count) for device 0."""
    if not haveDeps:
        return 0
    props = amdgpu_exec._runtime_module.hip_get_device_props(0)
    return int(props.get("multiprocessor_count", 0))


def _buildMxHeader(sol_dict: dict, M: int, N: int, batch: int) -> list:
    """Build the kernelArgs header for a stridedBatched MX kernel."""
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
    return args


def _buildTnMxF8Args(sol_dict: dict, M: int, N: int, batch: int, K: int,
                     D_arr, C_arr, A_arr, mxsa_arr, B_arr, mxsb_arr,
                     blockK: int = 32, alpha: float = 1.0, beta: float = 0.0) -> list:
    """Build typed args for TN stridedBatched MX F8->float32 GEMM.

    TN layout: lda = M (K stride over column-major A), ldb = N.
    Scale A: shape (K//blockK, M) per batch -> ld_mxsa = K//blockK.
    Scale B: shape (K//blockK, N) per batch -> ld_mxsb = K//blockK.
    """
    args = _buildMxHeader(sol_dict, M, N, batch)
    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])
    args.extend([D_arr, C_arr, A_arr, mxsa_arr, B_arr, mxsb_arr])

    kBlocks = K // blockK
    lda, ldb, ldd, ldc = M, N, M, M
    stride_a, stride_b, stride_d, stride_c = M * K, N * K, M * N, M * N
    ld_mxsa, stride_mxsa = kBlocks, M * kBlocks
    ld_mxsb, stride_mxsb = kBlocks, N * kBlocks
    args.extend([
        np.uint32(ldd), np.uint32(stride_d),
        np.uint32(ldc), np.uint32(stride_c),
        np.uint32(lda), np.uint32(stride_a),
        np.uint32(ld_mxsa), np.uint32(stride_mxsa),
        np.uint32(ldb), np.uint32(stride_b),
        np.uint32(ld_mxsb), np.uint32(stride_mxsb),
    ])
    args.extend([np.float32(alpha), np.float32(beta)])
    return args


def _buildTnMxF4Args(sol_dict: dict, M: int, N: int, batch: int, K: int,
                     D_arr, C_arr, A_arr, mxsa_arr, B_arr, mxsb_arr,
                     blockK: int = 32, alpha: float = 1.0, beta: float = 0.0) -> list:
    """Build typed args for TN stridedBatched MX F4->float32 GEMM.

    F4 uses lda = M (logical F4 elements per K-row). Scale strides identical to F8.
    """
    args = _buildMxHeader(sol_dict, M, N, batch)
    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])
    args.extend([D_arr, C_arr, A_arr, mxsa_arr, B_arr, mxsb_arr])

    kBlocks = K // blockK
    lda, ldb, ldd, ldc = M, N, M, M
    stride_a, stride_b, stride_d, stride_c = M * K, N * K, M * N, M * N
    ld_mxsa, stride_mxsa = kBlocks, M * kBlocks
    ld_mxsb, stride_mxsb = kBlocks, N * kBlocks
    args.extend([
        np.uint32(ldd), np.uint32(stride_d),
        np.uint32(ldc), np.uint32(stride_c),
        np.uint32(lda), np.uint32(stride_a),
        np.uint32(ld_mxsa), np.uint32(stride_mxsa),
        np.uint32(ldb), np.uint32(stride_b),
        np.uint32(ld_mxsb), np.uint32(stride_mxsb),
    ])
    args.extend([np.float32(alpha), np.float32(beta)])
    return args


def _allocMxF8Batched(M: int, N: int, K: int, batch: int, blockK: int, rng):
    """Allocate F8 input buffers and all-1 E8 scale buffers.

    A and B: float32 in [-0.5, 0.5], quantized to float8_e4m3fn, stored as
    column-major uint8 (TN layout: lda=M). Scales: 0x7F = 1.0 for all.
    """
    try:
        import ml_dtypes
        fp8Dtype = ml_dtypes.float8_e4m3fn
    except ImportError:
        pytest.skip("ml_dtypes not installed")

    valsA = rng.uniform(-0.5, 0.5, (M, K)).astype(np.float32).astype(fp8Dtype)
    valsB = rng.uniform(-0.5, 0.5, (N, K)).astype(np.float32).astype(fp8Dtype)
    A_np = np.asfortranarray(valsA)
    B_np = np.asfortranarray(valsB)
    rawA = np.frombuffer(A_np.ravel(order="F").tobytes(), dtype=np.uint8)
    rawB = np.frombuffer(B_np.ravel(order="F").tobytes(), dtype=np.uint8)
    A_buf = np.tile(rawA, batch)
    B_buf = np.tile(rawB, batch)

    kBlocks = K // blockK
    mxsa_buf = np.full(batch * M * kBlocks, 0x7F, dtype=np.uint8)
    mxsb_buf = np.full(batch * N * kBlocks, 0x7F, dtype=np.uint8)
    C_buf = np.zeros(M * N * batch, dtype=np.float32)
    D_buf = np.zeros(M * N * batch, dtype=np.float32)
    return A_buf, B_buf, mxsa_buf, mxsb_buf, C_buf, D_buf, A_np, B_np


def _allocMxF4Batched(M: int, N: int, K: int, batch: int, blockK: int, rng):
    """Allocate F4 input buffers (2 elements/byte) and all-1 E8 scale buffers.

    A_mem: (K, M//2) bytes per batch (TN F4 layout, lda=M logical elements).
    B_mem: (K, N//2) bytes per batch. Scales: 0x7F = 1.0.
    """
    rawA_one = rng.integers(0, 256, K * (M // 2), dtype=np.uint8)
    rawB_one = rng.integers(0, 256, K * (N // 2), dtype=np.uint8)
    A_buf = np.tile(rawA_one, batch)
    B_buf = np.tile(rawB_one, batch)

    kBlocks = K // blockK
    mxsa_buf = np.full(batch * M * kBlocks, 0x7F, dtype=np.uint8)
    mxsb_buf = np.full(batch * N * kBlocks, 0x7F, dtype=np.uint8)
    C_buf = np.zeros(M * N * batch, dtype=np.float32)
    D_buf = np.zeros(M * N * batch, dtype=np.float32)

    # Unpack to logical (M, K) and (N, K) for the Python reference.
    A_logical = unpackFloat4(rawA_one.reshape(K, M // 2)).T.astype(np.float32)
    B_logical = unpackFloat4(rawB_one.reshape(K, N // 2)).T.astype(np.float32)
    return A_buf, B_buf, mxsa_buf, mxsb_buf, C_buf, D_buf, A_logical, B_logical


def _runMxF8Batched(entry: dict, M: int, N: int, batch: int, K: int,
                    blockK: int, rtol: float, atol: float, label: str):
    """Execute one MX F8 TN stridedBatched kernel and verify against reference."""
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]
    rng = np.random.default_rng(seed=M * 1000 + N + K)

    A_buf, B_buf, mxsa_buf, mxsb_buf, C_buf, D_buf, A_np, B_np = \
        _allocMxF8Batched(M, N, K, batch, blockK, rng)

    result_holder: dict = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np.float32).copy()

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    mxsa_in = amdgpu_exec.InputArray(mxsa_buf)
    mxsb_in = amdgpu_exec.InputArray(mxsb_buf)

    args = _buildTnMxF8Args(sol_dict, M, N, batch, K,
                             D_io, C_in, A_in, mxsa_in, B_in, mxsb_in,
                             blockK=blockK)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    D_gpu = result_holder["D_gpu"]

    # All-1 scales -> plain float32 GEMM is the expected reference.
    kBlocks = K // blockK
    sa = np.full((M, kBlocks), 0x7F, dtype=np.uint8)
    sb = np.full((N, kBlocks), 0x7F, dtype=np.uint8)
    D_ref_one = gemmMx(A_np.astype(np.float32), B_np.astype(np.float32),
                       sa, sb, blockK=blockK)
    D_ref = np.tile(np.asfortranarray(D_ref_one).ravel(order="F"), batch)
    assertClose(D_gpu, D_ref, rtol=rtol, atol=atol, label=label)


def _runMxF4Batched(entry: dict, M: int, N: int, batch: int, K: int,
                    blockK: int, rtol: float, atol: float, label: str):
    """Execute one MX F4 TN stridedBatched kernel and verify against reference."""
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]
    rng = np.random.default_rng(seed=M * 1000 + N + K + 7)

    A_buf, B_buf, mxsa_buf, mxsb_buf, C_buf, D_buf, A_logical, B_logical = \
        _allocMxF4Batched(M, N, K, batch, blockK, rng)

    result_holder: dict = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np.float32).copy()

    D_io = amdgpu_exec.InOutArray(D_buf)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    mxsa_in = amdgpu_exec.InputArray(mxsa_buf)
    mxsb_in = amdgpu_exec.InputArray(mxsb_buf)

    args = _buildTnMxF4Args(sol_dict, M, N, batch, K,
                             D_io, C_in, A_in, mxsa_in, B_in, mxsb_in,
                             blockK=blockK)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    D_gpu = result_holder["D_gpu"]

    kBlocks = K // blockK
    sa = np.full((M, kBlocks), 0x7F, dtype=np.uint8)
    sb = np.full((N, kBlocks), 0x7F, dtype=np.uint8)
    D_ref_one = gemmMx(A_logical, B_logical, sa, sb, blockK=blockK)
    D_ref = np.tile(np.asfortranarray(D_ref_one).ravel(order="F"), batch)
    assertClose(D_gpu, D_ref, rtol=rtol, atol=atol, label=label)


# ---------------------------------------------------------------------------
# Session-scoped compiled solution fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mxF8Kernels():
    """Compile F8->float32 MX (E8 scale) TN solutions from YAML group 0."""
    return _compileMxSolutions(0)


@pytest.fixture(scope="session")
def mxF4Kernels():
    """Compile F4->float32 MX (E8 scale) TN solutions from YAML group 1."""
    return _compileMxSolutions(1)


# ---------------------------------------------------------------------------
# Task 4.5 - GPU correctness tests.
# ---------------------------------------------------------------------------


@requires_gfx950
@pytest.mark.parametrize("size", _mxProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _mxProblemSizes])
def test_mxfp8_f8s_tn_correctness(mxF8Kernels, size):
    """MX F8->float32 TN stridedBatched correctness with all-1 E8 scales."""
    if not haveDeps:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in mxF8Kernels if _filterMxSolution(e)]
    if not entries:
        pytest.skip("no MX F8 solution compiled on this GPU")
    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runMxF8Batched(entry, M, N, batch, K, _mxBlockK, rtolMx, atolMx,
                        label=f"mx-f8 M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _mxProblemSizes,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _mxProblemSizes])
def test_mxfp4_f4s_tn_correctness(mxF4Kernels, size):
    """MX F4->float32 TN stridedBatched correctness with all-1 E8 scales."""
    if not haveDeps:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in mxF4Kernels if _filterMxSolution(e)]
    if not entries:
        pytest.skip("no MX F4 solution compiled on this GPU")
    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        _runMxF4Batched(entry, M, N, batch, K, _mxBlockK, rtolMx, atolMx,
                        label=f"mx-f4 M{M}N{N}B{batch}K{K} {sid}")
