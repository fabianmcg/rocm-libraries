# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M6 test suite: sparse 2:4 compression helpers (task 6.4).

All tests are CPU-only (pure NumPy). GPU sparse GEMM tests are skipped
pending a gfx950 sparse kernel YAML — see fixtures/m6_sparse_notes.txt.
"""

from __future__ import annotations

import numpy as np
import pytest

from Tensile.client.sparse import compress24, decompress24


# ===========================================================================
# Helpers
# ===========================================================================


def _make24Sparse(M: int, K: int, dtype, seed: int = 0) -> np.ndarray:
    """Return a random (M, K) matrix with exactly 2 non-zeros per group of 4.

    The non-zeros occupy positions 0 and 1 within each group; positions 2 and 3
    are zeroed. This guarantees a deterministic round-trip through compress24.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K // 4, 4)).astype(np.float32)
    A[:, :, 2] = 0.0
    A[:, :, 3] = 0.0
    return A.reshape(M, K).astype(dtype)


# ===========================================================================
# Task 6.4 — compress24 / decompress24 round-trip
# ===========================================================================


class TestCompress24:
    """Round-trip: decompress24(compress24(A)) reproduces the sparse input."""

    @pytest.mark.parametrize("shape,dtype", [
        ((256, 256), np.float32),
        ((512, 512), np.float32),
        ((256, 512), np.float32),
    ])
    def test_roundtrip(self, shape, dtype):
        M, K = shape
        A = _make24Sparse(M, K, dtype)
        compressed, metadata = compress24(A)
        reconstructed = decompress24(compressed, metadata, shape)
        np.testing.assert_array_equal(reconstructed, A)

    def test_compressed_shape(self):
        # K=128 → K//2=64 compressed columns; K//4=32 groups → (32+1)//2=16 meta bytes.
        A = _make24Sparse(64, 128, np.float32)
        compressed, metadata = compress24(A)
        assert compressed.shape == (64, 64)
        assert metadata.shape == (64, 16)
        assert metadata.dtype == np.uint8

    def test_compressed_shape_k4(self):
        # K=4 (odd K//4=1 group) → 1 meta byte with low nibble only.
        A = _make24Sparse(8, 4, np.float32)
        compressed, metadata = compress24(A)
        assert compressed.shape == (8, 2)
        assert metadata.shape == (8, 1)

    def test_metadata_dtype_is_uint8(self):
        A = _make24Sparse(32, 32, np.float32)
        _, metadata = compress24(A)
        assert metadata.dtype == np.uint8

    def test_compressed_dtype_matches_input(self):
        A = _make24Sparse(32, 32, np.float16)
        compressed, _ = compress24(A)
        assert compressed.dtype == np.float16

    def test_selects_largest_magnitude(self):
        """compress24 keeps the 2 values with the largest absolute value."""
        A = np.array([[3.0, -5.0, 1.0, 2.0]], dtype=np.float32)
        compressed, metadata = compress24(A)
        reconstructed = decompress24(compressed, metadata, A.shape)
        # Largest-magnitude: positions 1 (-5.0) and 0 (3.0); positions 2,3 zeroed.
        assert reconstructed[0, 0] == pytest.approx(3.0)
        assert reconstructed[0, 1] == pytest.approx(-5.0)
        assert reconstructed[0, 2] == 0.0
        assert reconstructed[0, 3] == 0.0

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            compress24(np.zeros((4, 4, 4), dtype=np.float32))

    def test_k_not_divisible_by_4_raises(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            compress24(np.zeros((4, 6), dtype=np.float32))


# ===========================================================================
# Task 6.4 — decompress24 from hand-crafted metadata
# ===========================================================================


class TestDecompress24FromKnown:
    """Verify decompress24 against manually computed expected outputs.

    Metadata nibble encoding: nibble = idx0 | (idx1 << 2)
    where idx0 < idx1 are 0-based positions (0-3) of the two non-zeros.
    Two nibbles packed per byte: byte = low_nibble | (high_nibble << 4).
    """

    def test_case_positions_01_and_23(self):
        """Group 0: positions (0,1); group 1: positions (2,3).

        nibble0 = 0 | (1<<2) = 4 = 0x4
        nibble1 = 2 | (3<<2) = 14 = 0xE
        byte = 0x4 | (0xE << 4) = 0xE4
        """
        compressed = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        metadata = np.array([[0xE4]], dtype=np.uint8)
        expected = np.array([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]], dtype=np.float32)
        result = decompress24(compressed, metadata, (1, 8))
        np.testing.assert_array_equal(result, expected)

    def test_case_positions_12_and_13(self):
        """Group 0: positions (1,2); group 1: positions (1,3).

        nibble0 = 1 | (2<<2) = 9 = 0x9
        nibble1 = 1 | (3<<2) = 13 = 0xD
        byte = 0x9 | (0xD << 4) = 0xD9
        """
        compressed = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        metadata = np.array([[0xD9]], dtype=np.uint8)
        expected = np.array([[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]], dtype=np.float32)
        result = decompress24(compressed, metadata, (1, 8))
        np.testing.assert_array_equal(result, expected)

    def test_case_two_rows(self):
        """Two-row matrix with different patterns per row.

        Row 0: byte = 0xE4  (same as test_case_positions_01_and_23)
        Row 1: nibble0 = 9 (positions 1,2); nibble1 = 12 (positions 0,3)
               nibble1: 0 | (3<<2) = 12 = 0xC
               byte = 0x9 | (0xC << 4) = 0xC9
        """
        compressed = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ], dtype=np.float32)
        metadata = np.array([[0xE4], [0xC9]], dtype=np.uint8)
        expected = np.array([
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0],
            [0.0, 5.0, 6.0, 0.0, 7.0, 0.0, 0.0, 8.0],
        ], dtype=np.float32)
        result = decompress24(compressed, metadata, (2, 8))
        np.testing.assert_array_equal(result, expected)

    def test_case_four_groups_one_row(self):
        """One row with 4 groups (16 columns, 2 metadata bytes).

        Group 0: positions (0,3) → nibble = 0 | (3<<2) = 12 = 0xC
        Group 1: positions (1,2) → nibble = 1 | (2<<2) = 9  = 0x9
        Group 2: positions (0,2) → nibble = 0 | (2<<2) = 8  = 0x8
        Group 3: positions (1,3) → nibble = 1 | (3<<2) = 13 = 0xD
        byte 0 = 0xC | (0x9 << 4) = 0x9C
        byte 1 = 0x8 | (0xD << 4) = 0xD8
        """
        compressed = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32
        )
        metadata = np.array([[0x9C, 0xD8]], dtype=np.uint8)
        expected = np.array(
            [[1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 7.0, 0.0, 8.0]],
            dtype=np.float32,
        )
        result = decompress24(compressed, metadata, (1, 16))
        np.testing.assert_array_equal(result, expected)

    def test_shape_mismatch_raises(self):
        compressed = np.zeros((2, 4), dtype=np.float32)
        metadata = np.zeros((2, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            decompress24(compressed, metadata, (4, 8))


# ===========================================================================
# GPU sparse tests — skipped, see fixtures/m6_sparse_notes.txt
# ===========================================================================


class TestSparseGpuGfx950:
    """GPU sparse GEMM correctness tests.

    Skipped: no gfx950 sparse-kernel YAML was located at M6-partial time.
    See Tensile/client/tests/fixtures/m6_sparse_notes.txt for details.
    """

    def test_gpu_sparse_placeholder(self):
        pytest.skip(
            "GPU sparse GEMM tests skipped — no gfx950 sparse kernel YAML "
            "found at M6-partial. See fixtures/m6_sparse_notes.txt."
        )
