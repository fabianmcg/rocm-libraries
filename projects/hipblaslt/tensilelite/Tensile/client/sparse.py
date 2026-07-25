# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""AMD 2:4 structured-sparsity compression helpers.

Implements the same metadata byte layout as DataInitialization.cpp:compressSparseArray.
Each group of 4 elements along the K dimension contributes a 4-bit metadata nibble:
  meta_nibble = idx0 | (idx1 << 2)
where idx0 < idx1 are the 0-based positions (0–3) of the two selected elements.
Two nibbles are packed per byte (low nibble = first group, high nibble = second group).
"""

from __future__ import annotations

import numpy as np


def compress24(A: np.ndarray) -> tuple:
    """Compress a dense matrix to AMD 2:4 sparse format.

    Selects the 2 largest-magnitude elements per group of 4 along axis 1.
    Returns (compressed, metadata) where compressed has shape (M, K//2) and
    metadata has shape (M, ceil(K//4 / 2)) = (M, (K//4+1)//2) with dtype uint8.

    A must have shape (M, K) with K divisible by 4.
    """
    if A.ndim != 2:
        raise ValueError("compress24 requires a 2-D input array")
    M, K = A.shape
    if K % 4 != 0:
        raise ValueError(f"K={K} must be divisible by 4 for 2:4 sparsity")

    groups = A.reshape(M, K // 4, 4)

    # Indices of the 2 largest-magnitude elements per group, kept in ascending order.
    mag_order = np.argsort(np.abs(groups), axis=-1, kind="stable")
    top2 = np.sort(mag_order[:, :, 2:], axis=-1)  # (M, K//4, 2)

    meta_idx0 = top2[:, :, 0]  # (M, K//4)
    meta_idx1 = top2[:, :, 1]  # (M, K//4)

    m_idx = np.arange(M)[:, None]
    g_idx = np.arange(K // 4)[None, :]
    val0 = groups[m_idx, g_idx, meta_idx0]  # (M, K//4)
    val1 = groups[m_idx, g_idx, meta_idx1]  # (M, K//4)

    # Interleave: compressed[:, 2*g] = val0[:, g], compressed[:, 2*g+1] = val1[:, g].
    compressed = np.stack([val0, val1], axis=-1).reshape(M, K // 2)

    # Pack 4-bit nibbles: low nibble = first group in each pair, high = second.
    # When K//4 is odd the last byte carries only a low nibble; pad with a zero.
    meta_nibbles = (meta_idx0 | (meta_idx1 << 2)).astype(np.uint8)  # (M, K//4)
    num_groups = K // 4
    num_bytes = (num_groups + 1) // 2
    if num_groups % 2 == 1:
        meta_nibbles = np.concatenate(
            [meta_nibbles, np.zeros((M, 1), dtype=np.uint8)], axis=1
        )
    meta_pairs = meta_nibbles.reshape(M, num_bytes, 2)
    metadata = (meta_pairs[:, :, 0] | (meta_pairs[:, :, 1] << np.uint8(4))).astype(np.uint8)

    return compressed, metadata


def decompress24(
    compressed: np.ndarray,
    metadata: np.ndarray,
    shape: tuple,
) -> np.ndarray:
    """Reconstruct a dense matrix from AMD 2:4 sparse format.

    compressed: (M, K//2) array of non-zero values.
    metadata:   (M, K//8) uint8 array of packed 4-bit index nibbles.
    shape:      (M, K) output shape.

    Returns a dense array of dtype matching compressed, with zeros in the
    positions not covered by the two selected indices per group of 4.
    """
    M, K = shape
    num_meta_bytes = (K // 4 + 1) // 2
    if compressed.shape != (M, K // 2):
        raise ValueError(
            f"compressed shape {compressed.shape} does not match shape {shape}"
        )
    if metadata.shape != (M, num_meta_bytes):
        raise ValueError(
            f"metadata shape {metadata.shape} does not match shape {shape}"
        )

    # Unpack two nibbles per byte into per-group nibbles.
    # When K//4 is odd the last byte has only a low nibble (high nibble is zero).
    num_groups = K // 4
    num_pairs = num_groups // 2
    meta_nibbles = np.zeros((M, num_groups), dtype=np.uint8)
    meta_nibbles[:, 0::2] = metadata & np.uint8(0xF)
    if num_pairs > 0:
        meta_nibbles[:, 1::2] = (metadata[:, :num_pairs] >> np.uint8(4)) & np.uint8(0xF)

    meta_idx0 = (meta_nibbles & np.uint8(0x3)).astype(int)
    meta_idx1 = ((meta_nibbles >> np.uint8(2)) & np.uint8(0x3)).astype(int)

    comp_groups = compressed.reshape(M, K // 4, 2)
    dense_groups = np.zeros((M, K // 4, 4), dtype=compressed.dtype)

    m_idx = np.arange(M)[:, None]
    g_idx = np.arange(K // 4)[None, :]

    dense_groups[m_idx, g_idx, meta_idx0] = comp_groups[:, :, 0]
    dense_groups[m_idx, g_idx, meta_idx1] = comp_groups[:, :, 1]

    return dense_groups.reshape(shape)
