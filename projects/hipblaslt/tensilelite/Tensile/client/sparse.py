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


def _packNibbles(metaIdx0: np.ndarray, metaIdx1: np.ndarray, M: int, numGroups: int) -> np.ndarray:
    """Pack per-group 4-bit metadata into bytes (2 nibbles per byte, low then high)."""
    numBytes = (numGroups + 1) // 2
    nibbles = (metaIdx0 | (metaIdx1 << 2)).astype(np.uint8)
    if numGroups % 2 == 1:
        nibbles = np.concatenate([nibbles, np.zeros((M, 1), dtype=np.uint8)], axis=1)
    pairs = nibbles.reshape(M, numBytes, 2)
    return (pairs[:, :, 0] | (pairs[:, :, 1] << np.uint8(4))).astype(np.uint8)


def _unpackNibbles(metadata: np.ndarray, M: int, numGroups: int) -> np.ndarray:
    """Unpack bytes into per-group 4-bit nibbles (2 nibbles per byte, low then high)."""
    numPairs = numGroups // 2
    nibbles = np.zeros((M, numGroups), dtype=np.uint8)
    nibbles[:, 0::2] = metadata & np.uint8(0xF)
    if numPairs > 0:
        nibbles[:, 1::2] = (metadata[:, :numPairs] >> np.uint8(4)) & np.uint8(0xF)
    return nibbles


def compress24(A: np.ndarray) -> tuple:
    """Compress a dense matrix to AMD 2:4 sparse format.

    Selects the 2 largest-magnitude elements per group of 4 along axis 1.
    Returns (compressed, metadata) where compressed has shape (M, K//2) and
    metadata has shape (M, (K//4+1)//2) with dtype uint8.
    A must have shape (M, K) with K divisible by 4.
    """
    if A.ndim != 2:
        raise ValueError("compress24 requires a 2-D input array")
    M, K = A.shape
    if K % 4 != 0:
        raise ValueError(f"K={K} must be divisible by 4 for 2:4 sparsity")
    groups = A.reshape(M, K // 4, 4)
    # Keep the 2 largest-magnitude elements per group, in ascending index order.
    magOrder = np.argsort(np.abs(groups), axis=-1, kind="stable")
    top2 = np.sort(magOrder[:, :, 2:], axis=-1)
    mIdx = np.arange(M)[:, None]
    gIdx = np.arange(K // 4)[None, :]
    val0 = groups[mIdx, gIdx, top2[:, :, 0]]
    val1 = groups[mIdx, gIdx, top2[:, :, 1]]
    compressed = np.stack([val0, val1], axis=-1).reshape(M, K // 2)
    metadata = _packNibbles(top2[:, :, 0], top2[:, :, 1], M, K // 4)
    return compressed, metadata


def decompress24(compressed: np.ndarray, metadata: np.ndarray, shape: tuple) -> np.ndarray:
    """Reconstruct a dense matrix from AMD 2:4 sparse format.

    compressed: (M, K//2) array of non-zero values.
    metadata:   (M, (K//4+1)//2) uint8 array of packed 4-bit index nibbles.
    shape:      (M, K) output shape.
    Returns a dense array with zeros in positions not selected by the metadata.
    """
    M, K = shape
    numMetaBytes = (K // 4 + 1) // 2
    if compressed.shape != (M, K // 2):
        raise ValueError(f"compressed shape {compressed.shape} does not match shape {shape}")
    if metadata.shape != (M, numMetaBytes):
        raise ValueError(f"metadata shape {metadata.shape} does not match shape {shape}")
    nibbles = _unpackNibbles(metadata, M, K // 4)
    metaIdx0 = (nibbles & np.uint8(0x3)).astype(int)
    metaIdx1 = ((nibbles >> np.uint8(2)) & np.uint8(0x3)).astype(int)
    compGroups = compressed.reshape(M, K // 4, 2)
    denseGroups = np.zeros((M, K // 4, 4), dtype=compressed.dtype)
    mIdx = np.arange(M)[:, None]
    gIdx = np.arange(K // 4)[None, :]
    denseGroups[mIdx, gIdx, metaIdx0] = compGroups[:, :, 0]
    denseGroups[mIdx, gIdx, metaIdx1] = compGroups[:, :, 1]
    return denseGroups.reshape(shape)
