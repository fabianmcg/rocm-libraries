# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MX block-scaled type decoders and unpackers.

Decoders: UE8M0 (E8) and E5M3 scale types.
Unpackers: Float4 (E2M1), Float6 E2M3, BFloat6 E3M2 packed element formats.

All public functions accept NumPy uint8 arrays and return float32 arrays.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Scale decoders
# ---------------------------------------------------------------------------


def decodeE8(scaleBytes: np.ndarray) -> np.ndarray:
    """Decode UE8M0 (E8) scale bytes to float32.

    Each byte encodes an unsigned exponent: value = 2^(byte - 127).
    Byte 0xFF is NaN (matches DataTypes_E8.hpp operator float).
    Byte 0x00 gives 2^-127 (smallest representable value, not a special case).
    """
    data = np.asarray(scaleBytes, dtype=np.uint8)
    # Suppress overflow for 0xFF: exp2(128) = inf, overridden by NaN below.
    with np.errstate(over="ignore"):
        result = np.exp2(data.astype(np.float32) - 127.0)
    return np.where(data == 0xFF, np.float32(np.nan), result)


def decodeE5m3(data: np.ndarray) -> np.ndarray:
    """Decode E5M3 unsigned-float bytes to float32.

    E5M3 is an unsigned 8-bit float: 5 exponent bits (bias 15), 3 mantissa bits,
    no sign bit.  Normal (exp > 0): 2^(exp-15) * (1 + mant/8).
    Subnormal (exp == 0): 2^(1-15) * (mant/8) = 2^-14 * (mant/8).
    Byte 0xFF is the only NaN (matches DataTypes_E5M3.hpp cast_from_uf8).
    Byte 0x00 decodes to zero (subnormal with mant=0).
    """
    data = np.asarray(data, dtype=np.uint8)
    i = data.astype(np.int32)
    exp = i >> 3
    mant = i & 0x7
    normal = np.exp2(np.where(exp > 0, (exp - 15).astype(np.float32), np.float32(-14.0)))
    coeff = np.where(exp > 0, 1.0 + mant / 8.0, mant / 8.0).astype(np.float32)
    result = normal * coeff
    return np.where(data == 0xFF, np.float32(np.nan), result).astype(np.float32)


# ---------------------------------------------------------------------------
# Float4 (E2M1) helpers and unpacker
# ---------------------------------------------------------------------------


def _decodeE2m1Array(nibbles: np.ndarray) -> np.ndarray:
    """Decode an array of 4-bit E2M1 values (0..15) to float32.

    Format: [s][e1][e0][m] — 1-bit sign, 2-bit exponent (bias 1), 1-bit mantissa.
    Subnormal (exp == 0): (-1)^sign * (mant * 0.5).
    Normal (exp > 0): (-1)^sign * 2^(exp-1) * (1 + mant * 0.5).
    E2M1 has no NaN or Inf — all 16 bit patterns are finite.
    """
    nibbles = np.asarray(nibbles, dtype=np.uint8)
    sign = (nibbles >> 3) & np.uint8(1)
    exp = (nibbles >> 1) & np.uint8(3)
    mant = nibbles & np.uint8(1)
    magnitude = np.where(
        exp == 0,
        mant.astype(np.float32) * 0.5,
        np.exp2(exp.astype(np.float32) - 1.0) * (1.0 + mant.astype(np.float32) * 0.5),
    )
    return np.where(sign == 1, -magnitude, magnitude).astype(np.float32)


def unpackFloat4(packed: np.ndarray) -> np.ndarray:
    """Unpack Float4 (E2M1) packed bytes to float32.

    Each input byte encodes two Float4 elements:
      low nibble  (bits [3:0]) → element at even index (2j)
      high nibble (bits [7:4]) → element at odd index  (2j+1)

    Input shape: (..., K//2) uint8.
    Output shape: (..., K) float32.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    low = packed & np.uint8(0xF)
    high = packed >> np.uint8(4)
    # Interleave: axis -1 pairs become [..., 2j] = low, [..., 2j+1] = high.
    interleaved = np.stack([_decodeE2m1Array(low), _decodeE2m1Array(high)], axis=-1)
    return interleaved.reshape(packed.shape[:-1] + (packed.shape[-1] * 2,))


# ---------------------------------------------------------------------------
# Float6 6-bit element unpacking helpers
# ---------------------------------------------------------------------------


def _unpack6bitFlat(packed: np.ndarray, nElements: int) -> np.ndarray:
    """Extract nElements 6-bit values from a flat byte array (little-endian bit order).

    Every 3 bytes encode 4 six-bit values; nElements must be a multiple of 4.
    Within each 3-byte group (b0, b1, b2):
      elem 4g+0 = b0[5:0]
      elem 4g+1 = b0[7:6] | (b1[3:0] << 2)
      elem 4g+2 = b1[7:4] | (b2[1:0] << 4)
      elem 4g+3 = b2[7:2]
    """
    nGroups = nElements // 4
    b = np.asarray(packed[: nGroups * 3], dtype=np.uint32).reshape(nGroups, 3)
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    e0 = b0 & 0x3F
    e1 = ((b0 >> 6) & 0x3) | ((b1 & 0xF) << 2)
    e2 = ((b1 >> 4) & 0xF) | ((b2 & 0x3) << 4)
    e3 = (b2 >> 2) & 0x3F
    return np.stack([e0, e1, e2, e3], axis=1).ravel().astype(np.uint8)


def _decodeFloat6E2m3Array(vals: np.ndarray) -> np.ndarray:
    """Decode an array of 6-bit E2M3 values to float32.

    Format: [s][e1][e0][m2][m1][m0] — 1-bit sign, 2-bit exp (bias 1), 3-bit mant.
    Normal (exp > 0): (-1)^s * 2^(exp-1) * (1 + mant/8).
    Subnormal (exp == 0): (-1)^s * (mant/8).
    """
    vals = np.asarray(vals, dtype=np.uint8)
    sign = (vals >> 5) & np.uint8(1)
    exp = (vals >> 3) & np.uint8(3)
    mant = vals & np.uint8(7)
    magnitude = np.where(
        exp == 0,
        mant.astype(np.float32) / 8.0,
        np.exp2(exp.astype(np.float32) - 1.0) * (1.0 + mant.astype(np.float32) / 8.0),
    )
    return np.where(sign == 1, -magnitude, magnitude).astype(np.float32)


def _decodeBfloat6E3m2Array(vals: np.ndarray) -> np.ndarray:
    """Decode an array of 6-bit E3M2 values to float32.

    Format: [s][e2][e1][e0][m1][m0] — 1-bit sign, 3-bit exp (bias 3), 2-bit mant.
    Normal (exp > 0): (-1)^s * 2^(exp-3) * (1 + mant/4).
    Subnormal (exp == 0): (-1)^s * 2^-2 * (mant/4).
    """
    vals = np.asarray(vals, dtype=np.uint8)
    sign = (vals >> 5) & np.uint8(1)
    exp = (vals >> 2) & np.uint8(7)
    mant = vals & np.uint8(3)
    magnitude = np.where(
        exp == 0,
        np.float32(0.25) * mant.astype(np.float32) / 4.0,
        np.exp2(exp.astype(np.float32) - 3.0) * (1.0 + mant.astype(np.float32) / 4.0),
    )
    return np.where(sign == 1, -magnitude, magnitude).astype(np.float32)


def unpackFloat6E2m3(packed: np.ndarray, nElements: int) -> np.ndarray:
    """Unpack Float6 E2M3 packed bytes to float32.

    nElements must be a multiple of 4; packed must have exactly nElements * 3 // 4 bytes.
    Elements are stored sequentially with 6-bit little-endian packing: every 3 bytes
    contain 4 elements.

    Input: flat uint8 of length nElements * 3 // 4.
    Output: float32 of length nElements.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    raw = _unpack6bitFlat(packed, nElements)
    return _decodeFloat6E2m3Array(raw)


def unpackBfloat6E3m2(packed: np.ndarray, nElements: int) -> np.ndarray:
    """Unpack BFloat6 E3M2 packed bytes to float32.

    nElements must be a multiple of 4; packed must have exactly nElements * 3 // 4 bytes.
    Elements are stored sequentially with 6-bit little-endian packing: every 3 bytes
    contain 4 elements.

    Input: flat uint8 of length nElements * 3 // 4.
    Output: float32 of length nElements.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    raw = _unpack6bitFlat(packed, nElements)
    return _decodeBfloat6E3m2Array(raw)
