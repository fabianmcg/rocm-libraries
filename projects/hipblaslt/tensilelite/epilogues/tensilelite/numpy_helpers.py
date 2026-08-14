# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numpy input generation and reference helpers for epilogue kernels."""
import math
import struct as _struct

import ml_dtypes
import numpy as np


def randBf16(rng, shape, scale=0.1):
    """Draw uniform fp32 in [0, scale), cast to bfloat16."""
    return (rng.random(shape, dtype=np.float32) * scale).astype(ml_dtypes.bfloat16)


def randGamma(rng, n):
    """Draw gamma in [0.5, 1.5) as (fp32, bfloat16) pair."""
    g = rng.random(n, dtype=np.float32) + 0.5
    return g, g.astype(ml_dtypes.bfloat16)


def sideInputDtype(typeChar):
    """Map a PartialRMS side-input type char to its numpy dtype."""
    dtypes = {"b": ml_dtypes.bfloat16, "s": np.float32}
    key = str(typeChar).lower()
    if key not in dtypes:
        raise ValueError(f"unrecognised PartialRMS side-input type char: {typeChar!r}")
    return dtypes[key]


def partialSumSq(hEff, nHidden, mt0):
    """Compute per-MT0-tile Σx² over free0: returns (rows, ceil(nHidden/mt0)) f32."""
    nD = math.ceil(nHidden / mt0)
    out = np.zeros((hEff.shape[0], nD), dtype=np.float32)
    for t in range(nD):
        lo = t * mt0
        hi = min((t + 1) * mt0, nHidden)
        out[:, t] = np.sum(hEff[:, lo:hi] ** 2, axis=1)
    return out


def partialAmax(dEff, nHidden, mt0, fp8_max=448.0):
    """Per-MT0-tile amax over free0 of |D|, scaled by 1/fp8_max."""
    nD = math.ceil(nHidden / mt0)
    out = np.zeros((dEff.shape[0], nD), dtype=np.float32)
    for t in range(nD):
        lo = t * mt0
        hi = min((t + 1) * mt0, nHidden)
        out[:, t] = np.max(np.abs(dEff[:, lo:hi]), axis=1) / fp8_max
    return out


def rmsDenom(rowSumSq, invD, eps):
    """Per-row RMS denominator sqrt(invD * rowSumSq + eps) as (rows,) float32."""
    return np.sqrt(np.asarray(rowSumSq, dtype=np.float32) * invD + eps).astype(np.float32)


def swizzleMxScaleGfx950(scale2d):
    """Pre-swizzle an (mTiles, nTiles) e8m0 grid into the GFX950 device layout.

    Rows are padded to a multiple of 32 and cols to a multiple of 8; returns a
    flat uint8 buffer of length paddedRows*paddedCols whose byte order matches
    the epilogue's on-device swizzle (DGen::preSwizzleScalesGFX950).
    """
    scale2d = np.ascontiguousarray(scale2d, dtype=np.uint8)
    mTiles, nTiles = scale2d.shape
    paddedRows = ((mTiles + 31) // 32) * 32
    paddedCols = ((nTiles + 7) // 8) * 8
    padded = np.zeros((paddedRows, paddedCols), dtype=np.uint8)
    padded[:mTiles, :nTiles] = scale2d
    view = padded.reshape(paddedRows // 32, 2, 16, paddedCols // 8, 2, 4)
    view = view.transpose(0, 3, 5, 2, 4, 1)
    return np.ascontiguousarray(view).reshape(-1)


def tileQuantReference(dEff_f32, q0, q1, fp8Max=448.0):
    """Compute per-tile dynamic fp8 quantization reference outputs.

    Returns (quantScale, dFp8) where quantScale has shape [ceil(M/q0), ceil(N/q1)]
    float32 and dFp8 has the same shape as dEff_f32 in OCP e4m3.
    dEff_f32 is the f32 effective D before quantization (alpha already applied);
    the caller is responsible for passing alpha * (A @ B).
    """
    import math
    import numpy as np
    import ml_dtypes

    M, N = dEff_f32.shape
    mT = math.ceil(M / q0)
    nT = math.ceil(N / q1)
    scale = np.zeros((mT, nT), dtype=np.float32)
    out   = np.zeros((M, N),   dtype=np.float32)
    for ti in range(mT):
        for tj in range(nT):
            mStart, mEnd = ti * q0, min((ti + 1) * q0, M)
            nStart, nEnd = tj * q1, min((tj + 1) * q1, N)
            blk  = dEff_f32[mStart:mEnd, nStart:nEnd]
            amax = float(np.max(np.abs(blk))) if blk.size else 0.0
            scale[ti, tj] = amax / fp8Max
            if amax > 0:
                out[mStart:mEnd, nStart:nEnd] = blk * (fp8Max / amax)
    return scale, out.astype(ml_dtypes.float8_e4m3fn)


def _e8m0ScaleByte(amax, fp8Max=448.0):
    """Return (scaleByte, quantMult) for one MX block given its absolute maximum."""
    if amax == 0.0:
        return 0, 0.0
    scaleF = amax / fp8Max
    bits = _struct.unpack('<I', _struct.pack('<f', scaleF))[0]
    expByte = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    sb = max(0, min(254, expByte + (1 if mant != 0 else 0)))
    qExpField = max(1, min(254, 254 - sb))
    quantMult = _struct.unpack('<f', _struct.pack('<I', qExpField << 23))[0]
    return sb, quantMult


def mxfp8QuantReference(dEff_f32, q0=1, q1=32):
    """Compute per-block e8m0 MX dynamic fp8 quantization reference outputs.

    Returns (mxScale, dFp8) where mxScale is a flat uint8 array in the GFX950
    pre-swizzled layout and dFp8 has the same shape as dEff_f32 in OCP e4m3.
    dEff_f32 is the f32 effective D before quantization (alpha already applied).

    Orientation depends on q0/q1:
      q0=1, q1=32 (old path): 32-element blocks along axis-1 (free1/M_tokens).
        Scale shape (mT, nT) = (N_hidden, M_tokens/32); outer ti, inner tj.
      q0=32, q1=1 (new path): 32-element blocks along axis-0 (free0/N_hidden).
        Scale shape (nT, mT) = (M_tokens, N_hidden/32); outer tj, inner ti.
    """
    M, N = dEff_f32.shape
    mT = math.ceil(M / q0)
    nT = math.ceil(N / q1)
    out = np.zeros((M, N), dtype=np.float32)

    if q0 == 1 and q1 == 32:
        # Old path: block along axis-1 (free1), scale shape (mT, nT).
        scale = np.zeros((mT, nT), dtype=np.uint8)
        for ti in range(mT):
            for tj in range(nT):
                mStart, mEnd = ti * q0, min((ti + 1) * q0, M)
                nStart, nEnd = tj * q1, min((tj + 1) * q1, N)
                blk = dEff_f32[mStart:mEnd, nStart:nEnd]
                amax = float(np.max(np.abs(blk))) if blk.size else 0.0
                sb, quantMult = _e8m0ScaleByte(amax)
                scale[ti, tj] = sb
                if quantMult > 0.0:
                    out[mStart:mEnd, nStart:nEnd] = blk * quantMult
    else:
        # New path (q0=32, q1=1): block along axis-0 (free0), scale shape (nT, mT).
        scale = np.zeros((nT, mT), dtype=np.uint8)
        for tj in range(nT):
            for ti in range(mT):
                mStart, mEnd = ti * q0, min((ti + 1) * q0, M)
                nStart, nEnd = tj * q1, min((tj + 1) * q1, N)
                blk = dEff_f32[mStart:mEnd, nStart:nEnd]
                amax = float(np.max(np.abs(blk))) if blk.size else 0.0
                sb, quantMult = _e8m0ScaleByte(amax)
                scale[tj, ti] = sb
                if quantMult > 0.0:
                    out[mStart:mEnd, nStart:nEnd] = blk * quantMult

    dFp8 = out.astype(ml_dtypes.float8_e4m3fn)
    assert np.all(np.isfinite(dFp8.astype(np.float32))), \
        "mxfp8QuantReference: NaN in fp8 D (ceiling exponent should prevent overflow)"
    return swizzleMxScaleGfx950(scale), dFp8


def partialRmsMxfp8Reference(aRow, bRow, gammaBf16, mt0, q0, q1):
    """Combined K1 reference (free0=N_hidden, free1=M).

    Returns (sumsqRef, mxScaleRef, dFp8Ref):
      sumsqRef  : [M, ceil(N/mt0)] f32 = per-MT0-tile Σh1² (pre-gamma).
      mxScaleRef: flat uint8 GFX950 pre-swizzled scale buffer for gamma·h1 blocks.
      dFp8Ref   : [N, M] fp8 e4m3 = MX-quantized gamma·h1 (free0×free1 layout).

    gammaBf16 is cast bf16→f32 to match the GPU (which loads gamma as bf16).
    """
    h1 = np.asarray(aRow).astype(np.float32) @ np.asarray(bRow).astype(np.float32)
    gammaF = np.asarray(gammaBf16).astype(np.float32)
    sumsqRef = partialSumSq(h1, h1.shape[1], mt0)
    dOutT = (h1 * gammaF[np.newaxis, :]).T
    mxScaleRef, dFp8Ref = mxfp8QuantReference(dOutT, q0, q1)
    return sumsqRef, mxScaleRef, dFp8Ref


def rmsNormReference(aRow, bRow, gammaBf16, invD, eps):
    """End-to-end RMSNorm reference: bf16(A@B * gamma) / rms(A@B), float32 (M, nHidden).

    Reference: D = bf16(h1 * gamma) / sqrt(invD * Σ(h1²) + eps), where h1 = aRow @ bRow.
    """
    h1 = np.asarray(aRow).astype(np.float32) @ np.asarray(bRow).astype(np.float32)
    h1Gamma = (h1 * np.asarray(gammaBf16).astype(np.float32)[np.newaxis, :]).astype(
        ml_dtypes.bfloat16).astype(np.float32)
    denom = rmsDenom((h1 ** 2).sum(axis=1), invD, eps)
    return h1Gamma / denom[:, np.newaxis]
