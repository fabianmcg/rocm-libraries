# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numpy input generation and reference helpers for epilogue kernels."""
import math

import ml_dtypes
import numpy as np


def randBf16(rng, shape, scale=0.1):
    """Draw uniform fp32 in [0, scale), cast to bfloat16."""
    return (rng.random(shape, dtype=np.float32) * scale).astype(ml_dtypes.bfloat16)


def randGamma(rng, n):
    """Draw gamma in [0.5, 1.5) as (fp32, bfloat16) pair."""
    g = rng.random(n, dtype=np.float32) + 0.5
    return g, g.astype(ml_dtypes.bfloat16)


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


def rmsNormReference(aRow, bRow, gammaBf16, invD, eps):
    """End-to-end RMSNorm reference: bf16(A@B * gamma) / rms(A@B), float32 (M, nHidden).

    Reference: D = bf16(h1 * gamma) / sqrt(invD * Σ(h1²) + eps), where h1 = aRow @ bRow.
    """
    h1 = np.asarray(aRow).astype(np.float32) @ np.asarray(bRow).astype(np.float32)
    h1Gamma = (h1 * np.asarray(gammaBf16).astype(np.float32)[np.newaxis, :]).astype(
        ml_dtypes.bfloat16).astype(np.float32)
    denom = rmsDenom((h1 ** 2).sum(axis=1), invD, eps)
    return h1Gamma / denom[:, np.newaxis]
