# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pure-numpy SwiGLU reference for global-split GEMM+SwiGLU validation.

The reference is geometry-free: it knows only M, K, N_gemm and the input
matrices.  All tile/wave/MT1/wg_n details are irrelevant here.

Global-split spec:
  D = (A.T @ B)[:, :n_out] columns are gate; [:, n_out:] columns are up.
  Output D[m, c] = up[m, c] * silu(gate[m, c])  for c in [0, n_out).
"""
import numpy as np


def swiglu_reference(a_kxm_f32, b_kxn_f32, alpha=1.0):
    """Compute the fused GEMM+SwiGLU reference output.

    Args:
        a_kxm_f32: Input A, shape (K, M), float32.
        b_kxn_f32: Input B, shape (K, N_gemm), float32.  N_gemm must be even.
        alpha:     Scalar multiplier applied before the gate/up split, float.

    Returns:
        Output D, shape (M, N_out) where N_out = N_gemm // 2, float32.
    """
    d = alpha * (a_kxm_f32.T @ b_kxn_f32)    # (M, N_gemm)
    n_out = d.shape[1] // 2
    gate = d[:, :n_out]
    up   = d[:, n_out:]
    silu = gate * (1.0 / (1.0 + np.exp(-gate.astype(np.float64))).astype(np.float32))
    return (up * silu).astype(np.float32)
