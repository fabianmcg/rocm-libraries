# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pure-Python / NumPy GEMM reference and correctness helpers.

All functions are CPU-only and require only numpy. No GPU, no amdgpu_exec.

Tolerance constants are derived from each dtype's machine epsilon with
headroom for K-length accumulation. Cross-reference against the C++ client's
Reference.hpp:37-48 formula (|a-b| < tol*(|a|+|b|+1)) if a test is flaky.
"""

from __future__ import annotations

import numpy as np

# Tolerance constants — rtol/atol pairs per dtype.
# bf16: machine epsilon ~7.8e-3, headroom for accumulation errors.
RTOL_BF16: float = 2e-2
ATOL_BF16: float = 2e-2
# fp16: machine epsilon ~9.8e-4.
RTOL_FP16: float = 1e-3
ATOL_FP16: float = 1e-3
# fp32: machine epsilon ~1.2e-7.
RTOL_FP32: float = 1e-5
ATOL_FP32: float = 1e-5


def gemm(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: np.ndarray | None = None,
) -> np.ndarray:
    """Compute D = alpha * (A @ B) + beta * C in float64 precision.

    A: (M, K), B: (K, N), C: (M, N) or None (treated as zero).
    Returns D as float64 for maximum precision.
    """
    A_f = A.astype(np.float64)
    B_f = B.astype(np.float64)
    D = alpha * (A_f @ B_f)
    if beta != 0.0 and C is not None:
        D += beta * C.astype(np.float64)
    return D


def gemmFp16(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: np.ndarray | None = None,
) -> np.ndarray:
    """Reference GEMM for fp16 inputs, fp32 accumulation (HPA mode), fp16 output.

    Upcast A and B to float32, compute in float32, downcast to float16.
    This matches the GPU kernel's HPA accumulation behaviour.

    A: (M, K) fp16, B: (K, N) fp16, C: (M, N) fp16 or None.
    Returns D as float16.
    """
    A_f = A.astype(np.float32)
    B_f = B.astype(np.float32)
    D = alpha * (A_f @ B_f)
    if beta != 0.0 and C is not None:
        D += beta * C.astype(np.float32)
    return D.astype(np.float16)


def gemmBf16(
    A,
    B,
    alpha: float = 1.0,
    beta: float = 0.0,
    C=None,
):
    """Reference GEMM for bf16 inputs, fp32 accumulation (HPA mode), bf16 output.

    Uses ml_dtypes.bfloat16 for input/output; intermediate computation in float32.
    This matches the GPU kernel's HPA accumulation behaviour.

    A: (M, K) bfloat16, B: (K, N) bfloat16, C: (M, N) bfloat16 or None.
    Returns D as ml_dtypes.bfloat16.
    """
    try:
        import ml_dtypes
    except ImportError as exc:
        raise ImportError("ml_dtypes is required for gemmBf16") from exc

    A_f = np.asarray(A, dtype=np.float32)
    B_f = np.asarray(B, dtype=np.float32)
    D = alpha * (A_f @ B_f)
    if beta != 0.0 and C is not None:
        D += beta * np.asarray(C, dtype=np.float32)
    return D.astype(ml_dtypes.bfloat16)


def assertClose(
    gpu: np.ndarray,
    ref: np.ndarray,
    rtol: float,
    atol: float,
    label: str = "output",
) -> None:
    """Assert gpu ≈ ref within tolerance, reporting the worst offender on failure."""
    diff = np.abs(gpu.astype(np.float64) - ref.astype(np.float64))
    tol = atol + rtol * np.abs(ref.astype(np.float64))
    bad = np.where(diff > tol)
    if len(bad[0]) == 0:
        return
    idx0 = bad[0][0]
    # Handle both 1-D and 2-D arrays for reporting.
    if gpu.ndim == 2:
        idx1 = bad[1][0]
        got = gpu[idx0, idx1]
        expected = ref[idx0, idx1]
        loc = f"row={idx0}, col={idx1}"
    else:
        got = gpu[idx0]
        expected = ref[idx0]
        loc = f"idx={idx0}"
    max_abs = float(np.nanmax(diff))
    raise AssertionError(
        f"{label} mismatch: {len(bad[0])} elements out of tolerance. "
        f"max_abs={max_abs:.3e}, first bad {loc}: "
        f"gpu={float(got):.6f} ref={float(expected):.6f}"
    )
