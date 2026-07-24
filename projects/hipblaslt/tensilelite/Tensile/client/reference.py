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


def gemmInt8(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: np.ndarray | None = None,
    outputInt8: bool = False,
) -> np.ndarray:
    """Reference GEMM for int8 inputs with int32 accumulation (HPA mode).

    Widen A and B to int32, compute in int32. When outputInt8=True, apply
    round-half-to-even (matching std::nearbyint) then saturate to [-128, 127].
    The float64 intermediate ensures correct banker's rounding regardless of
    the accumulation magnitude.

    A: (M, K) int8, B: (K, N) int8, C: (M, N) int32 or int8 or None.
    Returns D as int32 or int8 depending on outputInt8.
    """
    A32 = A.astype(np.int32)
    B32 = B.astype(np.int32)
    D = alpha * (A32 @ B32)
    if beta != 0.0 and C is not None:
        D += beta * C.astype(np.int32)
    if not outputInt8:
        return D.astype(np.int32)
    # float64 intermediate for round-half-to-even matching std::nearbyint.
    D64 = D.astype(np.float64)
    return np.clip(np.round(D64), -128, 127).astype(np.int8)


def toXf32(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to XFloat32 by zeroing the lower 13 mantissa bits.

    Matches DataTypes_XFloat32.hpp float_to_XFloat32: u.p &= 0xFFFFE000.
    The input must be float32; the output is float32 with truncated mantissa.
    """
    arr_c = np.ascontiguousarray(arr, dtype=np.float32)
    return (arr_c.view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32)


def gemmFp8(
    A,
    B,
    dtypeA,
    dtypeB,
    dtypeOut,
    alpha: float = 1.0,
    beta: float = 0.0,
    C=None,
) -> np.ndarray:
    """Reference GEMM for fp8 inputs with float32 accumulation (HPA mode).

    Upcast A and B to float32 via ml_dtypes, compute in float32, downcast
    to dtypeOut. This matches the GPU kernel's fp8 HPA behaviour.

    A: (M, K) fp8 array (dtypeA is an ml_dtypes fp8 dtype).
    B: (K, N) fp8 array (dtypeB is an ml_dtypes fp8 dtype).
    C: (M, N) array with dtype dtypeOut, or None.
    dtypeOut: output numpy dtype (e.g. np.float32).
    Returns D as dtypeOut.
    """
    A_f = np.asarray(A, dtype=dtypeA).astype(np.float32)
    B_f = np.asarray(B, dtype=dtypeB).astype(np.float32)
    D = alpha * (A_f @ B_f)
    if beta != 0.0 and C is not None:
        D += beta * np.asarray(C, dtype=np.float32)
    return D.astype(dtypeOut)


def gemmXf32(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: np.ndarray | None = None,
) -> np.ndarray:
    """Reference GEMM for XFloat32 inputs with float32 accumulation.

    Apply toXf32 to A and B (truncate mantissa), then compute in float32.
    This matches the GPU kernel's XF32 math-op behaviour.

    A: (M, K) float32, B: (K, N) float32, C: (M, N) float32 or None.
    Returns D as float32.
    """
    A_xf = toXf32(np.asarray(A, dtype=np.float32))
    B_xf = toXf32(np.asarray(B, dtype=np.float32))
    D = alpha * (A_xf.astype(np.float32) @ B_xf.astype(np.float32))
    if beta != 0.0 and C is not None:
        D += beta * np.asarray(C, dtype=np.float32)
    return D.astype(np.float32)


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
