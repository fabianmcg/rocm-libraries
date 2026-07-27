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
# bf16: machine epsilon ~7.8e-3; C++ client uses AlmostEqualTolerance_BFloat16=0.1
# applied as absDiff < 0.1*(|a|+|b|+1). RTOL=0.1 approximates that formula.
RTOL_BF16: float = 0.1
ATOL_BF16: float = 0.1
# fp16: machine epsilon ~9.8e-4.
RTOL_FP16: float = 1e-3
ATOL_FP16: float = 1e-3
# fp32: machine epsilon ~1.2e-7.
RTOL_FP32: float = 1e-5
ATOL_FP32: float = 1e-5
# int8: integer arithmetic is exact; zero tolerance.
RTOL_INT8: float = 0.0
ATOL_INT8: float = 0.0
# xf32: 10-bit mantissa (lower 13 of 23 bits zeroed); accumulation headroom.
RTOL_XF32: float = 1e-3
ATOL_XF32: float = 1e-3
# fp8/bf8: 3–4 mantissa bits; significant accumulation headroom needed.
RTOL_FP8: float = 0.1
ATOL_FP8: float = 0.1


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


def _mxBlockPartial(
    A_f: np.ndarray,
    B_f: np.ndarray,
    sa: np.ndarray,
    sb: np.ndarray,
    blockK: int,
) -> np.ndarray:
    """Accumulate block-scaled partial sums over all K-blocks.

    A_f: (M, K) float32, B_f: (N, K) float32.
    sa: (M, kBlocks) float32 decoded scales for A.
    sb: (N, kBlocks) float32 decoded scales for B.
    Returns (M, N) float32 accumulated result.
    """
    M, K = A_f.shape
    N = B_f.shape[0]
    kBlocks = K // blockK
    D = np.zeros((M, N), dtype=np.float32)
    for kb in range(kBlocks):
        kStart = kb * blockK
        partial = A_f[:, kStart:kStart + blockK] @ B_f[:, kStart:kStart + blockK].T
        D += partial * np.outer(sa[:, kb], sb[:, kb])
    return D


def gemmMx(
    A: np.ndarray,
    B: np.ndarray,
    scaleA: np.ndarray,
    scaleB: np.ndarray,
    blockK: int,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: np.ndarray | None = None,
) -> np.ndarray:
    """Reference MX block-scaled GEMM with E8 (UE8M0) scale tensors.

    A: (M, K) float32 — operand A already unpacked to logical float32.
    B: (N, K) float32 — operand B already unpacked to logical float32.
    scaleA: (M, K//blockK) uint8 — E8-encoded per-(row, K-block) scale for A.
    scaleB: (N, K//blockK) uint8 — E8-encoded per-(row, K-block) scale for B.
    blockK: number of K elements per MX scale block (e.g. 32 for gfx950).

    D[m, n] = alpha * sum_kb(sa[m,kb] * sb[n,kb] * sum_{k in kb}(A[m,k] * B[n,k])) + beta*C[m,n]

    Returns float32 D of shape (M, N).
    """
    from .mx_types import decodeE8

    A_f = np.asarray(A, dtype=np.float32)
    B_f = np.asarray(B, dtype=np.float32)
    sa = decodeE8(np.asarray(scaleA, dtype=np.uint8))  # (M, kBlocks)
    sb = decodeE8(np.asarray(scaleB, dtype=np.uint8))  # (N, kBlocks)
    D = _mxBlockPartial(A_f, B_f, sa, sb, blockK)
    D *= alpha
    if beta != 0.0 and C is not None:
        D += beta * np.asarray(C, dtype=np.float32)
    return D


# ---------------------------------------------------------------------------
# Epilogue activation arg-count table (mirrors Tensile/Activation.py extraArgs).
# 'all' and 'hipblaslt_all' use the max over all types (tanh=2).
# ---------------------------------------------------------------------------
_ACT_ARG_COUNT: dict[str, int] = {
    "none": 0,
    "relu": 0,
    "gelu": 0,
    "geluscaling": 1,
    "sigmoid": 0,
    "silu": 0,
    "dgelu": 0,
    "tanh": 2,
    "swish": 1,
    "all": 2,
    "hipblaslt_all": 2,
}


def applyBias(D: np.ndarray, bias: np.ndarray, biasSource: str) -> np.ndarray:
    """Add bias to D (float64). Returns a new array without mutating D.

    biasSource:
      "row"    → bias has shape (N,), broadcast over rows.
      "col"    → bias has shape (M,), broadcast over columns.
      "matrix" → bias has shape (M, N), elementwise add.
    """
    if biasSource == "row":
        return D + bias[None, :]
    if biasSource == "col":
        return D + bias[:, None]
    return D + bias


def applyActivation(
    D: np.ndarray,
    name: str,
    args: list | None = None,
) -> np.ndarray:
    """Apply a named activation elementwise in float64.

    Matches Reference.cpp:746-853 exactly. Returns a new float64 array.
    args is the list of activation scalar parameters (tanh: [a0, a1], swish: [beta]).
    """
    x = D.astype(np.float64)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name in ("gelu", "geluscaling"):
        k0 = 0.7978845608028654
        k1 = 0.044715
        inner = k0 * x * (1.0 + k1 * x * x)
        result = 0.5 * x * (1.0 + np.tanh(inner))
        if name == "geluscaling" and args:
            result = result * float(args[0])
        return result
    if name == "dgelu":
        k0, k1 = 0.0535161, 0.398942
        k2, k3 = 0.0356774, 0.797885
        p3 = x ** 3
        x1 = k0 * p3 + k1 * x
        xx = k2 * p3 + k3 * x
        x2 = 4.0 / (np.exp(-xx) + np.exp(xx)) ** 2
        return 0.5 * np.tanh(xx) + x1 * x2 + 0.5
    if name == "silu":
        return x / (1.0 + np.exp(-x))
    if name == "swish":
        beta = float(args[0]) if args else 1.0
        return x / (1.0 + np.exp(-x * beta))
    if name == "tanh":
        a0 = float(args[0]) if args else 1.0
        a1 = float(args[1]) if args and len(args) > 1 else 1.0
        return a1 * np.tanh(a0 * x)
    raise ValueError(f"unknown activation: {name!r}")


def applyScaleAb(
    A: np.ndarray,
    B: np.ndarray,
    scaleA,
    scaleB,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale A and B before matmul. scaleA/scaleB may be scalar or vector."""
    return A * scaleA, B * scaleB


def applyScaleCd(
    C: np.ndarray | None,
    D: np.ndarray,
    scaleC,
    scaleD,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Scale C and D elementwise. scaleC/scaleD may be scalar or vector."""
    scaledC = C * scaleC if C is not None else None
    return scaledC, D * scaleD


def applyScaleAlphaVec(
    D: np.ndarray,
    scaleVec: np.ndarray,
    factorDim: int,
) -> np.ndarray:
    """Scale D by a vector. factorDim=0: per-row, factorDim=1: per-column."""
    if factorDim == 0:
        return D * scaleVec[:, None]
    return D * scaleVec[None, :]


def computeAmaxD(D: np.ndarray) -> float:
    """Return the absolute maximum of D as a Python float."""
    return float(np.max(np.abs(D)))


def gemmGrouped(groups: list) -> list:
    """Reference for grouped GEMM: compute one gemm() per group dict.

    Each group dict must contain 'A', 'B', and optionally 'alpha', 'beta', 'C'.
    Returns a list of float64 result arrays, one per group.
    """
    results = []
    for g in groups:
        D = gemm(g["A"], g["B"], g.get("alpha", 1.0), g.get("beta", 0.0), g.get("C"))
        results.append(D)
    return results


def computeETensor(D: np.ndarray) -> np.ndarray:
    """Return a copy of D before any output cast (the pre-cast accumulator)."""
    return D.copy()


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
