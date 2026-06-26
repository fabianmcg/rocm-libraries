# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the K2 auxiliary reduction kernel (gfx950, fp32).

Exercises a range of (M, N_tiles_N, N_hidden, eps) shapes, verifying:
  rstdBuf[m] = rsqrt((Σ_t partialBuf[m,t]) / N_hidden + eps)   for m in [0, M)

partialBuf is 2D [M_padded, N_tiles_N] (row-major fp32).
One wave (64 lanes) per row; 6-stage ds_bpermute butterfly reduces N_tiles_N
partials per row. N_hidden is a runtime divisor (not baked into the binary).

Two test modes:
  1. Unit: synthetic random 2D partialBuf, compare GPU vs reference fp32.
  2. Chained: use the per-tile sumsq_ref from K1 as partialBuf input.

One session-scoped fixture compiles a single binary (no params — N_hidden is runtime).
"""

import math
import os
import sys

import numpy as np
import pytest

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if TENSILE_ROOT not in sys.path:
    sys.path.insert(0, TENSILE_ROOT)

try:
    import amdgpu_exec
    _HAVE_DEPS = True
except ImportError:
    _HAVE_DEPS = False

requires_gfx950 = pytest.mark.skipif(
    not _HAVE_DEPS or not (lambda: amdgpu_exec.get_chip().startswith("gfx950"))(),
    reason="requires amdgpu_exec and a gfx950 GPU",
)

# ---------------------------------------------------------------------------
# Test parameters: (M, N_tiles_N, N_hidden, eps, label)
# N_tiles_N covers 1 (trivial), 4, 8, 32, 64 (exact wave), 128 (multi-iter), 256.
# ---------------------------------------------------------------------------

_SHAPES = [
    # (M, N_tiles_N, N_hidden, eps, label)
    (    1,   1,    64, 1e-5, "M1_Nt1_N64"),
    (    4,   1,    64, 1e-5, "M4_Nt1_N64"),
    (   16,   4,   256, 1e-5, "M16_Nt4_N256"),
    (   64,   4,   256, 1e-5, "M64_Nt4_N256"),
    (  256,   8,   512, 1e-5, "M256_Nt8_N512"),
    ( 1024,  32,  2048, 1e-5, "M1024_Nt32_N2048"),
    (  128,  64,  4096, 1e-5, "M128_Nt64_N4096"),
    (  256, 128,  8192, 1e-5, "M256_Nt128_N8192"),
    (   32, 256, 16384, 1e-5, "M32_Nt256_N16384"),
    # Non-multiple N_hidden: N_tiles_N*64 != N_hidden (exercises runtime divisor).
    (   64,   4,   200, 1e-5, "M64_Nt4_N200_nonmul"),
    # Small eps.
    (  256,   4,   256, 1e-6, "M256_Nt4_N256_eps1e6"),
]

_M_VALUES_SHAPE = [1, 16, 255, 256, 257, 1024, 4093]
_EPS_VALUES = [1e-5, 1e-6]


# ---------------------------------------------------------------------------
# Session-scoped fixture: compile one binary (no N_hidden param)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def k2_kernel():
    """Build and compile the single K2 aux-reduction kernel binary."""
    sys.path.insert(0, TENSILE_ROOT)
    from tensile_gemm_rmsnorm_gemm_example import build_aux_reduction_asm

    chip = amdgpu_exec.get_chip()
    _asm_str, kernel_name, hsaco = build_aux_reduction_asm(chip)
    return kernel_name, hsaco, chip


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _rstd_ref(partial_2d: np.ndarray, N_hidden: int, eps: float) -> np.ndarray:
    """Scalar fp32 reference: sum partials per row, then rsqrt."""
    totals = partial_2d.sum(axis=1).astype(np.float64)
    return (1.0 / np.sqrt(totals / N_hidden + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: run K2 and return (rstd_gpu, rstd_ref)
# ---------------------------------------------------------------------------

def _run_k2(kernel_name: str, hsaco: bytes,
            M: int, N_tiles_N: int, N_hidden: int, eps: float,
            partial_buf_2d: np.ndarray):
    """Execute K2 and return (rstd_gpu[M_padded], ref[M])."""
    M_padded = math.ceil(M / 4) * 4  # ROWS_PER_WG = 4
    partial_padded = np.zeros((M_padded, N_tiles_N), dtype=np.float32, order='C')
    partial_padded[:M, :] = partial_buf_2d[:M, :]
    partial_flat = np.ascontiguousarray(partial_padded)

    rstd_buf = np.zeros(M_padded, dtype=np.float32)

    args = [
        amdgpu_exec.InputArray(partial_flat),
        amdgpu_exec.InOutArray(rstd_buf),
        np.uint32(M),
        np.uint32(N_tiles_N),
        np.uint32(N_hidden),
        np.float32(eps),
    ]

    result_holder = {}

    def capture(arguments):
        result_holder["rstd_gpu"] = np.asarray(arguments[1].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(math.ceil(M / 4), 1, 1),
        block_dim=(256, 1, 1),
        num_iterations=1,
        verify_fn=capture,
    )

    rstd_gpu = result_holder["rstd_gpu"]
    ref = _rstd_ref(partial_buf_2d[:M, :], N_hidden, eps)
    return rstd_gpu, ref


# ---------------------------------------------------------------------------
# Unit tests: synthetic 2D partialBuf
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("M,N_tiles_N,N_hidden,eps,label", _SHAPES,
                          ids=[s[4] for s in _SHAPES])
def test_k2_shape(k2_kernel, M, N_tiles_N, N_hidden, eps, label):
    """Verify K2 rstdBuf for each (M, N_tiles_N, N_hidden, eps) shape."""
    kernel_name, hsaco, chip = k2_kernel

    rng = np.random.default_rng(seed=M * 100 + N_tiles_N)
    partial_2d = rng.random((M, N_tiles_N), dtype=np.float32) * 10.0 + 1e-3

    rstd_gpu, ref = _run_k2(kernel_name, hsaco, M, N_tiles_N, N_hidden, eps, partial_2d)

    rtol, atol = 1e-4, 1e-4
    diff = np.abs(rstd_gpu[:M] - ref)
    tol  = atol + rtol * np.abs(ref)
    bad  = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
    n_bad = len(bad)
    assert n_bad == 0, (
        f"rstdBuf mismatch: M={M} N_tiles_N={N_tiles_N} N_hidden={N_hidden} eps={eps} ({label}): "
        f"{n_bad} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(rstd_gpu[:M] - ref)):.3e}, "
        f"first bad row={bad[0]}: "
        f"gpu={rstd_gpu[bad[0]]:.6f} ref={ref[bad[0]]:.6f}"
    )


@requires_gfx950
def test_k2_edge_values(k2_kernel):
    """Verify K2 handles edge values: zero, very large, very small."""
    kernel_name, hsaco, chip = k2_kernel

    N_tiles_N = 1
    N_hidden  = 64
    eps       = 1e-5
    M         = 8

    partial_2d = np.array(
        [[0.0], [1e-6], [0.1], [1.0], [10.0], [100.0], [1e4], [1e6]],
        dtype=np.float32,
    )
    rstd_gpu, ref = _run_k2(kernel_name, hsaco, M, N_tiles_N, N_hidden, eps, partial_2d)

    rtol, atol = 1e-4, 1e-4
    diff = np.abs(rstd_gpu[:M] - ref)
    tol  = atol + rtol * np.abs(ref)
    bad  = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
    assert len(bad) == 0, (
        f"edge-value mismatch: N_tiles_N={N_tiles_N} N_hidden={N_hidden} eps={eps}: "
        f"bad={bad.tolist()}"
    )


@requires_gfx950
def test_k2_bounds_guard(k2_kernel):
    """Waves with wave_row >= M must not write to rstdBuf.

    Use M=97 (not a multiple of 4 = ROWS_PER_WG), fill rstdBuf with sentinel,
    and verify entries at [M, M_padded) are untouched.
    """
    kernel_name, hsaco, chip = k2_kernel

    M         = 97   # not a multiple of 4
    N_tiles_N = 1
    N_hidden  = 64
    eps       = 1e-5
    M_padded  = math.ceil(M / 4) * 4

    sentinel    = np.float32(-999.0)
    partial_2d  = np.ones((M_padded, N_tiles_N), dtype=np.float32) * 2.0
    rstd_buf    = np.full(M_padded, sentinel, dtype=np.float32)
    partial_flat = np.ascontiguousarray(partial_2d)

    args = [
        amdgpu_exec.InputArray(partial_flat),
        amdgpu_exec.InOutArray(rstd_buf),
        np.uint32(M),
        np.uint32(N_tiles_N),
        np.uint32(N_hidden),
        np.float32(eps),
    ]

    result_holder = {}

    def capture(arguments):
        result_holder["rstd_gpu"] = np.asarray(arguments[1].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(math.ceil(M / 4), 1, 1),
        block_dim=(256, 1, 1),
        num_iterations=1,
        verify_fn=capture,
    )

    rstd_gpu = result_holder["rstd_gpu"]

    # Rows [M, M_padded) must retain the sentinel.
    tail = rstd_gpu[M:M_padded]
    corrupted = np.where(tail != sentinel)[0]
    assert len(corrupted) == 0, (
        f"bounds guard failed: M={M} M_padded={M_padded}: "
        f"{len(corrupted)} entries beyond M were written, "
        f"first at row {M + corrupted[0]}"
    )

    # Rows [0, M) must be valid.
    ref = _rstd_ref(partial_2d[:M, :], N_hidden, eps)
    rtol, atol = 1e-4, 1e-4
    diff = np.abs(rstd_gpu[:M] - ref)
    tol  = atol + rtol * np.abs(ref)
    bad  = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
    assert len(bad) == 0, (
        f"valid-row mismatch: M={M} N_tiles_N={N_tiles_N}: bad={bad.tolist()}"
    )


# ---------------------------------------------------------------------------
# Chained test: use K1 reference per-tile sumsq as partialBuf input
# ---------------------------------------------------------------------------

_K1_SHAPES = [
    (64,   64,  1,  64,  "M64_N64_K64_Nt1"),
    (128,  64,  1, 128,  "M128_N64_K128_Nt1"),
    (256, 256,  4, 4096, "M256_N256_K4096_Nt4"),
]


@requires_gfx950
@pytest.mark.parametrize("M,N,N_tiles_N,K,label", _K1_SHAPES, ids=[s[4] for s in _K1_SHAPES])
def test_k2_chained_with_k1(k2_kernel, M, N, N_tiles_N, K, label):
    """Feed K1 reference per-tile sumsq_ref into K2 and compare against double ref."""
    kernel_name, hsaco, chip = k2_kernel

    try:
        import ml_dtypes
    except ImportError:
        pytest.skip("ml_dtypes not available")

    eps = 1e-5
    MT1 = N // N_tiles_N  # tile width
    rng = np.random.default_rng(seed=M * 10000 + K)
    a_f32  = np.asfortranarray(rng.random((K, M), dtype=np.float32) * 0.1)
    w0_f32 = np.asfortranarray(rng.random((K, N), dtype=np.float32) * 0.1)
    a_bf16  = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(w0_f32.astype(ml_dtypes.bfloat16))

    a_ref  = np.asarray(a_bf16).astype(np.float32)
    w0_ref = np.asarray(w0_bf16).astype(np.float32)
    h1     = a_ref.T @ w0_ref    # M x N, fp32

    # Build 2D per-tile partial sums.
    sumsq_2d = np.zeros((M, N_tiles_N), dtype=np.float32)
    for t in range(N_tiles_N):
        col_lo = t * MT1
        col_hi = min((t + 1) * MT1, N)
        sumsq_2d[:, t] = np.sum(h1[:, col_lo:col_hi] ** 2, axis=1)

    rstd_gpu, ref = _run_k2(kernel_name, hsaco, M, N_tiles_N, N, eps, sumsq_2d)

    rtol, atol = 1e-4, 1e-4
    diff = np.abs(rstd_gpu[:M] - ref)
    tol  = atol + rtol * np.abs(ref)
    bad  = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
    assert len(bad) == 0, (
        f"chained K1→K2 mismatch: M={M} N={N} N_tiles_N={N_tiles_N} K={K} ({label}): "
        f"{len(bad)} rows out of tolerance, "
        f"max_abs={np.nanmax(np.abs(rstd_gpu[:M] - ref)):.3e}, "
        f"first bad row={bad[0]}: "
        f"gpu={rstd_gpu[bad[0]]:.6f} ref={ref[bad[0]]:.6f}"
    )
