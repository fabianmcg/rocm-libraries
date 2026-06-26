# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the K2 auxiliary reduction kernel (gfx950, fp32).

Exercises a range of (M, N_hidden, eps) shapes, verifying:
  rstdBuf[m] = rsqrt(partialBuf[m] / N_hidden + eps)   for m in [0, M)

Two test modes:
  1. Unit: synthetic random partialBuf, compare GPU vs reference fp32.
  2. Chained: use the sumsq_ref output of K1 as partialBuf input.

Session-scoped fixtures compile one HSACO per N_hidden value.
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
# Test parameters
# ---------------------------------------------------------------------------

_N_HIDDEN = [64, 4096]

_M_VALUES = [1, 16, 255, 256, 257, 1024, 2048, 4093, 65535]

_EPS_VALUES = [1e-5, 1e-6]

# Shapes to test in the parametrised test: (M, N_hidden, eps, label).
_SHAPES = [
    (M, N, eps, f"M{M}_N{N}_eps{eps:.0e}")
    for M in _M_VALUES
    for N in _N_HIDDEN
    for eps in _EPS_VALUES
]


# ---------------------------------------------------------------------------
# Session-scoped fixture: compile once per N_hidden
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", params=_N_HIDDEN, ids=[f"N{n}" for n in _N_HIDDEN])
def k2_kernel(request):
    """Build and compile the K2 aux-reduction kernel once per N_hidden."""
    sys.path.insert(0, TENSILE_ROOT)
    from tensile_gemm_rmsnorm_gemm_example import build_aux_reduction_asm

    N_hidden = request.param
    chip = amdgpu_exec.get_chip()
    _asm_str, kernel_name, hsaco = build_aux_reduction_asm(chip, N_hidden)
    return kernel_name, hsaco, N_hidden, chip


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _rstd_ref(partial_buf: np.ndarray, N_hidden: int, eps: float) -> np.ndarray:
    """Scalar fp32 reference for rstdBuf."""
    return np.array(
        [1.0 / math.sqrt(float(v) / N_hidden + eps) for v in partial_buf],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Helper: run one shape and return GPU/reference arrays
# ---------------------------------------------------------------------------

def _run_k2(kernel_name: str, hsaco: bytes, M: int, N_hidden: int, eps: float,
            partial_buf_in: np.ndarray):
    """Execute the K2 kernel and return (rstd_gpu, rstd_ref)."""
    M_padded = math.ceil(M / 256) * 256
    partial_buf_padded = np.zeros(M_padded, dtype=np.float32)
    partial_buf_padded[:M] = partial_buf_in[:M]

    rstd_buf = np.zeros(M_padded, dtype=np.float32)

    args = [
        amdgpu_exec.InputArray(partial_buf_padded),
        amdgpu_exec.InOutArray(rstd_buf),
        np.uint32(M),
        np.float32(eps),
    ]

    result_holder = {}

    def capture(arguments):
        result_holder["rstd_gpu"] = np.asarray(arguments[1].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(math.ceil(M / 256), 1, 1),
        block_dim=(256, 1, 1),
        num_iterations=1,
        verify_fn=capture,
    )

    rstd_gpu = result_holder["rstd_gpu"]
    ref = _rstd_ref(partial_buf_padded[:M], N_hidden, eps)
    return rstd_gpu, ref


# ---------------------------------------------------------------------------
# Unit tests: synthetic partialBuf
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("M,N_hidden,eps,label", _SHAPES, ids=[s[3] for s in _SHAPES])
def test_k2_shape(k2_kernel, M, N_hidden, eps, label):
    """Verify K2 rstdBuf for each (M, N_hidden, eps) shape."""
    kernel_name, hsaco, fixture_N, chip = k2_kernel
    if fixture_N != N_hidden:
        pytest.skip(f"fixture compiled for N={fixture_N}, not N={N_hidden}")

    rng = np.random.default_rng(seed=M * 100 + N_hidden)
    partial_buf = rng.random(M, dtype=np.float32) * 10.0 + 1e-3

    rstd_gpu, ref = _run_k2(kernel_name, hsaco, M, N_hidden, eps, partial_buf)

    rtol, atol = 1e-4, 1e-4
    diff = np.abs(rstd_gpu[:M] - ref)
    tol  = atol + rtol * np.abs(ref)
    bad  = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
    n_bad = len(bad)
    assert n_bad == 0, (
        f"rstdBuf mismatch: M={M} N_hidden={N_hidden} eps={eps} ({label}): "
        f"{n_bad} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(rstd_gpu[:M] - ref)):.3e}, "
        f"first bad row={bad[0]}: "
        f"gpu={rstd_gpu[bad[0]]:.6f} ref={ref[bad[0]]:.6f}"
    )


@requires_gfx950
@pytest.mark.parametrize("N_hidden,eps", [
    (64, 1e-5), (64, 1e-6), (4096, 1e-5)
], ids=["N64_eps1e-5", "N64_eps1e-6", "N4096_eps1e-5"])
def test_k2_edge_values(k2_kernel, N_hidden, eps):
    """Verify K2 handles edge values: zero, very large, very small."""
    kernel_name, hsaco, fixture_N, chip = k2_kernel
    if fixture_N != N_hidden:
        pytest.skip(f"fixture compiled for N={fixture_N}, not N={N_hidden}")

    M = 8
    # Edge values: near-zero, moderate, large.
    partial_buf = np.array(
        [0.0, 1e-6, 0.1, 1.0, 10.0, 100.0, 1e4, 1e6],
        dtype=np.float32,
    )
    # For zero input, partialBuf[0]/N + eps = eps, so rstd = 1/sqrt(eps).
    rstd_gpu, ref = _run_k2(kernel_name, hsaco, M, N_hidden, eps, partial_buf)

    rtol, atol = 1e-4, 1e-4
    diff = np.abs(rstd_gpu[:M] - ref)
    tol  = atol + rtol * np.abs(ref)
    bad  = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
    assert len(bad) == 0, (
        f"edge-value mismatch: N={N_hidden} eps={eps}: "
        f"bad={bad.tolist()}"
    )


@requires_gfx950
@pytest.mark.parametrize("N_hidden", _N_HIDDEN, ids=[f"N{n}" for n in _N_HIDDEN])
def test_k2_bounds_guard(k2_kernel, N_hidden):
    """Threads with global id >= M must not write to rstdBuf.

    Fill rstdBuf with a sentinel value, run a partial-block grid (M < 256),
    and verify that entries beyond M retain the sentinel.
    """
    kernel_name, hsaco, fixture_N, chip = k2_kernel
    if fixture_N != N_hidden:
        pytest.skip(f"fixture compiled for N={fixture_N}, not N={N_hidden}")

    M = 100          # not a multiple of 256 -> partial block
    eps = 1e-5
    M_padded = 256   # one full block

    sentinel = np.float32(-999.0)
    partial_buf = np.ones(M_padded, dtype=np.float32) * 2.0
    rstd_buf    = np.full(M_padded, sentinel, dtype=np.float32)

    args = [
        amdgpu_exec.InputArray(partial_buf),
        amdgpu_exec.InOutArray(rstd_buf),
        np.uint32(M),
        np.float32(eps),
    ]

    result_holder = {}

    def capture(arguments):
        result_holder["rstd_gpu"] = np.asarray(arguments[1].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
        num_iterations=1,
        verify_fn=capture,
    )

    rstd_gpu = result_holder["rstd_gpu"]

    # Rows [M, M_padded) must remain untouched.
    tail = rstd_gpu[M:M_padded]
    corrupted = np.where(tail != sentinel)[0]
    assert len(corrupted) == 0, (
        f"bounds guard failed: {len(corrupted)} entries beyond M={M} were "
        f"written (expected sentinel {sentinel}), first at row {M + corrupted[0]}"
    )

    # Rows [0, M) must be valid.
    ref = _rstd_ref(partial_buf[:M], N_hidden, eps)
    rtol, atol = 1e-4, 1e-4
    diff = np.abs(rstd_gpu[:M] - ref)
    tol  = atol + rtol * np.abs(ref)
    bad  = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
    assert len(bad) == 0, (
        f"valid-row mismatch: M={M} N={N_hidden}: bad={bad.tolist()}"
    )


# ---------------------------------------------------------------------------
# Chained test: use K1 reference sumsq_ref as partialBuf input
# ---------------------------------------------------------------------------

_K1_SHAPES = [
    (64,  64,   64,   "M64_N64_K64"),
    (128, 64,  128,   "M128_N64_K128"),
    (256, 64, 4096,   "M256_N64_K4096"),
]


@requires_gfx950
@pytest.mark.parametrize("M,N,K,label", _K1_SHAPES, ids=[s[3] for s in _K1_SHAPES])
def test_k2_chained_with_k1(k2_kernel, M, N, K, label):
    """Feed K1 reference sumsq_ref into K2 and compare against double ref."""
    kernel_name, hsaco, N_hidden, chip = k2_kernel
    # Only run when the fixture N_hidden matches the N dimension.
    if N_hidden != N:
        pytest.skip(f"fixture compiled for N={N_hidden}, need N={N}")

    try:
        import ml_dtypes
    except ImportError:
        pytest.skip("ml_dtypes not available")

    eps = 1e-5
    rng = np.random.default_rng(seed=M * 10000 + K)
    a_f32  = np.asfortranarray(rng.random((K, M), dtype=np.float32) * 0.1)
    w0_f32 = np.asfortranarray(rng.random((K, N), dtype=np.float32) * 0.1)
    a_bf16  = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(w0_f32.astype(ml_dtypes.bfloat16))

    # K1 reference sumsq.
    a_ref   = np.asarray(a_bf16).astype(np.float32)
    w0_ref  = np.asarray(w0_bf16).astype(np.float32)
    h1      = a_ref.T @ w0_ref    # M x N, fp32
    sumsq   = np.sum(h1 ** 2, axis=1).astype(np.float32)   # M, fp32 raw sum

    rstd_gpu, ref = _run_k2(kernel_name, hsaco, M, N_hidden, eps, sumsq)

    rtol, atol = 1e-4, 1e-4
    diff = np.abs(rstd_gpu[:M] - ref)
    tol  = atol + rtol * np.abs(ref)
    bad  = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
    assert len(bad) == 0, (
        f"chained K1→K2 mismatch: M={M} N={N} K={K} ({label}): "
        f"{len(bad)} rows out of tolerance, "
        f"max_abs={np.nanmax(np.abs(rstd_gpu[:M] - ref)):.3e}, "
        f"first bad row={bad[0]}: "
        f"gpu={rstd_gpu[bad[0]]:.6f} ref={ref[bad[0]]:.6f}"
    )
