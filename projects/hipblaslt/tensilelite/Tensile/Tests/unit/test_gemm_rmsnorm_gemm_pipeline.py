# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end pipeline test for K1 → K2 → K3 (gfx950, bf16).

Verifies that chaining:
  K1: GEMM + PartialRMS  → D (=h2), 2D partialBuf [M_padded, N_tiles_N]
  K2: aux rsqrt          → rstdBuf[M]
  K3: GEMM + RstdScale   → y

produces y matching the numpy reference within bf16 tolerance.

partialBuf is 2D: shape (M_padded_k1, N_tiles_N), C-order fp32.
N_tiles_N = ceil(N_hidden / MT1_k1). K2 is a single binary (N_hidden is runtime).

For wg_n=1: MacroTile1=64, shapes use N_hidden=N_out=64.
For wg_n=2: MacroTile1=128, shapes use N_hidden=N_out=128.

The first shape in each wg_n group (M=64) also validates intermediate
buffers (partialBuf and rstdBuf) against numpy references.
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
    import ml_dtypes
    _HAVE_DEPS = True
except ImportError:
    _HAVE_DEPS = False

requires_gfx950 = pytest.mark.skipif(
    not _HAVE_DEPS or not (lambda: amdgpu_exec.get_chip().startswith("gfx950"))(),
    reason="requires amdgpu_exec + ml_dtypes and a gfx950 GPU",
)

# wg_n values to exercise; MacroTile1 = 64 * wg_n.
_WG_N = [1, 2]

# Shapes per wg_n: (M, label).
# N_hidden = N_out = 64 * wg_n (determined at fixture time).
_SHAPES_WGN1 = [
    ( 64,  "M64_N64_J64"),
    (2048, "M2048_N64_J64"),
    (1000, "M1000_N64_J64"),
    ( 513, "M513_N64_J64"),
]

_SHAPES_WGN2 = [
    ( 64,  "M64_N128_J128"),
    (2048, "M2048_N128_J128"),
    (1000, "M1000_N128_J128"),
    ( 513, "M513_N128_J128"),
]

_SHAPES_BY_WGN = {1: _SHAPES_WGN1, 2: _SHAPES_WGN2}

# K (outer contraction for K1) — same for all shapes.
_K = 256

# Epsilon for K2.
_EPS = 1e-5


# ---------------------------------------------------------------------------
# Session-scoped fixture: build all three kernels once per wg_n
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", params=_WG_N, ids=[f"wgN{w}" for w in _WG_N])
def pipeline_kernels(request):
    """Build K1, K2, K3 kernels once per wg_n.

    Returns (k1_sol, k1_name, k1_hsaco, k2_name, k2_hsaco,
             k3_sol, k3_name, k3_hsaco, chip, wg_n, N).
    """
    sys.path.insert(0, TENSILE_ROOT)
    from tensile_gemm_rmsnorm_gemm_example import (
        setup_tensile,
        build_k1_solution,
        build_k3_solution,
        build_aux_reduction_asm,
        generate_asm,
    )

    wg_n = request.param
    N    = 64 * wg_n    # MacroTile1 = N_hidden = N_out

    chip = amdgpu_exec.get_chip()
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)

    k1_sol = build_k1_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    k3_sol = build_k3_solution(chip, assembler, isaInfoMap,
                               N_hidden=N, N_out=N, wg_n=wg_n)

    k1_asm, k1_name = generate_asm(k1_sol, assembler, debugConfig)
    k3_asm, k3_name = generate_asm(k3_sol, assembler, debugConfig)
    _k2_asm, k2_name, k2_hsaco = build_aux_reduction_asm(chip)

    k1_hsaco = amdgpu_exec.compile_asm_to_hsaco(k1_asm, chip)
    k3_hsaco = amdgpu_exec.compile_asm_to_hsaco(k3_asm, chip)

    return (k1_sol, k1_name, k1_hsaco,
            k2_name, k2_hsaco,
            k3_sol, k3_name, k3_hsaco,
            chip, wg_n, N)


# ---------------------------------------------------------------------------
# Helper: run full pipeline for one (M, N, K) shape, return results
# ---------------------------------------------------------------------------

def _run_pipeline(pipeline_kernels_fixture, M, K, eps, validate_intermediates):
    """Execute K1 → K2 → K3 on shared device buffers.

    Returns:
        y_gpu_f32:   fp32 upcast of GPU y output, shape (M, N_out)
        y_ref_f32:   fp32 upcast of numpy reference y, shape (M, N_out)
        pb_gpu:      2D partialBuf from GPU (M, N_tiles_N), fp32
        sumsq_ref:   numpy 2D reference sum-of-squares (M, N_tiles_N), fp32
        rstd_gpu:    rstdBuf from GPU (M,), fp32
        rstd_ref:    numpy reference rstd (M,), fp32
    """
    from tensile_gemm_rmsnorm_gemm_example import compute_sk3_dp_args

    (k1_sol, k1_name, k1_hsaco,
     k2_name, k2_hsaco,
     k3_sol, k3_name, k3_hsaco,
     chip, wg_n, N) = pipeline_kernels_fixture

    N_hidden = N
    N_out    = N

    MT0_k1  = k1_sol["MacroTile0"]
    MT1_k1  = k1_sol["MacroTile1"]
    MT0_k3  = k3_sol["MacroTile0"]
    M_padded_k1 = math.ceil(M / MT0_k1) * MT0_k1
    M_padded_k3 = math.ceil(M / MT0_k3) * MT0_k3
    # K2: ROWS_PER_WG=4, one wave per row.
    M_padded_k2 = math.ceil(M / 4) * 4
    N_tiles_N   = math.ceil(N_hidden / MT1_k1)

    numWG_k1 = math.ceil(M / MT0_k1) * N_tiles_N
    numWG_k3 = math.ceil(M / MT0_k3) * math.ceil(N_out / N_out)

    rng = np.random.default_rng(seed=M * 100000 + N * 100 + K)

    # K1 inputs.
    a_f32   = np.asfortranarray(rng.random((K, M),        dtype=np.float32) * 0.1)
    w0_f32  = np.asfortranarray(rng.random((K, N_hidden), dtype=np.float32) * 0.1)
    a_bf16  = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(w0_f32.astype(ml_dtypes.bfloat16))
    gamma_f32  = rng.random(N_hidden, dtype=np.float32) + 0.5
    gamma_bf16 = gamma_f32.astype(ml_dtypes.bfloat16)

    # K3 inputs.
    w1_f32  = np.asfortranarray(rng.random((N_hidden, N_out), dtype=np.float32) * 0.1)
    w1_bf16 = np.asfortranarray(w1_f32.astype(ml_dtypes.bfloat16))

    # Shared buffers.
    c_k1_bf16  = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order='F')
    # K1 writes D as (M, N_hidden) col-major with strideD0=M.
    d_bf16     = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order='F')
    # 2D partialBuf [M_padded_k1, N_tiles_N], C-order fp32.
    partial_buf_2d = np.zeros((M_padded_k1, N_tiles_N), dtype=np.float32, order='C')
    rstd_buf   = np.zeros(M_padded_k2, dtype=np.float32)
    c_k3_bf16  = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')
    y_bf16     = np.zeros((M, N_out),  dtype=ml_dtypes.bfloat16, order='F')
    # K3 expects A in (N_hidden, M) col-major layout; filled by transposing d_bf16.
    h2_for_k3  = np.zeros((N_hidden, M), dtype=ml_dtypes.bfloat16, order='F')

    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    # Numpy reference.
    a_ref     = np.asarray(a_bf16).astype(np.float32)
    w0_ref    = np.asarray(w0_bf16).astype(np.float32)
    h1        = a_ref.T @ w0_ref                     # M x N_hidden, fp32
    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)
    h2_ref    = np.asarray((h1 * gamma_ref[np.newaxis, :]).astype(ml_dtypes.bfloat16))
    # 2D partialBuf reference: per-tile Σx² over MT1-wide column blocks.
    sumsq_ref = np.zeros((M, N_tiles_N), dtype=np.float32)
    for t in range(N_tiles_N):
        col_lo = t * MT1_k1
        col_hi = min((t + 1) * MT1_k1, N_hidden)
        sumsq_ref[:, t] = np.sum(h1[:, col_lo:col_hi] ** 2, axis=1)
    rstd_ref  = (1.0 / np.sqrt(sumsq_ref.sum(axis=1) / N_hidden + eps)).astype(np.float32)
    # w1_ref shape (N_hidden, N_out); K3 computes h2 @ w1_ref = (M,N_hidden)@(N_hidden,N_out).
    w1_ref    = np.asarray(w1_bf16).astype(np.float32)
    h3        = h2_ref.astype(np.float32) @ w1_ref
    y_ref     = (h3 * rstd_ref[:, np.newaxis]).astype(ml_dtypes.bfloat16)

    # ---- K1 args ----
    sk1 = compute_sk3_dp_args(M, N_hidden, K, k1_sol)
    su1 = k1_sol.get("StaggerU", 0)
    sm1 = k1_sol.get("StaggerUMapping", 0)
    ss1 = k1_sol.get("_staggerStrideShift", 0)
    sw1 = (sm1 << 13) | ((ss1 << 8) & 0x1F00) | (su1 & 0xFF)
    ki0_k1 = np.uint32((sw1 << 16) | (k1_sol["GlobalSplitU"] & 0x3FFF))
    ki1_k1 = np.uint32(
        (k1_sol.get("WorkGroupMappingXCC", 1) << 16) | (k1_sol["WorkGroupMapping"] & 0xFFFF)
    )
    d_inout  = amdgpu_exec.InOutArray(d_bf16)
    pb_inout = amdgpu_exec.InOutArray(partial_buf_2d)

    args_k1 = [
        np.uint32(1), ki0_k1, ki1_k1, np.uint32(numWG_k1),
        np.uint32(M), np.uint32(N_hidden), np.uint32(1), np.uint32(K),
        d_inout,
        amdgpu_exec.InputArray(c_k1_bf16),
        amdgpu_exec.InputArray(a_bf16),
        amdgpu_exec.InputArray(w0_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk1["iters_per_tile"], sk1["magic_iters_per_tile"], sk1["shift_iters_per_tile"],
        sk1["sk_iters_per_wg"], sk1["sk_grid"], sk1["sk_tiles"],
        amdgpu_exec.InputArray(gamma_bf16),
        pb_inout,
        np.uint32(N_tiles_N),               # NTilesN (slot 32)
    ]

    pb_result = {}

    def capture_k1(arguments):
        pb_flat = np.asarray(arguments[31].array).copy()
        pb_result["pb_gpu"] = pb_flat.reshape(M_padded_k1, N_tiles_N)

    amdgpu_exec.execute_hsaco(
        hsaco=k1_hsaco, kernel_name=k1_name, arguments=args_k1,
        grid_dim=(numWG_k1, 1, 1), block_dim=(k1_sol["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture_k1 if validate_intermediates else None,
    )

    # Transpose K1's D output from (M, N_hidden) col-major to (N_hidden, M) col-major
    # for K3 which expects A in (N_hidden x M) layout with strideA0=N_hidden.
    np.copyto(h2_for_k3, d_bf16.T)

    # ---- K2 args ----
    # Pad rows to M_padded_k2; pass flat 2D array as contiguous buffer.
    partial_buf_padded_k2 = np.zeros((M_padded_k2, N_tiles_N), dtype=np.float32, order='C')
    partial_buf_padded_k2[:M, :] = partial_buf_2d[:M, :]
    partial_buf_k2_flat = np.ascontiguousarray(partial_buf_padded_k2)
    rstd_inout = amdgpu_exec.InOutArray(rstd_buf)

    args_k2 = [
        amdgpu_exec.InputArray(partial_buf_k2_flat),
        rstd_inout,
        np.uint32(M),
        np.uint32(N_tiles_N),
        np.uint32(N_hidden),
        np.float32(eps),
    ]

    rstd_result = {}

    def capture_k2(arguments):
        rstd_result["rstd_gpu"] = np.asarray(arguments[1].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=k2_hsaco, kernel_name=k2_name, arguments=args_k2,
        grid_dim=(math.ceil(M / 4), 1, 1), block_dim=(256, 1, 1),
        num_iterations=1, verify_fn=capture_k2 if validate_intermediates else None,
    )

    # ---- K3 args ----
    rstd_padded_k3 = np.zeros(M_padded_k3, dtype=np.float32)
    rstd_padded_k3[:M] = rstd_buf[:M]

    sk3 = compute_sk3_dp_args(M, N_out, N_hidden, k3_sol)
    su3 = k3_sol.get("StaggerU", 0)
    sm3 = k3_sol.get("StaggerUMapping", 0)
    ss3 = k3_sol.get("_staggerStrideShift", 0)
    sw3 = (sm3 << 13) | ((ss3 << 8) & 0x1F00) | (su3 & 0xFF)
    ki0_k3 = np.uint32((sw3 << 16) | (k3_sol["GlobalSplitU"] & 0x3FFF))
    ki1_k3 = np.uint32(
        (k3_sol.get("WorkGroupMappingXCC", 1) << 16) | (k3_sol["WorkGroupMapping"] & 0xFFFF)
    )
    y_inout = amdgpu_exec.InOutArray(y_bf16)

    args_k3 = [
        np.uint32(1), ki0_k3, ki1_k3, np.uint32(numWG_k3),
        np.uint32(M), np.uint32(N_out), np.uint32(1), np.uint32(N_hidden),
        y_inout,
        amdgpu_exec.InputArray(c_k3_bf16),
        amdgpu_exec.InputArray(h2_for_k3),
        amdgpu_exec.InputArray(w1_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(N_hidden), np.uint32(0),
        np.uint32(N_hidden), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk3["iters_per_tile"], sk3["magic_iters_per_tile"], sk3["shift_iters_per_tile"],
        sk3["sk_iters_per_wg"], sk3["sk_grid"], sk3["sk_tiles"],
        amdgpu_exec.InputArray(rstd_padded_k3),
    ]

    y_result = {}

    def capture_k3(arguments):
        y_result["y_gpu"] = np.asarray(arguments[8].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=k3_hsaco, kernel_name=k3_name, arguments=args_k3,
        grid_dim=(numWG_k3, 1, 1), block_dim=(k3_sol["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture_k3,
    )

    y_gpu_f32 = y_result["y_gpu"].astype(np.float32)
    y_ref_f32 = np.asarray(y_ref).astype(np.float32)
    pb_gpu    = pb_result.get("pb_gpu", partial_buf_2d)[:M, :]
    rstd_gpu  = rstd_result.get("rstd_gpu", rstd_buf)[:M]

    return y_gpu_f32, y_ref_f32, pb_gpu, sumsq_ref, rstd_gpu, rstd_ref


# ---------------------------------------------------------------------------
# Parametrised pipeline test
# ---------------------------------------------------------------------------

def _shape_id(wg_n, label):
    return f"wgN{wg_n}_{label}"


@requires_gfx950
@pytest.mark.parametrize("M,label", _SHAPES_WGN1, ids=[s[1] for s in _SHAPES_WGN1])
def test_pipeline_shape_wgn1(pipeline_kernels, M, label):
    """Verify pipeline y output for wg_n=1 shapes."""
    _wg_n = pipeline_kernels[9]
    if _wg_n != 1:
        pytest.skip(f"fixture wg_n={_wg_n}, need wg_n=1")

    validate_intermediates = (M == 64)
    y_gpu_f32, y_ref_f32, pb_gpu, sumsq_ref, rstd_gpu, rstd_ref = \
        _run_pipeline(pipeline_kernels, M, _K, _EPS, validate_intermediates)

    _check_y(y_gpu_f32, y_ref_f32, M, label)

    if validate_intermediates:
        _check_partialBuf(pb_gpu, sumsq_ref, M, label)
        _check_rstdBuf(rstd_gpu, rstd_ref, M, label)


@requires_gfx950
@pytest.mark.parametrize("M,label", _SHAPES_WGN2, ids=[s[1] for s in _SHAPES_WGN2])
def test_pipeline_shape_wgn2(pipeline_kernels, M, label):
    """Verify pipeline y output for wg_n=2 shapes."""
    _wg_n = pipeline_kernels[9]
    if _wg_n != 2:
        pytest.skip(f"fixture wg_n={_wg_n}, need wg_n=2")

    validate_intermediates = (M == 64)
    y_gpu_f32, y_ref_f32, pb_gpu, sumsq_ref, rstd_gpu, rstd_ref = \
        _run_pipeline(pipeline_kernels, M, _K, _EPS, validate_intermediates)

    _check_y(y_gpu_f32, y_ref_f32, M, label)

    if validate_intermediates:
        _check_partialBuf(pb_gpu, sumsq_ref, M, label)
        _check_rstdBuf(rstd_gpu, rstd_ref, M, label)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _check_y(y_gpu_f32, y_ref_f32, M, label):
    """Assert y matches reference within bf16 tolerance."""
    rtol, atol = 2e-2, 2e-2
    bad = np.where(
        ~np.isfinite(y_gpu_f32[:M]) |
        (np.abs(y_gpu_f32[:M] - y_ref_f32[:M]) > atol + rtol * np.abs(y_ref_f32[:M]))
    )
    n_bad = len(bad[0])
    assert n_bad == 0, (
        f"y mismatch ({label}): {n_bad} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(y_gpu_f32[:M] - y_ref_f32[:M])):.3e}, "
        f"first bad row={bad[0][0]}, col={bad[1][0]}: "
        f"gpu={y_gpu_f32[bad[0][0], bad[1][0]]:.6f} "
        f"ref={y_ref_f32[bad[0][0], bad[1][0]]:.6f}"
    )


def _check_partialBuf(pb_gpu, sumsq_ref, M, label):
    """Assert 2D partialBuf matches sum-of-squares reference within fp32 tolerance.

    pb_gpu and sumsq_ref have shape (M, N_tiles_N).
    """
    rtol, atol = 1e-4, 1e-4
    bad = np.where(
        ~np.isfinite(pb_gpu[:M, :]) |
        (np.abs(pb_gpu[:M, :] - sumsq_ref) > atol + rtol * np.abs(sumsq_ref))
    )
    n_bad = len(bad[0])
    assert n_bad == 0, (
        f"partialBuf mismatch ({label}): {n_bad} entries out of tolerance. "
        f"max_abs={np.nanmax(np.abs(pb_gpu[:M, :] - sumsq_ref)):.3e}, "
        f"first bad [{bad[0][0]},{bad[1][0]}]: "
        f"gpu={pb_gpu[bad[0][0], bad[1][0]]:.6f} ref={sumsq_ref[bad[0][0], bad[1][0]]:.6f}"
    )


def _check_rstdBuf(rstd_gpu, rstd_ref, M, label):
    """Assert rstdBuf matches reference within fp32 tolerance."""
    rtol, atol = 1e-4, 1e-4
    bad = np.where(
        ~np.isfinite(rstd_gpu[:M]) |
        (np.abs(rstd_gpu[:M] - rstd_ref) > atol + rtol * np.abs(rstd_ref))
    )
    n_bad = len(bad[0])
    assert n_bad == 0, (
        f"rstdBuf mismatch ({label}): {n_bad} rows out of tolerance. "
        f"max_abs={np.nanmax(np.abs(rstd_gpu[:M] - rstd_ref)):.3e}, "
        f"first bad row={bad[0][0]}: "
        f"gpu={rstd_gpu[bad[0][0]]:.6f} ref={rstd_ref[bad[0][0]]:.6f}"
    )
