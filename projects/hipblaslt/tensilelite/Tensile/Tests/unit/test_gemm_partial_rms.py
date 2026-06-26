# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+PartialRMS (K1) Subtile epilogue (gfx950, bf16).

Exercises a comprehensive set of (M, K, N_hidden) shapes, verifying both:
  - D output (bf16, tol=2e-2): h1 * gamma
  - partialBuf (fp32, tol=1e-4): 2D [M_padded, N_tiles_N] per-tile Σx²

partialBuf is 2D with shape [M_padded, N_tiles_N], N_tiles_N = ceil(N_hidden/MT1).
Element (m, t) = Σ_{n in tile t columns} h1[m,n]² (raw sum, not divided by N).

The fixture is parametrised over wg_n (MIWaveGroup[1]):
  wg_n=1: single-wave butterfly reduction
  wg_n=2: cross-wave LDS reduction

N_hidden is parametrised separately and is independent of MT1 (no row-containment).
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

# MIWaveGroup[1] values to exercise: single-wave and cross-wave (LDS) reduction.
_WG_N = [1, 2]

# N_hidden values to test — may be any multiple of MT1=64 or non-multiple.
_N_HIDDEN = [64, 128, 256]

# Shape matrix: (M, K, label).
_SHAPES = [
    # --- full-tile M, varying K ---
    (   64,    1,  "M64_K1"),
    (   64,   32,  "M64_K32"),
    (   64,   64,  "M64_K64"),
    (   64,   96,  "M64_K96"),
    (   64,  128,  "M64_K128"),
    (   64,  256,  "M64_K256"),
    (   64,  512,  "M64_K512"),
    (   64, 1024,  "M64_K1024"),
    (   64, 4096,  "M64_K4096"),
    # --- larger full-tile M ---
    (  128,  128,  "M128_K128"),
    (  256,   64,  "M256_K64"),
    (  512,  512,  "M512_K512"),
    ( 1024,  128,  "M1024_K128"),
    ( 2048, 4096,  "M2048_K4096"),
    # --- edge-tile M (non-multiples of MT0=64) ---
    (    1,   64,  "M1_K64"),
    (   16,   64,  "M16_K64"),
    (   32,   64,  "M32_K64"),
    (   48,   64,  "M48_K64"),
    (   80,   96,  "M80_K96"),
    (  100,  128,  "M100_K128"),
    (  130,   37,  "M130_K37"),
    (  200,   64,  "M200_K64"),
    (  513,  256,  "M513_K256"),
    ( 1000, 1024,  "M1000_K1024"),
    # --- prime K ---
    (   64,   31,  "M64_K31"),
    (   64,   97,  "M64_K97"),
    (   64,  127,  "M64_K127"),
    (  128,   61,  "M128_K61"),
    ( 1024, 4093,  "M1024_K4093"),
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures: build + compile once per wg_n
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", params=_WG_N, ids=[f"wgN{w}" for w in _WG_N])
def k1_kernel(request):
    """Build, assemble, and compile the K1 PartialRMS kernel once per wg_n."""
    sys.path.insert(0, TENSILE_ROOT)
    from tensile_gemm_rmsnorm_gemm_example import (
        setup_tensile,
        build_k1_solution,
        generate_asm,
    )

    wg_n = request.param
    chip = amdgpu_exec.get_chip()
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    solution = build_k1_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    asm_str, kernel_name = generate_asm(solution, assembler, debugConfig)
    hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
    return solution, kernel_name, hsaco, chip


# ---------------------------------------------------------------------------
# Helper: run one shape and return comparison data
# ---------------------------------------------------------------------------

def _run_shape(solution, kernel_name, hsaco, chip, M, K, N_hidden):
    from tensile_gemm_rmsnorm_gemm_example import compute_sk3_dp_args

    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    N   = N_hidden

    N_tiles_N = math.ceil(N / MT1)
    M_padded  = math.ceil(M / MT0) * MT0
    numWG     = math.ceil(M / MT0) * N_tiles_N

    rng = np.random.default_rng(seed=M * 10000 + K)

    a_f32  = np.asfortranarray(rng.random((K, M), dtype=np.float32) * 0.1)
    w0_f32 = np.asfortranarray(rng.random((K, N), dtype=np.float32) * 0.1)
    a_bf16  = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(w0_f32.astype(ml_dtypes.bfloat16))

    c_bf16      = np.zeros((M, N), dtype=ml_dtypes.bfloat16, order='F')
    d_bf16      = np.zeros((M, N), dtype=ml_dtypes.bfloat16, order='F')
    # 2D partialBuf, C-order so flat layout is (m*N_tiles_N + t)*4.
    partial_buf = np.zeros((M_padded, N_tiles_N), dtype=np.float32, order='C')

    gamma_f32  = rng.random(N, dtype=np.float32) + 0.5
    gamma_bf16 = gamma_f32.astype(ml_dtypes.bfloat16)

    # Numpy reference (bf16-rounded inputs to match kernel precision).
    a_ref     = np.asarray(a_bf16).astype(np.float32)
    w0_ref    = np.asarray(w0_bf16).astype(np.float32)
    h1        = a_ref.T @ w0_ref                            # M x N, fp32
    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)
    d_ref     = (h1 * gamma_ref[np.newaxis, :]).astype(ml_dtypes.bfloat16)

    # 2D sumsq_ref: per-tile Σx² over MT1-wide column blocks.
    sumsq_ref = np.zeros((M, N_tiles_N), dtype=np.float32)
    for t in range(N_tiles_N):
        col_lo = t * MT1
        col_hi = min((t + 1) * MT1, N)
        sumsq_ref[:, t] = np.sum(h1[:, col_lo:col_hi] ** 2, axis=1)

    sk_args      = compute_sk3_dp_args(M, N, K, solution)
    stagger_u    = solution.get("StaggerU", 0)
    su_map       = solution.get("StaggerUMapping", 0)
    ss_shift     = solution.get("_staggerStrideShift", 0)
    su_word      = (su_map << 13) | ((ss_shift << 8) & 0x1F00) | (stagger_u & 0xFF)
    kernel_info0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))
    wgmxcc       = solution.get("WorkGroupMappingXCC", 1)
    kernel_info1 = np.uint32((wgmxcc << 16) | (solution["WorkGroupMapping"] & 0xFFFF))

    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    args = [
        np.uint32(1), kernel_info0, kernel_info1, np.uint32(numWG),
        np.uint32(M), np.uint32(N), np.uint32(1), np.uint32(K),
        amdgpu_exec.InOutArray(d_bf16),
        amdgpu_exec.InputArray(c_bf16),
        amdgpu_exec.InputArray(a_bf16),
        amdgpu_exec.InputArray(w0_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk_args["iters_per_tile"],
        sk_args["magic_iters_per_tile"],
        sk_args["shift_iters_per_tile"],
        sk_args["sk_iters_per_wg"],
        sk_args["sk_grid"],
        sk_args["sk_tiles"],
        amdgpu_exec.InputArray(gamma_bf16),
        amdgpu_exec.InOutArray(partial_buf),
        np.uint32(N_tiles_N),              # NTilesN (new arg)
    ]

    result_holder = {}

    def capture(arguments):
        result_holder["d_gpu"]  = np.asarray(arguments[8].array).copy()
        pb_flat = np.asarray(arguments[31].array).copy()
        result_holder["pb_gpu"] = pb_flat.reshape(M_padded, N_tiles_N)

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(numWG, 1, 1),
        block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1,
        verify_fn=capture,
    )

    d_gpu_f32 = result_holder["d_gpu"].astype(np.float32)
    d_ref_f32 = np.asarray(d_ref).astype(np.float32)
    pb_gpu    = result_holder["pb_gpu"]

    return d_gpu_f32, d_ref_f32, pb_gpu, sumsq_ref, M


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("N_hidden", _N_HIDDEN, ids=[f"N{n}" for n in _N_HIDDEN])
@pytest.mark.parametrize("M,K,label", _SHAPES, ids=[s[2] for s in _SHAPES])
def test_k1_shape(k1_kernel, M, K, label, N_hidden):
    """Verify K1 (PartialRMS) outputs D and 2D partialBuf for shape M×N_hidden×K."""
    solution, kernel_name, hsaco, chip = k1_kernel

    d_gpu_f32, d_ref_f32, pb_gpu, sumsq_ref, M_actual = \
        _run_shape(solution, kernel_name, hsaco, chip, M, K, N_hidden)

    MT1 = solution["MacroTile1"]
    N_tiles_N = math.ceil(N_hidden / MT1)

    # Check D (bf16 output, always upcast to fp32 before comparing).
    rtol_d, atol_d = 2e-2, 2e-2
    bad_d = np.where(
        ~np.isfinite(d_gpu_f32[:M]) |
        (np.abs(d_gpu_f32[:M] - d_ref_f32[:M]) > atol_d + rtol_d * np.abs(d_ref_f32[:M]))
    )
    n_bad_d = len(bad_d[0])
    assert n_bad_d == 0, (
        f"D mismatch: M={M} N={N_hidden} K={K} ({label}): "
        f"{n_bad_d} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(d_gpu_f32[:M] - d_ref_f32[:M])):.3e}, "
        f"first bad row={bad_d[0][0]}, col={bad_d[1][0]}: "
        f"gpu={d_gpu_f32[bad_d[0][0], bad_d[1][0]]:.6f} "
        f"ref={d_ref_f32[bad_d[0][0], bad_d[1][0]]:.6f}"
    )

    # Check 2D partialBuf (fp32 Σx², compare first M rows).
    rtol_p, atol_p = 1e-4, 1e-4
    bad_p = np.where(
        ~np.isfinite(pb_gpu[:M, :]) |
        (np.abs(pb_gpu[:M, :] - sumsq_ref) > atol_p + rtol_p * np.abs(sumsq_ref))
    )
    n_bad_p = len(bad_p[0])
    assert n_bad_p == 0, (
        f"partialBuf mismatch: M={M} N={N_hidden} K={K} N_tiles_N={N_tiles_N} ({label}): "
        f"{n_bad_p} entries out of tolerance. "
        f"max_abs={np.nanmax(np.abs(pb_gpu[:M, :] - sumsq_ref)):.3e}, "
        f"first bad [{bad_p[0][0]},{bad_p[1][0]}]: "
        f"gpu={pb_gpu[bad_p[0][0], bad_p[1][0]]:.6f} "
        f"ref={sumsq_ref[bad_p[0][0], bad_p[1][0]]:.6f}"
    )
