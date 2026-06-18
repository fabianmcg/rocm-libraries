# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+RMSNorm Subtile epilogue (gfx950, bf16).

Exercises a comprehensive set of (M, N, K) shapes, including:
  - Full-tile M (multiples of MacroTile0=64) and edge-tile M
  - Full-tile K (multiples of DepthU=64), partial K (<64), and multi-DepthU K
  - A single N=64 (== MacroTile1; row-containment invariant)

The test reuses the build/generate/compile path from tensile_rmsnorm_example.py
and verifies numerical correctness against a numpy fp32 reference.
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

# ---------------------------------------------------------------------------
# Shape matrix
# N is determined by the kernel's MacroTile1 (== 64 * wg_n) — row-containment
# invariant.  The fixture is parametrized over wg_n: wg_n=1 is the single-wave
# path; wg_n=2 exercises the cross-wave LDS reduction (MIWaveGroup[1]==2).
#
# M categories:
#   full    : exact multiples of MT0=64
#   edge    : non-multiples (SubtileMGuard handles masking)
#
# K categories:
#   sub     : K < DepthU=64  (partial single iteration)
#   full    : K == DepthU    (exactly one iteration)
#   partial : 64 < K < 128   (one full + one partial iteration)
#   multi   : K is large multiple of DepthU
#   prime   : prime K         (exercises arbitrary remainder)
# ---------------------------------------------------------------------------
# MIWaveGroup[1] values to exercise: single-wave and cross-wave (LDS) reduction.
_WG_N = [1, 2]

_SHAPES = [
    # (M, K,  label)
    # --- full-tile M, varying K ---
    (   64,    1,  "M64_K1"),       # minimum K, sub-DepthU
    (   64,   32,  "M64_K32"),      # half DepthU
    (   64,   64,  "M64_K64"),      # exactly one DepthU
    (   64,   96,  "M64_K96"),      # 1.5× DepthU
    (   64,  128,  "M64_K128"),     # 2× DepthU (baseline)
    (   64,  256,  "M64_K256"),     # 4× DepthU
    (   64,  512,  "M64_K512"),     # 8× DepthU
    (   64, 1024,  "M64_K1024"),    # 16× DepthU
    (   64, 4096,  "M64_K4096"),    # typical LLM hidden dim
    # --- larger full-tile M ---
    (  128,  128,  "M128_K128"),    # 2× tile M, 2× DepthU K
    (  256,   64,  "M256_K64"),     # 4× tile M
    (  512,  512,  "M512_K512"),    # medium square
    ( 1024,  128,  "M1024_K128"),   # tall M
    ( 2048, 4096,  "M2048_K4096"),  # LLM-scale: seq_len × hidden
    # --- edge-tile M (non-multiples of MT0=64) ---
    (    1,   64,  "M1_K64"),       # single row
    (   16,   64,  "M16_K64"),      # quarter tile
    (   32,   64,  "M32_K64"),      # half tile
    (   48,   64,  "M48_K64"),      # three-quarter tile
    (   80,   96,  "M80_K96"),      # one full + edge tile, partial K
    (  100,  128,  "M100_K128"),    # non-multiple M, 2× DepthU K
    (  130,   37,  "M130_K37"),     # multiple edge tiles, sub-DepthU K
    (  200,   64,  "M200_K64"),     # 3 full + 1 edge tile
    (  513,  256,  "M513_K256"),    # just over 8 tiles
    ( 1000, 1024,  "M1000_K1024"),  # large non-multiple M
    # --- prime K ---
    (   64,   31,  "M64_K31"),      # prime, sub-DepthU
    (   64,   97,  "M64_K97"),      # prime, > DepthU
    (   64,  127,  "M64_K127"),     # prime, just under 2× DepthU
    (  128,   61,  "M128_K61"),     # prime K, 2× tile M
    ( 1024, 4093,  "M1024_K4093"),  # large prime K, large M
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures: build + compile once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", params=_WG_N, ids=[f"wgN{w}" for w in _WG_N])
def rmsnorm_kernel(request):
    """Build, assemble, and compile the RMSNorm kernel once per wg_n value."""
    sys.path.insert(0, TENSILE_ROOT)
    from tensile_rmsnorm_example import (
        setup_tensile,
        build_rmsnorm_solution,
        generate_asm,
    )

    wg_n = request.param
    chip = amdgpu_exec.get_chip()
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    solution = build_rmsnorm_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    asm_str, kernel_name = generate_asm(solution, assembler, debugConfig)
    hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
    return solution, kernel_name, hsaco, chip


# ---------------------------------------------------------------------------
# Helper: run one shape and return (gpu_output_f32, reference_f32)
# ---------------------------------------------------------------------------

def _run_shape(solution, kernel_name, hsaco, chip, M, K, eps=1e-5):
    from tensile_rmsnorm_example import compute_sk3_dp_args

    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    N   = MT1  # row-containment invariant: N == MacroTile1 (== 64 * wg_n)

    numWG = math.ceil(M / MT0) * math.ceil(N / MT1)

    rng = np.random.default_rng(seed=M * 10000 + K)  # deterministic per shape

    a_f32 = np.asfortranarray(rng.random((K, M), dtype=np.float32) * 0.1)
    b_f32 = np.asfortranarray(rng.random((K, N), dtype=np.float32) * 0.1)
    a_bf16 = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    b_bf16 = np.asfortranarray(b_f32.astype(ml_dtypes.bfloat16))

    c_bf16 = np.zeros((M, N), dtype=ml_dtypes.bfloat16, order='F')
    d_bf16 = np.zeros((M, N), dtype=ml_dtypes.bfloat16, order='F')

    gamma_f32  = rng.random(N, dtype=np.float32) + 0.5
    gamma_bf16 = gamma_f32.astype(ml_dtypes.bfloat16)

    # numpy reference (fp32 throughout; bf16-rounded inputs)
    a_ref     = np.asarray(a_bf16).astype(np.float32)
    b_ref     = np.asarray(b_bf16).astype(np.float32)
    d_gemm    = a_ref.T @ b_ref
    ss        = np.mean(d_gemm**2, axis=1, keepdims=True)
    rstd      = (1.0 / np.sqrt(ss + eps)).astype(np.float32)
    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)
    d_ref_f32 = (d_gemm * rstd * gamma_ref[np.newaxis, :]).astype(np.float32)

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
        amdgpu_exec.InputArray(b_bf16),
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
        np.float32(eps),
    ]

    result_holder = {}

    def capture(arguments):
        result_holder["d_gpu"] = np.asarray(arguments[8].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(numWG, 1, 1),
        block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1,
        verify_fn=capture,
    )

    d_gpu_f32 = np.asarray(result_holder["d_gpu"]).astype(np.float32)
    return d_gpu_f32, d_ref_f32


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("M,K,label", _SHAPES, ids=[s[2] for s in _SHAPES])
def test_rmsnorm_shape(rmsnorm_kernel, M, K, label):
    """Verify fused GEMM+RMSNorm output matches numpy reference for shape M×N×K."""
    solution, kernel_name, hsaco, chip = rmsnorm_kernel
    N = solution["MacroTile1"]
    d_gpu, d_ref = _run_shape(solution, kernel_name, hsaco, chip, M, K)

    # Tolerances: bf16 output rounding dominates (~1.5 ULP at 8-bit mantissa).
    rtol, atol = 2e-2, 2e-2

    bad = np.where(
        ~np.isfinite(d_gpu) | (np.abs(d_gpu - d_ref) > atol + rtol * np.abs(d_ref))
    )
    n_bad = len(bad[0])

    assert n_bad == 0, (
        f"Shape M={M} N={N} K={K} ({label}): "
        f"{n_bad} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(d_gpu - d_ref)):.3e}, "
        f"first bad row={bad[0][0]}, col={bad[1][0]}: "
        f"gpu={d_gpu[bad[0][0], bad[1][0]]:.6f} ref={d_ref[bad[0][0], bad[1][0]]:.6f}"
    )
