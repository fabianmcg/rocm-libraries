# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+PartialRMS (K1) Subtile epilogue (gfx950, bf16).

Exercises a comprehensive set of (M, K, N_hidden) shapes, verifying both:
  - D output (bf16, tol=2e-2): h1 * gamma
  - partialBuf (fp32, tol=1e-4): 2D [M_padded, N_tiles_N] per-tile Σx²

partialBuf is 2D with shape [M_padded, N_tiles_N], N_tiles_N = ceil(N_hidden/MT1).

The fixture is parametrised over (wg0_waves, wg1_waves), which independently
control MIWaveGroup[0] and MIWaveGroup[1]:
  MacroTile0 = MatrixInstM * MIWaveTile[0] * MIWaveGroup[0] = 16 * 4 * wg0_waves
  MacroTile1 = MatrixInstN * MIWaveTile[1] * MIWaveGroup[1] = 16 * 4 * wg1_waves

Exercising both axes catches codegen bugs (such as the VMulLOU32 literal-operand
error that appeared when MacroTile0 > 64) that are invisible when only one wave
dimension is varied.  Shapes are generated at runtime from the actual MT0 so the
coverage remains correct for any (wg0, wg1) combination.

N_hidden is parametrised separately and is independent of MT1 (no row-containment).
"""

import math
import os
import sys

import numpy as np
import pytest

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TOOLS_DIR    = os.path.join(TENSILE_ROOT, "tools")
for _d in (TENSILE_ROOT, TOOLS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

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

# (wg0_waves, wg1_waves) — independent axes for MIWaveGroup[0] and MIWaveGroup[1].
# MacroTile0 = 16 * 4 * wg0_waves, MacroTile1 = 16 * 4 * wg1_waves.
# wg0=1,wg1=1 → MT 64×64   (inline-literal range, baseline)
# wg0=1,wg1=2 → MT 64×128  (wg1 cross-wave LDS; small MT0)
# wg0=2,wg1=1 → MT 128×64  (MT0 first outside inline-literal range)
# wg0=4,wg1=1 → MT 256×64  (triggered the original VMulLOU32 literal bug)
# wg0=4,wg1=2 → MT 256×128 (default high-perf config)
_WG_CONFIGS = [
    (1, 1),
    (1, 2),
    (2, 1),
    (4, 1),
    (4, 2),
]

# N_hidden values to test — may be any multiple of MT1 or non-multiple.
# 192 = 3*64, 320 = 5*64: non-power-of-2 multiples (odd N_tiles_N).
# 4096, 8192: large hidden dims typical in LLM inference.
_N_HIDDEN = [64, 128, 192, 256, 320, 4096, 8192]

# K values that exercise: minimal (K=1, K=32), typical (K=64, K=256, K=512),
# large (K=4096, K=8192), and several prime sizes (K=31, K=97, K=127, K=4093).
_K_VALUES = [1, 32, 64, 96, 128, 256, 512, 1024, 4096, 31, 97, 127, 4093, 8192]


def _m_shapes_for_mt0(mt0):
    """Return a list of (M, label) pairs that give representative coverage for MT0.

    Includes:
      - Several exact multiples of MT0 (full-tile M).
      - Several non-multiples at important boundary fractions (MT0/4, MT0/2, MT0-1,
        MT0+1, 2*MT0-1) and prime values near those boundaries.
      - Very small M (1, 3, 7) to stress the tail-loop path.
      - Very large M (4096, 8192) for throughput coverage.
    """
    shapes = []

    def add(m, tag):
        shapes.append((m, f"M{m}_{tag}"))

    # Full-tile multiples.
    for mult in [1, 2, 4, 8, 16, 32]:
        add(mt0 * mult, f"{mult}xMT0")

    # Boundary fractions — clamp to at least 1.
    for frac_num, frac_den, tag in [
        (1, 4, "MT0d4"),
        (1, 2, "MT0d2"),
    ]:
        m = max(1, mt0 * frac_num // frac_den)
        add(m, tag)

    # Near-boundary values.
    for delta, tag in [(-1, "MT0m1"), (1, "MT0p1"), (mt0 - 1, "2MT0m1")]:
        m = max(1, mt0 + delta)
        add(m, tag)

    # Small primes.
    for m, tag in [(1, "M1"), (3, "M3"), (7, "M7"), (31, "M31")]:
        if m not in [s[0] for s in shapes]:
            add(m, tag)

    # Large M for throughput.
    for m, tag in [(4096, "M4096"), (8192, "M8192")]:
        add(m, tag)

    # Deduplicate while preserving order.
    seen = set()
    result = []
    for m, label in shapes:
        if m not in seen:
            seen.add(m)
            result.append((m, label))
    return result


# ---------------------------------------------------------------------------
# Session-scoped fixture: build + compile once per (wg0_waves, wg1_waves)
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=_WG_CONFIGS,
    ids=[f"wg0x{c[0]}_wg1x{c[1]}" for c in _WG_CONFIGS],
)
def k1_kernel(request):
    """Build, assemble, and compile the K1 PartialRMS kernel for each tile config."""
    sys.path.insert(0, TENSILE_ROOT)
    from gemm_partialrms_colv2_helpers import setup_tensile, build_k1_solution, generate_asm

    wg0Waves, wg1Waves = request.param
    chip = amdgpu_exec.get_chip()

    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    # wt0=4 (not 8 as in the default YAML) — use miOverride to specify the full MI spec.
    miOverride = [16, 16, 32, 1, 1, 4, 4, wg0Waves, wg1Waves]
    solution = build_k1_solution(chip, assembler, isaInfoMap, miOverride=miOverride)
    asm_str, kernel_name = generate_asm(solution, assembler, debugConfig)
    hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
    return solution, kernel_name, hsaco, chip


# ---------------------------------------------------------------------------
# Helper: run one shape and return comparison data
# ---------------------------------------------------------------------------

def _run_shape(solution, kernel_name, hsaco, chip, M, K, N_hidden):
    from gemm_partialrms_colv2_helpers import compute_sk3_dp_args

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
    partial_buf = np.zeros((M_padded, N_tiles_N), dtype=np.float32, order='C')

    gamma_f32  = rng.random(N, dtype=np.float32) + 0.5
    gamma_bf16 = gamma_f32.astype(ml_dtypes.bfloat16)

    # Numpy reference (bf16-rounded inputs to match kernel precision).
    a_ref     = np.asarray(a_bf16).astype(np.float32)
    w0_ref    = np.asarray(w0_bf16).astype(np.float32)
    h1        = a_ref.T @ w0_ref
    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)
    d_ref     = (h1 * gamma_ref[np.newaxis, :]).astype(ml_dtypes.bfloat16)

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

    return d_gpu_f32, d_ref_f32, pb_gpu, sumsq_ref


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("N_hidden", _N_HIDDEN, ids=[f"N{n}" for n in _N_HIDDEN])
@pytest.mark.parametrize("K", _K_VALUES, ids=[f"K{k}" for k in _K_VALUES])
def test_k1_shape(k1_kernel, K, N_hidden):
    """Verify K1 (PartialRMS) outputs D and 2D partialBuf across (wg0, wg1, K, N_hidden).

    M shapes are derived at runtime from the fixture's actual MacroTile0 so the
    coverage adapts automatically to any (wg0_waves, wg1_waves) configuration.
    """
    solution, kernel_name, hsaco, chip = k1_kernel
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    N_tiles_N = math.ceil(N_hidden / MT1)

    for M, m_label in _m_shapes_for_mt0(MT0):
        d_gpu_f32, d_ref_f32, pb_gpu, sumsq_ref = \
            _run_shape(solution, kernel_name, hsaco, chip, M, K, N_hidden)

        rtol_d, atol_d = 2e-2, 2e-2
        bad_d = np.where(
            ~np.isfinite(d_gpu_f32[:M]) |
            (np.abs(d_gpu_f32[:M] - d_ref_f32[:M]) > atol_d + rtol_d * np.abs(d_ref_f32[:M]))
        )
        n_bad_d = len(bad_d[0])
        assert n_bad_d == 0, (
            f"D mismatch: MT0={MT0} MT1={MT1} {m_label} N={N_hidden} K={K}: "
            f"{n_bad_d} elements out of tolerance. "
            f"max_abs={np.nanmax(np.abs(d_gpu_f32[:M] - d_ref_f32[:M])):.3e}, "
            f"first bad row={bad_d[0][0]}, col={bad_d[1][0]}: "
            f"gpu={d_gpu_f32[bad_d[0][0], bad_d[1][0]]:.6f} "
            f"ref={d_ref_f32[bad_d[0][0], bad_d[1][0]]:.6f}"
        )

        rtol_p, atol_p = 1e-4, 1e-4
        bad_p = np.where(
            ~np.isfinite(pb_gpu[:M, :]) |
            (np.abs(pb_gpu[:M, :] - sumsq_ref) > atol_p + rtol_p * np.abs(sumsq_ref))
        )
        n_bad_p = len(bad_p[0])
        assert n_bad_p == 0, (
            f"partialBuf mismatch: MT0={MT0} MT1={MT1} {m_label} "
            f"N={N_hidden} K={K} N_tiles_N={N_tiles_N}: "
            f"{n_bad_p} entries out of tolerance. "
            f"max_abs={np.nanmax(np.abs(pb_gpu[:M, :] - sumsq_ref)):.3e}, "
            f"first bad [{bad_p[0][0]},{bad_p[1][0]}]: "
            f"gpu={pb_gpu[bad_p[0][0], bad_p[1][0]]:.6f} "
            f"ref={sumsq_ref[bad_p[0][0], bad_p[1][0]]:.6f}"
        )
