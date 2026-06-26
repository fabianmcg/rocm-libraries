# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end pipeline test for K1 → colv2 (gfx950, bf16).

Verifies that chaining:
  K1: GEMM + PartialRMS  → D (bf16 col-major M×N_hidden), partialBuf (f32 row-major M_padded×N_tiles_N)
  colv2: RMSNorm in-place → D /= sqrt(inv_d * sum_t(partialBuf[m,t]) + eps)

produces D matching the numpy reference within bf16 tolerance.

The fixture is parametrised over wg_n (MIWaveGroup[1]):
  MacroTile0 = 256  (wg0_waves=4, MIWaveTile [8,4])
  MacroTile1 = 16 * wg_n * 4

M shapes are derived at runtime from the actual MT0.
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

# wg_n controls MIWaveGroup[1]; wg_n=4 excluded due to LDS limits.
_WG_N_VALUES = [1, 2]

# K values (outer contraction for K1).
_K_VALUES = [64, 256, 4096]

_EPS = 1e-5

def _n_shapes_for_mt1(mt1):
    """Return N_hidden values covering aligned and irregular (partial last tile) cases."""
    shapes = []
    # Tile-aligned counts: baseline, non-power-of-2 butterfly, larger count.
    for mult in [1, 3, 7]:
        shapes.append(mt1 * mult)
    # Irregular: partial last tile (1, mt1//2-1, mt1-1 elements in last tile).
    for mult in [1, 3]:
        for offset in [1, mt1 // 2 - 1, mt1 - 1]:
            n = mt1 * mult + offset
            if n not in shapes:
                shapes.append(n)
    return shapes


def _m_shapes_for_mt0(mt0):
    """Return (M, label) pairs giving representative M coverage for a given MT0."""
    shapes = []

    def add(m, tag):
        shapes.append((m, f"M{m}_{tag}"))

    # Tile-aligned multiples.
    for mult in [1, 2, 4, 8, 16, 32]:
        add(mt0 * mult, f"{mult}xMT0")

    # Tile-boundary edge cases.
    for m, tag in [
        (max(1, mt0 // 4),  "MT0d4"),
        (max(1, mt0 // 2),  "MT0d2"),
        (max(1, mt0 - 1),   "MT0m1"),
        (mt0 + 1,           "MT0p1"),
        (2 * mt0 - 1,       "2MT0m1"),
    ]:
        add(m, tag)

    # Small irregular sizes (test edge handling for tiny M).
    for m, tag in [(1, "M1"), (3, "M3"), (7, "M7"), (31, "M31")]:
        if m not in [s[0] for s in shapes]:
            add(m, tag)

    # Irregular sizes with awkward remainders mod MT0 (= 256):
    # primes and values that stress the boundary workgroup.
    for m, tag in [
        (127,  "M127_irr"),   # 127 mod 256 = 127
        (193,  "M193_irr"),   # prime
        (317,  "M317_irr"),   # prime, > MT0
        (511,  "M511_irr"),   # 511 mod 256 = 255 (worst-case tail)
        (769,  "M769_irr"),   # prime, 3×256 + 1
        (1009, "M1009_irr"),  # prime
        (1537, "M1537_irr"),  # 6×256 + 1
        (4097, "M4097_irr"),  # 4096 + 1
        (8191, "M8191_irr"),  # 8192 - 1 (near colv2 M limit)
    ]:
        if m not in [s[0] for s in shapes]:
            add(m, tag)

    seen = set()
    result = []
    for m, label in shapes:
        if m not in seen:
            seen.add(m)
            result.append((m, label))
    return result


# ---------------------------------------------------------------------------
# Session-scoped fixture: build K1 and colv2 once per wg_n.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=_WG_N_VALUES,
    ids=[f"wg_n{n}" for n in _WG_N_VALUES],
)
def pipeline_kernels(request):
    """Build K1 (GEMM+PartialRMS) and colv2 (RMSNorm) for one wg_n value.

    Returns (k1_sol, k1_name, k1_hsaco, cd_name, cd_hsaco, chip, MT1).
    """
    sys.path.insert(0, TENSILE_ROOT)
    from gemm_partialrms_colv2_helpers import setup_tensile, build_k1_solution, generate_asm
    from Colv2Generator import build_colv2

    wg_n = request.param
    chip = amdgpu_exec.get_chip()

    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    k1_sol = build_k1_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    k1_asm, k1_name = generate_asm(k1_sol, assembler, debugConfig)
    k1_hsaco = amdgpu_exec.compile_asm_to_hsaco(k1_asm, chip)

    # Build colv2 once with M=1; the hsaco is valid for all M <= 32767
    # because the kernel reads M from kernargs at runtime.
    cd_asm, cd_name = build_colv2(chip, M=1)
    cd_hsaco = amdgpu_exec.compile_asm_to_hsaco(cd_asm, chip)

    MT1 = k1_sol["MacroTile1"]

    return (k1_sol, k1_name, k1_hsaco, cd_name, cd_hsaco, chip, MT1)


# ---------------------------------------------------------------------------
# Helper: run full two-kernel pipeline for one (M, K) shape.
# ---------------------------------------------------------------------------

def _run_pipeline(pipeline_kernels_fixture, M, K, N_hidden, eps):
    from gemm_partialrms_colv2_helpers import compute_sk3_dp_args, _pack_kernel_info

    (k1_sol, k1_name, k1_hsaco, cd_name, cd_hsaco, chip, MT1) = pipeline_kernels_fixture

    MT0 = k1_sol["MacroTile0"]
    N_tiles_N   = math.ceil(N_hidden / MT1)
    M_padded_k1 = math.ceil(M / MT0) * MT0
    numWG_k1    = math.ceil(M / MT0) * N_tiles_N

    rng = np.random.default_rng(seed=M * 100000 + N_hidden * 100 + K)

    a_f32   = np.asfortranarray(rng.random((K, M),        dtype=np.float32) * 0.1)
    w0_f32  = np.asfortranarray(rng.random((K, N_hidden), dtype=np.float32) * 0.1)
    a_bf16  = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(w0_f32.astype(ml_dtypes.bfloat16))
    gamma_f32  = rng.random(N_hidden, dtype=np.float32) + 0.5
    gamma_bf16 = gamma_f32.astype(ml_dtypes.bfloat16)

    # Numpy reference.
    a_ref     = np.asarray(a_bf16).astype(np.float32)
    w0_ref    = np.asarray(w0_bf16).astype(np.float32)
    h1        = a_ref.T @ w0_ref                              # (M, N_hidden) f32
    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)

    # Per-tile sum-of-squares reference.
    sumsq_ref = np.zeros((M, N_tiles_N), dtype=np.float32)
    for t in range(N_tiles_N):
        col_lo = t * MT1
        col_hi = min((t + 1) * MT1, N_hidden)
        sumsq_ref[:, t] = np.sum(h1[:, col_lo:col_hi] ** 2, axis=1)

    # End-to-end D reference: bf16(h1*gamma) / sqrt(inv_d * Σsumsq + eps).
    inv_d        = 1.0 / N_hidden
    h1_gamma_f32 = (h1 * gamma_ref[np.newaxis, :]).astype(ml_dtypes.bfloat16).astype(np.float32)
    row_sums     = sumsq_ref.sum(axis=1)
    rms_denom    = np.sqrt(inv_d * row_sums + eps)
    d_ref_f32    = h1_gamma_f32 / rms_denom[:, np.newaxis]

    # Device buffers.
    d_bf16         = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order='F')
    c_k1_bf16      = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order='F')
    partial_buf_2d = np.zeros((M_padded_k1, N_tiles_N), dtype=np.float32, order='C')
    ws_dummy       = np.zeros(4, dtype=np.float32)
    flags_dummy    = np.zeros(4, dtype=np.float32)

    sk1            = compute_sk3_dp_args(M, N_hidden, K, k1_sol)
    ki0_k1, ki1_k1 = _pack_kernel_info(k1_sol)

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
    ]

    # Capture K1 intermediates by closure; no index arithmetic needed.
    pb_captured = {}

    def capture_k1(_arguments):
        pb_captured["pb"] = np.asarray(pb_inout.array).reshape(M_padded_k1, N_tiles_N).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=k1_hsaco, kernel_name=k1_name, arguments=args_k1,
        grid_dim=(numWG_k1, 1, 1), block_dim=(k1_sol["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture_k1,
    )

    # Re-wrap so colv2 re-uploads K1's D and divides it in-place.
    d_colv2_inout = amdgpu_exec.InOutArray(d_bf16)
    pb_for_colv2  = np.ascontiguousarray(partial_buf_2d)

    args_cd = [
        d_colv2_inout,
        amdgpu_exec.InputArray(pb_for_colv2),
        np.int32(M), np.int32(N_hidden), np.int32(N_tiles_N),
        np.float32(inv_d), np.float32(eps),
    ]

    grid_cd  = (math.ceil(M / 256), math.ceil(N_hidden / 256), 1)
    block_cd = (256, 1, 1)

    d_final_captured = {}

    def capture_colv2(_arguments):
        raw = np.asarray(d_colv2_inout.array)
        if raw.dtype == ml_dtypes.bfloat16:
            d_final_captured["d"] = raw.astype(np.float32)
        else:
            # Raw uint16: decode via shift trick.
            d_final_captured["d"] = (raw.astype(np.uint32) << 16).view(np.float32)

    amdgpu_exec.execute_hsaco(
        hsaco=cd_hsaco, kernel_name=cd_name, arguments=args_cd,
        grid_dim=grid_cd, block_dim=block_cd,
        num_iterations=1, verify_fn=capture_colv2,
    )

    d_final_f32 = d_final_captured["d"]
    # Ensure shape (M, N_hidden); may come back flat or F-order.
    if d_final_f32.ndim == 1:
        d_final_f32 = np.reshape(d_final_f32, (M, N_hidden), order='F')

    pb_gpu = pb_captured.get("pb", partial_buf_2d)

    return d_final_f32, d_ref_f32, pb_gpu[:M, :], sumsq_ref, M


# ---------------------------------------------------------------------------
# Parametrised pipeline test.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("K", _K_VALUES, ids=[f"K{k}" for k in _K_VALUES])
def test_pipeline_shape(pipeline_kernels, K):
    """Verify pipeline D output across (wg_n, M, K, N_hidden).

    M shapes are derived at runtime from the fixture's actual MacroTile0.
    N_hidden is swept over _N_TILE_MULTS * MT1, covering both single-tile and
    multi-tile (butterfly-reduction) colv2 paths.
    Intermediate buffers (partialBuf) are validated for the first full-tile
    M shape at K=256 (runs once per N-mult, covering both tile counts).
    """
    (k1_sol, *_rest) = pipeline_kernels
    MT0 = k1_sol["MacroTile0"]
    MT1 = _rest[-1]  # last element of fixture tuple is MT1

    n_shapes = _n_shapes_for_mt1(MT1)
    for M, m_label in _m_shapes_for_mt0(MT0):
        for N_hidden in n_shapes:
            validate_intermediates = (M == MT0 and K == 256)
            d_final_f32, d_ref_f32, pb_gpu, sumsq_ref, _M = _run_pipeline(
                pipeline_kernels, M, K, N_hidden, _EPS
            )
            _check_d(d_final_f32, d_ref_f32, M, f"{m_label}_N{N_hidden}")
            if validate_intermediates:
                _check_partialBuf(pb_gpu, sumsq_ref, M, f"{m_label}_N{N_hidden}")


# ---------------------------------------------------------------------------
# Assertion helpers.
# ---------------------------------------------------------------------------

def _check_d(d_gpu_f32, d_ref_f32, M, label):
    """Assert final RMSNorm D output matches reference within bf16 tolerance."""
    rtol, atol = 2e-2, 2e-2
    gpu = d_gpu_f32[:M]
    ref = d_ref_f32[:M]
    bad = np.where(
        ~np.isfinite(gpu) |
        (np.abs(gpu - ref) > atol + rtol * np.abs(ref))
    )
    n_bad = len(bad[0])
    assert n_bad == 0, (
        f"D mismatch ({label}): {n_bad} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(gpu - ref)):.3e}, "
        f"first bad row={bad[0][0]}, col={bad[1][0]}: "
        f"gpu={gpu[bad[0][0], bad[1][0]]:.6f} "
        f"ref={ref[bad[0][0], bad[1][0]]:.6f}"
    )


def _check_partialBuf(pb_gpu, sumsq_ref, M, label):
    """Assert partialBuf intermediate matches reference within tight tolerance."""
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
