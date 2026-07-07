# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end pipeline test for K1 → partial_rms_epilogue (gfx950, bf16).

Verifies that chaining:
  K1: GEMM + PartialRMS  → D (bf16 col-major M×N_hidden), partialBuf (f32 row-major M_padded×N_tiles_N)
  partial_rms_epilogue: RMSNorm in-place → D /= sqrt(invD * sum_t(partialBuf[m,t]) + eps)

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
        (8191, "M8191_irr"),  # 8192 - 1 (near partial_rms_epilogue M limit)
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
# Session-scoped fixture: build K1 and partial_rms_epilogue once per wg_n.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=_WG_N_VALUES,
    ids=[f"wg_n{n}" for n in _WG_N_VALUES],
)
def pipeline_kernels(request):
    """Build K1 (GEMM+PartialRMS) and partial_rms_epilogue (RMSNorm) for one wg_n value.

    Returns (k1Sol, k1Name, k1Hsaco, cdName, cdHsaco, chip, MT1).
    """
    sys.path.insert(0, TENSILE_ROOT)
    sys.path.insert(0, TOOLS_DIR)
    from gemm_partialrms_colv2_helpers import setup_tensile, build_k1_solution, generate_asm
    from PartialRmsEpilogueGenerator import build_partial_rms_epilogue

    wgN = request.param
    chip = amdgpu_exec.get_chip()

    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    k1Sol = build_k1_solution(chip, assembler, isaInfoMap, wgN=wgN)
    k1Asm, k1Name = generate_asm(k1Sol, assembler, debugConfig)
    k1Hsaco = amdgpu_exec.compile_asm_to_hsaco(k1Asm, chip)

    # Build partial_rms_epilogue once with M=1; the hsaco is valid for all M <= 32767
    # because the kernel reads M from kernargs at runtime.
    cdAsm, cdName = build_partial_rms_epilogue(chip, M=1)
    cdHsaco = amdgpu_exec.compile_asm_to_hsaco(cdAsm, chip)

    MT1 = k1Sol["MacroTile1"]

    return (k1Sol, k1Name, k1Hsaco, cdName, cdHsaco, chip, MT1)


# ---------------------------------------------------------------------------
# Helper: run full two-kernel pipeline for one (M, K) shape.
# ---------------------------------------------------------------------------

def _run_pipeline(pipelineKernelsFixture, M, K, nHidden, eps):
    from gemm_partialrms_colv2_helpers import compute_sk3_dp_args, _pack_kernel_info

    (k1Sol, k1Name, k1Hsaco, cdName, cdHsaco, chip, MT1) = pipelineKernelsFixture

    MT0 = k1Sol["MacroTile0"]
    nTilesN   = math.ceil(nHidden / MT1)
    mPaddedK1 = math.ceil(M / MT0) * MT0
    numWgK1   = math.ceil(M / MT0) * nTilesN

    rng = np.random.default_rng(seed=M * 100000 + nHidden * 100 + K)

    aF32   = np.asfortranarray(rng.random((K, M),       dtype=np.float32) * 0.1)
    w0F32  = np.asfortranarray(rng.random((K, nHidden), dtype=np.float32) * 0.1)
    aBf16  = np.asfortranarray(aF32.astype(ml_dtypes.bfloat16))
    w0Bf16 = np.asfortranarray(w0F32.astype(ml_dtypes.bfloat16))
    gammaF32  = rng.random(nHidden, dtype=np.float32) + 0.5
    gammaBf16 = gammaF32.astype(ml_dtypes.bfloat16)

    # Numpy reference.
    aRef     = np.asarray(aBf16).astype(np.float32)
    w0Ref    = np.asarray(w0Bf16).astype(np.float32)
    h1       = aRef.T @ w0Ref                              # (M, nHidden) f32
    gammaRef = np.asarray(gammaBf16).astype(np.float32)

    # Per-tile sum-of-squares reference.
    sumsqRef = np.zeros((M, nTilesN), dtype=np.float32)
    for t in range(nTilesN):
        colLo = t * MT1
        colHi = min((t + 1) * MT1, nHidden)
        sumsqRef[:, t] = np.sum(h1[:, colLo:colHi] ** 2, axis=1)

    # End-to-end D reference: bf16(h1*gamma) / sqrt(invD * Σsumsq + eps).
    invD        = 1.0 / nHidden
    h1GammaF32  = (h1 * gammaRef[np.newaxis, :]).astype(ml_dtypes.bfloat16).astype(np.float32)
    rowSums     = sumsqRef.sum(axis=1)
    rmsDenom    = np.sqrt(invD * rowSums + eps)
    dRefF32     = h1GammaF32 / rmsDenom[:, np.newaxis]

    # Device buffers.
    dBf16        = np.zeros((M, nHidden), dtype=ml_dtypes.bfloat16, order='F')
    cK1Bf16      = np.zeros((M, nHidden), dtype=ml_dtypes.bfloat16, order='F')
    partialBuf2d = np.zeros((mPaddedK1, nTilesN), dtype=np.float32, order='C')
    wsDummy      = np.zeros(4, dtype=np.float32)
    flagsDummy   = np.zeros(4, dtype=np.float32)

    sk1            = compute_sk3_dp_args(M, nHidden, K, k1Sol)
    ki0K1, ki1K1 = _pack_kernel_info(k1Sol)

    dInout  = amdgpu_exec.InOutArray(dBf16)
    pbInout = amdgpu_exec.InOutArray(partialBuf2d)

    argsK1 = [
        np.uint32(1), ki0K1, ki1K1, np.uint32(numWgK1),
        np.uint32(M), np.uint32(nHidden), np.uint32(1), np.uint32(K),
        dInout,
        amdgpu_exec.InputArray(cK1Bf16),
        amdgpu_exec.InputArray(aBf16),
        amdgpu_exec.InputArray(w0Bf16),
        amdgpu_exec.InputArray(wsDummy),
        amdgpu_exec.InputArray(flagsDummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk1["iters_per_tile"], sk1["magic_iters_per_tile"], sk1["shift_iters_per_tile"],
        sk1["sk_iters_per_wg"], sk1["sk_grid"], sk1["sk_tiles"],
        amdgpu_exec.InputArray(gammaBf16),
        pbInout,
    ]

    # Capture K1 intermediates by closure; no index arithmetic needed.
    pbCaptured = {}

    def capture_k1(_arguments):
        pbCaptured["pb"] = np.asarray(pbInout.array).reshape(mPaddedK1, nTilesN).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=k1Hsaco, kernel_name=k1Name, arguments=argsK1,
        grid_dim=(numWgK1, 1, 1), block_dim=(k1Sol["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture_k1,
    )

    # Re-wrap so partial_rms_epilogue re-uploads K1's D and divides it in-place.
    dCdInout = amdgpu_exec.InOutArray(dBf16)
    pbForCd  = np.ascontiguousarray(partialBuf2d)

    argsCd = [
        dCdInout,
        amdgpu_exec.InputArray(pbForCd),
        np.int32(M), np.int32(nHidden), np.int32(nTilesN),
        np.float32(invD), np.float32(eps),
    ]

    gridCd  = (math.ceil(M / 256), math.ceil(nHidden / 256), 1)
    blockCd = (256, 1, 1)

    dFinalCaptured = {}

    def capture_partial_rms_epilogue(_arguments):
        raw = np.asarray(dCdInout.array)
        if raw.dtype == ml_dtypes.bfloat16:
            dFinalCaptured["d"] = raw.astype(np.float32)
        else:
            # Raw uint16: decode via shift trick.
            dFinalCaptured["d"] = (raw.astype(np.uint32) << 16).view(np.float32)

    amdgpu_exec.execute_hsaco(
        hsaco=cdHsaco, kernel_name=cdName, arguments=argsCd,
        grid_dim=gridCd, block_dim=blockCd,
        num_iterations=1, verify_fn=capture_partial_rms_epilogue,
    )

    dFinalF32 = dFinalCaptured["d"]
    # Ensure shape (M, nHidden); may come back flat or F-order.
    if dFinalF32.ndim == 1:
        dFinalF32 = np.reshape(dFinalF32, (M, nHidden), order='F')

    pbGpu = pbCaptured.get("pb", partialBuf2d)

    return dFinalF32, dRefF32, pbGpu[:M, :], sumsqRef, M


# ---------------------------------------------------------------------------
# Parametrised pipeline test.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("K", _K_VALUES, ids=[f"K{k}" for k in _K_VALUES])
def test_pipeline_shape(pipeline_kernels, K):
    """Verify pipeline D output across (wg_n, M, K, nHidden).

    M shapes are derived at runtime from the fixture's actual MacroTile0.
    nHidden is swept over _N_TILE_MULTS * MT1, covering both single-tile and
    multi-tile (butterfly-reduction) partial_rms_epilogue paths.
    Intermediate buffers (partialBuf) are validated for the first full-tile
    M shape at K=256 (runs once per N-mult, covering both tile counts).
    """
    (k1Sol, *_rest) = pipeline_kernels
    MT0 = k1Sol["MacroTile0"]
    MT1 = _rest[-1]  # last element of fixture tuple is MT1.

    nShapes = _n_shapes_for_mt1(MT1)
    for M, mLabel in _m_shapes_for_mt0(MT0):
        for nHidden in nShapes:
            validateIntermediates = (M == MT0 and K == 256)
            dFinalF32, dRefF32, pbGpu, sumsqRef, _M = _run_pipeline(
                pipeline_kernels, M, K, nHidden, _EPS
            )
            _check_d(dFinalF32, dRefF32, M, f"{mLabel}_N{nHidden}")
            if validateIntermediates:
                _check_partial_buf(pbGpu, sumsqRef, M, f"{mLabel}_N{nHidden}")


# ---------------------------------------------------------------------------
# _MAX_M guard test.
# ---------------------------------------------------------------------------

def test_partial_rms_epilogue_max_m_guard():
    """Verify that build_partial_rms_epilogue raises ValueError for M >= 32768."""
    from PartialRmsEpilogueGenerator import build_partial_rms_epilogue
    with pytest.raises(ValueError, match="exceeds"):
        build_partial_rms_epilogue("gfx950", M=32768, N=256, K=64)


# ---------------------------------------------------------------------------
# Assertion helpers.
# ---------------------------------------------------------------------------

def _check_d(dGpuF32, dRefF32, M, label):
    """Assert final RMSNorm D output matches reference within bf16 tolerance."""
    rtol, atol = 2e-2, 2e-2
    gpu = dGpuF32[:M]
    ref = dRefF32[:M]
    bad = np.where(
        ~np.isfinite(gpu) |
        (np.abs(gpu - ref) > atol + rtol * np.abs(ref))
    )
    nBad = len(bad[0])
    assert nBad == 0, (
        f"D mismatch ({label}): {nBad} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(gpu - ref)):.3e}, "
        f"first bad row={bad[0][0]}, col={bad[1][0]}: "
        f"gpu={gpu[bad[0][0], bad[1][0]]:.6f} "
        f"ref={ref[bad[0][0], bad[1][0]]:.6f}"
    )


def _check_partial_buf(pbGpu, sumsqRef, M, label):
    """Assert partialBuf intermediate matches reference within tight tolerance."""
    rtol, atol = 1e-4, 1e-4
    bad = np.where(
        ~np.isfinite(pbGpu[:M, :]) |
        (np.abs(pbGpu[:M, :] - sumsqRef) > atol + rtol * np.abs(sumsqRef))
    )
    nBad = len(bad[0])
    assert nBad == 0, (
        f"partialBuf mismatch ({label}): {nBad} entries out of tolerance. "
        f"max_abs={np.nanmax(np.abs(pbGpu[:M, :] - sumsqRef)):.3e}, "
        f"first bad [{bad[0][0]},{bad[1][0]}]: "
        f"gpu={pbGpu[bad[0][0], bad[1][0]]:.6f} ref={sumsqRef[bad[0][0], bad[1][0]]:.6f}"
    )
