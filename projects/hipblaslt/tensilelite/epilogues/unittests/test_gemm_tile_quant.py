# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+TileQuant (K1) Subtile epilogue (gfx950, bf16 in / fp8 out).

Exercises a set of (M, K, N_hidden) shapes and alpha values, verifying:
  - quantScale (fp32 [ceil(N_hidden/q0), ceil(M/q1)], tol=1e-3): per-tile amax/448
  - D output (OCP e4m3 fp8, compared as float32 with tol=0.1): scaled GEMM result

D is the fp8 output; QuantScale is the only epilogue side buffer.
free0 = N_hidden direction (tiled by q0), free1 = M_tokens direction (tiled by q1).
"""

import math
import struct

import numpy as np
import pytest

from epilogue_test_common import (
    amdgpu_exec, ml_dtypes,
    requires_gfx950, yamlPath,
    enumerateSolutions, assertClose,
)
from epilogues.tensilelite.partialrms_helpers import (
    compute_sk3_dp_args, _pack_kernel_info, compileSolution, buildSubtileArgs,
)
from epilogues.tensilelite.numpy_helpers import randBf16, tileQuantReference
from epilogues.tensilelite.yaml_solution_builder import readTestAxes

_K1_YAML = yamlPath("gemm_tile_quant_k1.yaml")

_K1_SOLUTIONS = enumerateSolutions("gemm_tile_quant_k1.yaml")

_K1_AXES = {}
try:
    _K1_AXES = readTestAxes(_K1_YAML, "K1", mt1=1)  # mt1 placeholder; expanded below
except Exception:
    pass

_KN_PAIRS = _K1_AXES.get("KNHidden", [])


# ---------------------------------------------------------------------------
# Session-scoped fixture: one instance per solution in the K1 YAML.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _K1_SOLUTIONS],
    ids=[sid for _sol, sid in _K1_SOLUTIONS],
)
def k1_tq_kernel(request):
    """Assemble and compile one K1 TileQuant solution from the benchmark YAML."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


# ---------------------------------------------------------------------------
# Helpers: build kernel args, then execute and collect results.
# ---------------------------------------------------------------------------

def _build_kernel_args(solution, M, K, nHidden, q0, q1, alpha=1.0, zeroInput=False):
    """Allocate buffers, compute references, and build the kernarg list.

    Returns (args, quantScale_buf, dFortran_buf, qsRef, dFp8Ref, qsMTiles, qsNTiles).
    """
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    numWG = math.ceil(nHidden / MT0) * math.ceil(M / MT1)

    rng = np.random.default_rng(seed=M * 10000 + K)
    aRow = randBf16(rng, (M, K))
    bRow = randBf16(rng, (K, nHidden))
    if zeroInput:
        aRow = np.zeros((M, K), dtype=ml_dtypes.bfloat16)

    bFortran  = np.asfortranarray(bRow)
    aFortranT = np.asfortranarray(aRow.T)
    cFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    # D is OCP e4m3 fp8; allocate Fortran order (nHidden × M).
    dFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.float8_e4m3fn, order="F")

    qsMTiles = math.ceil(nHidden / q0)
    qsNTiles = math.ceil(M / q1)
    quantScale = np.zeros((qsMTiles, qsNTiles), dtype=np.float32, order="C")

    # Transpose h1 to (nHidden, M) to match free0=nHidden, free1=M layout.
    h1T = (aRow.astype(np.float32) @ bRow.astype(np.float32)).T
    qsRef, dFp8Ref = tileQuantReference(alpha * h1T, q0, q1)

    skArgs = compute_sk3_dp_args(nHidden, M, K, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    # TileQuant uses UseBeta=False: no beta slot in kernarg; epilogue at slot 27.
    args = buildSubtileArgs(
        nHidden, M, K, numWG,
        amdgpu_exec.InOutArray(dFortran), amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        skArgs, ki0, ki1, [amdgpu_exec.InOutArray(quantScale)],
        alpha=np.float32(alpha), hasBeta=False,
    )
    return args, quantScale, dFortran, qsRef, dFp8Ref, qsMTiles, qsNTiles, numWG


def _run_shape(solution, kernelName, hsaco, chip, M, K, nHidden, alpha=1.0, zeroInput=False):
    """Run K1 TileQuant for one (M, K, nHidden, alpha) configuration.

    Returns (qs_gpu, qs_ref, d_gpu_f32, d_ref_f32).
    """
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    q0 = solution.get("_DQuantSize0",MT0)
    q1 = solution.get("_DQuantSize1",MT1)

    args, quantScale, dFortran, qsRef, dFp8Ref, qsMTiles, qsNTiles, numWG = \
        _build_kernel_args(solution, M, K, nHidden, q0, q1, alpha=alpha, zeroInput=zeroInput)

    result_holder = {}

    def capture(arguments):
        # D is at arg slot 8; Fortran (nHidden × M) → row-major (nHidden, M).
        dRaw = np.asarray(arguments[8].array)
        result_holder["d_gpu"] = dRaw.reshape(nHidden, M, order="F")
        # quantScale is at slot 27 (first epilogue arg; no beta slot since UseBeta=False).
        result_holder["qs_gpu"] = np.asarray(arguments[27].array).copy().reshape(qsMTiles, qsNTiles)

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    return (
        result_holder["qs_gpu"],
        qsRef,
        result_holder["d_gpu"].astype(np.float32),
        dFp8Ref.astype(np.float32),
    )


def _check(solution, kernelName, hsaco, chip, M, K, nHidden, alpha=1.0):
    """Run and assert correctness; return a descriptive label on pass."""
    q0 = solution.get("_DQuantSize0",solution["MacroTile0"])
    q1 = solution.get("_DQuantSize1",solution["MacroTile1"])
    numWgM = math.ceil(M / solution["MacroTile1"])
    numWgN = math.ceil(nHidden / solution["MacroTile0"])
    label = (
        f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
        f"Q=[{q0},{q1}] M={M} K={K} N={nHidden} "
        f"wgM={numWgM} wgN={numWgN} alpha={alpha}"
    )
    qsGpu, qsRef, dGpuF32, dRefF32 = _run_shape(
        solution, kernelName, hsaco, chip, M, K, nHidden, alpha=alpha
    )
    assertClose(qsGpu, qsRef, label, rtol=1e-3, atol=1e-3, kind="quantScale")
    assertClose(dGpuF32, dRefF32, label, rtol=0.1, atol=0.1, kind="D_fp8")


# ---------------------------------------------------------------------------
# Test: KNHidden sweep from YAML axes (basic correctness, alpha=1).
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize(
    "K,N_hidden",
    _KN_PAIRS,
    ids=[f"K{k}-N{n}" for k, n in _KN_PAIRS],
)
def test_k1_tq_shape(k1_tq_kernel, K, N_hidden):
    """Verify K1 TileQuant outputs: quantScale and fp8 D."""
    solution, kernelName, hsaco, chip = k1_tq_kernel
    MT1 = solution["MacroTile1"]

    for M, mLabel in readTestAxes(_K1_YAML, "K1", mt1=MT1)["M"]:
        qsGpu, qsRef, dGpuF32, dRefF32 = _run_shape(
            solution, kernelName, hsaco, chip, M, K, N_hidden
        )
        label = f"MT0={solution['MacroTile0']} MT1={MT1} {mLabel} N={N_hidden} K={K}"

        assertClose(qsGpu, qsRef, label, rtol=1e-3, atol=1e-3, kind="quantScale")

        # Compare fp8-encoded values as float32. OCP e4m3 has ~3 mantissa bits;
        # rtol=0.1 accounts for the coarser fp8 quantisation grid.
        assertClose(dGpuF32, dRefF32, label, rtol=0.1, atol=0.1, kind="D_fp8")


# ---------------------------------------------------------------------------
# Test: multi-workgroup shapes in M and N.
# ---------------------------------------------------------------------------

# Exercises multiple workgroups in M (numWG_M > 1) and/or N, and sub-macro-tile
# Q shapes (Q=[16,16] or Q=[32,32]) which put nQTilesM > 1 inside a wave when
# wg_m=2. These cases stress the waveM offset in _tileByteOffset for
# MIWaveGroup=[2,1] kernels.
#
# With MT1=128, numWG_M = ceil(M / MT1):
#   M=256 -> numWG_M=2   M=512 -> numWG_M=4
# With MT0=64, numWG_N = ceil(N_hidden / MT0):
#   N=256 -> numWG_N=4   N=1024 -> numWG_N=16
_MULTI_WG_SHAPES = [
    # (M, K, N_hidden)
    (256,  64,  128),   # numWG_M=2, numWG_N=2: multi-WG M
    (512,  64,  128),   # numWG_M=4, numWG_N=2: more M workgroups
    (128,  64,  256),   # numWG_M=1, numWG_N=4: multi-WG N only
    (256,  64,  256),   # numWG_M=2, numWG_N=4: both multi-WG
    (256, 128,  256),   # numWG_M=2, numWG_N=4, larger K
    (512,  64, 1024),   # numWG_M=4, numWG_N=16: large both
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _MULTI_WG_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _MULTI_WG_SHAPES],
)
def test_k1_tq_multiWG_shape(k1_tq_kernel, M, K, N_hidden):
    """Verify K1 TileQuant for multi-workgroup shapes (numWG_M > 1 and/or numWG_N > 1)."""
    solution, kernelName, hsaco, chip = k1_tq_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Test: non-tile-aligned M and N (partial quant-tile boundary at the edge).
# ---------------------------------------------------------------------------

# These shapes have M or N not divisible by Q, exercising the ceiling-division
# boundary path in both the reference and the runtime ceil(N/Q1) computation.
_UNALIGNED_SHAPES = [
    # (M, K, N_hidden) — chosen so N is not a multiple of q0, M not of q1.
    ( 17,  64,  80),    # M=17 not MT1-aligned, N=80 not MT0-aligned
    (113,  64, 200),    # non-round M and N
    (200,  64, 113),    # swap of above
    (129, 128, 129),    # one past a power-of-two in both dims
    (255,  64, 255),    # one below 256 in both dims
    (257,  64, 257),    # one above 256 in both dims
    (  1,  64,  64),    # M=1: single row edge case
    ( 64,  64,   1),    # N=1: single column edge case (N < q0)
    (  1,  64,   1),    # both M=1 and N=1
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _UNALIGNED_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _UNALIGNED_SHAPES],
)
def test_k1_tq_unaligned(k1_tq_kernel, M, K, N_hidden):
    """Verify K1 TileQuant on shapes where M or N is not tile-aligned."""
    solution, kernelName, hsaco, chip = k1_tq_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Test: non-unit alpha (pre-quantization alpha scaling).
# ---------------------------------------------------------------------------

# Alpha is applied before amax, so quantScale and D must both reflect it.
# Tests alpha=2 (scale up), alpha=0.5 (scale down), and alpha=-1 (sign flip).
_ALPHA_SHAPES = [
    # (M, K, N_hidden, alpha)
    (128,  64, 128, 2.0),
    (128,  64, 128, 0.5),
    (128,  64, 128, -1.0),
    (256,  64, 256, 2.0),   # multi-WG with non-unit alpha
    (128, 256, 128, 0.25),  # larger K
    (512,  64, 512, 3.0),   # large grid, alpha=3
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden,alpha",
    _ALPHA_SHAPES,
    ids=[f"M{m}-K{k}-N{n}-a{a}" for m, k, n, a in _ALPHA_SHAPES],
)
def test_k1_tq_alpha(k1_tq_kernel, M, K, N_hidden, alpha):
    """Verify K1 TileQuant correctly applies alpha before quantization."""
    solution, kernelName, hsaco, chip = k1_tq_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden, alpha=alpha)


# ---------------------------------------------------------------------------
# Test: varying K (reduction depth).
# ---------------------------------------------------------------------------

# K affects the accumulator magnitude and how far fp32 accumulators are from
# zero. Small K → nearly-zero accumulators; large K → larger magnitudes that
# stress fp8 saturation.
_K_SHAPES = [
    # (M, K, N_hidden)
    (128,   1, 128),    # K=1: minimal reduction
    (128,  32, 128),    # K=32: sub-DepthU
    (128,  64, 128),    # K=64: one DepthU tile
    (128, 128, 128),    # K=128: two DepthU tiles
    (128, 512, 128),    # K=512: eight tiles
    (128, 1024, 128),   # K=1024: sixteen tiles
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _K_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _K_SHAPES],
)
def test_k1_tq_k_sweep(k1_tq_kernel, M, K, N_hidden):
    """Verify K1 TileQuant across a range of reduction depths."""
    solution, kernelName, hsaco, chip = k1_tq_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Test: large production-scale shapes.
# ---------------------------------------------------------------------------

_LARGE_SHAPES = [
    # (M, K, N_hidden) — representative LLM decode/prefill sizes.
    ( 512,  128, 4096),
    (1024,  128, 4096),
    ( 512,  128, 8192),
    (4096,  128, 4096),
    (2048,  512, 4096),
    (1024, 4096, 1024),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _LARGE_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _LARGE_SHAPES],
)
def test_k1_tq_large(k1_tq_kernel, M, K, N_hidden):
    """Verify K1 TileQuant on production-scale LLM shapes."""
    solution, kernelName, hsaco, chip = k1_tq_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Test: all-zero input -> amax=0 produces all-zero D and all-zero QuantScale.
# ---------------------------------------------------------------------------

_ALL_ZERO_SHAPES = [
    (128, 64, 128),
    (256, 64, 256),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _ALL_ZERO_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _ALL_ZERO_SHAPES],
)
def test_k1_tq_all_zero(k1_tq_kernel, M, K, N_hidden):
    """amax=0 (all-zero A) must yield all-zero D and all-zero QuantScale."""
    solution, kernelName, hsaco, chip = k1_tq_kernel
    qsGpu, qsRef, dGpuF32, dRefF32 = _run_shape(
        solution, kernelName, hsaco, chip, M, K, N_hidden, zeroInput=True,
    )
    label = (f"all-zero MT{solution['MacroTile0']}x{solution['MacroTile1']}"
             f" M={M} K={K} N={N_hidden}")
    assert np.all(qsGpu == 0.0), f"{label}: QuantScale not all zero"
    assert np.all(dGpuF32 == 0.0), f"{label}: D not all zero"
    assertClose(qsGpu, qsRef, label, rtol=1e-3, atol=1e-3, kind="quantScale")
    assertClose(dGpuF32, dRefF32, label, rtol=0.1, atol=0.1, kind="D_fp8")


_SUBROW_SHAPES = [
    (128,  64, 128),
    ( 65,  64,  65),
    (128,  64,  65),
    (256,  64, 256),
    (129,  64, 200),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _SUBROW_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _SUBROW_SHAPES],
)
def test_k1_tq_subrow(k1_tq_kernel, M, K, N_hidden):
    """Verify sub-row TileQuant (q0<mfmaM) row addressing and OOB-row suppression."""
    solution, kernelName, hsaco, chip = k1_tq_kernel
    if solution.get("_DQuantSize0",solution["MacroTile0"]) >= solution["MatrixInstM"]:
        pytest.skip("not a sub-row (q0<mfmaM) solution")
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Debug test: verify fp8 store path reads accumulators in the expected order.
#
# Strategy: inject a position-encoded value float(linId) into each accumulator
# element acc[n, m, k], where linId = n*mmaM*rowsPerLane + m*rowsPerLane + k + 1.
# Set quantMult=1.0 and scaleDequant=1/448 so the store path writes fp8(linId)
# without any additional scaling (since the TileQuant epilogue writes
# acc * quantMult into D, and quantMult=1 means acc passes through unchanged).
#
# The expected mapping from (linId, laneId) to (global_row, global_col) is:
#   col = laneId % mfmaN
#   rowGroup = laneId // mfmaN
#   n = (col + N_base_in_wave) // mfmaN   <- simplified: col determines n via
#       col_in_wave = n * mfmaN + col_lane, so n = col // mfmaN is always 0
#       since col_lane = laneId % mfmaN < mfmaN.  Wait — in a multi-N wave:
#       global_col_within_wave = n * mfmaN + (laneId % mfmaN)
#   global_row_within_wave = m * mfmaM + rowGroup * rowsPerLane + k
#
# So given global position (row, col) within the wave tile:
#   n = col // mfmaN
#   col_lane = col % mfmaN                 (= laneId % mfmaN)
#   rowGroup * rowsPerLane + offset = row % mfmaM ... this is more complex.
#   m = row // mfmaM
#   k = row % mfmaM   ... but rowsPerLane may be <mfmaM; rowGroup covers the rest.
#
# The cleaner check: for each output element, compute what linId the store should
# have used, then verify D_fp8[row, col] == fp8(expected_linId).
#
# NOTE: The TileQuant emitter writes quantMult=1.0 into the quantMult VGPRs.
# The _applyScaleInPlace path is SKIPPED (debug mode replaces it with
# _injectKnownValues). The store path therefore converts acc values as-is to fp8.
# Since linId values (1..mmaN*mmaM*rowsPerLane) are all small integers well within
# fp8 range (<=448), fp8(float(linId)) == float(linId) exactly for those values
# that are representable (e4m3 has 3 mantissa bits; integers 1-8 are exact, larger
# ones may round to the nearest representable value).
# ---------------------------------------------------------------------------


def _fp8_representable(val):
    """Convert float val to OCP e4m3 fp8 and back to float32, for comparison."""
    arr = np.array([val], dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
    return float(arr.astype(np.float32)[0])


def _expected_linid(n, m, k, mmaM, rowsPerLane):
    """Position-encoded id injected into acc[n, m, k]."""
    return n * mmaM * rowsPerLane + m * rowsPerLane + k + 1


def _build_debug_reference(nHidden, M, sol):
    """Build reference D array for the debug-inject test.

    For each wave tile covering output rows [waveRowBase, waveRowBase+mmaM*mfmaM)
    and cols [waveColBase, waveColBase+mmaN*mfmaN):
      D[waveRowBase + m*mfmaM + rowGroup*rowsPerLane + k,
        waveColBase + n*mfmaN + laneMod]  = fp8(linId)
    where linId = n*mmaM*rowsPerLane + m*rowsPerLane + k + 1,
    and laneMod = col % mfmaN (all lanes with the same laneMod get the same value).

    All lanes in a wavegroup compute the same (n,m,k) -> linId mapping, so the
    value at each (row, col) is determined solely by the tile index (n,m,k).
    """
    mfmaM = sol["MatrixInstM"]
    mfmaN = sol["MatrixInstN"]
    waveSize = sol["WavefrontSize"]
    rowsPerLane = (mfmaM * mfmaN) // waveSize
    wg = sol["MIWaveGroup"]
    wgM = wg[0]
    wgN = wg[1]
    mmaM = (sol["MacroTile0"] // mfmaM) // wgM   # free0=nHidden dimension
    mmaN = (sol["MacroTile1"] // mfmaN) // wgN   # free1=M dimension

    MT0 = sol["MacroTile0"]   # N_hidden direction (free0)
    MT1 = sol["MacroTile1"]   # M direction (free1)

    ref = np.zeros((nHidden, M), dtype=np.float32)

    # Iterate over all workgroups and all waves within each workgroup.
    # WG0/wgM tile the nHidden (free0) direction; WG1/wgN tile M (free1).
    for wgIdN in range(math.ceil(nHidden / MT0)):
        for wgIdM in range(math.ceil(M / MT1)):
            for wvM in range(wgM):
                for wvN in range(wgN):
                    # Base output coordinates for this wave.
                    nhBase = wgIdN * MT0 + wvM * mmaM * mfmaM   # free0 = nHidden
                    mBase  = wgIdM * MT1 + wvN * mmaN * mfmaN   # free1 = M
                    for n in range(mmaN):        # M-tile index (free1)
                        for m in range(mmaM):    # nHidden-tile index (free0)
                            for k in range(rowsPerLane):
                                linId = _expected_linid(n, m, k, mmaM, rowsPerLane)
                                expected = _fp8_representable(float(linId))
                                # The value depends only on (n, m, k), not laneId, so
                                # every lane writes the same value to its own position:
                                #   free0 (nHidden) = m*mfmaM + rowGroup*rowsPerLane + k
                                #   free1 (M)       = n*mfmaN + (laneId % mfmaN)
                                for laneId in range(waveSize):
                                    colLane = laneId % mfmaN     # free1 (M) within mfma
                                    rowGroup = laneId // mfmaN   # free0 (nHidden) within mfma
                                    nh = nhBase + m * mfmaM + rowGroup * rowsPerLane + k
                                    mm = mBase + n * mfmaN + colLane
                                    if nh < nHidden and mm < M:
                                        ref[nh, mm] = expected
    return ref


@requires_gfx950
def test_tq_store_layout(k1_tq_kernel):
    """Verify that the fp8 D store reads accumulators in N-outer M-inner k-inner order.

    Injects position-encoded values into each accumulator element via
    _TQ_DebugInjectKnown=True, then checks that each D output element contains
    the fp8 encoding of the value that should have been written by the store path
    if it reads accumulators in the documented (n, m, k) order.

    A mismatch reveals that the store path reads a different VGPR order than
    the TileQuant emitter assumed when writing the scaled values.
    """
    solution, _kernelName, _hsaco, _chip = k1_tq_kernel

    # Compile a fresh kernel with the debug-inject flag set.
    solution["_TQ_DebugInjectKnown"] = True
    try:
        kernelName, hsaco, chip = compileSolution(solution)
    finally:
        # Reset the debug flag so later tests using this fixture are unaffected.
        solution["_TQ_DebugInjectKnown"] = False

    # Use a single-WG shape that fits within exactly one macro tile.
    M = solution["MacroTile1"]
    K = 64
    nHidden = solution["MacroTile0"]
    q0 = solution.get("_DQuantSize0",nHidden)
    q1 = solution.get("_DQuantSize1",M)

    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    numWG = math.ceil(nHidden / MT0) * math.ceil(M / MT1)

    # Allocate buffers — input values don't matter since accumulators are overwritten.
    rng = np.random.default_rng(seed=42)
    aRow = randBf16(rng, (M, K))
    bRow = randBf16(rng, (K, nHidden))
    bFortran  = np.asfortranarray(bRow)
    aFortranT = np.asfortranarray(aRow.T)
    cFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    dFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.float8_e4m3fn, order="F")

    qsMTiles = math.ceil(nHidden / q0)
    qsNTiles = math.ceil(M / q1)
    quantScale = np.zeros((qsMTiles, qsNTiles), dtype=np.float32, order="C")

    skArgs = compute_sk3_dp_args(nHidden, M, K, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    args = buildSubtileArgs(
        nHidden, M, K, numWG,
        amdgpu_exec.InOutArray(dFortran), amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        skArgs, ki0, ki1, [amdgpu_exec.InOutArray(quantScale)],
        alpha=np.float32(1.0), hasBeta=False,
    )

    result_holder = {}

    def capture(arguments):
        dRaw = np.asarray(arguments[8].array)
        result_holder["d_gpu"] = dRaw.reshape(nHidden, M, order="F").astype(np.float32)
        result_holder["qs_gpu"] = np.asarray(arguments[27].array).copy().reshape(qsMTiles, qsNTiles)

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    dGpu = result_holder["d_gpu"]   # shape (nHidden, M), float32
    dRef = _build_debug_reference(nHidden, M, solution)

    label = (f"debug-inject MT{MT0}x{MT1} M={M} K={K} N={nHidden} "
             f"Q=[{q0},{q1}]")

    # Print first few values to aid diagnosis on failure.
    print(f"\n[test_tq_store_layout] {label}")
    print(f"  dGpu[0:4, 0:8] = {dGpu[:4, :8]}")
    print(f"  dRef[0:4, 0:8] = {dRef[:4, :8]}")

    bad = np.where(np.abs(dGpu - dRef) > 0.5)
    nBad = len(bad[0])
    if nBad > 0:
        r0, c0 = bad[0][0], bad[1][0]
        # Determine which (n, m, k) the reference expects at this position.
        mfmaM = solution["MatrixInstM"]
        mfmaN = solution["MatrixInstN"]
        waveSize = solution["WavefrontSize"]
        rowsPerLane = (mfmaM * mfmaN) // waveSize
        wg = solution["MIWaveGroup"]
        wgM, wgN = wg[0], wg[1]
        mmaM = (MT1 // mfmaM) // wgM
        mmaN = (MT0 // mfmaN) // wgN
        # Decode reference expected value.
        refVal = dRef[r0, c0]
        print(f"  First bad element: dGpu[{r0},{c0}]={dGpu[r0,c0]:.3f} "
              f"dRef[{r0},{c0}]={refVal:.3f}")
        print(f"  Pattern: mfmaM={mfmaM} mfmaN={mfmaN} mmaM={mmaM} mmaN={mmaN} "
              f"rowsPerLane={rowsPerLane} wgM={wgM} wgN={wgN}")
        # Show column pattern: which dGpu cols differ from ref in row 0.
        row0_diff = np.where(np.abs(dGpu[:, 0] - dRef[:, 0]) > 0.5)[0]
        print(f"  Bad cols in row=0: {row0_diff[:16]}")
        row0_ratio = np.where(dRef[:, 0] > 0, dGpu[:, 0] / np.maximum(dRef[:, 0], 1e-9), np.nan)
        print(f"  dGpu[:,0] / dRef[:,0] (first 16): {row0_ratio[:16]}")

    assert nBad == 0, (
        f"{label}: {nBad}/{dGpu.size} elements mismatch between injected D and expected "
        f"store-layout reference. First bad: dGpu[{bad[0][0]},{bad[1][0]}]="
        f"{dGpu[bad[0][0],bad[1][0]]:.3f} expected {dRef[bad[0][0],bad[1][0]]:.3f}"
    )
