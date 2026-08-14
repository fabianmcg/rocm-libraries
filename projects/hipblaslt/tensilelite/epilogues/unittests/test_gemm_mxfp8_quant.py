# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+MXFP8Quant (K1) Subtile epilogue (gfx950, bf16 in / fp8 out).

Exercises a set of (M, K, nHidden) shapes and alpha values, verifying:
  - MXScale (uint8 [ceil(nHidden/q0), ceil(M/q1)], exact byte equality): e8m0 per-block scale
  - D output (OCP e4m3 fp8, compared as float32 with tol=0.1): quantized GEMM result

D is the fp8 output; MXScale is the only epilogue side buffer (1 byte per block).
free0 = nHidden direction (tiled by q0), free1 = M_tokens direction (tiled by q1).
"""

import math

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
from epilogues.tensilelite.numpy_helpers import randBf16, mxfp8QuantReference
from epilogues.tensilelite.yaml_solution_builder import readTestAxes

_k1Yaml = yamlPath("gemm_mxfp8_quant_k1.yaml")

_k1Solutions = enumerateSolutions("gemm_mxfp8_quant_k1.yaml")

_k1Axes = {}
try:
    _k1Axes = readTestAxes(_k1Yaml, "K1", mt1=1)  # mt1 placeholder; expanded below
except Exception:
    pass

_knPairs = _k1Axes.get("KNHidden", [])


# ---------------------------------------------------------------------------
# Session-scoped fixture: one instance per solution in the K1 YAML.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _k1Solutions],
    ids=[sid for _sol, sid in _k1Solutions],
)
def k1_mx_kernel(request):
    """Assemble and compile one K1 MXFP8Quant solution from the benchmark YAML."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


# ---------------------------------------------------------------------------
# Helpers: build kernel args, then execute and collect results.
# ---------------------------------------------------------------------------

def _build_kernel_args(solution, M, K, nHidden, q0, q1, alpha=1.0, zeroInput=False):
    """Allocate buffers, compute references, and build the kernarg list.

    Returns (args, mxScale_buf, dFortran_buf, mxScaleRef, dFp8Ref, mT, nT).
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

    mT = math.ceil(nHidden / q0)
    nT = math.ceil(M / q1)
    # Free dim (nT = M_tokens) padded to ×32; kblock dim (mT = N_hidden/q0) padded to ×8.
    # Matches the (nT, mT) grid passed to swizzleMxScaleGfx950 in the q0=32 path.
    paddedRows = ((nT + 31) // 32) * 32
    paddedCols = ((mT + 7) // 8) * 8
    # MXScale: GFX950 pre-swizzled, flat uint8 [paddedRows*paddedCols].
    mxScale = np.zeros(paddedRows * paddedCols, dtype=np.uint8)

    # Transpose h1 to (nHidden, M) to match free0=nHidden, free1=M layout.
    h1T = (aRow.astype(np.float32) @ bRow.astype(np.float32)).T
    mxScaleRef, dFp8Ref = mxfp8QuantReference(alpha * h1T, q0, q1)

    skArgs = compute_sk3_dp_args(nHidden, M, K, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    # MXFP8Quant uses UseBeta=False: no beta slot in kernarg; epilogue at slot 27.
    args = buildSubtileArgs(
        nHidden, M, K, numWG,
        amdgpu_exec.InOutArray(dFortran), amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        skArgs, ki0, ki1, [amdgpu_exec.InOutArray(mxScale)],
        alpha=np.float32(alpha), hasBeta=False,
    )
    return args, mxScale, dFortran, mxScaleRef, dFp8Ref, mT, nT, numWG


def _runShape(solution, kernelName, hsaco, chip, M, K, nHidden, alpha=1.0, zeroInput=False):
    """Run K1 MXFP8Quant for one (M, K, nHidden, alpha) configuration.

    Returns (mx_gpu, mx_ref, d_gpu_f32, d_ref_f32).
    """
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    q0 = solution.get("_DQuantSize0",MT0)
    q1 = solution.get("_DQuantSize1",MT1)

    args, mxScale, dFortran, mxScaleRef, dFp8Ref, mT, nT, numWG = \
        _build_kernel_args(solution, M, K, nHidden, q0, q1, alpha=alpha, zeroInput=zeroInput)

    result_holder = {}

    def capture(arguments):
        dRaw = np.asarray(arguments[8].array)
        result_holder["d_gpu"] = dRaw.reshape(nHidden, M, order="F")
        # mxScale is at slot 27 (first epilogue arg; no beta slot since UseBeta=False).
        result_holder["mx_gpu"] = np.asarray(arguments[27].array).copy().reshape(-1)

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )
    return (
        result_holder["mx_gpu"].view(np.uint8),
        mxScaleRef,
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
    mxGpu, mxRef, dGpuF32, dRefF32 = _runShape(
        solution, kernelName, hsaco, chip, M, K, nHidden, alpha=alpha
    )
    # MXScale bytes must match exactly; e8m0 bytes are deterministic.
    assert np.array_equal(mxGpu, mxRef), (
        f"{label}: MXScale byte mismatch. "
        f"gpu={mxGpu.ravel()[:8]} ref={mxRef.ravel()[:8]}"
    )
    # MXFP8Quant uses identical power-of-two quantMult on GPU and reference
    # (verified by MXScale exact match). Raw D differences from bfloat16 MFMA
    # vs float32 matmul can cause 1-ULP e4m3 differences (max relative: 1/8 =
    # 12.5%), so rtol=0.15 is the correct tight tolerance for this epilogue.
    assertClose(dGpuF32, dRefF32, label, rtol=0.15, atol=0.15, kind="D_fp8")


# ---------------------------------------------------------------------------
# Test: KNHidden sweep from YAML axes (basic correctness, alpha=1).
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize(
    "K,nHidden",
    _knPairs,
    ids=[f"K{k}-N{n}" for k, n in _knPairs],
)
def test_k1_mx_shape(k1_mx_kernel, K, nHidden):
    """Verify K1 MXFP8Quant outputs: MXScale bytes and fp8 D."""
    solution, kernelName, hsaco, chip = k1_mx_kernel
    MT1 = solution["MacroTile1"]

    for M, mLabel in readTestAxes(_k1Yaml, "K1", mt1=MT1)["M"]:
        mxGpu, mxRef, dGpuF32, dRefF32 = _runShape(
            solution, kernelName, hsaco, chip, M, K, nHidden
        )
        label = f"MT0={solution['MacroTile0']} MT1={MT1} {mLabel} N={nHidden} K={K}"
        assert np.array_equal(mxGpu, mxRef), (
            f"{label}: MXScale byte mismatch. "
            f"gpu={mxGpu.ravel()[:8]} ref={mxRef.ravel()[:8]}"
        )
        # MXFP8Quant uses identical power-of-two quantMult on GPU and reference
        # (verified by MXScale exact match). Raw D differences from bfloat16 MFMA
        # vs float32 matmul can cause 1-ULP e4m3 differences (max relative:
        # 1/8 = 12.5%), so rtol=0.15 is the correct tight tolerance.
        assertClose(dGpuF32, dRefF32, label, rtol=0.15, atol=0.15, kind="D_fp8")


# ---------------------------------------------------------------------------
# Test: multi-workgroup shapes in M and N.
# ---------------------------------------------------------------------------

_multiWgShapes = [
    # (M, K, nHidden)
    (256,  64,  128),   # numWG_M=2, numWG_N=2: multi-WG M
    (512,  64,  128),   # numWG_M=4, numWG_N=2: more M workgroups
    (128,  64,  256),   # numWG_M=1, numWG_N=4: multi-WG N only
    (256,  64,  256),   # numWG_M=2, numWG_N=4: both multi-WG
    (256, 128,  256),   # numWG_M=2, numWG_N=4, larger K
    (512,  64, 1024),   # numWG_M=4, numWG_N=16: large both
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden",
    _multiWgShapes,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _multiWgShapes],
)
def test_k1_mx_multiWG_shape(k1_mx_kernel, M, K, nHidden):
    """Verify K1 MXFP8Quant for multi-workgroup shapes (numWG_M > 1 and/or numWG_N > 1)."""
    solution, kernelName, hsaco, chip = k1_mx_kernel
    _check(solution, kernelName, hsaco, chip, M, K, nHidden)


# ---------------------------------------------------------------------------
# Test: non-tile-aligned M and N (partial quant-tile boundary at the edge).
# ---------------------------------------------------------------------------

_unalignedShapes = [
    # (M, K, nHidden)
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
    "M,K,nHidden",
    _unalignedShapes,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _unalignedShapes],
)
def test_k1_mx_unaligned(k1_mx_kernel, M, K, nHidden):
    """Verify K1 MXFP8Quant on shapes where M or N is not tile-aligned."""
    solution, kernelName, hsaco, chip = k1_mx_kernel
    _check(solution, kernelName, hsaco, chip, M, K, nHidden)


# ---------------------------------------------------------------------------
# Test: non-unit alpha (pre-quantization alpha scaling).
# ---------------------------------------------------------------------------

_ALPHA_SHAPES = [
    # (M, K, nHidden, alpha)
    (128,  64, 128, 2.0),
    (128,  64, 128, 0.5),
    (128,  64, 128, -1.0),
    (256,  64, 256, 2.0),   # multi-WG with non-unit alpha
    (128, 256, 128, 0.25),  # larger K
    (512,  64, 512, 3.0),   # large grid, alpha=3
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden,alpha",
    _ALPHA_SHAPES,
    ids=[f"M{m}-K{k}-N{n}-a{a}" for m, k, n, a in _ALPHA_SHAPES],
)
def test_k1_mx_alpha(k1_mx_kernel, M, K, nHidden, alpha):
    """Verify K1 MXFP8Quant correctly applies alpha before quantization."""
    solution, kernelName, hsaco, chip = k1_mx_kernel
    _check(solution, kernelName, hsaco, chip, M, K, nHidden, alpha=alpha)


# ---------------------------------------------------------------------------
# Test: varying K (reduction depth).
# ---------------------------------------------------------------------------

_kShapes = [
    # (M, K, nHidden)
    (128,   1, 128),    # K=1: minimal reduction
    (128,  32, 128),    # K=32: sub-DepthU
    (128,  64, 128),    # K=64: one DepthU tile
    (128, 128, 128),    # K=128: two DepthU tiles
    (128, 512, 128),    # K=512: eight tiles
    (128, 1024, 128),   # K=1024: sixteen tiles
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden",
    _kShapes,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _kShapes],
)
def test_k1_mx_k_sweep(k1_mx_kernel, M, K, nHidden):
    """Verify K1 MXFP8Quant across a range of reduction depths."""
    solution, kernelName, hsaco, chip = k1_mx_kernel
    _check(solution, kernelName, hsaco, chip, M, K, nHidden)


# ---------------------------------------------------------------------------
# Test: large production-scale shapes.
# ---------------------------------------------------------------------------

_LARGE_SHAPES = [
    # (M, K, nHidden) — representative LLM decode/prefill sizes.
    ( 512,  128, 4096),
    (1024,  128, 4096),
    ( 512,  128, 8192),
    (4096,  128, 4096),
    (2048,  512, 4096),
    (1024, 4096, 1024),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden",
    _LARGE_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _LARGE_SHAPES],
)
def test_k1_mx_large(k1_mx_kernel, M, K, nHidden):
    """Verify K1 MXFP8Quant on production-scale LLM shapes."""
    solution, kernelName, hsaco, chip = k1_mx_kernel
    _check(solution, kernelName, hsaco, chip, M, K, nHidden)


# ---------------------------------------------------------------------------
# Test: all-zero input -> amax=0 produces all-zero D and all-zero MXScale.
# ---------------------------------------------------------------------------

_allZeroShapes = [
    (128, 64, 128),
    (256, 64, 256),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden",
    _allZeroShapes,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _allZeroShapes],
)
def test_k1_mx_all_zero(k1_mx_kernel, M, K, nHidden):
    """amax=0 (all-zero A) must yield all-zero D and all-zero MXScale."""
    solution, kernelName, hsaco, chip = k1_mx_kernel
    mxGpu, mxRef, dGpuF32, dRefF32 = _runShape(
        solution, kernelName, hsaco, chip, M, K, nHidden, zeroInput=True,
    )
    label = (f"all-zero MT{solution['MacroTile0']}x{solution['MacroTile1']}"
             f" M={M} K={K} N={nHidden}")
    assert np.all(mxGpu == 0), f"{label}: MXScale not all zero"
    assert np.all(dGpuF32 == 0.0), f"{label}: D not all zero"
    assert np.array_equal(mxGpu, mxRef), f"{label}: MXScale gpu != ref"
    # MXFP8Quant uses identical power-of-two quantMult on GPU and reference
    # (verified by MXScale exact match). Raw D differences from bfloat16 MFMA
    # vs float32 matmul can cause 1-ULP e4m3 differences (max relative: 1/8 =
    # 12.5%), so rtol=0.15 is the correct tight tolerance for this epilogue.
    assertClose(dGpuF32, dRefF32, label, rtol=0.15, atol=0.15, kind="D_fp8")


# ---------------------------------------------------------------------------
# Test: sub-row path (q0 < mfmaM).
# ---------------------------------------------------------------------------

_SUBROW_SHAPES = [
    (128,  64, 128),
    ( 65,  64,  65),
    (128,  64,  65),
    (256,  64, 256),
    (129,  64, 200),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden",
    _SUBROW_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _SUBROW_SHAPES],
)
def test_k1_mx_subrow(k1_mx_kernel, M, K, nHidden):
    """Verify sub-row MXFP8Quant (q0 < mfmaM) row addressing and OOB-row suppression."""
    solution, kernelName, hsaco, chip = k1_mx_kernel
    if solution.get("_DQuantSize0",solution["MacroTile0"]) >= solution["MatrixInstM"]:
        pytest.skip("not a sub-row (q0<mfmaM) solution")
    _check(solution, kernelName, hsaco, chip, M, K, nHidden)
