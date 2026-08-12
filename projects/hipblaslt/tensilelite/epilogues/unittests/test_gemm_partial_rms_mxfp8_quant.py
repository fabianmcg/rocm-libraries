# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+PartialRMS+MXFP8Quant (K1) epilogue (gfx950).

Data flow: acc=A*B → PartialRMS (Σx² to partialBuf, gamma applied in-place)
           → MXFP8Quant (amax to MXScale, quantMult applied) → fp8 D store.

Verifies:
  - partialBuf (f32 [M_padded, n_d], single half): per-MT0-tile Σh1² pre-gamma,
    matched within rtol=atol=1e-4.
  - MXScale (uint8 [ceil(N/q0), ceil(M/q1)]): e8m0 per-block scale, exact bytes.
  - D (fp8 e4m3 OCP): MX-quantized gamma·h1, matched as f32 within rtol=atol=0.15.

Kernarg layout (hasBeta=False):
  args[0..26]  : scalar/ptr kernel args (flags, ki0/ki1, numWG, sizes, D/C/A/B,
                 strides, alpha, SK decomp args) — same as mxfp8-only.
  args[27]     : gamma bf16 pointer (InputArray)
  args[28]     : partialBuf f32 pointer (InOutArray)
  args[29]     : mxScale u8 pointer (InOutArray)
  args[30..33] : batchOffset{D,C,A,B} = np.uint64(0) (batch=1, DP-only).
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
from epilogues.tensilelite.numpy_helpers import (
    randBf16, randGamma, partialRmsMxfp8Reference,
)
from epilogues.tensilelite.yaml_solution_builder import readTestAxes

_K1_YAML = yamlPath("gemm_partial_rms_mxfp8_quant_k1.yaml")

_K1_SOLUTIONS = enumerateSolutions("gemm_partial_rms_mxfp8_quant_k1.yaml")

_K1_AXES = {}
try:
    _K1_AXES = readTestAxes(_K1_YAML, "K1", mt1=1)
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
def k1_combo_kernel(request):
    """Assemble and compile one K1 PartialRMS+MXFP8Quant solution."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


# ---------------------------------------------------------------------------
# Helper: run one shape, return comparison data.
# ---------------------------------------------------------------------------

def _run_shape(solution, kernelName, hsaco, chip, M, K, nHidden, alpha=1.0,
               zeroInput=False):
    """Run K1 combined epilogue for one (M, K, nHidden) shape.

    Returns (pb_gpu, sumsq_ref, mx_gpu, mx_ref, d_gpu_f32, d_ref_f32, mPadded, nD).
    """
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    q0 = solution.get("_MXFP8QuantQ0", MT0)
    q1 = solution.get("_MXFP8QuantQ1", MT1)

    nD = math.ceil(nHidden / MT0)
    mPadded = math.ceil(M / MT1) * MT1
    numWG = math.ceil(nHidden / MT0) * math.ceil(M / MT1)

    rng = np.random.default_rng(seed=M * 10000 + K)
    aRow = randBf16(rng, (M, K))
    bRow = randBf16(rng, (K, nHidden))
    if zeroInput:
        aRow = np.zeros((M, K), dtype=ml_dtypes.bfloat16)

    gammaF32, gammaBf16 = randGamma(rng, nHidden)

    # Kernel operands: free0=nHidden, free1=M (transposed convention).
    bFortran  = np.asfortranarray(bRow)
    aFortranT = np.asfortranarray(aRow.T)
    cFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    # D is OCP fp8 e4m3.
    dFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.float8_e4m3fn, order="F")
    # partialBuf is single-half (PartialRMSQuant=False): shape [M_padded, n_d].
    partialBuf = np.zeros((mPadded, nD), dtype=np.float32, order="C")

    mT = math.ceil(nHidden / q0)
    nT = math.ceil(M / q1)
    mxScale = np.zeros((mT, nT), dtype=np.uint8, order="C")

    # Combined reference: Σx² (pre-gamma), e8m0 MXScale, fp8 D (post-gamma).
    sumsqRef, mxScaleRef, dFp8Ref = partialRmsMxfp8Reference(
        aRow, bRow, gammaBf16, MT0, q0, q1
    )

    skArgs = compute_sk3_dp_args(nHidden, M, K, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    # Epilogue order: [gamma, partialBuf, mxScale] — see module docstring.
    epilogueArgs = [
        amdgpu_exec.InputArray(gammaBf16),
        amdgpu_exec.InOutArray(partialBuf),
        amdgpu_exec.InOutArray(mxScale),
    ]
    args = buildSubtileArgs(
        nHidden, M, K, numWG,
        amdgpu_exec.InOutArray(dFortran), amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        skArgs, ki0, ki1, epilogueArgs,
        alpha=np.float32(alpha), hasBeta=False,
    )

    result_holder = {}

    def capture(arguments):
        dRaw = np.asarray(arguments[8].array)
        result_holder["d_gpu"] = dRaw.reshape(nHidden, M, order="F")
        # args[28] = partialBuf InOutArray (gamma=27, partialBuf=28, mxScale=29).
        pb = np.asarray(arguments[28].array).copy().reshape(mPadded, nD)
        result_holder["pb_gpu"] = pb
        mx = np.asarray(arguments[29].array).copy().reshape(mT, nT)
        result_holder["mx_gpu"] = mx

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    return (
        result_holder["pb_gpu"],
        sumsqRef,
        result_holder["mx_gpu"].view(np.uint8),
        mxScaleRef,
        result_holder["d_gpu"].astype(np.float32),
        dFp8Ref.astype(np.float32),
        mPadded,
        nD,
    )


def _check(solution, kernelName, hsaco, chip, M, K, nHidden, alpha=1.0):
    """Run and assert correctness; return a descriptive label on pass."""
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    q0 = solution.get("_MXFP8QuantQ0", MT0)
    q1 = solution.get("_MXFP8QuantQ1", MT1)
    label = (
        f"MT{MT0}x{MT1} Q=[{q0},{q1}] M={M} K={K} N={nHidden} alpha={alpha}"
    )
    pbGpu, sumsqRef, mxGpu, mxRef, dGpuF32, dRefF32, mPadded, _ = _run_shape(
        solution, kernelName, hsaco, chip, M, K, nHidden, alpha=alpha
    )
    assertClose(pbGpu[:M, :], sumsqRef, label, rtol=1e-4, atol=1e-4,
                kind="partialBuf_sumsq")
    assert np.array_equal(mxGpu, mxRef), (
        f"{label}: MXScale byte mismatch. "
        f"gpu={mxGpu.ravel()[:8]} ref={mxRef.ravel()[:8]}"
    )
    assertClose(dGpuF32, dRefF32, label, rtol=0.15, atol=0.15, kind="D_fp8")


# ---------------------------------------------------------------------------
# Test: KNHidden sweep from YAML axes (basic correctness, alpha=1).
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize(
    "K,N_hidden",
    _KN_PAIRS,
    ids=[f"K{k}-N{n}" for k, n in _KN_PAIRS],
)
def test_k1_combo_shape(k1_combo_kernel, K, N_hidden):
    """Verify K1 PartialRMS+MXFP8Quant outputs: partialBuf Σx², MXScale bytes, fp8 D."""
    solution, kernelName, hsaco, chip = k1_combo_kernel
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]

    for M, mLabel in readTestAxes(_K1_YAML, "K1", mt1=MT1)["M"]:
        pbGpu, sumsqRef, mxGpu, mxRef, dGpuF32, dRefF32, mPadded, nD = _run_shape(
            solution, kernelName, hsaco, chip, M, K, N_hidden
        )
        label = f"MT0={MT0} MT1={MT1} {mLabel} N={N_hidden} K={K}"
        assertClose(pbGpu[:M, :], sumsqRef, label, rtol=1e-4, atol=1e-4,
                    kind="partialBuf_sumsq")
        assert np.array_equal(mxGpu, mxRef), (
            f"{label}: MXScale byte mismatch. "
            f"gpu={mxGpu.ravel()[:8]} ref={mxRef.ravel()[:8]}"
        )
        assertClose(dGpuF32, dRefF32, label, rtol=0.15, atol=0.15, kind="D_fp8")


# ---------------------------------------------------------------------------
# Test: multi-workgroup shapes.
# ---------------------------------------------------------------------------

_MULTI_WG_SHAPES = [
    (256,  64,  128),
    (512,  64,  128),
    (128,  64,  256),
    (256,  64,  256),
    (256, 128,  256),
    (512,  64, 1024),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _MULTI_WG_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _MULTI_WG_SHAPES],
)
def test_k1_combo_multiWG(k1_combo_kernel, M, K, N_hidden):
    """Verify combined epilogue for multi-workgroup shapes."""
    solution, kernelName, hsaco, chip = k1_combo_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Test: non-tile-aligned M and N.
# ---------------------------------------------------------------------------

_UNALIGNED_SHAPES = [
    ( 17,  64,  80),
    (113,  64, 200),
    (200,  64, 113),
    (129, 128, 129),
    (255,  64, 255),
    (  1,  64,  64),
    ( 64,  64,   1),
    (  1,  64,   1),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _UNALIGNED_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _UNALIGNED_SHAPES],
)
def test_k1_combo_unaligned(k1_combo_kernel, M, K, N_hidden):
    """Verify combined epilogue on shapes where M or N is not tile-aligned."""
    solution, kernelName, hsaco, chip = k1_combo_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Test: K sweep.
# ---------------------------------------------------------------------------

_K_SHAPES = [
    (128,   1, 128),
    (128,  32, 128),
    (128,  64, 128),
    (128, 128, 128),
    (128, 512, 128),
    (128, 1024, 128),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _K_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _K_SHAPES],
)
def test_k1_combo_k_sweep(k1_combo_kernel, M, K, N_hidden):
    """Verify combined epilogue across a range of reduction depths."""
    solution, kernelName, hsaco, chip = k1_combo_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Test: large production shapes.
# ---------------------------------------------------------------------------

_LARGE_SHAPES = [
    ( 512,  128, 4096),
    (1024,  128, 4096),
    ( 512,  128, 8192),
    (2048,  512, 4096),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,N_hidden",
    _LARGE_SHAPES,
    ids=[f"M{m}-K{k}-N{n}" for m, k, n in _LARGE_SHAPES],
)
def test_k1_combo_large(k1_combo_kernel, M, K, N_hidden):
    """Verify combined epilogue on production-scale LLM shapes."""
    solution, kernelName, hsaco, chip = k1_combo_kernel
    _check(solution, kernelName, hsaco, chip, M, K, N_hidden)


# ---------------------------------------------------------------------------
# Test: all-zero input → partialBuf=0, MXScale=0, D=0.
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
def test_k1_combo_all_zero(k1_combo_kernel, M, K, N_hidden):
    """All-zero A input must yield Σx²=0, MXScale=0, D=0."""
    solution, kernelName, hsaco, chip = k1_combo_kernel
    pbGpu, sumsqRef, mxGpu, mxRef, dGpuF32, dRefF32, mPadded, _ = _run_shape(
        solution, kernelName, hsaco, chip, M, K, N_hidden, zeroInput=True
    )
    label = (f"all-zero MT{solution['MacroTile0']}x{solution['MacroTile1']}"
             f" M={M} K={K} N={N_hidden}")
    assert np.all(pbGpu[:M, :] == 0.0), f"{label}: partialBuf not all zero"
    assert np.all(mxGpu == 0), f"{label}: MXScale not all zero"
    assert np.all(dGpuF32 == 0.0), f"{label}: D not all zero"
    assert np.array_equal(mxGpu, mxRef), f"{label}: MXScale gpu != ref"
    assertClose(pbGpu[:M, :], sumsqRef, label, rtol=1e-4, atol=1e-4,
                kind="partialBuf_sumsq")
