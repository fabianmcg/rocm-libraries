# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for MXFP8Quant (K1) with StreamKForceDPOnly=0 (K-split, gfx950).

Exercises the non-DP Stream-K code path where each output tile's K dimension is
split across multiple workgroups:
  - Role B (starter): reads other WGs' partials, does fixup, then runs the
    quantization epilogue on the now-complete accumulator.
  - Role C (non-starter): writes raw partial accumulators and exits without
    running the epilogue.
  - Role A (complete tile): no K-split, runs epilogue directly.

The GEMM math and reference are identical to the DP tests; only the launch
config changes (sk_grid > tiles, zeroed workspace + flags buffers).

MXScale tolerance: exact byte match. D tolerance: rtol=atol=0.15 (fp8 rounding).
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
    compute_sk_split_args, _pack_kernel_info, compileSolution,
    buildSubtileArgs, makeSKSplitBuffers,
)
from epilogues.tensilelite.numpy_helpers import randBf16, mxfp8QuantReference

_SK_SPLIT_YAML = yamlPath("gemm_mxfp8_quant_k1_sk_split.yaml")
_SK_SPLIT_SOLUTIONS = enumerateSolutions("gemm_mxfp8_quant_k1_sk_split.yaml")


def test_mxfp8_sk_split_blocked():
    """Explicitly signal when MXFP8Quant SKFDPO0 solutions are unavailable."""
    if not _SK_SPLIT_SOLUTIONS:
        pytest.skip(
            "MXFP8Quant SKFDPO0 blocked: SkPartialIdx aliases sgprBeta, absent for UseBeta=False"
        )


@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _SK_SPLIT_SOLUTIONS],
    ids=[sid for _sol, sid in _SK_SPLIT_SOLUTIONS],
)
def mx_sk_split_kernel(request):
    """Assemble and compile one K1 MXFP8Quant SK-split solution."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


def _run_mx_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid,
                     alpha=1.0, rng=None):
    """Run MXFP8Quant K1 with non-DP Stream-K for one (M, K, nHidden, sk_grid) config.

    Returns (mx_gpu, mx_ref, d_gpu_f32, d_ref_f32).
    """
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    q0 = solution.get("_DQuantSize0", MT0)
    q1 = solution.get("_DQuantSize1", MT1)

    if rng is None:
        rng = np.random.default_rng(seed=M * 10000 + K + nHidden)

    aRow = randBf16(rng, (M, K))
    bRow = randBf16(rng, (K, nHidden))

    bFortran  = np.asfortranarray(bRow)
    aFortranT = np.asfortranarray(aRow.T)
    cFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    dFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.float8_e4m3fn, order="F")

    mT = math.ceil(nHidden / q0)
    nT = math.ceil(M / q1)
    paddedRows = ((nT + 31) // 32) * 32
    paddedCols = ((mT + 7) // 8) * 8
    mxScale = np.zeros(paddedRows * paddedCols, dtype=np.uint8)

    # Reference: free0=nHidden, free1=M; h1T is (nHidden, M).
    h1T = (aRow.astype(np.float32) @ bRow.astype(np.float32)).T
    mxScaleRef, dFp8Ref = mxfp8QuantReference(alpha * h1T, q0, q1)

    skArgs = compute_sk_split_args(nHidden, M, K, solution, sk_grid)
    ki0, ki1 = _pack_kernel_info(solution)
    ws, flags = makeSKSplitBuffers(solution, sk_grid)

    # MXFP8Quant has UseBeta=False.
    args = buildSubtileArgs(
        nHidden, M, K, sk_grid,
        amdgpu_exec.InOutArray(dFortran), amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        skArgs, ki0, ki1, [amdgpu_exec.InOutArray(mxScale)],
        alpha=np.float32(alpha), hasBeta=False,
        wsArray=amdgpu_exec.InputArray(ws),
        flagsArray=amdgpu_exec.InputArray(flags),
    )

    # With wsArray and hasBeta=False, mxScale shifts from DP-only slot 27 to slot 29.
    mx_slot = 29
    result_holder = {}

    def capture(arguments):
        dRaw = np.asarray(arguments[8].array)
        result_holder["d_gpu"] = dRaw.reshape(nHidden, M, order="F").astype(np.float32)
        result_holder["mx_gpu"] = np.asarray(arguments[mx_slot].array).copy().reshape(-1)

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(sk_grid, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    return (
        result_holder["mx_gpu"].view(np.uint8),
        mxScaleRef,
        result_holder["d_gpu"],
        dFp8Ref.astype(np.float32),
    )


def _check_mx_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid,
                        alpha=1.0):
    """Run and assert correctness; return a descriptive label on pass."""
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    q0 = solution.get("_DQuantSize0", MT0)
    q1 = solution.get("_DQuantSize1", MT1)
    tiles = math.ceil(nHidden / MT0) * math.ceil(M / MT1)
    ipt   = max(1, math.ceil(K / solution["DepthU"]))
    label = (
        f"MT{MT0}x{MT1} Q=[{q0},{q1}] M={M} K={K} N={nHidden} "
        f"tiles={tiles} ipt={ipt} sk_grid={sk_grid} alpha={alpha}"
    )
    mxGpu, mxRef, dGpuF32, dRefF32 = _run_mx_sk_split(
        solution, kernelName, hsaco, M, K, nHidden, sk_grid, alpha=alpha,
    )
    assert np.array_equal(mxGpu, mxRef), (
        f"{label}: MXScale byte mismatch. "
        f"gpu={mxGpu.ravel()[:8]} ref={mxRef.ravel()[:8]}"
    )
    assertClose(dGpuF32, dRefF32, label, rtol=0.15, atol=0.15, kind="D_fp8")


# ---------------------------------------------------------------------------
# Test: MXFP8 dynquant with K-split (Roles A, B, C exercised).
# ---------------------------------------------------------------------------

# Complete-tile config: sk_grid == tiles, each WG processes a full tile (Role A).
# K-split config: sk_grid == 2*tiles, 2 WGs per tile, exercising Role B (fixup+store)
# and Role C (write partial, exit). Verifies epilogue + quantization run only after fixup.
_MX_SK_SHAPES = [
    # (M, K, nHidden, sk_grid, note)
    # complete: each WG owns a full tile (no K-split); tiles=2, sk_grid=2 → Role A
    (128, 128, 128, 2, "complete_2tile"),
    # K-split 2-way: tiles=2, sk_grid=4 → 2 WGs/tile (Role B + Role C)
    (128, 128, 128, 4, "ksplit_2way"),
    # K-split 2-way, larger K: tiles=4, ipt=4, sk_grid=8 → 2 WGs/tile (2 iters each)
    (128, 256, 256, 8, "ksplit_2way_largeK"),
    # larger M: tiles=8 (4 nHid × 2 M), sk_grid=16 → 2 WGs/tile (K-split)
    (256, 256, 256, 16, "ksplit_2way_M256"),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden,sk_grid,note",
    _MX_SK_SHAPES,
    ids=[note for *_, note in _MX_SK_SHAPES],
)
def test_mx_sk_split(mx_sk_split_kernel, M, K, nHidden, sk_grid, note):
    """Verify MXFP8Quant K1 MXScale bytes and fp8 D on SK non-DP (K-split) path."""
    solution, kernelName, hsaco, chip = mx_sk_split_kernel
    _check_mx_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid)


# ---------------------------------------------------------------------------
# Test: non-unit alpha with K-split.
# ---------------------------------------------------------------------------

_MX_ALPHA_SK_SHAPES = [
    (128, 128, 128, 4, 2.0, "alpha2_ksplit"),
    (128, 128, 128, 4, 0.5, "alpha05_ksplit"),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden,sk_grid,alpha,note",
    _MX_ALPHA_SK_SHAPES,
    ids=[note for *_, note in _MX_ALPHA_SK_SHAPES],
)
def test_mx_sk_split_alpha(mx_sk_split_kernel, M, K, nHidden, sk_grid, alpha, note):
    """Verify MXFP8Quant K1 correctly applies alpha before quantization on K-split path."""
    solution, kernelName, hsaco, chip = mx_sk_split_kernel
    _check_mx_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid,
                        alpha=alpha)
