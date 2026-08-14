# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for PartialRMS (K1) with StreamKForceDPOnly=0 (K-split path, gfx950).

Exercises the non-DP Stream-K code path where each output tile's K dimension is
split across multiple workgroups:
  - Role B (partial-tile starter): reads other WGs' partials, does fixup, then
    runs the epilogue on the now-complete accumulator.
  - Role C (partial-tile non-starter): writes raw partial accumulators and exits
    without running the epilogue.
  - Role A (complete tile): not K-split; runs epilogue directly.

The GEMM math and reference are identical to the DP tests; only the launch
config changes (sk_grid > tiles, zeroed workspace + flags buffers).

D tolerance: rtol=atol=2e-2. partialBuf tolerance: rtol=atol=1e-4.
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
from epilogues.tensilelite.numpy_helpers import randBf16, randGamma, partialSumSq, sideInputDtype

_SK_SPLIT_YAML = yamlPath("gemm_partial_rms_k1_sk_split.yaml")
_SK_SPLIT_SOLUTIONS = enumerateSolutions("gemm_partial_rms_k1_sk_split.yaml")


@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _SK_SPLIT_SOLUTIONS],
    ids=[sid for _sol, sid in _SK_SPLIT_SOLUTIONS],
)
def prms_sk_split_kernel(request):
    """Assemble and compile one K1 PartialRMS SK-split solution."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


def _run_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid,
                  residualAdd=False, rng=None):
    """Run PartialRMS K1 with non-DP Stream-K for one (M, K, nHidden, sk_grid) config.

    Returns (d_gpu_f32, d_ref_f32, pb_gpu, sumsq_ref).
    """
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    nD = math.ceil(nHidden / MT0)
    mPadded = math.ceil(M / MT1) * MT1

    if rng is None:
        rng = np.random.default_rng(seed=M * 10000 + K + nHidden)

    aRow = randBf16(rng, (M, K))
    bRow = randBf16(rng, (K, nHidden))

    # Kernel operands: A=bFortran (K×nHidden), B=aFortranT (K×M), Fortran order.
    bFortran  = np.asfortranarray(bRow)
    aFortranT = np.asfortranarray(aRow.T)
    cFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    dFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    partialBuf = np.zeros((mPadded, nD), dtype=np.float32, order="C")

    gammaType = solution.get("PartialRMSGammaType", "b")
    resType   = solution.get("PartialRMSResidualType", "b")

    gammaF32, gammaBf16 = randGamma(rng, nHidden)
    gammaBuf = gammaF32 if str(gammaType).lower() == "s" else gammaBf16
    gammaEff = np.asarray(gammaBuf).astype(np.float32)

    if residualAdd:
        residualF32 = (rng.random((M, nHidden), dtype=np.float32) - 0.5) * 0.2
        residualBuf = np.ascontiguousarray(residualF32.astype(sideInputDtype(resType)))
        residualEff = np.asarray(residualBuf).astype(np.float32)
    else:
        residualBuf = None
        residualEff = None

    h1 = aRow.astype(np.float32) @ bRow.astype(np.float32)
    hEff = h1 + residualEff if residualAdd else h1
    dRef = (hEff * gammaEff[np.newaxis, :]).astype(ml_dtypes.bfloat16)
    sumsqRef = partialSumSq(hEff, nHidden, MT0)

    skArgs = compute_sk_split_args(nHidden, M, K, solution, sk_grid)
    ki0, ki1 = _pack_kernel_info(solution)
    ws, flags = makeSKSplitBuffers(solution, sk_grid)

    dInout  = amdgpu_exec.InOutArray(dFortran)
    pbInout = amdgpu_exec.InOutArray(partialBuf)

    epilogueArgs = [amdgpu_exec.InputArray(gammaBuf), pbInout]
    if residualAdd:
        epilogueArgs.append(amdgpu_exec.InputArray(residualBuf))

    args = buildSubtileArgs(
        nHidden, M, K, sk_grid,
        dInout, amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        skArgs, ki0, ki1, epilogueArgs,
        wsArray=amdgpu_exec.InputArray(ws),
        flagsArray=amdgpu_exec.InputArray(flags),
    )

    # With wsArray inserted, epilogue args shift by 2 from DP-only slot numbering.
    # DP-only: D=8, gamma=28, pb=29. SK-split: D=8, gamma=30, pb=31.
    # The residual arg (if present) is at slot 32; pb is always slot 31.
    pb_slot = 31
    result_holder = {}

    def capture(arguments):
        dRaw = np.asarray(arguments[8].array)
        result_holder["d_gpu"] = dRaw.reshape(nHidden, M, order="F").T.astype(np.float32)
        result_holder["pb_gpu"] = np.asarray(arguments[pb_slot].array).copy().reshape(mPadded, nD)

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(sk_grid, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    return (
        result_holder["d_gpu"],
        np.asarray(dRef).astype(np.float32),
        result_holder["pb_gpu"],
        sumsqRef,
    )


def _check_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid,
                    residualAdd=None):
    """Run, assert correctness, and return the test label on pass.

    When residualAdd is None, it is read from the solution's PartialRMSResidualAdd
    field so callers don't have to track it separately.
    """
    if residualAdd is None:
        residualAdd = bool(solution.get("PartialRMSResidualAdd", False))
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    tiles = math.ceil(nHidden / MT0) * math.ceil(M / MT1)
    ipt   = max(1, math.ceil(K / solution["DepthU"]))
    label = (
        f"MT{MT0}x{MT1} M={M} K={K} N={nHidden} "
        f"tiles={tiles} ipt={ipt} sk_grid={sk_grid} residualAdd={residualAdd}"
    )
    dGpu, dRef, pbGpu, sumsqRef = _run_sk_split(
        solution, kernelName, hsaco, M, K, nHidden, sk_grid,
        residualAdd=residualAdd,
    )
    nD = math.ceil(nHidden / MT0)
    assertClose(dGpu, dRef, label, kind="D")
    assertClose(pbGpu[:M, :], sumsqRef,
                f"{label} n_d={nD}", rtol=1e-4, atol=1e-4, kind="partialBuf")


# ---------------------------------------------------------------------------
# Test: PartialRMS alone with K-split (Roles A, B, C exercised).
# ---------------------------------------------------------------------------

# Complete-tile config (sk_grid == tiles): every WG processes a full tile (Role A).
# K-split config (sk_grid == 2*tiles): 2 WGs per tile; one is Role B (fixup+store),
# the other is Role C (write partial, exit). Verifies epilogue runs only after fixup.
_PRMS_SK_SHAPES = [
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
    _PRMS_SK_SHAPES,
    ids=[note for *_, note in _PRMS_SK_SHAPES],
)
def test_prms_sk_split(prms_sk_split_kernel, M, K, nHidden, sk_grid, note):
    """Verify PartialRMS K1 D and partialBuf on SK non-DP (K-split) path."""
    solution, kernelName, hsaco, chip = prms_sk_split_kernel
    _check_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid)


# ---------------------------------------------------------------------------
# Test: PartialRMS + ResidualAdd with K-split.
# ---------------------------------------------------------------------------

_PRMS_RESIDUAL_SK_SHAPES = [
    (128, 128, 128, 2, "complete_2tile_residual"),
    (128, 128, 128, 4, "ksplit_2way_residual"),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden,sk_grid,note",
    _PRMS_RESIDUAL_SK_SHAPES,
    ids=[note for *_, note in _PRMS_RESIDUAL_SK_SHAPES],
)
def test_prms_residual_sk_split(prms_sk_split_kernel, M, K, nHidden, sk_grid, note):
    """Verify PartialRMS+ResidualAdd K1 D and partialBuf on SK non-DP path."""
    solution, kernelName, hsaco, chip = prms_sk_split_kernel
    if not solution.get("PartialRMSResidualAdd", False):
        pytest.skip("solution has PartialRMSResidualAdd=False")
    _check_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid,
                    residualAdd=True)


# ---------------------------------------------------------------------------
# Test: PartialRMS with f32 (and bf16) Gamma/Residual types, K-split path.
# ---------------------------------------------------------------------------

_SK_SPLIT_TYPES_SOLUTIONS = enumerateSolutions("gemm_partial_rms_k1_sk_split_types.yaml")


@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _SK_SPLIT_TYPES_SOLUTIONS],
    ids=[sid for _sol, sid in _SK_SPLIT_TYPES_SOLUTIONS],
)
def prms_sk_split_types_kernel(request):
    """Assemble and compile one K1 PartialRMS SK-split solution with configured side types."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


_PRMS_TYPES_SK_SHAPES = [
    (128, 128, 128, 2, "complete_2tile"),
    (128, 128, 128, 4, "ksplit_2way"),
]


@requires_gfx950
@pytest.mark.parametrize(
    "M,K,nHidden,sk_grid,note",
    _PRMS_TYPES_SK_SHAPES,
    ids=[note for *_, note in _PRMS_TYPES_SK_SHAPES],
)
def test_prms_sk_split_types(prms_sk_split_types_kernel, M, K, nHidden, sk_grid, note):
    """Verify PartialRMS K1 D and partialBuf with f32/bf16 Gamma+Residual (SK non-DP)."""
    solution, kernelName, hsaco, chip = prms_sk_split_types_kernel
    _check_sk_split(solution, kernelName, hsaco, M, K, nHidden, sk_grid)
