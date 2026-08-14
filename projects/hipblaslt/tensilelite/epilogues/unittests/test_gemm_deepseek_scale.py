# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+DeepseekScale fp32 software-rescale path (PGR=0/1/2, gfx950, fp8 in / f32 out).

Covers three flag combinations:
  - A-only: D = alpha * scaleA[m/Aq0, k/Aq1] * partial[m,n,k] + beta*C
  - B-only: D = alpha * scaleB[k/Bq0, n/Bq1] * partial[m,n,k] + beta*C
  - A+B:    D = alpha * scaleA[m/Aq0, k/Aq1] * scaleB[k/Bq0, n/Bq1] * partial[m,n,k] + beta*C

Default tile sizes: Aq0=128, Aq1=128, Bq0=1, Bq1=128.
Layout: free0=M (rows), free1=N (columns), bound=K.
A is stored as [K, M] (TransposeA=True), B as [K, N].
Scales are fp32 values in a reasonable range (e.g. [0.5, 1.5]).

Device layout for scale buffers (matches the 4-byte DirectToLds load in the kernel):
  scaleA -- logical fp32 [nRowGroups, nKBlocks], device flat fp32 [nRowGroups, nKBlocks, 64]:
    R = 64 (lanes per wave; rows per wave group for in-scope configs)
    nRowGroups = mPadded // R
    All 64 lane slots in each (rowGroup, kBlock) entry hold the same fp32 value.

  scaleB -- logical fp32 [nKBlocks, nNBlocks], device flat fp32 [nNBlocks, nKBlocks, 64]:
    All 64 lane slots in each (nBlock, kBlock) entry hold the same fp32 value.
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
    compute_sk3_dp_args, compute_sk_split_args, makeSKSplitBuffers,
    _pack_kernel_info, compileSolution, buildSubtileArgs,
)

_DSAB_MULTIK_YAML  = yamlPath("gemm_deepseek_scale_ab_multik.yaml")
_DSA_MULTIK_YAML   = yamlPath("gemm_deepseek_scale_a_multik.yaml")
_DSB_MULTIK_YAML   = yamlPath("gemm_deepseek_scale_b_multik.yaml")

_DSAB_MULTIK_SOLUTIONS = enumerateSolutions("gemm_deepseek_scale_ab_multik.yaml")
_DSA_MULTIK_SOLUTIONS  = enumerateSolutions("gemm_deepseek_scale_a_multik.yaml")
_DSB_MULTIK_SOLUTIONS  = enumerateSolutions("gemm_deepseek_scale_b_multik.yaml")

_DSAB_SK_SPLIT_YAML       = yamlPath("gemm_deepseek_scale_ab_sk_split.yaml")
_DSAB_SK_SPLIT_SOLUTIONS  = enumerateSolutions("gemm_deepseek_scale_ab_sk_split.yaml")


# ---------------------------------------------------------------------------
# Session-scoped fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _DSAB_MULTIK_SOLUTIONS],
    ids=[sid for _sol, sid in _DSAB_MULTIK_SOLUTIONS],
)
def dsab_multik_kernel(request):
    """Assemble and compile one multi-K DeepseekScaleAB solution (PGR=0)."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _DSA_MULTIK_SOLUTIONS],
    ids=[sid for _sol, sid in _DSA_MULTIK_SOLUTIONS],
)
def dsa_multik_kernel(request):
    """Assemble and compile one multi-K DeepseekScaleA-only solution (PGR=0)."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _DSB_MULTIK_SOLUTIONS],
    ids=[sid for _sol, sid in _DSB_MULTIK_SOLUTIONS],
)
def dsb_multik_kernel(request):
    """Assemble and compile one multi-K DeepseekScaleB-only solution (PGR=0)."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _DSAB_SK_SPLIT_SOLUTIONS],
    ids=[sid for _sol, sid in _DSAB_SK_SPLIT_SOLUTIONS],
)
def dsab_sk_split_kernel(request):
    """Assemble and compile one non-DP Stream-K DeepseekScaleAB solution."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


# ---------------------------------------------------------------------------
# Reference computations.
# ---------------------------------------------------------------------------

def numpy_ref_multiblock(A, B, scaleA, scaleB, alpha, beta, C):
    """Multi-K-block reference: D = alpha*sum_kb(sA[m,kb]*sB[kb,nb]*partial[m,n,kb]) + beta*C.

    A: fp8 [M,K], B: fp8 [K,N].
    scaleA: f32 [M, nKBlocks]  (nKBlocks = K // 128).
    scaleB: f32 [nKBlocks, ceil(N/128)].
    """
    A_f32 = np.asarray(A).astype(np.float32)
    B_f32 = np.asarray(B).astype(np.float32)
    M, K  = A_f32.shape
    N     = B_f32.shape[1]
    nKBlocks = K // 128
    nNBlocks = math.ceil(N / 128)
    result = np.zeros((M, N), dtype=np.float32)
    for kb in range(nKBlocks):
        partial = A_f32[:, kb * 128:(kb + 1) * 128] @ B_f32[kb * 128:(kb + 1) * 128, :]
        for nb in range(nNBlocks):
            n0, n1 = nb * 128, min((nb + 1) * 128, N)
            combined = scaleA[:, kb] * scaleB[kb, nb]  # [M]
            result[:, n0:n1] += combined[:, np.newaxis] * partial[:, n0:n1]
    return alpha * result + beta * np.asarray(C).astype(np.float32)


# ---------------------------------------------------------------------------
# Device-layout swizzle helpers.
# ---------------------------------------------------------------------------

# R = MatrixInstM * mma_m = 16 * 4 = 64 rows handled per wave for in-scope configs.
_ROWS_PER_WAVE = 64


def _swizzleScaleADevice(scaleA_padded: np.ndarray, nKBlocks: int) -> np.ndarray:
    """Convert logical scaleA [mPadded, nKBlocks] fp32 to the device layout.

    Device layout (flat fp32, length nRowGroups*nKBlocks*64):
      device[(rowGroup*nKBlocks + kb)*64 + lane] = scaleA[rowGroup*R, kb]
    where R=64 and all 64 lane slots hold the same fp32 value.
    """
    R = _ROWS_PER_WAVE
    mPadded = scaleA_padded.shape[0]
    assert mPadded % R == 0, f"mPadded ({mPadded}) must be divisible by R ({R})"
    nRowGroups = mPadded // R
    # Take the representative value for each wave group (all rows in a group are equal).
    rep = scaleA_padded[::R, :]  # [nRowGroups, nKBlocks]
    # Broadcast each value to all 64 lane slots.
    broadcast = np.repeat(rep[:, :, np.newaxis], R, axis=2)  # [nRowGroups, nKBlocks, 64]
    return broadcast.astype(np.float32).ravel()


def _swizzleScaleBDevice(scaleB_logical: np.ndarray, nKBlocks: int) -> np.ndarray:
    """Convert logical scaleB [nKBlocks, nNBlocks] fp32 to the device layout.

    Device layout (flat fp32, length nNBlocks*nKBlocks*64):
      device[(nb*nKBlocks + kb)*64 + lane] = scaleB[kb, nb]
    where all 64 lane slots hold the same fp32 value.
    """
    nNBlocks = scaleB_logical.shape[1]
    # Transpose to [nNBlocks, nKBlocks] so C-order matches (nb, kb) indexing.
    transposed = scaleB_logical.T  # [nNBlocks, nKBlocks]
    # Broadcast each value to all 64 lane slots.
    broadcast = np.repeat(transposed[:, :, np.newaxis], 64, axis=2)
    return broadcast.astype(np.float32).ravel()


# ---------------------------------------------------------------------------
# GPU execution helpers.
# ---------------------------------------------------------------------------

def _execute_and_compare(solution, kernelName, hsaco, M, N, K, numWG,
                         aFortran, bFortran, cFortran, dFortran,
                         epilogueArgs, alpha, beta=0.0):
    """Build kernel args, execute on GPU, and return D in row-major f32."""
    skArgs = compute_sk3_dp_args(M, N, K, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    args = buildSubtileArgs(
        M, N, K, numWG,
        amdgpu_exec.InOutArray(dFortran), amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(aFortran), amdgpu_exec.InputArray(bFortran),
        skArgs, ki0, ki1,
        [],                       # no epilogue args before the batchOffsets.
        alpha=np.float32(alpha),
        hasBeta=True,
        beta=np.float32(beta),
    )
    # Scaled-GEMM kernarg layout: ScaleABuf/ScaleBBuf follow the batchOffset block
    # (see Signature.py), so append them after buildSubtileArgs' trailing batchOffsets.
    args.extend(epilogueArgs)

    result_holder = {}

    def capture(arguments):
        result_holder["d_gpu"] = np.asarray(arguments[8].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    return result_holder["d_gpu"].reshape(M, N, order="F").astype(np.float32)


def _execute_and_compare_sk_split(solution, kernelName, hsaco, M, N, K, sk_grid,
                                  aFortran, bFortran, cFortran, dFortran,
                                  epilogueArgs, alpha, beta=0.0):
    """Build args, execute a non-DP Stream-K kernel on GPU, return D row-major f32.

    Launches sk_grid workgroups (numWG == sk_grid) with a zeroed workspace and
    flags buffer so K is split across workgroups and combined by the fixup.
    """
    skArgs = compute_sk_split_args(M, N, K, solution, sk_grid)
    ki0, ki1 = _pack_kernel_info(solution)
    ws, flags = makeSKSplitBuffers(solution, sk_grid)

    args = buildSubtileArgs(
        M, N, K, sk_grid,
        amdgpu_exec.InOutArray(dFortran), amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(aFortran), amdgpu_exec.InputArray(bFortran),
        skArgs, ki0, ki1,
        [],
        alpha=np.float32(alpha),
        hasBeta=True,
        beta=np.float32(beta),
        wsArray=amdgpu_exec.InputArray(ws),
        flagsArray=amdgpu_exec.InputArray(flags),
    )
    args.extend(epilogueArgs)

    result_holder = {}

    def capture(arguments):
        result_holder["d_gpu"] = np.asarray(arguments[8].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(sk_grid, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    return result_holder["d_gpu"].reshape(M, N, order="F").astype(np.float32)


def _make_nonzero_c(M, N, seed):
    """Generate a non-zero fp32 C matrix in [-1, 1] for beta tests."""
    rng = np.random.default_rng(seed=seed)
    return rng.random((M, N), dtype=np.float32) * 2 - 1


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

_TEST_SHAPES_MULTIK = [
    (m, n, k)
    for m in [7, 128]
    for n in [128, 256, 320]
    for k in [256, 512]
]


def _make_inputs_ab_multik(M, N, K, mPadded):
    """Generate fp8 A/B and fp32 scaleA/scaleB for a multi-K-block A+B test."""
    rng      = np.random.default_rng(seed=M * 100000 + N * 1000 + K + 3)
    nKBlocks = K // 128
    nNBlocks = math.ceil(N / 128)

    aKM = (rng.random((K, M), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn)
    bKN = (rng.random((K, N), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn)

    aFortran = np.asfortranarray(aKM)
    bFortran = np.asfortranarray(bKN)
    cFortran = np.zeros((M, N), dtype=np.float32, order="F")
    dFortran = np.zeros((M, N), dtype=np.float32, order="F")

    # scaleA: one fp32 per wave group (64 rows) per K-block. Broadcast to all rows in each group.
    nRowGroups = mPadded // _ROWS_PER_WAVE
    scaleAVals = rng.uniform(0.5, 1.5, size=(nRowGroups, nKBlocks)).astype(np.float32)
    scaleAPadded = np.repeat(scaleAVals, _ROWS_PER_WAVE, axis=0)  # [mPadded, nKBlocks]
    scaleARef = scaleAPadded[:M, :]  # [M, nKBlocks] fp32 values used by the reference

    # scaleB: one fp32 per K-block per N-block.
    scaleBRef = rng.uniform(0.5, 1.5, size=(nKBlocks, nNBlocks)).astype(np.float32)

    return (aFortran, bFortran, cFortran, dFortran,
            scaleAPadded, scaleBRef, scaleARef, aKM, bKN)


def _run_shape_ab_multik(solution, kernelName, hsaco, chip, M, N, K,
                         alpha=1.0, beta=0.0, cMatrix=None):
    """Run multi-K DeepseekScaleAB kernel for one (M, N, K) shape."""
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    (aFortran, bFortran, cFortran, dFortran,
     scaleAPadded, scaleBRef, scaleARef, aKM, bKN) = \
        _make_inputs_ab_multik(M, N, K, mPadded)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    aMK  = np.asarray(aKM).T
    dRef = numpy_ref_multiblock(aMK, np.asarray(bKN), scaleARef, scaleBRef,
                                alpha, beta, c_ref)

    nKBlocks = K // 128
    epilogueArgs = [amdgpu_exec.InputArray(_swizzleScaleADevice(scaleAPadded, nKBlocks)),
                    amdgpu_exec.InputArray(_swizzleScaleBDevice(scaleBRef, nKBlocks))]
    dGpu = _execute_and_compare(solution, kernelName, hsaco, M, N, K, numWG,
                                aFortran, bFortran, cFortran, dFortran,
                                epilogueArgs, alpha, beta=beta)
    return dGpu, dRef.astype(np.float32)


def _run_shape_a_multik(solution, kernelName, hsaco, chip, M, N, K,
                        alpha=1.0, beta=0.0, cMatrix=None):
    """Run multi-K DeepseekScaleA-only kernel for one (M, N, K) shape.

    scaleB side uses a unit scale (1.0 fp32) inside the mainloop.
    The reference uses scaleA only, with scaleB effectively 1.0.
    """
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    (aFortran, bFortran, cFortran, dFortran,
     scaleAPadded, _scaleBRef, scaleARef, aKM, bKN) = \
        _make_inputs_ab_multik(M, N, K, mPadded)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    # Unit scaleB (all 1.0): scaleB[kb, nb] = 1.0 for all blocks.
    nNBlocks = math.ceil(N / 128)
    nKBlocks = K // 128
    unitScaleBRef = np.ones((nKBlocks, nNBlocks), dtype=np.float32)
    aMK  = np.asarray(aKM).T
    dRef = numpy_ref_multiblock(aMK, np.asarray(bKN), scaleARef, unitScaleBRef,
                                alpha, beta, c_ref)

    epilogueArgs = [amdgpu_exec.InputArray(_swizzleScaleADevice(scaleAPadded, nKBlocks))]
    dGpu = _execute_and_compare(solution, kernelName, hsaco, M, N, K, numWG,
                                aFortran, bFortran, cFortran, dFortran,
                                epilogueArgs, alpha, beta=beta)
    return dGpu, dRef.astype(np.float32)


def _run_shape_b_multik(solution, kernelName, hsaco, chip, M, N, K,
                        alpha=1.0, beta=0.0, cMatrix=None):
    """Run multi-K DeepseekScaleB-only kernel for one (M, N, K) shape.

    scaleA side uses a unit scale (1.0 fp32) inside the mainloop.
    The reference uses scaleB only, with scaleA effectively 1.0.
    """
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    (aFortran, bFortran, cFortran, dFortran,
     _scaleAPadded, scaleBRef, _scaleARef, aKM, bKN) = \
        _make_inputs_ab_multik(M, N, K, mPadded)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    # Unit scaleA (all 1.0): scaleA[m, kb] = 1.0 for all rows and K-blocks.
    unitScaleARef = np.ones((M, K // 128), dtype=np.float32)
    aMK  = np.asarray(aKM).T
    dRef = numpy_ref_multiblock(aMK, np.asarray(bKN), unitScaleARef, scaleBRef,
                                alpha, beta, c_ref)

    nKBlocks = K // 128
    epilogueArgs = [amdgpu_exec.InputArray(_swizzleScaleBDevice(scaleBRef, nKBlocks))]
    dGpu = _execute_and_compare(solution, kernelName, hsaco, M, N, K, numWG,
                                aFortran, bFortran, cFortran, dFortran,
                                epilogueArgs, alpha, beta=beta)
    return dGpu, dRef.astype(np.float32)


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    _TEST_SHAPES_MULTIK,
    ids=[f"M{m}-N{n}-K{k}" for m, n, k in _TEST_SHAPES_MULTIK],
)
def test_deepseek_scale_ab_multik_shape(dsab_multik_kernel, M, N, K):
    """Verify multi-K DeepseekScaleAB output: per-K-block fp32 software rescale."""
    solution, kernelName, hsaco, chip = dsab_multik_kernel
    dGpu, dRef = _run_shape_ab_multik(solution, kernelName, hsaco, chip, M, N, K)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    _TEST_SHAPES_MULTIK,
    ids=[f"M{m}-N{n}-K{k}" for m, n, k in _TEST_SHAPES_MULTIK],
)
def test_deepseek_scale_a_multik_shape(dsa_multik_kernel, M, N, K):
    """Verify multi-K DeepseekScaleA-only: unit scaleB fallback in fp32 software rescale."""
    solution, kernelName, hsaco, chip = dsa_multik_kernel
    dGpu, dRef = _run_shape_a_multik(solution, kernelName, hsaco, chip, M, N, K)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"A-only multik M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    _TEST_SHAPES_MULTIK,
    ids=[f"M{m}-N{n}-K{k}" for m, n, k in _TEST_SHAPES_MULTIK],
)
def test_deepseek_scale_b_multik_shape(dsb_multik_kernel, M, N, K):
    """Verify multi-K DeepseekScaleB-only: unit scaleA fallback in fp32 software rescale."""
    solution, kernelName, hsaco, chip = dsb_multik_kernel
    dGpu, dRef = _run_shape_b_multik(solution, kernelName, hsaco, chip, M, N, K)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"B-only multik M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
def test_deepseek_scale_ab_multik_nonzero_beta(dsab_multik_kernel):
    """Verify multi-K A+B: beta=0.5, non-zero C to exercise the beta*C addition."""
    solution, kernelName, hsaco, chip = dsab_multik_kernel
    M, N, K = 128, 128, 256
    cMatrix = _make_nonzero_c(M, N, seed=55)
    dGpu, dRef = _run_shape_ab_multik(solution, kernelName, hsaco, chip, M, N, K,
                                      alpha=1.0, beta=0.5, cMatrix=cMatrix)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"A+B multik beta=0.5 M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
def test_deepseek_scale_ab_batched():
    """Batched (batch>1) execution is not yet exercised by this harness.

    The Python test harness builds kernel args via buildSubtileArgs, which
    hard-codes batch=1 and zero batch offsets. The kernel YAML declares
    Batched=True / StridedBatched=True, so the kernel itself supports
    strided-batched execution; however, wiring a multi-batch invocation
    through the Python amdgpu_exec path would require per-batch pointer
    arithmetic and is deferred to the C++ client e2e test.
    """
    pytest.skip(
        "batch>1 is not supported by the Python amdgpu_exec harness; "
        "multi-batch correctness is validated by the C++ client."
    )


def _run_shape_ab_sk_split(solution, kernelName, hsaco, chip, M, N, K, sk_grid,
                           alpha=1.0, beta=0.0, cMatrix=None):
    """Run a non-DP Stream-K DeepseekScaleAB kernel for one (M, N, K) shape."""
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0

    (aFortran, bFortran, cFortran, dFortran,
     scaleAPadded, scaleBRef, scaleARef, aKM, bKN) = \
        _make_inputs_ab_multik(M, N, K, mPadded)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    aMK  = np.asarray(aKM).T
    dRef = numpy_ref_multiblock(aMK, np.asarray(bKN), scaleARef, scaleBRef,
                                alpha, beta, c_ref)

    nKBlocks = K // 128
    epilogueArgs = [amdgpu_exec.InputArray(_swizzleScaleADevice(scaleAPadded, nKBlocks)),
                    amdgpu_exec.InputArray(_swizzleScaleBDevice(scaleBRef, nKBlocks))]
    dGpu = _execute_and_compare_sk_split(solution, kernelName, hsaco, M, N, K, sk_grid,
                                         aFortran, bFortran, cFortran, dFortran,
                                         epilogueArgs, alpha, beta=beta)
    return dGpu, dRef.astype(np.float32)


@requires_gfx950
def test_deepseek_scale_ab_sk_split(dsab_sk_split_kernel):
    """Verify non-DP Stream-K DeepseekScaleAB: K split across workgroups, combined
    by the workspace-reduction fixup with deferred alpha applied once."""
    solution, kernelName, hsaco, chip = dsab_sk_split_kernel
    M, N, K = 128, 256, 512
    MT0, MT1 = solution["MacroTile0"], solution["MacroTile1"]
    tiles = math.ceil(M / MT0) * math.ceil(N / MT1)
    sk_grid = 2 * tiles  # sk_grid > tiles -> each output tile is K-split across WGs.
    dGpu, dRef = _run_shape_ab_sk_split(solution, kernelName, hsaco, chip, M, N, K,
                                        sk_grid, alpha=2.0, beta=0.0)
    label = (f"MT{MT0}x{MT1} A+B sk-split M={M} N={N} K={K} sk_grid={sk_grid}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")
