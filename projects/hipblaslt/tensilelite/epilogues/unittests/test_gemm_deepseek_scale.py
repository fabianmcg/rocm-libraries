# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+DeepseekScale Subtile epilogue (gfx950, fp8 in / f32 out).

Covers three flag combinations:
  - A-only: D = alpha * scaleA[m] * (A_fp8 @ B_fp8) + beta*C
  - B-only: D = alpha * scaleB[n//128] * (A_fp8 @ B_fp8) + beta*C
  - A+B:    D = alpha * scaleA[m] * scaleB[n//128] * (A_fp8 @ B_fp8) + beta*C

Layout: free0=M (rows), free1=N (columns), bound=K.
A is stored as [K, M] (TransposeA=True), B as [K, N].
scaleA is E8M0 uint8 [M], one byte per output row.
scaleB is E8M0 uint8 [ceil(N/128)], one byte per 128-column N-block.

E8M0 format: value = 2^(byte - 127). Bytes in [120, 135] give values in [2^-7, 2^8],
keeping the product of scaleA * scaleB * fp8 accumulator in a reasonable fp32 range.

N-block note: N=320 is a valid partial-block test (3 blocks of 128: 0-127, 128-255, 256-319).
  The third scaleB element scales only the 64 columns [256, 320), which is correctly
  handled by the write-guard in the store path and the numpy reference's [:, :N] slice.
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

_DSA_YAML       = yamlPath("gemm_deepseek_scale_a_k1.yaml")
_DSB_YAML       = yamlPath("gemm_deepseek_scale_b_k1.yaml")
_DSAB_YAML      = yamlPath("gemm_deepseek_scale_ab_k1.yaml")
_DSAB_MULTIK_YAML = yamlPath("gemm_deepseek_scale_ab_multik.yaml")

_DSA_SOLUTIONS       = enumerateSolutions("gemm_deepseek_scale_a_k1.yaml")
_DSB_SOLUTIONS       = enumerateSolutions("gemm_deepseek_scale_b_k1.yaml")
_DSAB_SOLUTIONS      = enumerateSolutions("gemm_deepseek_scale_ab_k1.yaml")
_DSAB_MULTIK_SOLUTIONS = enumerateSolutions("gemm_deepseek_scale_ab_multik.yaml")

# A-only test shapes: M x N x K.
# Covers M=1 (partial block), M=7 (odd), M=128 (aligned), M=129 (crosses boundary).
_TEST_SHAPES_A = [
    (m, n, 128)
    for m in [1, 7, 128, 129]
    for n in [128, 256, 320, 512]
]

# B-only and A+B test shapes: same M/N grid as A-only for symmetric coverage.
_TEST_SHAPES_BN = [
    (m, n, 128)
    for m in [1, 7, 128, 129]
    for n in [128, 256, 320, 512]
]

# Non-unit alpha shapes: verify that alpha scales the final result correctly.
# C is zero so beta has no effect; only alpha is exercised here.
_TEST_SHAPES_NONUNIT_ALPHA = [
    (7, 128, 128),
    (128, 256, 128),
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _DSA_SOLUTIONS],
    ids=[sid for _sol, sid in _DSA_SOLUTIONS],
)
def dsa_kernel(request):
    """Assemble and compile one DeepseekScaleA solution from the benchmark YAML."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _DSB_SOLUTIONS],
    ids=[sid for _sol, sid in _DSB_SOLUTIONS],
)
def dsb_kernel(request):
    """Assemble and compile one DeepseekScaleB solution from the benchmark YAML."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _DSAB_SOLUTIONS],
    ids=[sid for _sol, sid in _DSAB_SOLUTIONS],
)
def dsab_kernel(request):
    """Assemble and compile one DeepseekScaleAB solution from the benchmark YAML."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


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


# ---------------------------------------------------------------------------
# Reference computations.
# ---------------------------------------------------------------------------

def numpy_ref_a_only(A, B, scaleA, alpha, beta, C):
    """Compute the A-only dequantization reference in f32.

    A: fp8 [M,K], B: fp8 [K,N], scaleA: f32 [M,1], C: f32 [M,N] -> D: f32 [M,N].
    """
    A_f32 = np.asarray(A).astype(np.float32)
    B_f32 = np.asarray(B).astype(np.float32)
    return alpha * (A_f32 * scaleA) @ B_f32 + beta * np.asarray(C).astype(np.float32)


def numpy_ref_b_only(A, B, scaleB, alpha, beta, C):
    """Compute the B-only dequantization reference in f32.

    scaleB: f32 [1, ceil(N/128)], broadcast over 128-column N-blocks.
    """
    A_f32 = np.asarray(A).astype(np.float32)
    B_f32 = np.asarray(B).astype(np.float32)
    N = B.shape[1]
    scale_broadcast = np.repeat(scaleB, 128, axis=1)[:, :N]  # [1, N].
    return alpha * (A_f32 @ B_f32) * scale_broadcast + beta * np.asarray(C).astype(np.float32)


def numpy_ref_ab(A, B, scaleA, scaleB, alpha, beta, C):
    """Compute the combined A+B dequantization reference in f32.

    scaleA: f32 [M,1], scaleB: f32 [1, ceil(N/128)].
    """
    A_f32 = np.asarray(A).astype(np.float32)
    B_f32 = np.asarray(B).astype(np.float32)
    N = B.shape[1]
    scale_broadcast = np.repeat(scaleB, 128, axis=1)[:, :N]  # [1, N].
    return alpha * (A_f32 * scaleA) @ B_f32 * scale_broadcast + beta * np.asarray(C).astype(np.float32)


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
# E8M0 helpers.
# ---------------------------------------------------------------------------

def _make_e8m0_bytes(shape, rng):
    """Generate random E8M0 exponent bytes in [120, 135] and decode to fp32.

    Returns (bytes_u8, fp32_values). Bytes in [120, 135] give scale factors
    2^(-7) to 2^(8), keeping products in a reasonable fp32 range.
    """
    exp_bytes = rng.integers(120, 136, size=shape, dtype=np.uint8)
    fp32_vals = np.ldexp(1.0, exp_bytes.astype(np.int32) - 127).astype(np.float32)
    return exp_bytes, fp32_vals


# ---------------------------------------------------------------------------
# Input generators.
# ---------------------------------------------------------------------------

def _make_inputs_a(M, N, K, mPadded):
    """Generate fp8 A/B and E8M0 scaleA/C for an A-only or A+B test."""
    rng = np.random.default_rng(seed=M * 100000 + N * 1000 + K)

    aKM = (rng.random((K, M), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn)
    bKN = (rng.random((K, N), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn)

    aFortran = np.asfortranarray(aKM)
    bFortran = np.asfortranarray(bKN)
    cFortran = np.zeros((M, N), dtype=np.float32, order="F")
    dFortran = np.zeros((M, N), dtype=np.float32, order="F")

    scaleABytes, scaleARef = _make_e8m0_bytes(M, rng)
    scaleAPadded = np.zeros(mPadded, dtype=np.uint8)
    scaleAPadded[:M] = scaleABytes

    return aFortran, bFortran, cFortran, dFortran, scaleAPadded, aKM, bKN, scaleARef


def _make_scaleB(N, rng):
    """Generate E8M0 scaleB bytes of shape [1, ceil(N/128)] and decode to fp32.

    Returns (bytes_u8, fp32_values).
    """
    nBlocks = math.ceil(N / 128)
    exp_bytes, fp32_vals = _make_e8m0_bytes((1, nBlocks), rng)
    return exp_bytes, fp32_vals


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
        epilogueArgs,
        alpha=np.float32(alpha),
        hasBeta=True,
        beta=np.float32(beta),
    )

    result_holder = {}

    def capture(arguments):
        result_holder["d_gpu"] = np.asarray(arguments[8].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    return result_holder["d_gpu"].reshape(M, N, order="F").astype(np.float32)


def _make_nonzero_c(M, N, seed):
    """Generate a non-zero fp32 C matrix in [-1, 1] for beta tests."""
    rng = np.random.default_rng(seed=seed)
    return rng.random((M, N), dtype=np.float32) * 2 - 1


def _run_shape_a(solution, kernelName, hsaco, chip, M, N, K,
                 alpha=1.0, beta=0.0, cMatrix=None):
    """Run DeepseekScaleA kernel for one (M, N, K) shape."""
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    aFortran, bFortran, cFortran, dFortran, scaleAPadded, aKM, bKN, scaleARef = \
        _make_inputs_a(M, N, K, mPadded)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    aMK   = np.asarray(aKM).T
    # scaleAPadded is uint8 (E8M0 bytes); scaleARef is the decoded fp32 for numpy reference.
    dRef  = numpy_ref_a_only(aMK, np.asarray(bKN), scaleARef[:, np.newaxis],
                              alpha, beta, c_ref)

    epilogueArgs = [amdgpu_exec.InputArray(scaleAPadded)]
    dGpu = _execute_and_compare(solution, kernelName, hsaco, M, N, K, numWG,
                                aFortran, bFortran, cFortran, dFortran,
                                epilogueArgs, alpha, beta=beta)
    return dGpu, dRef.astype(np.float32)


def _run_shape_b(solution, kernelName, hsaco, chip, M, N, K,
                 alpha=1.0, beta=0.0, cMatrix=None):
    """Run DeepseekScaleB kernel for one (M, N, K) shape."""
    MT0   = solution["MacroTile0"]
    numWG = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    rng = np.random.default_rng(seed=M * 100000 + N * 1000 + K + 1)
    aKM = (rng.random((K, M), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn)
    bKN = (rng.random((K, N), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn)

    aFortran = np.asfortranarray(aKM)
    bFortran = np.asfortranarray(bKN)
    cFortran = np.zeros((M, N), dtype=np.float32, order="F")
    dFortran = np.zeros((M, N), dtype=np.float32, order="F")
    scaleBBytes, scaleBRef = _make_scaleB(N, rng)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    aMK   = np.asarray(aKM).T
    dRef  = numpy_ref_b_only(aMK, np.asarray(bKN), scaleBRef, alpha, beta, c_ref)

    # Allocate scaleB with ceil(N/128) entries, one E8M0 byte per 128-column N-block.
    nPadded      = math.ceil(N / 128)
    scaleBPadded = np.zeros((1, nPadded), dtype=np.uint8)
    scaleBPadded[:, :scaleBBytes.shape[1]] = scaleBBytes
    epilogueArgs = [amdgpu_exec.InputArray(scaleBPadded)]
    dGpu = _execute_and_compare(solution, kernelName, hsaco, M, N, K, numWG,
                                aFortran, bFortran, cFortran, dFortran,
                                epilogueArgs, alpha, beta=beta)
    return dGpu, dRef.astype(np.float32)


def _run_shape_ab(solution, kernelName, hsaco, chip, M, N, K,
                  alpha=1.0, beta=0.0, cMatrix=None):
    """Run DeepseekScaleAB kernel for one (M, N, K) shape."""
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    aFortran, bFortran, cFortran, dFortran, scaleAPadded, aKM, bKN, scaleARef = \
        _make_inputs_a(M, N, K, mPadded)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    rng              = np.random.default_rng(seed=M * 100000 + N * 1000 + K + 2)
    scaleBBytes, scaleBRef = _make_scaleB(N, rng)

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    aMK   = np.asarray(aKM).T
    dRef  = numpy_ref_ab(aMK, np.asarray(bKN), scaleARef[:, np.newaxis], scaleBRef,
                         alpha, beta, c_ref)

    # Allocate scaleB with ceil(N/128) entries, one E8M0 byte per 128-column N-block.
    nPadded      = math.ceil(N / 128)
    scaleBPadded = np.zeros((1, nPadded), dtype=np.uint8)
    scaleBPadded[:, :scaleBBytes.shape[1]] = scaleBBytes
    # ScaleABuf first, ScaleBBuf second (matches kernel arg layout for A+B).
    epilogueArgs = [amdgpu_exec.InputArray(scaleAPadded),
                    amdgpu_exec.InputArray(scaleBPadded)]
    dGpu = _execute_and_compare(solution, kernelName, hsaco, M, N, K, numWG,
                                aFortran, bFortran, cFortran, dFortran,
                                epilogueArgs, alpha, beta=beta)
    return dGpu, dRef.astype(np.float32)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    _TEST_SHAPES_A,
    ids=[f"M{m}-N{n}-K128" for m, n, _ in _TEST_SHAPES_A],
)
def test_deepseek_scale_a_shape(dsa_kernel, M, N, K):
    """Verify DeepseekScaleA output for shape M x N x K."""
    solution, kernelName, hsaco, chip = dsa_kernel
    dGpu, dRef = _run_shape_a(solution, kernelName, hsaco, chip, M, N, K)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    _TEST_SHAPES_BN,
    ids=[f"M{m}-N{n}-K128" for m, n, _ in _TEST_SHAPES_BN],
)
def test_deepseek_scale_b_shape(dsb_kernel, M, N, K):
    """Verify DeepseekScaleB output for shape M x N x K."""
    solution, kernelName, hsaco, chip = dsb_kernel
    dGpu, dRef = _run_shape_b(solution, kernelName, hsaco, chip, M, N, K)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    _TEST_SHAPES_BN,
    ids=[f"M{m}-N{n}-K128" for m, n, _ in _TEST_SHAPES_BN],
)
def test_deepseek_scale_ab_shape(dsab_kernel, M, N, K):
    """Verify DeepseekScaleAB output for shape M x N x K."""
    solution, kernelName, hsaco, chip = dsab_kernel
    dGpu, dRef = _run_shape_ab(solution, kernelName, hsaco, chip, M, N, K)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
@pytest.mark.parametrize(
    "M,N,K",
    _TEST_SHAPES_NONUNIT_ALPHA,
    ids=[f"M{m}-N{n}-K{k}" for m, n, k in _TEST_SHAPES_NONUNIT_ALPHA],
)
def test_deepseek_scale_ab_nonunit_alpha(dsab_kernel, M, N, K):
    """Verify DeepseekScaleAB output with alpha=2.0 (non-unit alpha scaling)."""
    solution, kernelName, hsaco, chip = dsab_kernel
    dGpu, dRef = _run_shape_ab(solution, kernelName, hsaco, chip, M, N, K, alpha=2.0)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"alpha=2.0 M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
def test_deepseek_scale_ab_nonzero_beta_a(dsa_kernel):
    """Verify A-only: alpha=1.0, beta=0.5, non-zero C adds correctly."""
    solution, kernelName, hsaco, chip = dsa_kernel
    M, N, K = 128, 128, 128
    cMatrix = _make_nonzero_c(M, N, seed=42)
    dGpu, dRef = _run_shape_a(solution, kernelName, hsaco, chip, M, N, K,
                              alpha=1.0, beta=0.5, cMatrix=cMatrix)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"A-only beta=0.5 M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
def test_deepseek_scale_ab_nonzero_beta_b(dsb_kernel):
    """Verify B-only: alpha=1.0, beta=0.5, non-zero C adds correctly."""
    solution, kernelName, hsaco, chip = dsb_kernel
    M, N, K = 128, 128, 128
    cMatrix = _make_nonzero_c(M, N, seed=43)
    dGpu, dRef = _run_shape_b(solution, kernelName, hsaco, chip, M, N, K,
                              alpha=1.0, beta=0.5, cMatrix=cMatrix)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"B-only beta=0.5 M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


@requires_gfx950
def test_deepseek_scale_ab_nonzero_beta(dsab_kernel):
    """Verify A+B: alpha=1.0, beta=0.5, non-zero C: D = scale*acc + 0.5*C."""
    solution, kernelName, hsaco, chip = dsab_kernel
    M, N, K = 128, 128, 128
    cMatrix = _make_nonzero_c(M, N, seed=44)
    dGpu, dRef = _run_shape_ab(solution, kernelName, hsaco, chip, M, N, K,
                               alpha=1.0, beta=0.5, cMatrix=cMatrix)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"A+B beta=0.5 M={M} N={N} K={K}")
    assertClose(dGpu[:M, :N], dRef[:M, :N], label, rtol=2e-2, atol=2e-2, kind="D_f32")


_TEST_SHAPES_MULTIK = [
    (m, n, k)
    for m in [7, 128]
    for n in [128, 256, 320]
    for k in [256, 512]
]


def _make_inputs_ab_multik(M, N, K, mPadded):
    """Generate fp8 A/B and E8M0 scaleA/scaleB for a multi-K-block A+B test."""
    rng      = np.random.default_rng(seed=M * 100000 + N * 1000 + K + 3)
    nKBlocks = K // 128
    nNBlocks = math.ceil(N / 128)

    aKM = (rng.random((K, M), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn)
    bKN = (rng.random((K, N), dtype=np.float32) * 0.5).astype(ml_dtypes.float8_e4m3fn)

    aFortran = np.asfortranarray(aKM)
    bFortran = np.asfortranarray(bKN)
    cFortran = np.zeros((M, N), dtype=np.float32, order="F")
    dFortran = np.zeros((M, N), dtype=np.float32, order="F")

    # scaleA[M, nKBlocks]: each entry is one E8M0 byte; decoded to fp32 for reference.
    scaleABytes, scaleARef = _make_e8m0_bytes((M, nKBlocks), rng)
    scaleAPadded = np.zeros((mPadded, nKBlocks), dtype=np.uint8)
    scaleAPadded[:M, :] = scaleABytes

    # scaleB[nKBlocks, nNBlocks]: one E8M0 byte per (K-block, N-block) pair.
    scaleBBytes, scaleBRef = _make_e8m0_bytes((nKBlocks, nNBlocks), rng)

    return (aFortran, bFortran, cFortran, dFortran,
            scaleAPadded, scaleBBytes, scaleBRef, aKM, bKN, scaleARef)


def _run_shape_ab_multik(solution, kernelName, hsaco, chip, M, N, K,
                         alpha=1.0, beta=0.0):
    """Run multi-K DeepseekScaleAB kernel for one (M, N, K) shape."""
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    (aFortran, bFortran, cFortran, dFortran,
     scaleAPadded, scaleBBytes, scaleBRef, aKM, bKN, scaleARef) = \
        _make_inputs_ab_multik(M, N, K, mPadded)

    aMK  = np.asarray(aKM).T
    dRef = numpy_ref_multiblock(aMK, np.asarray(bKN), scaleARef, scaleBRef,
                                alpha, beta,
                                np.zeros((M, N), dtype=np.float32))

    # scaleA buffer: flat row-major [mPadded * nKBlocks] E8M0 uint8 array.
    # scaleB buffer: flat row-major [nKBlocks * nNBlocks] E8M0 uint8 array.
    epilogueArgs = [amdgpu_exec.InputArray(scaleAPadded.ravel()),
                    amdgpu_exec.InputArray(scaleBBytes.ravel())]
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
    """Verify multi-K DeepseekScaleAB output: per-K-block fp32 scale via WMMA operand."""
    solution, kernelName, hsaco, chip = dsab_multik_kernel
    dGpu, dRef = _run_shape_ab_multik(solution, kernelName, hsaco, chip, M, N, K)
    label = (f"MT{solution['MacroTile0']}x{solution['MacroTile1']} "
             f"M={M} N={N} K={K}")
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
