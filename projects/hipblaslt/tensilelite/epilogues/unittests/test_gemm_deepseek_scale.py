# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+DeepseekScale Subtile mainloop scale path (PGR=0/1/2, gfx950, fp8 in / f32 out).

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

Device layout for scale buffers (matches the b32 DirectToLds load in the kernel):
  scaleA -- logical uint8 [Mpadded, nKBlocks], device flat uint8 of length nRowGroups*nKBlocks*R*4:
    R = 64  (MatrixInstM * mma_m = 16 * 4; rows per wave for in-scope configs)
    rowGroup = m // R;  rowSlot = m % R
    device[((rowGroup*nKBlocks + kb)*R + rowSlot)*4 + b] = scaleA[m, kb]  for b in 0..3

  scaleB -- logical uint8 [nKBlocks, nNBlocks], device flat uint8 of length nNBlocks*nKBlocks*256:
    device[(nb*nKBlocks + kb)*256 + j] = scaleB[kb, nb]  for all j in 0..255
    (all 256 bytes in each (nb, kb) slot hold the same broadcast E8M0 byte)
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
# Device-layout swizzle helpers.
# ---------------------------------------------------------------------------

# R = MatrixInstM * mma_m = 16 * 4 = 64 rows handled per wave for in-scope configs.
_ROWS_PER_WAVE = 64


def _swizzleScaleADevice(scaleA_bytes: np.ndarray, nKBlocks: int) -> np.ndarray:
    """Convert logical scaleA [Mpadded, nKBlocks] uint8 to the device b32-LDS layout.

    Device layout (flat uint8, length nRowGroups*nKBlocks*R*4):
      device[((rowGroup*nKBlocks + kb)*R + rowSlot)*4 + b] = scaleA[m, kb]
    where R=64, rowGroup=m//R, rowSlot=m%R, b in 0..3 (broadcast).
    """
    R = _ROWS_PER_WAVE
    mPadded = scaleA_bytes.shape[0]
    assert mPadded % R == 0, f"mPadded ({mPadded}) must be divisible by R ({R})"
    nRowGroups = mPadded // R
    # Reshape to [nRowGroups, R, nKBlocks], then permute to [nRowGroups, nKBlocks, R].
    shaped = scaleA_bytes.reshape(nRowGroups, R, nKBlocks).transpose(0, 2, 1)
    # Broadcast each byte into 4 identical bytes: add axis, tile to length 4.
    broadcast = np.repeat(shaped[:, :, :, np.newaxis], 4, axis=3)
    return broadcast.ravel()


def _swizzleScaleBDevice(scaleB_bytes: np.ndarray, nKBlocks: int) -> np.ndarray:
    """Convert logical scaleB [nKBlocks, nNBlocks] uint8 to the device b32-LDS layout.

    Device layout (flat uint8, length nNBlocks*nKBlocks*256):
      device[(nb*nKBlocks + kb)*256 + j] = scaleB[kb, nb]  for all j in 0..255
    Each (nb, kb) slot holds 256 identical copies of the E8M0 byte.
    """
    nNBlocks = scaleB_bytes.shape[1]
    # Transpose to [nNBlocks, nKBlocks] so C-order matches (nb, kb) indexing.
    transposed = scaleB_bytes.T  # [nNBlocks, nKBlocks]
    # Broadcast each byte into 256 identical bytes.
    broadcast = np.repeat(transposed[:, :, np.newaxis], 256, axis=2)
    return broadcast.ravel()


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
                         alpha=1.0, beta=0.0, cMatrix=None):
    """Run multi-K DeepseekScaleAB kernel for one (M, N, K) shape."""
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    (aFortran, bFortran, cFortran, dFortran,
     scaleAPadded, scaleBBytes, scaleBRef, aKM, bKN, scaleARef) = \
        _make_inputs_ab_multik(M, N, K, mPadded)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    aMK  = np.asarray(aKM).T
    dRef = numpy_ref_multiblock(aMK, np.asarray(bKN), scaleARef, scaleBRef,
                                alpha, beta, c_ref)

    nKBlocks = K // 128
    epilogueArgs = [amdgpu_exec.InputArray(_swizzleScaleADevice(scaleAPadded, nKBlocks)),
                    amdgpu_exec.InputArray(_swizzleScaleBDevice(scaleBBytes, nKBlocks))]
    dGpu = _execute_and_compare(solution, kernelName, hsaco, M, N, K, numWG,
                                aFortran, bFortran, cFortran, dFortran,
                                epilogueArgs, alpha, beta=beta)
    return dGpu, dRef.astype(np.float32)


def _run_shape_a_multik(solution, kernelName, hsaco, chip, M, N, K,
                        alpha=1.0, beta=0.0, cMatrix=None):
    """Run multi-K DeepseekScaleA-only kernel for one (M, N, K) shape.

    scaleB side uses a unit scale inside the mainloop (0x7f = 1.0 in E8M0).
    The reference uses scaleA only, with scaleB effectively 1.0.
    """
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    (aFortran, bFortran, cFortran, dFortran,
     scaleAPadded, _scaleBBytes, _scaleBRef, aKM, bKN, scaleARef) = \
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

    scaleA side uses a unit scale inside the mainloop (0x7f = 1.0 in E8M0).
    The reference uses scaleB only, with scaleA effectively 1.0.
    """
    MT0     = solution["MacroTile0"]
    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0) * math.ceil(N / solution["MacroTile1"])

    (aFortran, bFortran, cFortran, dFortran,
     _scaleAPadded, scaleBBytes, scaleBRef, aKM, bKN, _scaleARef) = \
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
    epilogueArgs = [amdgpu_exec.InputArray(_swizzleScaleBDevice(scaleBBytes, nKBlocks))]
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
    """Verify multi-K DeepseekScaleAB output: per-K-block E8M0 scale via MFMA operand."""
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
    """Verify multi-K DeepseekScaleA-only: unit scaleB fallback path in MFMA operand."""
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
    """Verify multi-K DeepseekScaleB-only: unit scaleA fallback path in MFMA operand."""
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
     scaleAPadded, scaleBBytes, scaleBRef, aKM, bKN, scaleARef) = \
        _make_inputs_ab_multik(M, N, K, mPadded)

    if cMatrix is not None:
        cFortran = np.asfortranarray(cMatrix.astype(np.float32))

    c_ref = cMatrix if cMatrix is not None else np.zeros((M, N), dtype=np.float32)
    aMK  = np.asarray(aKM).T
    dRef = numpy_ref_multiblock(aMK, np.asarray(bKN), scaleARef, scaleBRef,
                                alpha, beta, c_ref)

    nKBlocks = K // 128
    epilogueArgs = [amdgpu_exec.InputArray(_swizzleScaleADevice(scaleAPadded, nKBlocks)),
                    amdgpu_exec.InputArray(_swizzleScaleBDevice(scaleBBytes, nKBlocks))]
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
