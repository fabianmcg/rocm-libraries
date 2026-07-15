# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+PartialRMSQuant (K1) Subtile epilogue (gfx950, bf16).

Exercises a set of (M, K, N_hidden) shapes, verifying:
  - D output (bf16 row-major M×N_hidden, tol=2e-2): h1 * gamma
  - partialBuf first half (fp32, tol=1e-4): 2D [M_padded, n_d] per-tile Σx² (pre-gamma)
  - partialBuf second half (fp32, tol=1e-4): 2D [M_padded, n_d] amax(|D|)/fp8_max (post-gamma)

The buffer has shape [2*M_padded, n_d]:
  - rows [0, M_padded)        → Σx² of h1 (pre-gamma)
  - rows [M_padded, 2*M_padded) → amax(|D|)/448 (post-gamma)
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
from epilogues.tensilelite.numpy_helpers import randBf16, randGamma, partialSumSq, partialAmax
from epilogues.tensilelite.yaml_solution_builder import readTestAxes

_K1_YAML = yamlPath("gemm_partial_rms_quant_k1.yaml")

_K1_SOLUTIONS = enumerateSolutions("gemm_partial_rms_quant_k1.yaml")

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
def k1_quant_kernel(request):
    """Assemble and compile one K1 PartialRMSQuant solution from the benchmark YAML."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


# ---------------------------------------------------------------------------
# Helper: run one shape and return comparison data.
# ---------------------------------------------------------------------------

def _run_shape(solution, kernelName, hsaco, chip, M, K, nHidden):
    """Run K1 quant for one (M, K, nHidden) shape.

    Returns (d_gpu_f32, d_ref_f32, pb_gpu, sumsq_ref, amax_ref).
    """
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    nD  = math.ceil(nHidden / MT0)
    mPadded = math.ceil(M / MT1) * MT1
    numWG   = math.ceil(nHidden / MT0) * math.ceil(M / MT1)

    rng = np.random.default_rng(seed=M * 10000 + K)

    aRow = randBf16(rng, (M, K))
    bRow = randBf16(rng, (K, nHidden))

    # Kernel operands: A=bFortran (K×nHidden), B=aFortranT (K×M), both Fortran order.
    bFortran  = np.asfortranarray(bRow)
    aFortranT = np.asfortranarray(aRow.T)
    cFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    dFortran  = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    # Buffer is doubled: first half Σx², second half amax/fp8_max.
    partialBuf = np.zeros((2 * mPadded, nD), dtype=np.float32, order="C")

    gammaF32, gammaBf16 = randGamma(rng, nHidden)

    # Reference computations.
    h1 = aRow.astype(np.float32) @ bRow.astype(np.float32)
    gammaF = np.asarray(gammaBf16).astype(np.float32)
    dOut = h1 * gammaF[np.newaxis, :]  # post-gamma D (fp32)
    sumsqRef = partialSumSq(h1, nHidden, MT0)       # first half (pre-gamma)
    amaxRef  = partialAmax(dOut, nHidden, MT0)       # second half (post-gamma /448)
    dRef = dOut.astype(ml_dtypes.bfloat16)

    skArgs = compute_sk3_dp_args(nHidden, M, K, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    dInout  = amdgpu_exec.InOutArray(dFortran)
    pbInout = amdgpu_exec.InOutArray(partialBuf)

    epilogueArgs = [amdgpu_exec.InputArray(gammaBf16), pbInout]

    args = buildSubtileArgs(
        nHidden, M, K, numWG,
        dInout, amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        skArgs, ki0, ki1, epilogueArgs,
    )

    result_holder = {}

    def capture(arguments):
        dRaw = np.asarray(arguments[8].array)
        result_holder["d_gpu"] = dRaw.reshape(nHidden, M, order="F").T.astype(np.float32)
        pb = np.asarray(arguments[29].array).copy().reshape(2 * mPadded, nD)
        result_holder["pb_gpu"] = pb

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    return (
        result_holder["d_gpu"],
        np.asarray(dRef).astype(np.float32),
        result_holder["pb_gpu"],
        sumsqRef,
        amaxRef,
        mPadded,
    )


# ---------------------------------------------------------------------------
# Parametrised test.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize(
    "K,N_hidden",
    _KN_PAIRS,
    ids=[f"K{k}-N{n}" for k, n in _KN_PAIRS],
)
def test_k1_quant_shape(k1_quant_kernel, K, N_hidden):
    """Verify K1 PartialRMSQuant outputs D, Σx² (first half), and amax/fp8_max (second half)."""
    solution, kernelName, hsaco, chip = k1_quant_kernel
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    nD  = math.ceil(N_hidden / MT0)

    for M, mLabel in readTestAxes(_K1_YAML, "K1", mt1=MT1)["M"]:
        dGpu, dRef, pbGpu, sumsqRef, amaxRef, mPadded = _run_shape(
            solution, kernelName, hsaco, chip, M, K, N_hidden
        )
        label = f"MT0={MT0} MT1={MT1} {mLabel} N={N_hidden} K={K}"
        assertClose(dGpu, dRef, label, kind="D")
        assertClose(pbGpu[:M, :], sumsqRef,
                    f"{label} n_d={nD} sumsq", rtol=1e-4, atol=1e-4, kind="partialBuf_sumsq")
        assertClose(pbGpu[mPadded:mPadded + M, :], amaxRef,
                    f"{label} n_d={nD} amax", rtol=1e-4, atol=1e-4, kind="partialBuf_amax")
