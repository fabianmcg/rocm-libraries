# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+PartialRMS (K1) Subtile epilogue (gfx950, bf16).

Exercises a comprehensive set of (M, K, N_hidden) shapes, verifying both:
  - D output (bf16 row-major M×N_hidden, tol=2e-2): h1 * gamma
  - partialBuf (fp32, tol=1e-4): 2D [M_padded, n_d] per-tile Σx²
    where n_d = ceil(N_hidden/MT0) (number of free0 tiles)

The kernel uses PartialRMSAxis=0 (free0 reduction): free0=N_hidden, free1=M.
D is produced in row-major order (M×N_hidden) by swapping A↔B^T.

The fixture is parametrised directly from gemm_partial_rms_k1_rowmajor.yaml,
which forks over all MatrixInstruction tile configs and both PartialRMSResidualAdd
values. N_hidden and K are read from the YAML TestAxes section; M shapes are
expanded per-solution against the solution's MacroTile1.
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
from epilogues.tensilelite.numpy_helpers import randBf16, randGamma, partialSumSq
from epilogues.tensilelite.yaml_solution_builder import readTestAxes

_K1_YAML = yamlPath("gemm_partial_rms_k1_rowmajor.yaml")

_K1_SOLUTIONS = enumerateSolutions("gemm_partial_rms_k1_rowmajor.yaml")

_K1_AXES = {}
try:
    _K1_AXES = readTestAxes(_K1_YAML, "K1", mt1=1)  # mt1 placeholder; expanded below
except Exception:
    pass

_N_HIDDEN = _K1_AXES.get("NHidden", [])
_K_VALUES  = _K1_AXES.get("K", [])


# ---------------------------------------------------------------------------
# Session-scoped fixture: one instance per solution in the K1 YAML.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _K1_SOLUTIONS],
    ids=[sid for _sol, sid in _K1_SOLUTIONS],
)
def k1_kernel(request):
    """Assemble and compile one K1 PartialRMS solution from the benchmark YAML."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    residualAdd = bool(solution.get("PartialRMSResidualAdd", False))
    return solution, kernelName, hsaco, chip, residualAdd


# ---------------------------------------------------------------------------
# Helper: run one shape and return comparison data.
# ---------------------------------------------------------------------------

def _run_shape(solution, kernelName, hsaco, chip, M, K, nHidden, residualAdd=False):
    """Run K1 for one (M, K, nHidden) shape; return (d_gpu_f32, d_ref_f32, pb_gpu, sumsq_ref).

    Kernel: free0=nHidden, free1=M. D is row-major (M, nHidden).
    partialBuf[token, t] = Σ_{i in MT0-tile t} h1_eff[token, i]²
    where h1_eff = h1 + residual when residualAdd, else h1.
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
    partialBuf = np.zeros((mPadded, nD), dtype=np.float32, order="C")

    gammaF32, gammaBf16 = randGamma(rng, nHidden)

    if residualAdd:
        residualF32 = (rng.random((M, nHidden), dtype=np.float32) - 0.5) * 0.2
        residualBf16 = residualF32.astype(ml_dtypes.bfloat16)
    else:
        residualBf16 = None

    # Reference: h1_eff = h1 [+ residual], D = bf16(h1_eff * gamma).
    h1    = aRow.astype(np.float32) @ bRow.astype(np.float32)
    h1Eff = h1 + np.asarray(residualBf16).astype(np.float32) if residualAdd else h1
    dRef  = (h1Eff * np.asarray(gammaBf16).astype(np.float32)[np.newaxis, :]).astype(
        ml_dtypes.bfloat16)
    sumsqRef = partialSumSq(h1Eff, nHidden, MT0)

    skArgs = compute_sk3_dp_args(nHidden, M, K, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    dInout  = amdgpu_exec.InOutArray(dFortran)
    pbInout = amdgpu_exec.InOutArray(partialBuf)

    epilogueArgs = [amdgpu_exec.InputArray(gammaBf16), pbInout]
    if residualAdd:
        epilogueArgs.append(amdgpu_exec.InputArray(np.ascontiguousarray(residualBf16)))

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
        result_holder["pb_gpu"] = np.asarray(arguments[31].array).copy().reshape(mPadded, nD)

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
    )


# ---------------------------------------------------------------------------
# Parametrised test.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("N_hidden", _N_HIDDEN, ids=[f"N{n}" for n in _N_HIDDEN])
@pytest.mark.parametrize("K", _K_VALUES, ids=[f"K{k}" for k in _K_VALUES])
def test_k1_shape(k1_kernel, K, N_hidden):
    """Verify K1 (PartialRMS axis=0) outputs D and 2D partialBuf.

    M shapes are expanded from the YAML TestAxes.K1 section against the
    solution's MacroTile1, so adding a new tile to the YAML automatically
    exercises the correct M coverage with no Python changes.
    """
    solution, kernelName, hsaco, chip, residualAdd = k1_kernel
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    nD  = math.ceil(N_hidden / MT0)

    for M, mLabel in readTestAxes(_K1_YAML, "K1", mt1=MT1)["M"]:
        dGpu, dRef, pbGpu, sumsqRef = _run_shape(
            solution, kernelName, hsaco, chip, M, K, N_hidden, residualAdd
        )
        label = f"MT0={MT0} MT1={MT1} {mLabel} N={N_hidden} K={K}"
        assertClose(dGpu, dRef, label, kind="D")
        assertClose(pbGpu[:M, :], sumsqRef,
                    f"{label} n_d={nD}", rtol=1e-4, atol=1e-4, kind="partialBuf")
