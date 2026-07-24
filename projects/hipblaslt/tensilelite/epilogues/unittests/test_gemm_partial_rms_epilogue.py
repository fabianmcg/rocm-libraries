# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end test: row-major GEMM+RMSNorm via K1 (PartialRMSAxis=0) → row_div (gfx950, bf16).

Runs the two-kernel pipeline and verifies the final output D against a numpy
GEMM + RMSNorm reference: D = (A @ B * gamma) / sqrt(mean((A @ B)^2) + eps).

K1 (PartialRMSAxis=0): free0=N_hidden, free1=M. Writes row-major D (M×N_hidden)
and partialBuf[token, tile] (f32, M_padded × n_d, n_d = ceil(N_hidden/MT0)).

row_div: divides D in-place using partialBuf, grid=(M, 1, 1), block=(N_hidden, 1, 1).

row_div is read from epilogues/kernels/row_div.s and requires a gfx950 GPU.
"""

import math
import os

import numpy as np
import pytest

from epilogue_test_common import (
    TENSILE_ROOT, amdgpu_exec, ml_dtypes,
    requires_gfx950, yamlPath,
    enumerateSolutions, assertClose,
)
from epilogues.tensilelite.partialrms_helpers import (
    compute_sk3_dp_args, _pack_kernel_info, compileSolution, buildSubtileArgs,
    buildRowDivArgs,
)
from epilogues.tensilelite.numpy_helpers import randBf16, randGamma, rmsNormReference
from epilogues.tensilelite.yaml_solution_builder import readTestAxes

_K1_YAML    = yamlPath("gemm_partial_rms_k1_rowmajor.yaml")
_ROW_DIV_S  = os.path.join(TENSILE_ROOT, "epilogues", "kernels", "row_div.s")
_ROW_DIV_NAME = "row_div"

_PIPELINE_SOLUTIONS = enumerateSolutions(
    "gemm_partial_rms_k1_rowmajor.yaml",
    predicate=lambda s: not s.get("PartialRMSResidualAdd", False),
)

_PIPELINE_AXES = {}
try:
    _PIPELINE_AXES = readTestAxes(_K1_YAML, "Pipeline", mt1=1)  # mt1 placeholder
except Exception:
    pass

_K_VALUES = _PIPELINE_AXES.get("K", [])

_EPS      = 1e-5
_RD_BLOCK = 128   # columns each row_div block processes


# ---------------------------------------------------------------------------
# Session-scoped fixture: one instance per no-residual K1 solution in the YAML.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _PIPELINE_SOLUTIONS],
    ids=[sid for _sol, sid in _PIPELINE_SOLUTIONS],
)
def pipeline_kernels(request):
    """Assemble K1 (no-residual) and compile row_div for one solution from the YAML.

    Returns (k1Sol, k1Name, k1Hsaco, rdHsaco, chip, MT0, MT1).
    """
    k1Sol = request.param
    k1Name, k1Hsaco, chip = compileSolution(k1Sol)

    if not os.path.exists(_ROW_DIV_S):
        pytest.skip(f"row_div.s not found at {_ROW_DIV_S}")
    with open(_ROW_DIV_S) as fh:
        rdAsm = fh.read()
    rdHsaco = amdgpu_exec.compile_asm_to_hsaco(rdAsm, chip)

    return (k1Sol, k1Name, k1Hsaco, rdHsaco, chip, k1Sol["MacroTile0"], k1Sol["MacroTile1"])


# ---------------------------------------------------------------------------
# Helper: run full two-kernel pipeline for one (M, K, nHidden) shape.
# ---------------------------------------------------------------------------

def _run_pipeline(pipelineKernelsFixture, M, K, nHidden, eps):
    """Run K1 then row_div and verify against numpy GEMM+RMSNorm.

    Reference: D = (A@B * gamma) / sqrt(mean((A@B)²) + eps).
    """
    (k1Sol, k1Name, k1Hsaco, rdHsaco, chip, MT0, MT1) = pipelineKernelsFixture

    nD      = math.ceil(nHidden / MT0)
    mPadded = math.ceil(M / MT1) * MT1
    numWgK1 = math.ceil(nHidden / MT0) * math.ceil(M / MT1)
    invD    = 1.0 / nHidden

    rng = np.random.default_rng(seed=M * 100000 + nHidden * 100 + K)

    aRow = randBf16(rng, (M, K))
    bRow = randBf16(rng, (K, nHidden))
    gammaF32, gammaBf16 = randGamma(rng, nHidden)

    # Reference.
    dRefF32 = rmsNormReference(aRow, bRow, gammaBf16, invD, eps)

    # Device buffers.
    bFortran   = np.asfortranarray(bRow)
    aFortranT  = np.asfortranarray(aRow.T)
    cFortran   = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    dFortran   = np.zeros((nHidden, M), dtype=ml_dtypes.bfloat16, order="F")
    partialBuf = np.zeros((mPadded, nD), dtype=np.float32, order="C")

    sk1 = compute_sk3_dp_args(nHidden, M, K, k1Sol)
    ki0, ki1 = _pack_kernel_info(k1Sol)

    dInout  = amdgpu_exec.InOutArray(dFortran)
    pbInout = amdgpu_exec.InOutArray(partialBuf)

    argsK1 = buildSubtileArgs(
        nHidden, M, K, numWgK1,
        dInout, amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        sk1, ki0, ki1,
        [amdgpu_exec.InputArray(gammaBf16), pbInout],
    )

    amdgpu_exec.execute_hsaco(
        hsaco=k1Hsaco, kernel_name=k1Name, arguments=argsK1,
        grid_dim=(numWgK1, 1, 1), block_dim=(k1Sol["NumThreads"], 1, 1),
        num_iterations=1,
    )

    # row_div operates on a contiguous row-major (M, nHidden) bf16 copy of D.
    dRow        = np.ascontiguousarray(np.asarray(dFortran).reshape(nHidden, M, order="F").T)
    partialBufM = np.ascontiguousarray(partialBuf[:M, :])
    dRowInout   = amdgpu_exec.InOutArray(dRow)

    nC     = _RD_BLOCK
    nSplit = nHidden // nC

    argsRd = buildRowDivArgs(dRowInout, amdgpu_exec.InputArray(partialBufM),
                             nHidden, nC, nD, invD, eps)

    dFinalCaptured = {}

    def captureFinal(_arguments):
        dFinalCaptured["d"] = np.asarray(dRowInout.array).astype(np.float32)

    amdgpu_exec.execute_hsaco(
        hsaco=rdHsaco, kernel_name=_ROW_DIV_NAME, arguments=argsRd,
        grid_dim=(M, nSplit, 1), block_dim=(64, 1, 1),
        num_iterations=1, verify_fn=captureFinal,
    )

    dGpuF32 = dFinalCaptured["d"]
    if dGpuF32.ndim == 1:
        dGpuF32 = dGpuF32.reshape(M, nHidden)

    assertClose(dGpuF32[:M], dRefF32[:M], f"M{M}_N{nHidden}_K{K}", kind="D")


# ---------------------------------------------------------------------------
# Parametrised pipeline test.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("K", _K_VALUES, ids=[f"K{k}" for k in _K_VALUES])
def test_pipeline_shape(pipeline_kernels, K):
    """Verify the final GEMM+RMSNorm output matches a numpy reference.

    M and NHidden shapes come from TestAxes.Pipeline in the YAML so no
    shape logic is hard-coded here.
    """
    (k1Sol, *_rest) = pipeline_kernels
    MT1 = _rest[-1]

    axes = readTestAxes(_K1_YAML, "Pipeline", mt1=MT1)
    for M, _mLabel in axes["M"]:
        for nHidden in axes["NHidden"]:
            _run_pipeline(pipeline_kernels, M, K, nHidden, _EPS)
