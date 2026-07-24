# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest suite for the fused GEMM+RstdScale (K3) Subtile epilogue (gfx950, bf16).

Exercises a comprehensive set of (M, N_hidden) shapes, verifying:
  - y output (bf16, tol=2e-2): (h2 @ W1.T) * rstd[:, None]

The fixture is parametrized over solutions enumerated from gemm_rstdscale_k3.yaml.
N_out is pinned to MacroTile1 = 16 * 4 * wg_n (row-containment invariant).
N_hidden (GEMM2 contraction dim) varies per shape; DepthU=64 is fixed in the
solution so iters_per_tile = ceil(N_hidden / 64) changes per shape.
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
from epilogues.tensilelite.yaml_solution_builder import problemSizesFromYaml

_K3_YAML = yamlPath("gemm_rstdscale_k3.yaml")

_K3_SOLUTIONS = enumerateSolutions("gemm_rstdscale_k3.yaml")

# Shape list read from YAML ProblemSizes: (M, N_out, 1, N_hidden, ...).
# Test uses sizes[0]=M and sizes[3]=N_hidden; N_out comes from the solution.
_RAW_SIZES = []
try:
    _RAW_SIZES = problemSizesFromYaml(_K3_YAML)
except Exception:
    pass

_SHAPES = [(s[0], s[3], f"M{s[0]}_K{s[3]}") for s in _RAW_SIZES]


# ---------------------------------------------------------------------------
# Session-scoped fixture: one instance per solution in the K3 YAML.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[sol for sol, _id in _K3_SOLUTIONS],
    ids=[sid for _sol, sid in _K3_SOLUTIONS],
)
def k3_kernel(request):
    """Assemble and compile one K3 RstdScale solution from the benchmark YAML."""
    solution = request.param
    kernelName, hsaco, chip = compileSolution(solution)
    return solution, kernelName, hsaco, chip


# ---------------------------------------------------------------------------
# Helper: run one shape and return comparison data.
# ---------------------------------------------------------------------------

def _run_shape(solution, kernelName, hsaco, chip, M, N_hidden):
    """Run K3 for one (M, N_hidden) shape; return (y_gpu_f32, y_ref_f32, M)."""
    MT0   = solution["MacroTile0"]
    N_out = solution["MacroTile1"]   # row-containment invariant

    mPadded = math.ceil(M / MT0) * MT0
    numWG   = math.ceil(M / MT0)    # == ceil(M/MT0) * ceil(N_out/N_out)

    rng = np.random.default_rng(seed=M * 10000 + N_hidden)

    h2F32  = np.asfortranarray(rng.random((N_hidden, M), dtype=np.float32) * 0.1)
    w1F32  = np.asfortranarray(rng.random((N_hidden, N_out), dtype=np.float32) * 0.1)
    h2Bf16 = np.asfortranarray(h2F32.astype(ml_dtypes.bfloat16))
    w1Bf16 = np.asfortranarray(w1F32.astype(ml_dtypes.bfloat16))

    cBf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order="F")
    yBf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order="F")

    rstdRef    = rng.random(M, dtype=np.float32) * 0.5 + 0.5
    rstdPadded = np.zeros(mPadded, dtype=np.float32)
    rstdPadded[:M] = rstdRef

    # Reference: h3 = h2.T @ w1, y = h3 * rstd[:, None].
    h3    = np.asarray(h2Bf16).astype(np.float32).T @ np.asarray(w1Bf16).astype(np.float32)
    yRef  = (h3 * rstdRef[:, np.newaxis]).astype(ml_dtypes.bfloat16)

    skArgs = compute_sk3_dp_args(M, N_out, N_hidden, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    # K3 operands: free0=M, free1=N_out, bound=N_hidden.
    # Strides follow buildSubtileArgs: free0,free0,bound,bound = M,M,N_hidden,N_hidden.
    args = buildSubtileArgs(
        M, N_out, N_hidden, numWG,
        amdgpu_exec.InOutArray(yBf16), amdgpu_exec.InputArray(cBf16),
        amdgpu_exec.InputArray(h2Bf16), amdgpu_exec.InputArray(w1Bf16),
        skArgs, ki0, ki1,
        [amdgpu_exec.InputArray(rstdPadded)],
    )

    result_holder = {}

    def capture(arguments):
        result_holder["y_gpu"] = np.asarray(arguments[8].array).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    return result_holder["y_gpu"].astype(np.float32), np.asarray(yRef).astype(np.float32), M


# ---------------------------------------------------------------------------
# Parametrised test.
# ---------------------------------------------------------------------------

@requires_gfx950
@pytest.mark.parametrize("M,N_hidden,label", _SHAPES, ids=[s[2] for s in _SHAPES])
def test_k3_shape(k3_kernel, M, N_hidden, label):
    """Verify K3 (RstdScale) output y for shape M x N_out x N_hidden."""
    solution, kernelName, hsaco, chip = k3_kernel
    N_out = solution["MacroTile1"]

    yGpu, yRef, _ = _run_shape(solution, kernelName, hsaco, chip, M, N_hidden)

    assertClose(yGpu[:M], yRef[:M],
                f"M={M} N_out={N_out} N_hidden={N_hidden} ({label})", kind="y")
