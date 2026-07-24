# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regression tests for Bug 2: PartialRMS wrong output at large problem sizes.

Covers the two failure modes discovered in the tuning sweep:
  Bug 2a: DirectToLds=0 + MT1=256 (wg_n=2)
  Bug 2b: DirectToLds=1 + MT1=128 (wg_n=1) — load-dependent, but confirmed

Both bugs produce wrong D and wrong partialBuf at M=N=K=8192. They are
not caught by the existing suite because it only tests DirectToLds=1 and
only tests problem sizes up to 8192×4096.

Each test fixture is a (solution, kernelName, hsaco) triple compiled once
per session. The parametrize IDs embed DTL/MT/DU/PGR/SU/PKA so failures
are immediately actionable.
"""

import math

import numpy as np
import pytest

from epilogue_test_common import (
    amdgpu_exec, ml_dtypes,
    requires_gfx950, yamlPath,
    assertClose,
)
from epilogues.tensilelite.partialrms_helpers import (
    compute_sk3_dp_args, _pack_kernel_info, compileSolution, buildSubtileArgs,
)
from epilogues.tensilelite.numpy_helpers import randBf16, randGamma, partialSumSq
from epilogues.tensilelite.yaml_solution_builder import (
    buildSolutionsFromYaml, solutionId,
)

# ---------------------------------------------------------------------------
# Enumerate all solutions from the expanded YAML, across all groups, and
# partition them into the two known-failing sets.
# ---------------------------------------------------------------------------

_YAML = yamlPath("gemm_partial_rms_k1_rowmajor.yaml")

# (problemIdx, groupIdx) pairs covering all 8 groups in the expanded YAML.
# problemIdx 0 = no-residual (4 groups), problemIdx 1 = residual (4 groups).
_ALL_GROUPS = [(pi, gi) for pi in range(2) for gi in range(4)]


def _collect_solutions():
    """Return two lists: (bug2a_sols, bug2b_sols).

    bug2a: DirectToLds=0, MT1=256 (wg_n=2)
    bug2b: DirectToLds=1, MT1=128 (wg_n=1)

    PrefetchAcrossPersistent=1 configs are excluded from both lists — they
    trigger a GPU hard fault (SIGABRT in amdgpu_exec.synchronize), which
    kills the process rather than raising a catchable exception. Those configs
    are tracked separately as Bug 3 in repro_bug3_pap1_dtl1_mt128.yaml.

    Returns [] for both when deps/GPU are unavailable.
    """
    if amdgpu_exec is None:
        return [], []
    try:
        from epilogues.tensilelite.partialrms_helpers import setup_tensile
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    except Exception:
        return [], []

    bug2a, bug2b = [], []
    for pi, gi in _ALL_GROUPS:
        try:
            sols = buildSolutionsFromYaml(
                _YAML, assembler, isaInfoMap, debugConfig,
                problemIdx=pi, groupIdx=gi,
            )
        except Exception:
            continue
        for s in sols:
            dtl = s.get("DirectToLds", 0)
            mt1 = s["MacroTile1"]
            pap = s.get("PrefetchAcrossPersistent", 0)
            if pap == 1:
                continue  # Bug 3: hard GPU fault — excluded from this suite.
            sid = (solutionId(s)
                   + f"_DTL{dtl}"
                   + f"_DU{s['DepthU']}"
                   + f"_PGR{s['PrefetchGlobalRead']}"
                   + f"_SU{s['StaggerU']}"
                   + f"_PKA{int(s.get('PreloadKernArgs', False))}")
            if dtl == 0 and mt1 == 256:
                bug2a.append((s, sid))
            elif dtl == 1 and mt1 == 128:
                bug2b.append((s, sid))

    return bug2a, bug2b


_BUG2A_SOLS, _BUG2B_SOLS = _collect_solutions()


# ---------------------------------------------------------------------------
# Shared kernel execution helper.
# ---------------------------------------------------------------------------

_M = 8192
_N_HIDDEN = 8192
_K = 8192


def _run_and_check(solution, kernelName, hsaco, label):
    """Run one kernel at M=N=K=8192 and assert D and partialBuf are correct."""
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    nD = math.ceil(_N_HIDDEN / MT0)
    mPadded = math.ceil(_M / MT1) * MT1
    numWG = math.ceil(_N_HIDDEN / MT0) * math.ceil(_M / MT1)

    rng = np.random.default_rng(seed=42)
    aRow = randBf16(rng, (_M, _K))
    bRow = randBf16(rng, (_K, _N_HIDDEN))
    _, gammaBf16 = randGamma(rng, _N_HIDDEN)

    bFortran  = np.asfortranarray(bRow)
    aFortranT = np.asfortranarray(aRow.T)
    cFortran  = np.zeros((_N_HIDDEN, _M), dtype=ml_dtypes.bfloat16, order="F")
    dFortran  = np.zeros((_N_HIDDEN, _M), dtype=ml_dtypes.bfloat16, order="F")
    partialBuf = np.zeros((mPadded, nD), dtype=np.float32, order="C")

    h1 = aRow.astype(np.float32) @ bRow.astype(np.float32)
    dRef = (h1 * np.asarray(gammaBf16).astype(np.float32)[np.newaxis, :]).astype(
        ml_dtypes.bfloat16)
    sumsqRef = partialSumSq(h1, _N_HIDDEN, MT0)

    skArgs = compute_sk3_dp_args(_N_HIDDEN, _M, _K, solution)
    ki0, ki1 = _pack_kernel_info(solution)

    dInout  = amdgpu_exec.InOutArray(dFortran)
    pbInout = amdgpu_exec.InOutArray(partialBuf)

    result = {}

    def capture(arguments):
        dRaw = np.asarray(arguments[8].array)
        result["d_gpu"] = dRaw.reshape(_N_HIDDEN, _M, order="F").T.astype(np.float32)
        result["pb_gpu"] = np.asarray(arguments[31].array).copy().reshape(mPadded, nD)

    residualAdd = bool(solution.get("PartialRMSResidualAdd", False))
    epilogueArgs = [amdgpu_exec.InputArray(gammaBf16), pbInout]
    if residualAdd:
        residualBf16 = np.zeros((_M, _N_HIDDEN), dtype=ml_dtypes.bfloat16)
        epilogueArgs.append(amdgpu_exec.InputArray(np.ascontiguousarray(residualBf16)))

    args = buildSubtileArgs(
        _N_HIDDEN, _M, _K, numWG,
        dInout, amdgpu_exec.InputArray(cFortran),
        amdgpu_exec.InputArray(bFortran), amdgpu_exec.InputArray(aFortranT),
        skArgs, ki0, ki1, epilogueArgs,
    )

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=(numWG, 1, 1), block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    assertClose(result["d_gpu"], np.asarray(dRef).astype(np.float32),
                label, kind="D")
    assertClose(result["pb_gpu"][:_M, :], sumsqRef,
                label, rtol=1e-4, atol=1e-4, kind="partialBuf")


# ---------------------------------------------------------------------------
# Bug 2a: DirectToLds=0 + MT1=256
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[s for s, _ in _BUG2A_SOLS],
    ids=[sid for _, sid in _BUG2A_SOLS],
)
def bug2a_kernel(request):
    solution = request.param
    kernelName, hsaco, _ = compileSolution(solution)
    return solution, kernelName, hsaco


@requires_gfx950
def test_bug2a_dtl0_mt256(bug2a_kernel):
    """Bug 2a: DirectToLds=0 + MT1=256 must produce correct D and partialBuf at 8192^3."""
    solution, kernelName, hsaco = bug2a_kernel
    label = f"DTL0_MT{solution['MacroTile0']}x{solution['MacroTile1']}_DU{solution['DepthU']}"
    _run_and_check(solution, kernelName, hsaco, label)


# ---------------------------------------------------------------------------
# Bug 2b: DirectToLds=1 + MT1=128
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[s for s, _ in _BUG2B_SOLS],
    ids=[sid for _, sid in _BUG2B_SOLS],
)
def bug2b_kernel(request):
    solution = request.param
    kernelName, hsaco, _ = compileSolution(solution)
    return solution, kernelName, hsaco


@requires_gfx950
def test_bug2b_dtl1_mt128(bug2b_kernel):
    """Bug 2b: DirectToLds=1 + MT1=128 must produce correct D and partialBuf at 8192^3."""
    solution, kernelName, hsaco = bug2b_kernel
    label = f"DTL1_MT{solution['MacroTile0']}x{solution['MacroTile1']}_DU{solution['DepthU']}"
    _run_and_check(solution, kernelName, hsaco, label)
