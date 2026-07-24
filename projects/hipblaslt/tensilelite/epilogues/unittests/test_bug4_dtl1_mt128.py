# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regression test for Bug 4: wrong D-tensor output in DirectToLds=1 + MT1=128 + PAP=0 kernels.

The bug manifests as a D-tensor correctness mismatch (not a crash) at M=N=K=8192.
It is data-dependent: seed=1784565775 is the confirmed trigger from the PAP-validation
sweep (run 1 of 5, Group A of repro_bug2_24configs.yaml).  Three additional seeds are
tested to guard against spurious passes.

Failing config set (40 kernels):
  DirectToLds=1, MT1=128, PrefetchAcrossPersistent=0, 1LDSBuffer=0,
  StreamKAtomic=0 (ForceDPOnly), MT0 in {64,128,256},
  DepthU in {64,128} (MT0=256 only has DU=64),
  PrefetchGlobalRead in {1,2}, StaggerU in {0,8},
  PreloadKernArgs in {False,True}.
"""

import math
import os

import numpy as np
import pytest

from epilogue_test_common import (
    amdgpu_exec, ml_dtypes,
    requires_gfx950,
    assertClose,
)
from epilogues.tensilelite.partialrms_helpers import (
    compute_sk3_dp_args, _pack_kernel_info, compileSolution, buildSubtileArgs,
)
from epilogues.tensilelite.numpy_helpers import randBf16, randGamma, partialSumSq
from epilogues.tensilelite.yaml_solution_builder import buildSolutionsFromYaml, solutionId

# ---------------------------------------------------------------------------
# YAML location — lives in epilogues/docs/, not epilogues/yaml/.
# ---------------------------------------------------------------------------

_YAML = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "epilogues", "docs", "repro_bug4_dtl1_mt128_datarace.yaml",
)

# Group indices (0=MT0=64, 1=MT0=128, 2=MT0=256) — all no-residual, problemIdx=0.
_ALL_GROUPS = [(0, gi) for gi in range(3)]

# Seeds: confirmed trigger first, then three additional seeds.
_SEEDS = [1784565775, 42, 12345, 999999937]

_M = 8192
_N_HIDDEN = 8192
_K = 8192


# ---------------------------------------------------------------------------
# Solution enumeration.
# ---------------------------------------------------------------------------

def _collect_solutions():
    """Return [(solution, sid)] for all DTLA1+MT1=128+PAP=0 configs.

    Returns [] when GPU or dependencies are unavailable.
    """
    if amdgpu_exec is None:
        return []
    try:
        from epilogues.tensilelite.partialrms_helpers import setup_tensile
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    except Exception:
        return []

    results = []
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
            # Defensive: the YAML already excludes PAP=1 and DTL=0, but guard
            # in case the solution pipeline overrides a parameter.
            if dtl != 1 or mt1 != 128 or pap != 0:
                continue
            sid = (
                solutionId(s)
                + f"_DTL{dtl}"
                + f"_DU{s['DepthU']}"
                + f"_PGR{s['PrefetchGlobalRead']}"
                + f"_SU{s['StaggerU']}"
                + f"_PKA{int(s.get('PreloadKernArgs', False))}"
            )
            results.append((s, sid))

    return results


_BUG4_SOLS = _collect_solutions()


# ---------------------------------------------------------------------------
# Kernel execution helper.
# ---------------------------------------------------------------------------

def _run_and_check(solution, kernelName, hsaco, label, seed):
    """Run the kernel at M=N=K=8192 with the given seed and check D and partialBuf."""
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    nD = math.ceil(_N_HIDDEN / MT0)
    mPadded = math.ceil(_M / MT1) * MT1
    numWG = math.ceil(_N_HIDDEN / MT0) * math.ceil(_M / MT1)

    rng = np.random.default_rng(seed=seed)
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

    epilogueArgs = [amdgpu_exec.InputArray(gammaBf16), pbInout]

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

    seedLabel = f"{label}_seed{seed}"
    assertClose(
        result["d_gpu"],
        np.asarray(dRef).astype(np.float32),
        seedLabel, kind="D",
    )
    assertClose(
        result["pb_gpu"][:_M, :],
        sumsqRef,
        seedLabel, rtol=1e-4, atol=1e-4, kind="partialBuf",
    )


# ---------------------------------------------------------------------------
# Session-scoped fixture: compile once per kernel.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=[s for s, _ in _BUG4_SOLS],
    ids=[sid for _, sid in _BUG4_SOLS],
)
def bug4_kernel(request):
    solution = request.param
    kernelName, hsaco, _ = compileSolution(solution)
    return solution, kernelName, hsaco


# ---------------------------------------------------------------------------
# Test: run all four seeds per kernel.
# ---------------------------------------------------------------------------

@requires_gfx950
def test_bug4_dtl1_mt128_datarace(bug4_kernel):
    """Bug 4: DirectToLds=1 + MT1=128 + PAP=0 must produce correct D at 8192³.

    Tested with the confirmed failure seed (1784565775) and three additional
    seeds to guard against false negatives.  Each seed is a sub-test call so
    a single seed failure is reported precisely without masking others.
    """
    solution, kernelName, hsaco = bug4_kernel
    label = (
        f"DTL1_MT{solution['MacroTile0']}x{solution['MacroTile1']}"
        f"_DU{solution['DepthU']}"
        f"_PGR{solution['PrefetchGlobalRead']}"
        f"_SU{solution['StaggerU']}"
    )
    for seed in _SEEDS:
        _run_and_check(solution, kernelName, hsaco, label, seed)
