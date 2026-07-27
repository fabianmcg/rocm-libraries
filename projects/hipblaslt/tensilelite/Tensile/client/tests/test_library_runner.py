# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M10 test suite: LibraryRunner — solution selection and Formocast prediction.

Covers Tasks 10.3–10.4 of the TensileLite Python client plan:
  10.3  LibraryRunner Python class (find_best, find_top_n, filter_by_predicate)
  10.4  GPU tests: load real library, predicate eval, top-N sort, Formocast range

Tests that touch the GPU require @requires_gfx950.  The library YAML is
discovered at session start from the tox tmp dir produced by prior test runs
(nan_bounds_check_odd_padding / Cijk_Ailk_Bjlk_HHS_BH_UserArgs_00). If not
found, GPU tests are skipped with a clear message.
"""

from __future__ import annotations

import glob
import os

import pytest

try:
    import tensilelite_runtime as _rt_mod
    HAVE_RUNTIME = True
except ImportError:
    _rt_mod = None
    HAVE_RUNTIME = False

from .conftest import requires_gfx950

# ---------------------------------------------------------------------------
# Fixture: locate the HHS TensileLibrary produced by the common test suite.
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
_TOX_TMP = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", ".tox", "unit", "tmp"))

# Glob for any HHS library in the tox tmp directory.
_HHS_GLOB = os.path.join(
    _TOX_TMP,
    "**",
    "Cijk_Ailk_Bjlk_HHS_BH_UserArgs_00",
    "**",
    "gfx950",
    "TensileLibrary.yaml",
)


def _findHhsLibrary() -> str | None:
    matches = glob.glob(_HHS_GLOB, recursive=True)
    return matches[0] if matches else None


@pytest.fixture(scope="session")
def hhs_library_path():
    """Absolute path to the HHS TensileLibrary.yaml, or skip if absent."""
    path = _findHhsLibrary()
    if path is None:
        pytest.skip(
            "HHS TensileLibrary.yaml not found in tox tmp dir;"
            " run 'tox -e py3' first to generate it"
        )
    return path


# ---------------------------------------------------------------------------
# Problem factory: Half*Half->Half NT GEMM with HPA, matching HHS predicates.
# The HHS library uses OperationIdentifier Contraction_l_Ailk_Bjlk_Cijk_Dijk
# which corresponds to transA=False, transB=True in GEMM_Strides.
# ---------------------------------------------------------------------------

def _makeHhsProblem(m: int = 256, n: int = 256, k: int = 256):
    rt = _rt_mod
    return rt.Problem(
        M=m, N=n, K=k,
        dtype_a="Half", dtype_b="Half", dtype_c="Half", dtype_d="Half",
        trans_a=False, trans_b=True,
        high_precision_accumulate=True,
    )


# ---------------------------------------------------------------------------
# TestLoadLibrary: load a real library and verify find_best returns a solution.
# ---------------------------------------------------------------------------


class TestLoadLibrary:
    @requires_gfx950
    def test_load_returns_library(self, hhs_library_path):
        """load_library returns a non-null Library object."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        lib = _rt_mod.load_library(hhs_library_path)
        assert lib is not None

    @requires_gfx950
    def test_find_best_returns_solution(self, hhs_library_path):
        """find_best returns a solution with a non-empty kernel name."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem()
        sol = runner.find_best(prob)
        assert sol is not None, "expected a matching solution for 256×256×256 HHS NT"
        assert sol.kernel_name, "kernel_name must not be empty"


# ---------------------------------------------------------------------------
# TestPredicateEval: hardware and task predicate evaluation.
# ---------------------------------------------------------------------------


class TestPredicateEval:
    @requires_gfx950
    def test_hardware_predicate_true_for_gfx950(self, hhs_library_path):
        """eval_hardware_predicate returns True for the actual gfx950 device."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem()
        sol = runner.find_best(prob)
        assert sol is not None
        assert sol.eval_hardware_predicate(runner.hardware)

    @requires_gfx950
    def test_task_predicate_true_for_matching_problem(self, hhs_library_path):
        """eval_task_predicate returns True when problem matches the solution."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem()
        sol = runner.find_best(prob)
        assert sol is not None
        assert sol.eval_task_predicate(runner.hardware, prob)

    @requires_gfx950
    def test_filter_by_predicate_keeps_all_matching(self, hhs_library_path):
        """filter_by_predicate keeps solutions that pass both predicates."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem()
        tops = runner.find_top_n(10, prob)
        filtered = runner.filter_by_predicate(tops, prob)
        # Every solution in tops already passed predicates during find_top_n.
        assert len(filtered) == len(tops)


# ---------------------------------------------------------------------------
# TestTopN: find_top_n returns solutions sorted by Formocast (descending).
# ---------------------------------------------------------------------------


class TestTopN:
    @requires_gfx950
    def test_find_top_n_returns_at_most_n(self, hhs_library_path):
        """find_top_n(5, ...) returns at most 5 solutions."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem()
        tops = runner.find_top_n(5, prob)
        assert len(tops) <= 5

    @requires_gfx950
    def test_find_top_n_nonempty(self, hhs_library_path):
        """find_top_n returns at least one solution for a matching problem."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem()
        tops = runner.find_top_n(5, prob)
        assert len(tops) >= 1, "expected at least one solution"

    @requires_gfx950
    def test_find_top_n_sorted_descending(self, hhs_library_path):
        """Solutions returned by find_top_n are sorted by Formocast descending."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem()
        tops = runner.find_top_n(10, prob)
        if len(tops) < 2:
            pytest.skip("fewer than 2 solutions; order cannot be verified")
        gflops = [_rt_mod.formocast_predict(s, prob) for s in tops]
        for i in range(len(gflops) - 1):
            assert gflops[i] >= gflops[i + 1], (
                f"solutions not sorted: gflops[{i}]={gflops[i]:.1f}"
                f" < gflops[{i+1}]={gflops[i+1]:.1f}"
            )


# ---------------------------------------------------------------------------
# TestFormocast: formocast_predict returns a plausible GFLOPS value.
# ---------------------------------------------------------------------------


class TestFormocast:
    @requires_gfx950
    def test_formocast_predict_in_range(self, hhs_library_path):
        """formocast_predict returns a GFLOPS value in [10, 5000]."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem()
        sol = runner.find_best(prob)
        assert sol is not None
        gflops = _rt_mod.formocast_predict(sol, prob)
        assert 10.0 <= gflops <= 5000.0, (
            f"formocast_predict returned {gflops:.1f} GFLOPS, expected [10, 5000]"
        )

    @requires_gfx950
    def test_formocast_predict_larger_problem(self, hhs_library_path):
        """formocast_predict handles larger problem sizes (1024×1024×1024)."""
        if not HAVE_RUNTIME:
            pytest.skip("tensilelite_runtime not installed")
        from Tensile.client.library_runner import LibraryRunner
        runner = LibraryRunner(hhs_library_path)
        prob = _makeHhsProblem(m=1024, n=1024, k=1024)
        sol = runner.find_best(prob)
        assert sol is not None
        gflops = _rt_mod.formocast_predict(sol, prob)
        assert gflops > 0.0, "formocast_predict must return positive GFLOPS"
