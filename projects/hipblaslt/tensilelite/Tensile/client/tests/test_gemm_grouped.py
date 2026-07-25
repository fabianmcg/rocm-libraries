# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M6 test suite: grouped GEMM reference (task 6.2) and workspace stubs.

Pure-Python tests (TestGemmGroupedReference) run under plain tox -e unit.
GPU and workspace tests require M10 bindings (not yet available).
"""

from __future__ import annotations

import numpy as np
import pytest

from Tensile.client.reference import gemm, gemmGrouped


# ===========================================================================
# Task 6.2 — Grouped GEMM reference (no GPU required)
# ===========================================================================


class TestGemmGroupedReference:
    """Verify gemmGrouped computes each group independently via gemm()."""

    def _group(self, M: int, N: int, K: int, alpha: float = 1.0,
                beta: float = 0.0, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        C = rng.standard_normal((M, N)).astype(np.float32)
        return {"A": A, "B": B, "alpha": alpha, "beta": beta, "C": C}

    def test_two_groups_match_individual(self):
        g0 = self._group(64, 64, 64, seed=0)
        g1 = self._group(32, 128, 64, alpha=2.0, beta=0.5, seed=1)
        results = gemmGrouped([g0, g1])
        assert len(results) == 2
        ref0 = gemm(g0["A"], g0["B"], g0["alpha"], g0["beta"], g0["C"])
        ref1 = gemm(g1["A"], g1["B"], g1["alpha"], g1["beta"], g1["C"])
        np.testing.assert_array_equal(results[0], ref0)
        np.testing.assert_array_equal(results[1], ref1)

    def test_four_groups_shapes_and_values(self):
        shapes = [(64, 64, 64), (128, 32, 64), (32, 128, 32), (16, 16, 16)]
        groups = [self._group(M, N, K, seed=i) for i, (M, N, K) in enumerate(shapes)]
        results = gemmGrouped(groups)
        assert len(results) == 4
        for i, g in enumerate(groups):
            ref = gemm(g["A"], g["B"], g["alpha"], g["beta"], g["C"])
            np.testing.assert_array_equal(results[i], ref,
                                          err_msg=f"group {i} mismatch")

    def test_eight_groups_independent(self):
        """Verify that groups do not share state — results are independent."""
        groups = [self._group(32, 32, 32, seed=i) for i in range(8)]
        results = gemmGrouped(groups)
        assert len(results) == 8
        for i, g in enumerate(groups):
            ref = gemm(g["A"], g["B"], g.get("alpha", 1.0),
                       g.get("beta", 0.0), g.get("C"))
            np.testing.assert_array_equal(results[i], ref,
                                          err_msg=f"group {i} mismatch")

    def test_returns_list(self):
        g = self._group(16, 16, 16)
        result = gemmGrouped([g])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].dtype == np.float64

    def test_no_c_group(self):
        """A group without 'C' is treated as beta=0 (C is None)."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4)).astype(np.float32)
        B = rng.standard_normal((4, 4)).astype(np.float32)
        g = {"A": A, "B": B}
        results = gemmGrouped([g])
        ref = gemm(A, B, alpha=1.0, beta=0.0, C=None)
        np.testing.assert_array_equal(results[0], ref)

    def test_poison_missing_key_raises(self):
        """A group missing the 'A' key must propagate a KeyError."""
        g = {"B": np.zeros((4, 4), dtype=np.float32)}
        with pytest.raises((KeyError, TypeError)):
            gemmGrouped([g])

    def test_poison_shape_mismatch_raises(self):
        """Incompatible A/B inner dimensions must propagate an exception."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((4, 3)).astype(np.float32)
        B = rng.standard_normal((5, 4)).astype(np.float32)  # inner dim 5 != 3.
        with pytest.raises((ValueError, Exception)):
            gemmGrouped([{"A": A, "B": B}])


# ===========================================================================
# Task 6.6 (stub) — GPU grouped GEMM and workspace binding (requires M10)
# ===========================================================================


class TestGemmGroupedWorkspaceSizes:
    """Workspace-size binding tests deferred to M6-full (requires M10 bindings).

    M10 exposes ContractionSolution and ContractionProblemGemm as Python types,
    which are needed to call grouped_gemm_workspace_size.
    """

    def test_placeholder(self):
        pytest.skip("requires M10 workspace binding (grouped_gemm_workspace_size)")
