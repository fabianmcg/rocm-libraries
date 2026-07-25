# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M6 test suite: grouped GEMM reference (task 6.2), arg builder (task 6.3).

Pure-Python tests (TestGemmGroupedReference, TestBuildGroupedGemmArgs) run
under plain tox -e unit.  GPU workspace tests in TestGemmGroupedWorkspaceSizes
are skipped because no grouped GEMM kernel YAML (groupedGemm: True) was found
in the repository — see fixtures/m6_grouped_notes.txt for details.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from Tensile.client.gemm_args import buildGroupedGemmArgs
from Tensile.client.reference import gemm, gemmGrouped
from .conftest import requires_gfx950


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
# Task 6.3 — buildGroupedGemmArgs (pure Python, no GPU required)
# ===========================================================================


def _makeMinimalGroup(M: int = 64, N: int = 64, K: int = 64) -> dict:
    """Return a minimal group dict for buildGroupedGemmArgs unit tests."""
    sol = {
        "KernArgsVersion": 2,
        "SupportCustomWGM": True,
        "SupportCustomStaggerU": False,
        "SupportUserGSU": False,
        "UseSFC": False,
        "UseUniversalArgs": True,
        "MacroTile0": 64,
        "MacroTile1": 64,
        "WorkGroupMapping": 8,
        "WorkGroupMappingXCC": 0,
        "WorkGroupMappingXCCGroup": 0,
        "StaggerU": 0,
        "StaggerUMapping": 0,
        "_staggerStrideShift": 0,
        "GlobalSplitU": 1,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "StreamK": 0,
        "StreamKAtomic": 0,
        "StridedBatched": True,
        "UseBeta": True,
        "GlobalAccumulation": 0,
        "ExpertSchedulingMode": 0,
        "HighPrecisionAccumulate": True,
        "ComputeDataType": 0,
    }
    pp = {
        "sizes": [M, N, K],
        "ldd": M, "ldc": M, "lda": M, "ldb": N,
        "alpha": 1.0, "beta": 0.0, "gsu": 1,
    }
    tensors = {"D": 0x1000, "C": 0x2000, "A": 0x3000, "B": 0x4000}
    return {"solutionParams": sol, "problemParams": pp, "tensors": tensors}


class TestBuildGroupedGemmArgs:
    """Verify buildGroupedGemmArgs byte layout and workspace layout (task 6.3)."""

    def test_returns_bytes_and_list(self):
        """buildGroupedGemmArgs returns (bytes, list) for a single group."""
        g = _makeMinimalGroup()
        top_level, layout = buildGroupedGemmArgs([g], [128], synchronizerPtr=0)
        assert isinstance(top_level, bytes)
        assert isinstance(layout, list)

    def test_workspace_layout_count_matches_groups(self):
        """Layout list has one entry per group."""
        groups = [_makeMinimalGroup(64, 64, 64), _makeMinimalGroup(128, 128, 128)]
        _, layout = buildGroupedGemmArgs(groups, [128, 128], synchronizerPtr=0)
        assert len(layout) == 2

    def test_workspace_layout_offsets_sequential(self):
        """Offsets advance by workspaceSizes[i] for each subsequent group."""
        sizes = [128, 256, 192]
        groups = [_makeMinimalGroup() for _ in sizes]
        _, layout = buildGroupedGemmArgs(groups, sizes, synchronizerPtr=0)
        assert layout[0][0] == 0
        assert layout[1][0] == 128
        assert layout[2][0] == 128 + 256

    def test_top_level_length_version2(self):
        """Version=2: header(16) + 3 pointers(24) = 40 bytes total."""
        g = _makeMinimalGroup()
        top_level, _ = buildGroupedGemmArgs([g], [64], synchronizerPtr=0)
        # gemm_count(4) + arg0(4) + arg1(4) + numWG(4) + argsPtr(8)
        # + Synchronizer(8) + Workspace(8) = 40.
        assert len(top_level) == 40

    def test_gemm_count_encodes_group_count_and_hbm(self):
        """Low 30 bits = N groups; high 2 bits = 1 (HBM argType)."""
        groups = [_makeMinimalGroup() for _ in range(3)]
        top_level, _ = buildGroupedGemmArgs(groups, [64, 64, 64], synchronizerPtr=0)
        gemmCount = struct.unpack_from("<I", top_level, 0)[0]
        assert (gemmCount & 0x3FFFFFFF) == 3
        assert (gemmCount >> 30) == 1

    def test_synchronizer_ptr_written_at_correct_offset(self):
        """Synchronizer device address is at offset 24 for version=2."""
        g = _makeMinimalGroup()
        sync_ptr = 0xDEADBEEF0000
        top_level, _ = buildGroupedGemmArgs([g], [64], synchronizerPtr=sync_ptr)
        # Header = 16 bytes; argsPtr = 8 bytes; Synchronizer starts at byte 24.
        sync_val = struct.unpack_from("<Q", top_level, 24)[0]
        assert sync_val == sync_ptr

    def test_workspace_slot_equals_args_ptr_plus_total_blob_size(self):
        """Workspace pointer = argsPtr + sum(workspaceSizes)."""
        g = _makeMinimalGroup()
        ws_ptr = 0x5000
        sizes = [128]
        top_level, _ = buildGroupedGemmArgs(
            [g], sizes, synchronizerPtr=0, argsPtr=ws_ptr
        )
        # argsPtr at offset 16, Workspace at offset 32 for version=2.
        args_val = struct.unpack_from("<Q", top_level, 16)[0]
        ws_val = struct.unpack_from("<Q", top_level, 32)[0]
        assert args_val == ws_ptr
        assert ws_val == ws_ptr + sum(sizes)

    def test_mbsk_raises_not_implemented(self):
        """globalAccumulation=3 (MBSK) must raise NotImplementedError."""
        g = _makeMinimalGroup()
        g["solutionParams"]["GlobalAccumulation"] = 3
        with pytest.raises(NotImplementedError, match="MBSK"):
            buildGroupedGemmArgs([g], [64], synchronizerPtr=0)

    def test_empty_groups_raises_value_error(self):
        """Passing an empty groups list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            buildGroupedGemmArgs([], [], synchronizerPtr=0)

    def test_mismatched_lengths_raises_value_error(self):
        """Mismatched groups and workspaceSizes lengths raise ValueError."""
        g = _makeMinimalGroup()
        with pytest.raises(ValueError):
            buildGroupedGemmArgs([g], [64, 64], synchronizerPtr=0)

    def test_four_groups_workspace_layout(self):
        """Four groups produce four workspace layout entries with correct offsets."""
        sizes = [64, 64, 64, 64]
        groups = [_makeMinimalGroup() for _ in sizes]
        _, layout = buildGroupedGemmArgs(groups, sizes, synchronizerPtr=0)
        assert len(layout) == 4
        expected_offsets = [0, 64, 128, 192]
        for i, (off, _blob) in enumerate(layout):
            assert off == expected_offsets[i], f"group {i}: offset {off} != {expected_offsets[i]}"


# ===========================================================================
# Task 6.6 — GPU grouped GEMM workspace test (requires M10 + grouped kernel)
# ===========================================================================


class TestGemmGroupedWorkspaceSizes:
    """GPU workspace-size binding tests (task 6.1).

    grouped_gemm_workspace_size is available in tensilelite_runtime (M10 done),
    but requires a ContractionSolution with groupedGemm=True — the binding
    returns 0 for non-grouped solutions.  No grouped GEMM kernel YAML was found
    in the repository.  See fixtures/m6_grouped_notes.txt for details.
    """

    @requires_gfx950
    def test_placeholder(self):
        pytest.skip(
            "no grouped GEMM kernel YAML (groupedGemm: True) found in repo; "
            "grouped_gemm_workspace_size binding exists but returns 0 for "
            "non-grouped solutions — see fixtures/m6_grouped_notes.txt"
        )
