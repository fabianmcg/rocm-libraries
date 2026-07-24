# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for epilogue_harness.yaml_solution_builder.enumerateAllSolutions.

Tests use in-memory YAML dicts written to temporary files so that rocisa is
not required (yaml.safe_load is used, not LibraryIO.readYAML).
"""

import os
import textwrap

import pytest
import yaml

from epilogues.epilogue_harness.yaml_solution_builder import (
    _CHIP_TO_KERNS_ARGS_VERSION,
    _injectInternalArgsSupport,
    _iterRawSolutions,
    enumerateAllSolutions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPILOGUES_YAML_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "epilogues", "yaml",
)


def _writeYaml(tmp_path, content: str) -> str:
    """Write a YAML string to a temp file and return its path."""
    path = str(tmp_path / "test.yaml")
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))
    return path


def _minimalYaml(solutions: list[dict]) -> dict:
    """Build the minimal BenchmarkProblems YAML dict with solutions in SSE."""
    return {
        "BenchmarkProblems": [
            [
                {"OperationType": "GEMM"},
                {
                    "BenchmarkCommonParameters": [],
                    "ForkParameters": [],
                    "BenchmarkFinalParameters": [
                        {"SolutionSummationExpansion": solutions}
                    ],
                },
            ]
        ]
    }


# ---------------------------------------------------------------------------
# _iterRawSolutions tests
# ---------------------------------------------------------------------------


def test_iterRawSolutions_empty_yaml(tmp_path):
    """YAML with no SolutionSummationExpansion yields an empty list."""
    path = _writeYaml(
        tmp_path,
        """
        BenchmarkProblems:
          - - OperationType: GEMM
            - BenchmarkFinalParameters:
                - ProblemSizes:
                    - Exact: [256, 256, 1, 256]
        """,
    )
    assert _iterRawSolutions(path) == []


def test_iterRawSolutions_list_yaml(tmp_path):
    """A non-dict (list-format) YAML returns an empty list without crashing."""
    path = str(tmp_path / "list.yaml")
    with open(path, "w") as f:
        yaml.dump([{"MinimumRequiredVersion": "5.0.0"}, "gfx950"], f)
    assert _iterRawSolutions(path) == []


def test_iterRawSolutions_single_solution(tmp_path):
    sol = {"MacroTile0": 64, "MacroTile1": 64, "KernArgsVersion": 2}
    data = _minimalYaml([sol])
    path = str(tmp_path / "single.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    result = _iterRawSolutions(path)
    assert len(result) == 1
    g, s, d = result[0]
    assert g == 0
    assert s == 0
    assert d["MacroTile0"] == 64


def test_iterRawSolutions_multiple_solutions(tmp_path):
    sols = [
        {"MacroTile0": 64, "MacroTile1": 64},
        {"MacroTile0": 128, "MacroTile1": 128},
    ]
    data = _minimalYaml(sols)
    path = str(tmp_path / "multi.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    result = _iterRawSolutions(path)
    assert len(result) == 2
    assert result[0][1] == 0
    assert result[1][1] == 1
    assert result[0][0] == result[1][0] == 0


def test_iterRawSolutions_multiple_groups(tmp_path):
    """Two BenchmarkFinalParameters groups with solutions get distinct group_idx."""
    data = {
        "BenchmarkProblems": [
            [
                {"OperationType": "GEMM"},
                {
                    "BenchmarkFinalParameters": [
                        {"SolutionSummationExpansion": [{"MT": 64}]},
                        {"SolutionSummationExpansion": [{"MT": 128}]},
                    ]
                },
            ]
        ]
    }
    path = str(tmp_path / "two_groups.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    result = _iterRawSolutions(path)
    assert len(result) == 2
    assert result[0][0] == 0
    assert result[1][0] == 1


# ---------------------------------------------------------------------------
# _injectInternalArgsSupport tests
# ---------------------------------------------------------------------------


def test_inject_reads_isp_block():
    sol = {
        "MacroTile0": 64,
        "InternalSupportParams": {
            "KernArgsVersion": 2,
            "SupportCustomWGM": True,
            "SupportCustomStaggerU": False,
            "SupportUserGSU": True,
            "UseSFC": False,
            "UseUniversalArgs": True,
        },
    }
    result = _injectInternalArgsSupport(sol, chip="gfx942")
    assert result["KernArgsVersion"] == 2
    assert result["SupportCustomWGM"] is True
    assert result["SupportUserGSU"] is True
    assert result["UseUniversalArgs"] is True
    # Original dict must not be mutated.
    assert "KernArgsVersion" not in sol


def test_inject_fallback_to_chip_table():
    sol = {"MacroTile0": 64}
    result = _injectInternalArgsSupport(sol, chip="gfx942")
    assert result["KernArgsVersion"] == _CHIP_TO_KERNS_ARGS_VERSION["gfx942"]
    assert result["SupportCustomWGM"] is False
    assert result["UseUniversalArgs"] is True


def test_inject_unknown_chip_raises():
    sol = {"MacroTile0": 64}
    with pytest.raises(NotImplementedError, match="unsupported chip"):
        _injectInternalArgsSupport(sol, chip="gfx9999")


def test_inject_version2_bitfield_keys():
    sol = {
        "GlobalSplitUCoalesced": True,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "InternalSupportParams": {"KernArgsVersion": 2},
    }
    result = _injectInternalArgsSupport(sol, chip="gfx942")
    assert result["GlobalSplitUCoalesced"] is True
    assert result["GlobalSplitUWorkGroupMappingRoundRobin"] is False


# ---------------------------------------------------------------------------
# enumerateAllSolutions tests
# ---------------------------------------------------------------------------


def test_enumerate_empty_yaml(tmp_path):
    """Input-spec YAMLs with no solutions return an empty list."""
    path = _writeYaml(
        tmp_path,
        """
        BenchmarkProblems:
          - - OperationType: GEMM
            - BenchmarkFinalParameters:
                - ProblemSizes:
                    - Exact: [256, 256, 1, 256]
        """,
    )
    assert enumerateAllSolutions(path, chip="gfx942") == []


def test_enumerate_single_solution(tmp_path):
    sol = {
        "MacroTile0": 64,
        "MacroTile1": 64,
        "InternalSupportParams": {
            "KernArgsVersion": 2,
            "SupportCustomWGM": False,
            "SupportCustomStaggerU": False,
            "SupportUserGSU": False,
            "UseSFC": False,
            "UseUniversalArgs": True,
        },
    }
    data = _minimalYaml([sol])
    path = str(tmp_path / "enum_single.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    result = enumerateAllSolutions(path, chip="gfx942")
    assert len(result) == 1
    g, s, d = result[0]
    assert g == 0
    assert s == 0
    assert d["KernArgsVersion"] == 2
    assert d["MacroTile0"] == 64


def test_enumerate_augments_with_chip_fallback(tmp_path):
    """When InternalSupportParams is absent, the chip table fills KernArgsVersion."""
    sol = {"MacroTile0": 128, "MacroTile1": 128}
    data = _minimalYaml([sol])
    path = str(tmp_path / "fallback.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    result = enumerateAllSolutions(path, chip="gfx942")
    assert result[0][2]["KernArgsVersion"] == _CHIP_TO_KERNS_ARGS_VERSION["gfx942"]


# ---------------------------------------------------------------------------
# Verify existing epilogue YAML files do not crash (return empty).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_name",
    [
        "gemm_partial_rms_k1.yaml",
        "gemm_partial_rms_k1_rowmajor.yaml",
    ],
)
def test_enumerate_existing_epilogue_yaml_empty(yaml_name):
    """Calling enumerateAllSolutions on epilogue input-spec YAMLs returns []."""
    path = os.path.join(_EPILOGUES_YAML_DIR, yaml_name)
    if not os.path.exists(path):
        pytest.skip(f"{yaml_name} not found")
    result = enumerateAllSolutions(path, chip="gfx942")
    # Input-spec YAMLs have no SolutionSummationExpansion.
    assert result == []
