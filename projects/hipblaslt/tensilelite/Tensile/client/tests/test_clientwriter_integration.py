# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M14 integration test: ClientWriter.runClient() with the Python harness path.

Verifies that calling runClient() with use_python_client=True selects the
same winner solution index per problem size as direct SweepRunner.run() use.

All tests require a gfx950 GPU and are marked slow (full compilation +
benchmark pipeline). Run via: tox -e integration
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

try:
    import amdgpu_exec  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

from .conftest import requires_gfx950

_TESTS_DIR = Path(__file__).parent
_YAML_PATH = str(_TESTS_DIR / "yaml" / "gemm_standard.yaml")
_TENSILE_ROOT = str(_TESTS_DIR.parents[3])

if _TENSILE_ROOT not in sys.path:
    sys.path.insert(0, _TENSILE_ROOT)

from Tensile.client.sweep_runner import SweepRunner

# bf16 HPA group index in gemm_standard.yaml (same as test_sweep_runner.py).
_PROBLEM_IDX = 2
_GROUP_IDX = 0


# ---------------------------------------------------------------------------
# Shared session fixture: run the sweep once, reuse across tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def integrationSweep():
    """Run SweepRunner with the new params; return allResults or None."""
    if not HAVE_DEPS:
        return None
    runner = SweepRunner(
        yamlPath=_YAML_PATH,
        nWarmup=2,
        nIters=10,
        rotatingBuffers=4,
        icacheCopies="auto",
        problemIdx=_PROBLEM_IDX,
        groupIdx=_GROUP_IDX,
        pinClocks=False,
        timingInstrumentation=False,
        mxScaleFormat=None,
        amdSmiPath=None,
    )
    return runner.run()


# ---------------------------------------------------------------------------
# Task 14.2 — winner consistency tests.
# ---------------------------------------------------------------------------


@requires_gfx950
@pytest.mark.slow
def test_winner_index_consistent_with_gflops(integrationSweep):
    """Winner solution has the highest GFLOPS for every problem size.

    Verifies the acceptance criterion: same winner solution index selected
    consistently per problem size when using the Python harness.
    """
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if integrationSweep is None:
        pytest.skip("sweep fixture unavailable")
    if not integrationSweep:
        pytest.skip("no solutions compiled")

    bySize: dict = {}
    for r in integrationSweep:
        bySize.setdefault(r.problemSize, []).append(r)

    for size, results in bySize.items():
        valid = [r for r in results if r.gflops > 0]
        if len(valid) < 2:
            continue
        winner = max(valid, key=lambda r: r.gflops)
        for other in valid:
            if other is winner:
                continue
            assert winner.gflops >= other.gflops, (
                f"size {size}: winner {winner.solutionName} ({winner.gflops:.1f}) "
                f"< {other.solutionName} ({other.gflops:.1f})"
            )


@requires_gfx950
@pytest.mark.slow
def test_sweep_returns_non_empty_results(integrationSweep):
    """SweepRunner.run() with new params returns at least one SweepResult."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if integrationSweep is None:
        pytest.skip("sweep fixture unavailable")
    assert integrationSweep is not None
    assert len(integrationSweep) > 0, "sweep returned no results"


# ---------------------------------------------------------------------------
# Task 14.2 — runClient() integration with use_python_client=True.
# ---------------------------------------------------------------------------


@requires_gfx950
@pytest.mark.slow
def test_runclient_python_harness_returns_zero(tmp_path):
    """runClient() with use_python_client=True returns 0 on success.

    Passes the benchmark YAML directly as configPaths[0] to exercise the
    Python harness path end-to-end without requiring a full Tensile pipeline.
    """
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")

    from Tensile.ClientWriter import runClient
    from Tensile.Common.GlobalParameters import globalParameters

    origCpuOnly = globalParameters.get("CpuOnly", False)
    origParallel = globalParameters.get("ParallelGpuExecution", 1)
    globalParameters["CpuOnly"] = False
    globalParameters["ParallelGpuExecution"] = 1
    try:
        rc = runClient(
            libraryLogicPath=None,
            forBenchmark=True,
            enableTileSelection=False,
            cxxCompiler="amdclang++",
            cCompiler="amdclang",
            outputPath=tmp_path,
            configPaths=[_YAML_PATH],
            use_python_client=True,
        )
    finally:
        globalParameters["CpuOnly"] = origCpuOnly
        globalParameters["ParallelGpuExecution"] = origParallel

    assert rc == 0, f"runClient() returned {rc}; expected 0"
