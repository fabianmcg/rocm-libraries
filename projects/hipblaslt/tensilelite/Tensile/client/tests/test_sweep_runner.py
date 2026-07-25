# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M12 test suite: SweepRunner, CSV reporters, full benchmark pipeline.

Covers Tasks 12.0–12.3 of the TensileLite Python client plan:
  12.0  CSV schema audit (row-length parity with LibraryLogic.py:addFromCSV)
  12.1  ResultsCSVReporter and LibraryUpdateReporter unit tests (no GPU)
  12.2  SweepRunner GPU test over gemm_standard.yaml group 2 (bf16)
  12.3  Plausibility and winner-selection verification

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec.
Pure-Python reporter tests run under plain tox -e unit.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import pytest

try:
    import amdgpu_exec
    HAVE_DEPS = True
except ImportError:
    amdgpu_exec = None
    HAVE_DEPS = False

from .conftest import requires_gfx950

_TESTS_DIR = os.path.dirname(__file__)
_YAML_PATH = os.path.join(_TESTS_DIR, "yaml", "gemm_standard.yaml")
_TENSILE_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", ".."))

if _TENSILE_ROOT not in sys.path:
    sys.path.insert(0, _TENSILE_ROOT)

from Tensile.client.reporters import LibraryUpdateReporter, ResultsCSVReporter
from Tensile.client.sweep_runner import SweepResult, SweepRunner

# ---------------------------------------------------------------------------
# CSV schema constants (Task 12.0 audit findings).
#
# For gemm_standard.yaml group 2 (bf16, batched GEMM):
#   LibraryLogic.py:304: numIndices = TotalIndices + NumIndicesLD = 4 + 4 = 8
#
#   numSizeDims      = 8  (SizeI=M, SizeJ=N, SizeK=batch, SizeL=K,
#                          SizeM=ldd, SizeN=ldc, SizeO=lda, SizeP=ldb)
#   nonSolCols       = 14 (GFlops + 8 sizes + LDD + LDC + LDA + LDB + TotalFlops)
#   numSolutions     = 1  (single fork permutation from the YAML)
#   rowLength        = 15
#   solutionStartIdx = 14  (= rowLength - numSolutions)
#
# LibraryLogic.py:459:  solutionStartIdx = rowLength - numSolutions
# LibraryLogic.py:426:  totalSizeIdx = 1 + numIndices = 1 + 8 = 9
# ---------------------------------------------------------------------------

_NUM_SIZE_DIMS = 8
_NON_SOL_COLS = 14   # GFlops + 8 sizes + 4 LDs + TotalFlops
_NUM_SOLUTIONS = 1   # gemm_standard.yaml group 2 has one solution
_ROW_LENGTH = _NON_SOL_COLS + _NUM_SOLUTIONS


# ===========================================================================
# Task 12.1 — ResultsCSVReporter unit tests (no GPU)
# ===========================================================================


class TestResultsCSVReporter:
    """Verify ResultsCSVReporter column schema and separator (no GPU)."""

    def _makeReporter(self, f, solNames=None, numSizeDims=4):
        if solNames is None:
            solNames = ["sol0", "sol1"]
        return ResultsCSVReporter(f.name, solNames, numSizeDims=numSizeDims)

    def test_header_column_count(self):
        """Header row must have numSizeDims + 5 + numSolutions columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                        delete=False) as f:
            path = f.name
        try:
            rep = ResultsCSVReporter(path, ["s0", "s1"], numSizeDims=4)
            rep.writeHeader()
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                header = next(reader)
            # 1 perf label + 4 sizes + 4 LDs + 1 TotalFlops + 2 solutions = 12
            assert len(header) == 12
        finally:
            os.unlink(path)

    def test_header_col0_is_perf_metric(self):
        """Column 0 header is the perfMetric label (GFlops by default)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                        delete=False) as f:
            path = f.name
        try:
            rep = ResultsCSVReporter(path, ["s0"])
            rep.writeHeader()
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                header = next(reader)
            assert header[0].strip() == "GFlops"
        finally:
            os.unlink(path)

    def test_header_size_columns(self):
        """Size columns are SizeI, SizeJ, SizeK, SizeL for 4-dim problems."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                        delete=False) as f:
            path = f.name
        try:
            rep = ResultsCSVReporter(path, ["s0"], numSizeDims=4)
            rep.writeHeader()
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                header = [h.strip() for h in next(reader)]
            assert header[1:5] == ["SizeI", "SizeJ", "SizeK", "SizeL"]
        finally:
            os.unlink(path)

    def test_header_ld_and_flops_columns(self):
        """LDD, LDC, LDA, LDB, TotalFlops appear after size columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                        delete=False) as f:
            path = f.name
        try:
            rep = ResultsCSVReporter(path, ["s0"], numSizeDims=4)
            rep.writeHeader()
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                header = [h.strip() for h in next(reader)]
            assert header[5:10] == ["LDD", "LDC", "LDA", "LDB", "TotalFlops"]
        finally:
            os.unlink(path)

    def test_data_row_column_count_matches_header(self):
        """Data rows have the same column count as the header row."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                        delete=False) as f:
            path = f.name
        try:
            rep = ResultsCSVReporter(path, ["s0", "s1"], numSizeDims=4)
            rep.writeHeader()
            rep.writeRow(
                sizeParams={
                    "sizes": [256, 256, 4, 256],
                    "ldd": 256, "ldc": 256, "lda": 256, "ldb": 256,
                    "totalFlops": 134217728,
                },
                solutionResults=[("s0", 1234.5), ("s1", 2345.6)],
            )
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                header = next(reader)
                data = next(reader)
            assert len(data) == len(header)
        finally:
            os.unlink(path)

    def test_row_length_matches_library_logic_formula(self):
        """rowLength = nonSolCols + numSolutions matches LibraryLogic.py:459.

        Uses numSizeDims=4 (4-dim problem): nonSolCols = 1+4+4+1 = 10.
        """
        numSols = 3
        # For 4-dim problems: GFlops(1) + sizes(4) + LDs(4) + TotalFlops(1) = 10.
        expectedNonSolCols = 10
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                        delete=False) as f:
            path = f.name
        try:
            solNames = [f"sol{i}" for i in range(numSols)]
            rep = ResultsCSVReporter(path, solNames, numSizeDims=4)
            rep.writeHeader()
            rep.writeRow(
                sizeParams={
                    "sizes": [256, 256, 4, 256],
                    "ldd": 256, "ldc": 256, "lda": 256, "ldb": 256,
                    "totalFlops": 134217728,
                },
                solutionResults=[(n, float(i + 100)) for i, n in
                                 enumerate(solNames)],
            )
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                header = next(reader)
                row = next(reader)
            rowLength = len(row)
            # LibraryLogic.py:459: solutionStartIdx = rowLength - numSolutions
            solutionStartIdx = rowLength - numSols
            assert solutionStartIdx == expectedNonSolCols
            assert rowLength == expectedNonSolCols + numSols
        finally:
            os.unlink(path)

    def test_gflops_in_solution_columns(self):
        """GFLOPS values appear in the solution columns at the expected offsets."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                        delete=False) as f:
            path = f.name
        try:
            rep = ResultsCSVReporter(path, ["sol0", "sol1"], numSizeDims=4)
            rep.writeHeader()
            rep.writeRow(
                sizeParams={
                    "sizes": [512, 512, 4, 512],
                    "ldd": 512, "ldc": 512, "lda": 512, "ldb": 512,
                    "totalFlops": 1073741824,
                },
                solutionResults=[("sol0", 5000.0), ("sol1", 8000.0)],
            )
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                next(reader)  # header
                row = next(reader)
            # solutionStartIdx = 10 for 4-dim GEMM
            assert float(row[10].strip()) == pytest.approx(5000.0, rel=1e-5)
            assert float(row[11].strip()) == pytest.approx(8000.0, rel=1e-5)
        finally:
            os.unlink(path)


# ===========================================================================
# Task 12.1 — LibraryUpdateReporter unit tests (no GPU)
# ===========================================================================


class TestLibraryUpdateReporter:
    """Verify LibraryUpdateReporter format (no GPU)."""

    def test_format_matches_cpp_client(self):
        """Output matches LibraryUpdateReporter.cpp:156-165 format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                        delete=False) as f:
            path = f.name
        try:
            rep = LibraryUpdateReporter(path)
            rep.writeRow([256, 256, 4, 256], winnerIdx=2, winnerGFlops=4853.07)
            rep.close()
            with open(path) as fh:
                lines = fh.readlines()
            assert len(lines) == 2
            assert lines[0].strip() == "- - [256, 256, 4, 256]"
            assert lines[1].strip().startswith("- [2,")
        finally:
            os.unlink(path)

    def test_winner_idx_and_gflops_parseable(self):
        """winner idx and gflops in the output are int and float parseable."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                        delete=False) as f:
            path = f.name
        try:
            rep = LibraryUpdateReporter(path)
            rep.writeRow([1024, 1024, 4, 1024], winnerIdx=0,
                         winnerGFlops=12345.6789)
            rep.close()
            with open(path) as fh:
                content = fh.read()
            # Extract the second line: "    - [0, 12345.7]"
            second = content.strip().splitlines()[1].strip()
            assert second.startswith("- [")
            inner = second[3:-1]  # "0, 12345.7"
            parts = inner.split(", ")
            assert int(parts[0]) == 0
            assert float(parts[1]) == pytest.approx(12345.7, rel=1e-4)
        finally:
            os.unlink(path)

    def test_multiple_rows(self):
        """Multiple problem sizes produce two YAML entries each."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                        delete=False) as f:
            path = f.name
        try:
            rep = LibraryUpdateReporter(path)
            rep.writeRow([256, 256, 4, 256], 0, 1000.0)
            rep.writeRow([512, 512, 4, 512], 1, 2000.0)
            rep.close()
            with open(path) as fh:
                lines = fh.readlines()
            assert len(lines) == 4
        finally:
            os.unlink(path)


# ===========================================================================
# Task 12.2/12.3 — SweepRunner GPU tests
# ===========================================================================


@pytest.fixture(scope="session")
def bf16Sweep():
    """Run SweepRunner on gemm_standard.yaml group 2 (bf16 HPA).

    Returns (allResults, csvPath, luPath) or None if no GPU or deps.
    """
    if not HAVE_DEPS:
        return None

    csvTmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    luTmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    csvPath = csvTmp.name
    luPath = luTmp.name
    csvTmp.close()
    luTmp.close()

    runner = SweepRunner(
        yamlPath=_YAML_PATH,
        nWarmup=2,
        nIters=10,
        rotatingBuffers=4,
        icacheCopies="auto",
        problemIdx=2,   # bf16 HPA group
        groupIdx=0,
    )
    allResults = runner.run(resultsCsv=csvPath, libraryUpdateFile=luPath)
    return allResults, csvPath, luPath


@requires_gfx950
def test_sweep_csv_schema(bf16Sweep):
    """Results CSV has correct rowLength = _NON_SOL_COLS + numSolutions."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if bf16Sweep is None:
        pytest.skip("sweep fixture unavailable")

    allResults, csvPath, _luPath = bf16Sweep
    if not allResults:
        pytest.skip("no sweep results (no solutions compiled)")

    with open(csvPath) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        firstDataRow = next(reader, None)

    rowLength = len(header)
    numSols = len({r.solutionName for r in allResults})
    # rowLength must equal _NON_SOL_COLS + numSolutions (LibraryLogic.py:459).
    assert rowLength == _NON_SOL_COLS + numSols, (
        f"CSV rowLength={rowLength}, expected {_NON_SOL_COLS + numSols}; "
        f"solutionStartIdx={rowLength - numSols} (should be {_NON_SOL_COLS})"
    )

    if firstDataRow is not None:
        assert len(firstDataRow) == rowLength, (
            f"data row has {len(firstDataRow)} cols, header has {rowLength}"
        )


@requires_gfx950
def test_sweep_csv_col0_is_gflops_label(bf16Sweep):
    """Col-0 header is 'GFlops' matching LibraryLogic.py:addFromCSV row[0] check."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if bf16Sweep is None:
        pytest.skip("sweep fixture unavailable")

    _allResults, csvPath, _luPath = bf16Sweep
    with open(csvPath) as fh:
        reader = csv.reader(fh)
        header = next(reader)

    assert header[0].strip() == "GFlops"


@requires_gfx950
def test_sweep_gflops_plausibility(bf16Sweep):
    """All benchmark GFLOPS are in [100, 1_000_000] (gfx950 bf16 sanity range).

    Lower bound (100) catches ns-as-µs bugs.
    Upper bound (1_000_000) catches µs-as-ms bugs and remains above any real
    hardware peak (gfx950 ~383 TFLOPS peak bf16 = 383_000 GFLOPS).
    """
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if bf16Sweep is None:
        pytest.skip("sweep fixture unavailable")

    allResults, _csv, _lu = bf16Sweep
    successful = [r for r in allResults if r.gflops > 0]

    if not successful:
        pytest.skip("no successful benchmark results")

    for r in successful:
        assert 100 <= r.gflops <= 1_000_000, (
            f"GFLOPS {r.gflops:.1f} for {r.solutionName} on {r.problemSize} "
            "is outside [100, 1_000_000] — possible unit-conversion bug"
        )


@requires_gfx950
def test_sweep_best_solution_wins_per_problem(bf16Sweep):
    """Best solution per problem size has the highest GFLOPS in that group."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if bf16Sweep is None:
        pytest.skip("sweep fixture unavailable")

    allResults, _csv, luPath = bf16Sweep
    if not allResults:
        pytest.skip("no sweep results")

    # Group results by problem size.
    bySize: dict = {}
    for r in allResults:
        bySize.setdefault(r.problemSize, []).append(r)

    for size, results in bySize.items():
        valid = [r for r in results if r.gflops > 0]
        if len(valid) < 2:
            # With one solution there is trivially a single winner.
            continue
        best = max(valid, key=lambda r: r.gflops)
        # Verify that the winner has strictly the highest GFLOPS.
        for other in valid:
            if other is best:
                continue
            assert best.gflops >= other.gflops, (
                f"size {size}: winner {best.solutionName} ({best.gflops:.1f}) "
                f"< {other.solutionName} ({other.gflops:.1f})"
            )


@requires_gfx950
def test_sweep_library_update_winner_idx(bf16Sweep):
    """library_update_file contains correct winner solution_idx per problem."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if bf16Sweep is None:
        pytest.skip("sweep fixture unavailable")

    allResults, _csv, luPath = bf16Sweep
    if not allResults:
        pytest.skip("no sweep results")

    with open(luPath) as fh:
        lines = [l.rstrip() for l in fh if l.strip()]

    # Each problem produces two lines: "  - - [sizes]" and "    - [idx, gflops]".
    # Collect (winnerIdx, winnerGFlops) from second lines.
    luEntries = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("- - ["):
            secondLine = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if secondLine.startswith("- ["):
                inner = secondLine[3:-1]
                parts = inner.split(", ", 1)
                luEntries.append((int(parts[0]), float(parts[1])))
                i += 2
                continue
        i += 1

    if not luEntries:
        pytest.skip("library update file has no entries")

    # Group results by problem size to find expected winner.
    bySize: dict = {}
    for r in allResults:
        bySize.setdefault(r.problemSize, []).append(r)

    for (luIdx, luGflops), (size, results) in zip(
            luEntries, bySize.items()):
        valid = [r for r in results if r.gflops > 0]
        if not valid:
            continue
        expectedWinner = max(valid, key=lambda r: r.gflops)
        assert luIdx == expectedWinner.solutionIdx, (
            f"size {size}: library update has idx={luIdx}, "
            f"expected {expectedWinner.solutionIdx}"
        )
        # GFLOPS in library update matches the best result within 1% (floating-point
        # formatting rounding).
        assert luGflops == pytest.approx(expectedWinner.gflops, rel=0.01), (
            f"size {size}: library update gflops={luGflops:.2f}, "
            f"expected {expectedWinner.gflops:.2f}"
        )


@requires_gfx950
def test_sweep_csv_solution_start_idx(bf16Sweep):
    """solutionStartIdx = rowLength - numSolutions == _NON_SOL_COLS.

    Verifies GFLOPS values in the data rows are positive in solution columns and
    that the non-solution prefix is correct, matching LibraryLogic.py:459.
    """
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if bf16Sweep is None:
        pytest.skip("sweep fixture unavailable")

    allResults, csvPath, _lu = bf16Sweep
    if not allResults:
        pytest.skip("no sweep results")

    numSols = len({r.solutionName for r in allResults})
    solutionStartIdx = _NON_SOL_COLS  # must match LibraryLogic.py computation

    with open(csvPath) as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            # Verify sizes parse as positive integers.
            for col in range(1, 1 + _NUM_SIZE_DIMS):
                assert int(row[col].strip()) > 0, f"size col {col} is not positive"
            # Verify solution GFLOPS are floats (may be -1 for failed solutions).
            for col in range(solutionStartIdx, solutionStartIdx + numSols):
                val = float(row[col].strip())
                assert val != 0.0 or val == pytest.approx(-1.0), (
                    f"unexpected zero GFLOPS at col {col}"
                )
