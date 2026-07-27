# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for SweepRunner in-sweep correctness verification (numElementsToValidate).

No-GPU tests (TestSelectReference, TestValidationCsvColumn, TestAggregateValidation)
run under plain tox -e unit.  GPU tests require gfx950 and amdgpu_exec.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np
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

from Tensile.client import reference
from Tensile.client.reporters import ResultsCSVReporter
from Tensile.client.sweep_runner import SweepResult, SweepRunner, _selectReference


# ===========================================================================
# TestSelectReference — no GPU required
# ===========================================================================


class TestSelectReference:
    """Verify _selectReference returns correct (npDtype, refFn, rtol, atol) tuples."""

    def test_fp32_returns_float32_and_gemm(self):
        """fp32 input/output (DataType=0) selects numpy float32 and reference.gemm."""
        solDict = {"DataType": 0, "DestDataType": 0, "StreamK": 0}
        result = _selectReference(solDict)
        assert result is not None
        npDtype, refFn, _rtol, _atol = result
        assert npDtype == np.float32
        assert refFn is reference.gemm

    def test_fp16_returns_float16_and_gemmFp16(self):
        """fp16 input/output (DataType=4) selects numpy float16 and reference.gemmFp16."""
        solDict = {"DataType": 4, "DestDataType": 4, "StreamK": 0}
        result = _selectReference(solDict)
        assert result is not None
        npDtype, refFn, _rtol, _atol = result
        assert npDtype == np.float16
        assert refFn is reference.gemmFp16

    def test_unsupported_dtype_returns_none(self):
        """Int8 (DataType=8) is not supported; result is None."""
        solDict = {"DataType": 8, "DestDataType": 8, "StreamK": 0}
        assert _selectReference(solDict) is None

    def test_mismatched_inout_dtype_returns_none(self):
        """Mismatched input/output dtypes are not verified; result is None."""
        solDict = {"DataType": 7, "DestDataType": 0, "StreamK": 0}
        assert _selectReference(solDict) is None

    def test_streamk_nonzero_returns_none(self):
        """StreamK!=0 solutions are not verified; result is None."""
        solDict = {"DataType": 0, "DestDataType": 0, "StreamK": 4}
        assert _selectReference(solDict) is None

    def test_bf16_returns_bfloat16_and_gemmBf16(self):
        """bf16 input/output (DataType=7) selects ml_dtypes.bfloat16 and reference.gemmBf16."""
        ml_dtypes = pytest.importorskip("ml_dtypes")
        from Tensile.client.reference import gemmBf16, RTOL_BF16, ATOL_BF16
        result = _selectReference({"DataType": 7, "DestDataType": 7, "StreamK": 0})
        assert result is not None
        npDtype, refFn, rtol, atol = result
        assert npDtype == ml_dtypes.bfloat16
        assert refFn is gemmBf16
        assert rtol == RTOL_BF16 and atol == ATOL_BF16


# ===========================================================================
# TestValidationCsvColumn — no GPU required
# ===========================================================================


class TestValidationCsvColumn:
    """Verify ResultsCSVReporter places the Validation column correctly."""

    def _writeAndRead(self, includeValidation, numSols=2, validation="PASS"):
        """Write a header + one data row; return (header, dataRow) as stripped lists."""
        solNames = [f"sol{i}" for i in range(numSols)]
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = f.name
            rep = ResultsCSVReporter(path, solNames, numSizeDims=4,
                                     includeValidation=includeValidation)
            rep.writeHeader()
            rep.writeRow(
                sizeParams={
                    "sizes": [256, 256, 4, 256],
                    "ldd": 256, "ldc": 256, "lda": 256, "ldb": 256,
                    "totalFlops": 134217728,
                },
                solutionResults=[(n, 1000.0 + i) for i, n in enumerate(solNames)],
                validation=validation,
            )
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                header = [h.strip() for h in next(reader)]
                dataRow = [c.strip() for c in next(reader)]
        finally:
            if path is not None:
                os.unlink(path)
        return header, dataRow

    def test_validation_column_present_when_enabled(self):
        """Header contains exactly one Validation column when includeValidation=True."""
        header, _ = self._writeAndRead(includeValidation=True)
        assert header.count("Validation") == 1

    def test_validation_column_after_totalflops_before_solutions(self):
        """Validation column is immediately after TotalFlops and before solution columns."""
        header, _ = self._writeAndRead(includeValidation=True, numSols=2)
        totalFlopsIdx = header.index("TotalFlops")
        validationIdx = header.index("Validation")
        # Validation must be right after TotalFlops.
        assert validationIdx == totalFlopsIdx + 1
        # sol0 must be right after Validation.
        assert header[validationIdx + 1] == "sol0"

    def test_validation_data_value_in_correct_column(self):
        """Data row has PASS at the Validation column index."""
        header, dataRow = self._writeAndRead(includeValidation=True, validation="PASS")
        validationIdx = header.index("Validation")
        assert dataRow[validationIdx] == "PASS"

    def test_library_logic_invariant_with_validation(self):
        """solutionStartIdx = rowLength - numSolutions holds with Validation column."""
        numSols = 2
        header, dataRow = self._writeAndRead(includeValidation=True, numSols=numSols)
        rowLength = len(header)
        solutionStartIdx = rowLength - numSols
        # Validation column sits before solutions, so solutionStartIdx must point to sol0.
        assert header[solutionStartIdx] == "sol0"
        # The data row length must match the header length.
        assert len(dataRow) == rowLength

    def test_no_validation_column_when_disabled(self):
        """Validation column absent when includeValidation=False (default)."""
        header, _ = self._writeAndRead(includeValidation=False)
        assert "Validation" not in header

    def test_backward_compat_default_no_validation(self):
        """Default ResultsCSVReporter (no includeValidation arg) has no Validation column."""
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = f.name
            rep = ResultsCSVReporter(path, ["s0", "s1"], numSizeDims=4)
            rep.writeHeader()
            rep.close()
            with open(path) as fh:
                reader = csv.reader(fh)
                header = [h.strip() for h in next(reader)]
        finally:
            if path is not None:
                os.unlink(path)
        assert "Validation" not in header


# ===========================================================================
# TestAggregateValidation — no GPU required
# ===========================================================================


def _makeResult(validation: str) -> SweepResult:
    """Build a minimal SweepResult with the given validation status."""
    from Tensile.client.harness import BenchmarkResult
    return SweepResult(
        solutionIdx=0,
        solutionName="sol",
        problemSize=(64, 64, 1, 64),
        benchmark=BenchmarkResult(timesNs=[], warmupN=0),
        gflops=1000.0,
        validation=validation,
    )


class TestAggregateValidation:
    """Verify _aggregateValidation logic without GPU."""

    def _makeRunner(self, numElementsToValidate):
        """Construct a SweepRunner that only parses the YAML path (no GPU)."""
        return SweepRunner(yamlPath=_YAML_PATH,
                           numElementsToValidate=numElementsToValidate)

    def test_all_pass_returns_pass(self):
        """All-PASS results aggregate to PASS."""
        runner = self._makeRunner(-1)
        results = [_makeResult("PASS"), _makeResult("PASS")]
        assert runner._aggregateValidation(results) == "PASS"

    def test_fail_among_pass_returns_fail(self):
        """First FAIL is returned when any solution fails."""
        runner = self._makeRunner(-1)
        failMsg = "FAIL:mismatch at idx=0"
        results = [_makeResult("PASS"), _makeResult(failMsg), _makeResult("PASS")]
        assert runner._aggregateValidation(results) == failMsg

    def test_all_skipped_returns_skipped(self):
        """All-SKIPPED results aggregate to SKIPPED."""
        runner = self._makeRunner(-1)
        results = [_makeResult("SKIPPED"), _makeResult("SKIPPED")]
        assert runner._aggregateValidation(results) == "SKIPPED"

    def test_validation_disabled_always_skipped(self):
        """When numElementsToValidate=0, aggregate is always SKIPPED."""
        runner = self._makeRunner(0)
        results = [_makeResult("PASS"), _makeResult("FAIL:whatever")]
        assert runner._aggregateValidation(results) == "SKIPPED"


# ===========================================================================
# GPU tests — require gfx950
# ===========================================================================


@requires_gfx950
def test_sweep_validate_all_pass():
    """SweepRunner with numElementsToValidate=-1 yields PASS for all fp32 results."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")

    csvPath = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csvPath = f.name
        runner = SweepRunner(
            yamlPath=_YAML_PATH,
            problemIdx=0,
            groupIdx=0,
            nWarmup=1,
            nIters=2,
            rotatingBuffers=2,
            numElementsToValidate=-1,
        )
        allResults = runner.run(resultsCsv=csvPath)
    finally:
        if csvPath is not None:
            os.unlink(csvPath)

    if not allResults:
        pytest.skip("no results compiled")

    successful = [r for r in allResults if r.gflops > 0]
    if not successful:
        pytest.skip("no successful benchmark results")

    for r in successful:
        assert r.validation == "PASS", (
            f"{r.solutionName} on {r.problemSize}: validation={r.validation}"
        )

    # Verify the CSV contains the Validation column and all rows say PASS.
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csvPath2 = f.name
    try:
        runner2 = SweepRunner(
            yamlPath=_YAML_PATH,
            problemIdx=0,
            groupIdx=0,
            nWarmup=1,
            nIters=2,
            rotatingBuffers=2,
            numElementsToValidate=-1,
        )
        runner2.run(resultsCsv=csvPath2)
        with open(csvPath2) as fh:
            reader = csv.reader(fh)
            header = [h.strip() for h in next(reader)]
            assert "Validation" in header, "CSV header missing Validation column"
            validationIdx = header.index("Validation")
            for row in reader:
                if not row:
                    continue
                assert row[validationIdx].strip() == "PASS", (
                    f"data row has Validation={row[validationIdx]!r}"
                )
    finally:
        os.unlink(csvPath2)


@requires_gfx950
def test_sweep_validate_sampled_pass():
    """SweepRunner with numElementsToValidate=16 yields PASS for successful fp32 results."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")

    csvPath = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csvPath = f.name
        runner = SweepRunner(
            yamlPath=_YAML_PATH,
            problemIdx=0,
            groupIdx=0,
            nWarmup=1,
            nIters=2,
            rotatingBuffers=2,
            numElementsToValidate=16,
        )
        allResults = runner.run(resultsCsv=csvPath)
    finally:
        if csvPath is not None:
            os.unlink(csvPath)

    if not allResults:
        pytest.skip("no results compiled")

    successful = [r for r in allResults if r.gflops > 0]
    if not successful:
        pytest.skip("no successful benchmark results")

    for r in successful:
        assert r.validation == "PASS", (
            f"{r.solutionName} on {r.problemSize}: validation={r.validation}"
        )
