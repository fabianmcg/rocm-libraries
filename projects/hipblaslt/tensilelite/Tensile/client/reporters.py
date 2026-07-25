# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CSV reporters matching the C++ client's ResultFileReporter and LibraryUpdateReporter.

CSV schema audit (Task 12.0)
----------------------------
Source: LibraryLogic.py:413-531 (addFromCSV) and ResultFileReporter.cpp.

ResultFileReporter.cpp builds its column order via CSVStackFile.setHeaderForKey()
in insertion order (CSVStackFile.cpp tracks m_keyOrder on first call per key).

LibraryLogic.py:304 sets:
  numIndices = problemType["TotalIndices"] + problemType["NumIndicesLD"]

For a standard batched GEMM (NT, Batched=True):
  TotalIndices = 4  (M, N, batch, K)
  NumIndicesLD = 4  (ldd, ldc, lda, ldb)
  numIndices   = 8

The problem sizes vector passed to reportValue_sizes has numIndices=8 elements,
yielding 8 SizeX columns (SizeI through SizeP).  The leading-dimension columns
(LDD, LDC, LDA, LDB) are then written separately (as stated by the C++ comment
"Values for these come separately").  Column layout:

  Col 0:    "GFlops"      (key: ProblemIndex; header = "GFlops" for DeviceEfficiency)
  Col 1:    "SizeI"       = M
  Col 2:    "SizeJ"       = N
  Col 3:    "SizeK"       = batch
  Col 4:    "SizeL"       = K
  Col 5:    "SizeM"       = ldd
  Col 6:    "SizeN"       = ldc
  Col 7:    "SizeO"       = lda
  Col 8:    "SizeP"       = ldb
  Col 9:    "LDD"         (leading dimension of D, duplicate of SizeM)
  Col 10:   "LDC"         (leading dimension of C, duplicate of SizeN)
  Col 11:   "LDA"         (leading dimension of A, duplicate of SizeO)
  Col 12:   "LDB"         (leading dimension of B, duplicate of SizeP)
  Col 13:   "TotalFlops"  (= 2*M*N*K*batch)
  Col 14+:  one column per solution, header = solution name, value = GFLOPS

addFromCSV (LibraryLogic.py:458-459) computes:
  rowLength        = len(row)             (from header row)
  solutionStartIdx = rowLength - numSolutions

For an 8-dim GEMM (numSizeDims=8) with N solutions, --csv-export-extra-cols false:
  rowLength = 14 + N, solutionStartIdx = 14.

Size tuple order returned by problemSizesFromYaml:
  (M, N, batch, K, ldd, ldc, lda, ldb)  — verified for NT batched GEMM.

Column separator: ", " (comma followed by space), matching CSVStackFile default.
GFLOPS formatted with 6 significant digits (matching std::setprecision(6)).

LibraryUpdateReporter.cpp:156-165 writes:
  - - [M, N, batch, K, ldd, ldc, lda, ldb]
    - [winnerIdx, winnerGFlops]
"""

from __future__ import annotations


def _formatFloat(v: float) -> str:
    """Format a float with 6 significant digits, matching std::setprecision(6)."""
    return f"{v:.6g}"


class ResultsCSVReporter:
    """Write results.csv matching ResultFileReporter.cpp column schema.

    Column order (--csv-export-extra-cols false, 4-dim GEMM):
      GFlops, SizeI, SizeJ, SizeK, SizeL, LDD, LDC, LDA, LDB, TotalFlops,
      <sol0_name>, <sol1_name>, ...

    The separator is ", " (comma-space) as in CSVStackFile.
    GFLOPS values are formatted with 6 significant digits.
    """

    def __init__(self, path: str, solutionNames: list,
                 numSizeDims: int = 4, perfMetric: str = "GFlops") -> None:
        self._path = path
        self._solutionNames = list(solutionNames)
        self._numSizeDims = numSizeDims
        self._perfMetric = perfMetric
        self._file = None
        self._probIdx = 0

    def _sizeHeaders(self) -> list:
        return [f"Size{chr(ord('I') + i)}" for i in range(self._numSizeDims)]

    def writeHeader(self) -> None:
        """Write the header row to the CSV file."""
        self._file = open(self._path, "w", newline="")
        headers = (
            [self._perfMetric]
            + self._sizeHeaders()
            + ["LDD", "LDC", "LDA", "LDB", "TotalFlops"]
            + self._solutionNames
        )
        self._file.write(", ".join(headers) + "\n")
        self._file.flush()

    def writeRow(self, sizeParams: dict, solutionResults: list) -> None:
        """Write one benchmark data row.

        sizeParams: dict with keys 'sizes' (sequence), 'ldd', 'ldc', 'lda',
                    'ldb', 'totalFlops'.
        solutionResults: list of (solution_name, gflops) pairs in header order.
        """
        sizes = sizeParams["sizes"]
        ldd = sizeParams["ldd"]
        ldc = sizeParams["ldc"]
        lda = sizeParams["lda"]
        ldb = sizeParams["ldb"]
        totalFlops = sizeParams["totalFlops"]

        gflopsMap = {name: gf for name, gf in solutionResults}
        gflopsValues = [
            _formatFloat(gflopsMap.get(name, -1.0))
            for name in self._solutionNames
        ]

        row = (
            [str(self._probIdx)]
            + [str(s) for s in sizes]
            + [str(ldd), str(ldc), str(lda), str(ldb), str(totalFlops)]
            + gflopsValues
        )
        self._file.write(", ".join(row) + "\n")
        self._file.flush()
        self._probIdx += 1

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None


class LibraryUpdateReporter:
    """Write the library-update YAML format from LibraryUpdateReporter.cpp:156-165.

    Output format per problem (matching C++ exactly):
      - - [M, N, batch, K]
        - [winnerIdx, winnerGFlops]
    """

    def __init__(self, path: str) -> None:
        self._file = open(path, "w")

    def writeRow(self, sizeParams: list, winnerIdx: int,
                 winnerGFlops: float) -> None:
        """Write winner info for one problem size."""
        sizeStr = ", ".join(str(s) for s in sizeParams)
        gfStr = _formatFloat(winnerGFlops)
        self._file.write(f"  - - [{sizeStr}]\n")
        self._file.write(f"    - [{winnerIdx}, {gfStr}]\n")
        self._file.flush()

    def close(self) -> None:
        """Close the output file."""
        self._file.close()
