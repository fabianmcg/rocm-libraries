> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 12 — CSV Output, SweepRunner, Full Benchmark Pipeline

**Executed by:** fresh implementor agent (after M8, M9, M10, M11)
**Reviewed before:** Milestone 13 begins

### Goal

Complete the benchmarking pipeline: CSV output matching the exact schema consumed by
`BenchmarkProblems.py` and `LibraryLogic.py`, GFLOPS-based solution ranking, and a full
sweep runner.

### Tasks

**12.0 — CSV schema audit (prerequisite)**
Read `Tensile/LibraryLogic.py` to find the exact lines that parse `results.csv`. The parser
at `LibraryLogic.py:459` uses positional arithmetic:
`solutionStartIdx = rowLength - numSolutions` — it never matches by column name. An
off-by-one in the number of size columns shifts every solution's GFLOPS to the wrong index,
silently corrupting library logic output.

The verification target for M12 is therefore **`rowLength` parity**: run the Python
`ResultsCSVReporter` and the C++ `ResultFileReporter` on the same YAML and verify that the
number of columns per data row is identical. Column-name matching alone is insufficient.

**Important:** the C++ reference run must use `--csv-export-extra-cols false` (default).
`ResultFileReporter` emits additional columns (`WinnerGFlops`, `WinnerIdx`) when
`--csv-export-extra-cols true` is set (`ResultFileReporter.cpp:43,50`). Comparing against a
run that includes extra columns would show a spurious mismatch even when the core schema is
correct.

**12.1 — `Tensile/client/reporters.py`**
- `ResultsCSVReporter(path)`: writes `results.csv` in the exact schema identified in 12.0.
  Matches `ResultFileReporter`.
- `LibraryUpdateReporter(path)`: writes the `[sizes] → [solution_idx, gflops]` format.
  Matches `LibraryUpdateReporter.cpp:156–165`.

**12.2 — `Tensile/client/sweep_runner.py`**
```python
class SweepRunner:
    def __init__(self, yaml_path, library_path=None, n_warmup=2, n_iters=10,
                 rotating_buffers=8, icache_copies="auto"): ...
    def run(self, results_csv=None, library_update_file=None,
            hw_monitor=False, bounds_check=None,
            rocprof_counters=None) -> list[BenchmarkResult]: ...
```
Orchestrates: enumerate solutions → compile or load → benchmark with rotating buffers and
module rotation → report. `icache_copies="auto"` calls `get_icache_module_copies(co_path)` on
the compiled `.co` file. Falls back to `icache_copies=1` (no rotation) **if `co_path` is not
provided or no file exists at `co_path`** — not 4, since an unjustified constant risks
over-rotation causing OOM. Log a warning when falling back. (This mirrors the `KernelRunner`
`co_path` fallback from M7 task 7.2.)

**12.3 — Write `test_sweep_runner.py`**
- Run `SweepRunner` over `gemm_standard.yaml`, verify `results.csv` written with correct
  schema (column names from 12.0).
- Verify best solution per problem size has the highest GFLOPS.
- Verify `library_update_file` contains the correct `solution_idx` for the winner.
- **Plausibility check**: assert `BenchmarkResult.gflops` for each tested problem is in
  [100, 2000] for gfx950 bf16. Catches unit-conversion bugs.

**12.4 — Cross-validate against C++ client**
Run C++ client on the same YAML and sizes. Top solution index and GFLOPS must agree within
**±2% for problems ≥1024², ±5% for smaller**. Document any discrepancies.

**Thermal-variance guard (avoid false CI failures):** GFLOPS measurements vary with GPU
temperature and clock throttling. If a tolerance assertion fails, **re-run the measurement up
to 3 times and compare using the best-of-3 GFLOPS**, not a single-run value. Alternatively, for
CI robustness, widen to ±5% (≥1024²) / ±10% (smaller) and reserve the tighter ±2%/±5% for
manual validation runs. Never fail CI on a single thermal outlier.

### Acceptance criteria
- CSV schema verified against the actual parser in `LibraryLogic.py:413–531` (`addFromCSV`) — cite the line. (`BenchmarkProblems.py` contains the CSV writer that matches this schema, not the parser itself.)
- GFLOPS plausibility check passes.
- Agreement with C++ client within stated tolerances for all tested sizes.
- No regressions.
