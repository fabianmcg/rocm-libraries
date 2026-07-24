> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 14 — Replace `ClientWriter.py` subprocess call with `SweepRunner`

**Executed by:** fresh implementor agent
**Reviewed before:** declared complete

### Goal

Replace `ClientWriter.py`'s `subprocess.Popen` call (at line 273; verify against current
source) with a direct call to `SweepRunner`, making the Python harness the actual execution
path in the Tensile pipeline — not just a parallel tool.

### Tasks

**14.1 — Refactor `ClientWriter.runClient()` and extend `SweepRunner`**
`ClientWriter.py:273`: replace the `subprocess.Popen(runScriptName, cwd=buildPath)` call
(the single-GPU path reached after the `writeRunScript(...)` call at line 270) with a direct
call to `SweepRunner`. **Modify `SweepRunner.__init__` from M12** by appending three keyword
parameters — do not create a new class. (Verify these line numbers against the current
`ClientWriter.py` before editing.)

**Parameter sourcing (no implicit `globalParameters` reads):** `SweepRunner` reads
`pin_clocks`, `timing_instrumentation`, and `mx_scale_format` **only from its own keyword
parameters** — it must NOT read `globalParameters` directly (this keeps `SweepRunner` a pure,
testable unit). The `runClient` call site passes them explicitly from `globalParameters` when
constructing `SweepRunner`, e.g.:
```python
runner = SweepRunner(
    yaml_path=configPaths[0],
    pin_clocks=globalParameters["PinClocks"],
    timing_instrumentation=globalParameters["TimingInstrumentation"],
    mx_scale_format=globalParameters["MXScaleFormat"],
)
return runner.run(...)
```

The following behaviors from `writeRunScript()` (`ClientWriter.py:314–382`; verify line
numbers against the current source) must be replicated:
- **Clock pinning:** when `pin_clocks` is set and `globalParameters["AMDSMIPath"]` is
  available, run `sudo <amd-smi-path> set -g 0 --fan 255` and
  `sudo <amd-smi-path> set -g 0 --perf-level HIGH` before benchmarking (matching
  `ClientWriter.py:340–341`), then sleep 1 second. Reset in a `finally` block via
  `sudo <amd-smi-path> reset -g 0 --clocks --fans` (matching `ClientWriter.py:380`). If `sudo`
  returns non-zero, raise `PermissionError` — do not silently continue. **`sudo` may be absent
  in container environments:** wrap the `subprocess.run(["sudo", ...])` call in
  `try/except FileNotFoundError` and, on `FileNotFoundError`, raise
  `PermissionError("sudo not found; clock pinning requires sudo")`. Add
  `pin_clocks: bool = False` to `SweepRunner.__init__`.
- **Timing instrumentation:** add `timing_instrumentation: bool = False` (matches the
  `--timing-instrumentation` flag appended at `ClientWriter.py:356`).
- **MX scale format:** add `mx_scale_format: str = None` (matches the `--mx-scale-format`
  flag appended at `ClientWriter.py:357`).

**Multi-GPU path:** The existing `runClient` calls `runClientParallel` when
`numGpus > 1 and forBenchmark` (`ClientWriter.py:256`). This parallel path is not
replaced by `SweepRunner` in M14. When `use_python_client=True` and `numGpus > 1`, fall
through to `runClientParallel` unchanged. Add a log warning that the Python harness path is
not used in multi-GPU mode.

Keep the old `subprocess.Popen` path behind a `use_python_client=False` flag (default
`True`) for rollback. Log a deprecation warning when `use_python_client=False`.

**14.2 — Write `test_clientwriter_integration.py`**
Run a full `BenchmarkProblems` → `LibraryLogic` → `ClientWriter` pipeline using the Python
harness path. Verify the **same winner solution index** is selected for every problem size
in the test YAML — byte-for-byte YAML identity is not achievable because GFLOPS differences
can flip winner selection and timestamps differ across runs.

**14.3 — CI / tox integration**
Add a `tox -e integration` environment that runs `test_clientwriter_integration.py` on
gfx950. Requires a GPU; marked appropriately.

### Acceptance criteria
- The **same winner solution index** is selected for every problem size in the test YAML
  between the Python and C++ client paths. Byte-for-byte YAML identity is not required and
  not achievable (GFLOPS differences can flip winner selection; timestamps differ across runs).
- `use_python_client=False` fallback restores C++ client behavior.
- `tox -e integration` passes on gfx950.
