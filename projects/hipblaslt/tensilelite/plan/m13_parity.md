> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 13 — End-to-End Parity Validation

**Executed by:** fresh implementor agent
**Reviewed before:** Milestone 14 begins

### Goal

Demonstrate full feature parity with the C++ client across its entire supported surface.
No new features — only comprehensive integration testing and a generated parity report.

### Tasks

**13.0 — Generate and preserve C++ client reference CSV**
Before writing `test_parity.py`: run the C++ client on every problem combination listed in
13.1 and save the raw CSV output to
`Tensile/client/tests/fixtures/cpp_client_reference.csv`. Commit this file alongside a
companion `cpp_client_reference_cmd.txt` containing the exact command used. `test_parity.py`
reads this file as its ground truth — it does not re-run the C++ client at test time. The
reviewer spot-check (M13 acceptance criteria) cites specific row numbers in this file.

**13.1 — Write `Tensile/client/tests/test_parity.py`**
For each combination below, run the Python harness on identical inputs to those in 13.0 and
compare GPU outputs and GFLOPS against the preserved reference CSV:
- fp32 GEMM, no epilogue, sizes (256, 512, 1024).
- bf16 GEMM, no epilogue, stridedBatched=True, sizes (256, 512, 1024, 2048, 4096).
- fp16 GEMM, row bias + Relu.
- int8 → int32 accumulation.
- fp8 E4M3 OCP, size (512, 512, 512).
- MX float8 + E8 scale, block_k=32.
- Grouped GEMM, 4 groups.
- Sparse GEMM, fp16.
- PartialRMS (K1) — already passing in epilogues.
- RstdScale (K3) — already passing in epilogues.
- StreamK=3 ForceDPOnly — already passing in epilogues.

GFLOPS agreement: ±2% for ≥1024², ±5% for smaller. GPU tensor output correctness for each
combination is already verified against the Python numpy reference in M1–M5; M13 does not
re-verify tensor correctness against the C++ client (the C++ client CSV records only GFLOPS,
not tensor values).

**13.2 — Generate `Tensile/client/parity_report.md`**
The test suite writes this file (not hand-authored). Content:
- Feature coverage table (feature, status, evidence).
- GFLOPS table (problem, Python harness, C++ client, delta%).
- Any remaining discrepancies with explanation.

**13.3 — Deprecation annotation**
Add to `client/main.cpp` above the copyright header:
```cpp
// Retained as a reference implementation.
// The Python harness in Tensile/client/ provides equivalent functionality.
// See Tensile/client/parity_report.md for parity validation results.
```

**13.4 — Update `AGENTS_reference.md`**
Document the Python harness as the primary testing path. Include instructions for:
- `tox -e unit` — runs all harness tests including profiler tests (via `requires_rocprof` marker)
- `tox -e unit -k requires_rocprof` — runs only profiler counter tests (no dedicated `tox -e profiler` env exists)
- `SweepRunner` — how to run a benchmark sweep from Python
- `LibraryRunner` — how to load a production library artifact and run best-solution dispatch

### Acceptance criteria
- All `test_parity.py` cases pass within stated tolerances.
- `parity_report.md` generated (not hand-written) and covers all listed feature combinations.
- GFLOPS delta within ±2% for all ≥1024² problems.
- **Human spot-check (reviewer requirement):** The reviewer manually cross-checks at least 3
  rows of `parity_report.md` against the raw C++ client CSV output for the same inputs. This
  guards against the report being self-validating (a test that produces wrong results within
  too-loose tolerances generates a passing report). The reviewer must confirm that the Python
  harness and C++ client GFLOPS agree within ±2% (≥1024²) or ±5% (smaller) for those rows,
  using the values from `cpp_client_reference.csv`. Do not require identical raw numbers —
  GFLOPS are thermally variable across runs and GPU units.
- Zero regressions across all milestones.
