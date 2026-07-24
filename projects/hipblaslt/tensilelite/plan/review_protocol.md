# Review Protocol and Dependency Graph

## Review protocol (applies to every milestone)

Each milestone must be reviewed by a **fresh reviewer agent** before the next begins. The reviewer receives:
1. `architectural_decisions.md` — resolved decisions all milestones depend on
2. This file — review rules and dependency graph
3. The specific milestone sub-plan
4. The full diff of changed files
5. The test output (`tox -e unit` or `pytest` run)

The reviewer checks:
- All acceptance criteria are met.
- New code follows the style in `~/.claude/CLAUDE.md` (camelBack, early returns, ~40 line functions, comments only for non-obvious WHY).
- No test is vacuously passing (e.g. skipped due to missing GPU without being flagged).
  Specifically: if a milestone's GPU-gated tests (`@requires_gfx950` / `requires_rocprof`) are
  **all skipped** (e.g. the review environment has no GPU), the reviewer must explicitly mark
  the milestone **"GPU tests not verified"** in the review report and treat it as only
  **partially reviewed**. Full approval requires a separate GPU verification run whose output is
  attached — pure-Python tests passing is not sufficient for a milestone that has GPU tests.
- **M0 only:** `Tensile/client/tests` appears in `pyproject.toml` `testpaths` and
  `tox -e unit -- --collect-only 2>&1 | grep client` lists at least one test. Absence
  causes all M1–M13 tests to be silently skipped with zero failures reported.
- For argument-layout milestones: at least 5 argument slots verified by reading `ContractionSolution.cpp` alongside `build_kernel_args` for the tested configuration.
- Poison-input tests are present and demonstrably fail before correctness tests run.
- No dead code or stubs left as "TODO later."

If review fails, the implementor agent receives the findings and fixes before the next milestone begins.

---

## Dependency graph

```
M0 (baseline + infrastructure)
 └── M1 (fp32/fp16/bf16 + strided-batched, build_kernel_args core)
      └── M2 (int8, XFloat32)
           └── M3 (fp8)
                └── M4 (MX block-scaled)
                     └── M5 (fused epilogues)
                          ├── M6-partial: 6.2 grouped ref, 6.4 sparse metadata,      ─┐
                          │    6.5 StreamK=4/5 args (NO nanobind dep; runs in          │
                          │    parallel with M7)                                       │
                          └── M7 (rotating buffers, I-cache, ELF binding only)         │
                               ├── M8 (sentinel bounds checking)                       │
                               ├── M9 (hw monitoring)                                  ├─ all → M12
                               └── M10 (library/predicate/Formocast,                   │
                                    calculateAuto* methods, workspace_size)            │
                                    ├── M11 (ROCprofiler-SDK)                          │
                                    └── M6-full: 6.1/6.3 grouped workspace-size        │
                                         binding (NEEDS M10)                          ─┘
                                              M12 (sweep/CSV pipeline)
                                               └── M13 (parity validation)
                                                    └── M14 (ClientWriter integration)
```

## Key sequencing decisions

- **M5 before M7:** M1–M5 restrict tests to kernels where `WorkGroupMapping != 0` (explicit WGM, no auto-computation needed) and `StaggerU` is a fixed non-zero constant — `SupportCustomWGM` is always `True` for every BenchmarkProblems-generated solution (set unconditionally by `defaultInternalSupportParams`), so filtering on `SupportCustomWGM=False` would yield zero kernels. The correct restriction is on the parameter value: skip solutions where `WorkGroupMapping == 0`. M7 provides the ELF binding for rotating buffers/I-cache (benchmarking), not argument construction.
- **M6 is split (see graph):** Only **M6-full** (tasks 6.1/6.3, the grouped-GEMM
  workspace-size binding) depends on M10 — `grouped_gemm_workspace_size` requires
  `ContractionSolution`/`ContractionProblemGemm` exposed in M10. **M6-partial** (tasks 6.2
  grouped reference, 6.4 sparse metadata, 6.5 StreamK=4/5 arg branches) has no nanobind
  dependency and may run after M5 in parallel with M7–M10. M6's `build_grouped_gemm_args`
  accepts `workspace_sizes: list[int]` and `synchronizer_ptr: int` as explicit parameters;
  during M6 testing, workspace sizes are obtained per M6 task 6.1 and hardcoded in a
  `_TEST_WORKSPACE_SIZES` dict in the test file. `calculateAuto*` bound methods likewise come
  from M10 (needed by later WGM/StaggerU-auto testing, not by M6-partial).
- **M8, M9, M10 are independent** of each other and run concurrently after M7.
- **M11 extends M10's nanobind module** infrastructure.
- **M12 waits** for M6, M8, M9, M10, and M11.
