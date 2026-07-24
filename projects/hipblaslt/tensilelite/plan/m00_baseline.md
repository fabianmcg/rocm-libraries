> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md), [`review_protocol.md`](review_protocol.md), and [`amdgpu_exec_reference.md`](amdgpu_exec_reference.md) before starting this milestone.

## Milestone 0 — Baseline Audit and Infrastructure Setup (Prerequisite)

**Executed by:** implementor agent
**Reviewed before:** Milestone 1 begins

### Goal

Create a fresh implementation branch, establish a clean runnable baseline, pin dependencies,
resolve the `epilogues/tensilelite` import-shadowing risk, and create the shared
infrastructure all subsequent milestones build on.

**This plan is not implemented on the epilogue development branch.** M0 task 0.0 creates a
dedicated branch from `develop` before any other work begins.

### Tasks

**0.0 — Create the implementation branch**

The `plan/` and `epilogues/` directories currently live only on `users/fabianmcg/gemm_rms`.
`epilogues/` is needed because subsequent milestones reference `yaml_solution_builder.py`,
`enumerate_all_solutions`, `HAVE_DEPS`, `requires_gfx950`, and the existing YAML files as
templates. Both directories must travel to the new branch.

Run all commands below from the **`rocm-libraries` repo root** (the directory that contains
`projects/hipblaslt/tensilelite/`):

```bash
# Export plan/ and epilogues/ from the epilogue branch.
# git archive paths are relative to the repo root.
git checkout users/fabianmcg/gemm_rms
git archive HEAD projects/hipblaslt/tensilelite/plan \
                  projects/hipblaslt/tensilelite/epilogues \
  | tar x -C /tmp/branch_export

# Branch from develop
git fetch origin develop
git checkout -b users/fabianmcg/python-client-replacement origin/develop

# Copy both directories into the new branch and commit
cp -r /tmp/branch_export/projects/hipblaslt/tensilelite/plan \
      projects/hipblaslt/tensilelite/plan
cp -r /tmp/branch_export/projects/hipblaslt/tensilelite/epilogues \
      projects/hipblaslt/tensilelite/epilogues
git add projects/hipblaslt/tensilelite/plan \
        projects/hipblaslt/tensilelite/epilogues
git commit -m "docs(tensilelite): import plan and epilogues from gemm_rms branch"
```

All implementation work through M14 happens on `users/fabianmcg/python-client-replacement`.
The epilogue branch (`users/fabianmcg/gemm_rms`) is left untouched.
PRs are based on `develop`.

**0.1 — Pin `amdgpu_exec` version**
- Run `pip show amdgpu_exec | grep Version` and record the version.
- Add `amdgpu_exec==<version>` directly to the `deps` lines of every `[testenv]` section in
  `tox.ini` that runs Python harness tests. Do NOT add to a new `requirements-dev.txt` —
  `tox.ini` reads `requirements.txt` (not `requirements-dev.txt`) for its `-r` deps, so a
  separate dev file is invisible to tox.

**0.2 — Rename `epilogues/tensilelite/` to `epilogues/epilogue_harness/`**
- Rename the subpackage and update all imports in `epilogues/` to use `epilogue_harness`.
- Verify that `import tensilelite` in a test still resolves to the installed package and not
  the local subdirectory. Add an assertion to `conftest.py`:
  `assert tensilelite.__file__ not in epilogues_path`.

**0.3 — Verify existing test suite passes**
- Run `tox -e unit` and record baseline pass/fail counts in
  `epilogues/unittests/baseline_report.txt`.

**0.4 — Create `Tensile/client/__init__.py`, `Tensile/client/harness.py`, and `Tensile/client/tests/conftest.py`**
Define `HAVE_DEPS` and `requires_gfx950` (and later `requires_rocprof`) in
`Tensile/client/tests/conftest.py`, following the same pattern as
`epilogues/unittests/epilogue_test_common.py:21–25`:
```python
try:
    import amdgpu_exec  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

requires_deps = pytest.mark.skipif(not HAVE_DEPS, reason="amdgpu_exec not installed")
requires_gfx950 = pytest.mark.skipif(
    not HAVE_DEPS or get_chip() != "gfx950", reason="requires gfx950 GPU"
)
```
Every test that invokes GPU code must be decorated with `@requires_deps` or `@requires_gfx950`.
Tests without these decorators must be pure-Python (no GPU, no `amdgpu_exec` import at call
time).

A shared low-level execution harness (`harness.py`) must expose:
- `class KernelRunner`: wraps one or more `GpuModule` instances (for module rotation) and
  pre-allocated `GpuBuffer` input/output pools. Provides
  `run(args, grid, block, n_warmup, n_iters) -> BenchmarkResult`.
- `class BufferPool(n_slots, size_bytes)`: pre-allocates `n_slots` independent `GpuBuffer`
  instances, cycles them round-robin via `next() -> GpuBuffer`.
- `@dataclass BenchmarkResult`: holds `times_ns: List[int]`, `warmup_n: int`, properties
  `mean_us`, `p50_us`, `p95_us`, `min_us`, `gflops(M, N, K)`.
- No GPU calls in `__init__`; `GpuModule` and `GpuBuffer` are passed in by the caller.
- Unit tests with mocked `GpuModule`/`GpuBuffer` (no GPU required).

**0.5 — Create `Tensile/client/reference.py`**
Shared numpy reference computation. Must implement:
- `gemm(A, B, alpha=1.0, beta=0.0, C=None) -> np.ndarray`: `D = alpha*(A@B) + beta*C` in
  float32.
- Tolerance constants: `RTOL_BF16=2e-2`, `ATOL_BF16=2e-2`, `RTOL_FP16=1e-3`,
  `ATOL_FP16=1e-3`, `RTOL_FP32=1e-5`, `ATOL_FP32=1e-5`.
- `assert_close(gpu, ref, rtol, atol, label)`: wraps `np.testing.assert_allclose` with a
  message showing the worst offender's index, got, expected.
- Unit tests (no GPU).

**0.6 — Create `Tensile/client/gemm_args.py` skeleton**
Stub file with the `build_kernel_args(solution_params, problem_params, tensors) -> bytes`
signature, a docstring describing the supported configuration subset, and `NotImplementedError`
for unsupported flag combinations. Milestones 1–6 fill in the branches.

**0.7 — Extend `epilogue_harness/yaml_solution_builder.py` with `enumerate_all_solutions`**
Generalizes the `while True / except (IndexError, KeyError): break` pattern from
`bench_gemm_rmsnorm.py:130–139`. Returns `list[(group_idx, solution_idx, solution_dict)]`.
Tested against both existing YAMLs.

**0.8 — Augment solution dicts with `InternalArgsSupport` fields**
`InternalArgsSupport.version`, `useSFC`, `gsu`, `wgm`, `staggerU` are runtime attributes of
the C++ `ContractionSolution` type, not present in the `yaml_solution_builder.py` output.

`enumerate_all_solutions` consumes **tuning YAMLs** — the output of BenchmarkProblems
(under `2_BenchmarkData/`), not the library YAMLs produced by LibraryLogic
(`3_LibraryLogic/`). The existing epilogue YAMLs in `epilogues/yaml/` are tuning YAMLs.
In tuning YAMLs, solution dicts appear under `BenchmarkFinalParameters[*].SolutionSummationExpansion[*]`
and contain `InternalSupportParams` as a sub-key on each solution dict.

Add a step in `enumerate_all_solutions` to read these from the solution dict's
`InternalSupportParams` block. The **correct YAML key names** (verified against actual
characterization YAMLs under `Tensile/Tests/unit/characterization/`) are:
- `KernArgsVersion` — NOT `version`
- `SupportCustomWGM` — NOT `SupportWgm`
- `SupportCustomStaggerU` — NOT `SupportStaggerU`
- `SupportUserGSU`
- `UseSFC`
- `UseUniversalArgs`

Also read the following solution-level YAML keys (NOT inside `InternalSupportParams`) needed
for `version >= 2` bit-fields:
- `GlobalSplitUCoalesced` → bit 15 of `internalArg0`
- `GlobalSplitUWorkGroupMappingRoundRobin` → bit 14 of `internalArg0`

For tuning YAMLs that do not have a compiled library YAML, **do not default to version=0**.
There is no `ArchitectureSet` field in any tuning YAML (confirmed by exhaustive repo grep).
Note: `defaultInternalSupportParams` in `GlobalParameters.py:430` already sets
`KernArgsVersion: 2` for all BenchmarkProblems-generated solutions, so in practice the
chip→version table below is only consulted for a narrow edge case (a YAML that has neither
`InternalSupportParams` nor a BenchmarkProblems-generated solution dict). It is retained as
a safety net for unexpected YAML formats. The only path is `amdgpu_exec.get_chip()`. Use
the following hardcoded chip→version table:
```python
_CHIP_TO_KERNS_ARGS_VERSION = {
    "gfx908": 0,
    "gfx90a": 0,
    "gfx940": 1,
    "gfx941": 1,
    "gfx942": 2,
    "gfx950": 2,
    "gfx1100": 1,
    "gfx1101": 1,
}
```
Raise `NotImplementedError(f"unsupported chip for KernArgsVersion lookup: {chip}")` for
chips not in the table — do not fall back to version=0 silently. Document the table in the
module docstring so it can be updated as new architectures are supported.

Grep all characterization YAMLs for `KernArgsVersion` values and assert none exceed 2.
Document the scan result in `baseline_report.txt`. Note: `AdaptiveGemmNTAB != 0` in
`Solution.py:1832–1836` force-promotes `KernArgsVersion` to 3 at generation time; the grep
will catch any such kernels already present.

**0.9 — Update `pyproject.toml` package-data, exclusions, and testpaths**
- Add `Tensile/client/tests/yaml/**` to `[tool.setuptools.package-data]` so YAML fixtures
  are included in source distributions.
- Add `Tensile/client/tests` to `[tool.setuptools.packages.find].exclude` so test files are
  not installed into production wheels.
- Add `Tensile/client/tests` to `[tool.pytest.ini_options] testpaths` in `pyproject.toml`.
  Without this, `tox -e unit` uses the existing `testpaths = ["Tensile/Tests", "rocisa/test"]`
  and silently skips every test added in M1–M13 — the run reports zero failures, making
  the omission easy to miss. **Task 0.9 must be verified complete before M0 review passes.**
  Verify by running `tox -e unit -- --collect-only 2>&1 | grep client` and confirming at
  least one test in `Tensile/client/tests/` is listed.

### Acceptance criteria
- `tox -e unit` passes with zero regressions after rename.
- `import tensilelite` resolves to the installed package in all test environments.
- `amdgpu_exec` version pinned directly in `tox.ini` `deps` (not in `requirements-dev.txt`).
- `harness.py`, `reference.py` unit-tested without GPU.
- `enumerate_all_solutions` tested against both existing YAMLs.

---
