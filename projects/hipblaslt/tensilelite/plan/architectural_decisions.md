# Architectural Decisions

**Read this file before starting any milestone.** These decisions are resolved and must not be relitigated during implementation.

---

## Argument slot layout (`build_kernel_args`)

`ContractionSolution.cpp:singleCallArgs` (lines 548–1130) is a runtime-conditional sequence, not a static table. The Python port replicates this function as `build_kernel_args(solution_params, problem_params, tensors) -> bytes`.

**Supported configuration subset** (M1–M5; extended incrementally per milestone):
- `stridedBatched` ∈ {True, False}
- `streamK` ∈ {0, 3} **for M1–M5**. M6 (task 6.5) incrementally extends support to `streamK` ∈ {0, 3, 4, 5} by adding the StreamK=4 and StreamK=5 argument branches. In M1–M5, any `streamK` value other than 0 or 3 (i.e. 1, 2, 4, or 5) must raise `NotImplementedError`.
- `streamKAtomic`: only the default value `0` (non-atomic) is supported. The unsupported case is `streamKAtomic=1`, which must raise `NotImplementedError` (it skips the workspace+flags block at line 653). The `=0` denotes the single supported value, not an added restriction.
- `groupedGemm=False`
- `GSU=1`, `globalAccumulation` ∈ {0, 1, 2}
- `useInitialStrides=False`
- `internalArgsSupport.version` ≤ 2, `useSFC=False`
- `expertSchedulingMode=0`, `debugKernel=False`

**Unsupported combinations raise `NotImplementedError`** — including:
- `GSU > 1 && streamK == 0` (emits workspace D/C pointers, not device pointers)
- `globalAccumulation=3` (MBSK; adds `dstD`/`Synchronizer`/`GSUSync` trailing slots per group)
- `useSFC=True` (different `internalArg1` packing)
- `version > 2` (different GSU bit-field: bits 0–11 for GSU, bits 12–13 for `ntaBit`/`ntbBit`)
- `expertSchedulingMode != 0` (adds `ESMRuntimeSupported` int32 slot at line 743)
- `debugKernel=True` (adds leading uint pointer slot at line 557)
- `streamKAtomic=1` (skips workspace+flags block at line 653)

**`InternalArgsSupport` YAML key names** (verified against characterization YAMLs):
- `KernArgsVersion` — NOT `version`
- `SupportCustomWGM` — NOT `SupportWgm`
- `SupportCustomStaggerU` — NOT `SupportStaggerU`
- `SupportUserGSU`, `UseSFC`, `UseUniversalArgs`

**`version >= 2` bit-fields** — solution-level YAML keys, NOT inside `InternalSupportParams`:
- `GlobalSplitUCoalesced` → bit 15 of `internalArg0`
- `GlobalSplitUWorkGroupMappingRoundRobin` → bit 14 of `internalArg0`

**`calculateAuto*` are `const` member functions of `ContractionSolution`** (`ContractionSolution.hpp:698–705`). They access `sizeMapping`, `wgmParamsCache`, and `staggerUParamsCache` (mutable member caches). They cannot be bound as free functions from dicts without internally constructing a full `ContractionSolution` C++ object. Therefore:

**M1 through M7 restrict tests exclusively to kernels where `WorkGroupMapping != 0` (explicit WGM, no auto-computation) and `StaggerU` is a fixed non-zero constant.** Every BenchmarkProblems-generated solution has `SupportCustomWGM=True` (set unconditionally by `defaultInternalSupportParams` in `GlobalParameters.py:430`) — filtering on `SupportCustomWGM=False` would yield zero test kernels. The correct restriction is on the parameter value itself: skip any solution where `WorkGroupMapping == 0` (which would require `calculateAutoWGM`). `build_kernel_args` reads WGM/StaggerU/GSU directly from the solution dict for solutions with explicit values.

M10 exposes `ContractionSolution` as a Python type and adds `calculate_auto_wgm`, `calculate_auto_gsu`, `calculate_auto_stagger_u` as bound methods on it.

**Chip→KernArgsVersion table** (for tuning YAMLs without compiled library YAMLs; no `ArchitectureSet` field exists in any tuning YAML):
```python
_CHIP_TO_KERNS_ARGS_VERSION = {
    "gfx908": 0, "gfx90a": 0,
    "gfx940": 1, "gfx941": 1,
    "gfx942": 2, "gfx950": 2,
    "gfx1100": 1, "gfx1101": 1,
}
```
Raise `NotImplementedError(f"unsupported chip for KernArgsVersion lookup: {chip}")` for unknown chips. Do not fall back to version=0 silently.

---

## ROCprofiler-SDK

`LD_PRELOAD` is not required. The nanobind module exports `rocprofiler_configure` and calls `rocprofiler_force_configure` from `PyInit`. The guard checks the **return code of `rocprofiler_force_configure`**: if it returns `ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED`, raise `RuntimeError`. Do NOT use `rocprofiler_is_initialized()` as the guard. Context creation happens inside `tool_init_impl` at import time. The GIL is released before `m_future.get()` in `fetch()`.

---

## `getMinKernelSizeToGwEnd`

Extracted from `client/main.cpp` — the entire `#if defined(__linux__)` block spanning lines 914–1025 (guard at 914, explanatory comment at 915–926, function body `getMinKernelSizeToGwEnd` at 927–1024, closing `#endif` at 1025) — into `client/src/ElfUtils.cpp` + `client/include/ElfUtils.hpp` and added to `tensilelite-client-common` as M7 task 7.0. Verify these line numbers against the current `main.cpp` before extracting.

---

## `tensilelite_runtime` module

Created once in M7 (with `get_icache_module_copies`). M10 **extends** the same module — M10 implementor receives M7's CMake file and nanobind source as input context and adds to them. Do not create a new target.

---

## Grouped GEMM dispatch path

`useUniversalArgs` defaults to `True` everywhere and every characterization YAML sets it explicitly. Only the universal-args path is implemented.

`requiredHostWorkspaceSizePerProblem` is problem-size-dependent (computed by `requiredHostSizeGroupedGemmSingle` at `ContractionSolution.cpp:3761–3780`, returning `h_args.size()` at line 3779; verify these line numbers against the current source before use). The binding requires `ContractionSolution` and `ContractionProblemGemm` exposed in M10. **Only the grouped-GEMM workspace-size binding (M6 tasks 6.1/6.3) depends on M10.** M6 tasks 6.2 (grouped reference), 6.4 (sparse metadata), and 6.5 (StreamK=4/5 arg branches) have no nanobind dependency and may proceed after M5 in parallel with M7–M10. See the dependency graph in `review_protocol.md`.

During M6 testing, workspace sizes are obtained by running the C++ client and hardcoded in a `_TEST_WORKSPACE_SIZES` dict in the test file.

---

## `origami`

Always available when `tensilelite-host` builds (`find_package(origami REQUIRED)` is unconditional). M10 task 10.0 audits origami headers before implementing the Formocast binding.

---

## Thread safety of `find_best_solution`

Do NOT release the GIL during `find_best_solution`. `SingleSolutionLibrary.hpp:170–173` contains a lazy workspace-size computation that is called inside the `solutionsGuard` lock but is not itself mutex-protected against concurrent re-entry. Releasing the GIL would permit a data race.

---

## Code location

- Production harness: `Tensile/client/`
- Nanobind modules: alongside `rocisa/`, following its CMake and editable-install pattern
- Epilogue-specific tests: `epilogues/epilogue_harness/` — imported from `users/fabianmcg/gemm_rms` by M0 task 0.0 (the `epilogues/` directory does not exist on `develop`); renamed from `epilogues/tensilelite/` in M0 task 0.2
- `amdgpu_exec`: **buildable from source.** The source repository is available at `~/amdgpu-exec/` (a standalone git repo; editable build via `pip install --no-build-isolation -e .` using scikit-build-core + CMake/Ninja — see `~/amdgpu-exec/README.md`). It is also installed into the active environment (`~/.tensile/lib/python<version>/site-packages/amdgpu_exec/`). Because the source is available, `amdgpu_exec` *can* be rebuilt and modified if strictly necessary. Nevertheless, new GPU primitives (e.g. `BoundedBuffer` for M8) are still implemented as standalone nanobind modules alongside `tensilelite_runtime` — a deliberate design choice to keep `amdgpu_exec` a general-purpose, separately-versioned dependency, not a limitation imposed by the wheel.
