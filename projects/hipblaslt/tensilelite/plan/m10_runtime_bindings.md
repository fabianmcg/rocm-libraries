> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 10 — nanobind: MasterSolutionLibrary, Predicates, Formocast

**Executed by:** fresh implementor agent (parallel with M8/M9 after M7)
**Reviewed before:** Milestone 12 begins

### Goal

Expose `MasterSolutionLibrary`, predicate evaluation, and `origami::Formocast` to Python,
enabling the harness to load and query production library artifacts.

### Tasks

**10.0 — origami header audit (prerequisite)**
Read the origami headers (found via `find_package(origami)` resolution in CMake) and
identify the fields of `Formocast::ProblemInfo` and `SizeMapping`. Confirm they can be
populated from Python-supplied M/N/K/dtype/tile values. If an adapter struct is needed,
design it before writing any binding code.

**10.1 — Extend the existing `tensilelite_runtime` nanobind module from M7**
The implementor receives M7's CMake file and nanobind source as input context. Do NOT create
a new CMake target — add the following bindings to the existing `tensilelite_runtime` target
and source file.

Add `grouped_gemm_workspace_size` (deferred from M6 because it requires `ContractionProblemGemm`):
```python
size: int = tensilelite_runtime.grouped_gemm_workspace_size(solution: Solution, problem: Problem)
```

Add `calculateAuto*` as **bound methods on `Solution`** (deferred from M7 because they are
`const` member functions of `ContractionSolution` accessing mutable member caches):
```python
solution.calculate_auto_wgm(problem: Problem, hardware: Hardware, skgrid: int) -> int
solution.calculate_auto_gsu(problem: Problem, hardware: Hardware) -> int
solution.calculate_auto_stagger_u(problem: Problem, hardware: Hardware, skgrid: int, auto_wgm: int) -> int
```
These unlock testing of kernels with `SupportCustomWGM=True` / `SupportCustomStaggerU=True`.

Additionally expose:
- `load_library(path: str) -> Library`
- `get_hardware(device_id=0) -> Hardware`
- `Library.find_best_solution(hw: Hardware, prob: Problem) -> Solution`
- `Library.find_top_solutions(hw: Hardware, prob: Problem, n: int) -> list[Solution]`
- `Solution.eval_hardware_predicate(hw: Hardware) -> bool`
- `Solution.eval_task_predicate(prob: Problem) -> bool`
- `Solution.kernel_name: str`
- `Solution.code_object_path: str`
- `Problem(M, N, K, dtype_a, dtype_b, dtype_c, dtype_d, trans_a, trans_b, **kwargs)`

Wraps `MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>`,
`ContractionHardware`, and the predicate hierarchy. Use `nb::keep_alive` for the
Library→Solution lifetime relationship.

Do NOT release the GIL during `find_best_solution`. The call path is:
`MasterSolutionLibrary::findBestSolution` acquires `solutionsGuard`, then calls
`SingleSolutionLibrary::findBestSolution`, which contains a lazy workspace-size computation
(`requiredHostSizeGroupedGemmSingle`) at `SingleSolutionLibrary.hpp:170–173`. That lazy
init is not itself mutex-protected against concurrent re-entry. Releasing the GIL inside
the binding would permit a second Python thread to enter the same path before the first
thread's lazy init completes, producing a data race. Keep the GIL held for the duration of
`find_best_solution` until the lazy-init path is audited and protected with its own mutex.

**10.2 — Formocast binding** (unconditional — `origami` is always available):
`tensilelite_runtime.formocast_predict(solution: Solution, problem: Problem) -> float`

`origami::Formocast` is a stateful class with `setProblem(ProblemInfo)`,
`setSolution(SizeMapping)`, and `predictedPerformance() const` — it is NOT callable as a
free function. The `Formocast::ProblemInfo` struct (M, N, NumBatches, K, bpeA, bpeB, bpeD,
bpeCompute, transA, transB, swizzleTensorA, swizzleTensorB, dataType) and `SizeMapping` are
origami-specific types that must be populated from Python-supplied values via an adapter.
No existing adapter is present in TensileLite source — **task 10.0 must produce the adapter
design before any binding code is written**.

The binding internally constructs a `Formocast` instance, calls `setProblem` and
`setSolution` with values derived from the `Solution` and `Problem` arguments (using the
adapter from 10.0), then returns `predictedPerformance()`.

**10.3 — `Tensile/client/library_runner.py`**
```python
class LibraryRunner:
    def __init__(self, library_path, co_paths): ...
    def find_best(self, M, N, K, **kwargs) -> KernelRunner: ...
    def find_top_n(self, n, M, N, K, **kwargs) -> list[KernelRunner]: ...
    def filter_by_predicate(self, solutions) -> list[Solution]: ...
```

**10.4 — Write `test_library_runner.py`**
- `TestLoadLibrary`: load a real `TensileLibrary.yaml`+`.co` built for gfx950, call
  `find_best`, verify a non-null solution is returned and the kernel runs to completion.
- `TestPredicateEval`: `eval_hardware_predicate` returns True for gfx950, False for a
  mismatched architecture string.
- `TestTopN`: `find_top_n(5, ...)` returns ≤5 solutions. Note: `MasterSolutionLibrary::findTopSolutions` returns solutions in library-internal order, not Formocast-ranked order. The binding must sort the returned list by `formocast_predict` descending before returning it to Python. The test asserts that GFLOPS predictions are non-increasing across the returned list.
- `TestFormocast`: `formocast_predict` returns a positive float.

### Acceptance criteria
- `load_library` + `find_best` + `GpuFunction.launch` runs a kernel to completion on a
  real library artifact.
- Predicate evaluation correctly filters incompatible solutions.
- Module follows the editable-install pattern of rocisa.
- origami header audit completed and findings documented before implementation begins.
- No regressions.
