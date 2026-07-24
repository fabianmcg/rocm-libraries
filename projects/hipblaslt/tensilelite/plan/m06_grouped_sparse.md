> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 6 — Grouped GEMM, Sparse GEMM, StreamK=4/5

**Executed by:** fresh implementor agent
**Reviewed before:** Milestone 7 begins

### Goal

Support grouped GEMM (correct dispatch path), sparse GEMM (2:4 metadata), and StreamK=4/5
argument construction.

### Tasks

**6.0 — Dispatch path (pre-determined, no investigation needed)**
`useUniversalArgs` defaults to `True` (`ContractionSolution.hpp:579`,
`GlobalParameters.py:440`) and every characterization YAML in the project sets it `True`
explicitly. The non-universal path is dead in practice. Implement the universal-args path
only.

The universal-args top-level kernel argument layout (`ContractionSolution.cpp:2024–2083`):
1. `gemm_count` (`uint32_t`): low 30 bits = problem count; high 2 bits = `argType`
   (1=HBM, 2=USERARGS). Emitted by `kernelArgs<false>` at line 1567.
2. `internalArgs` (`uint32_t`): packed GSU/WGM/StaggerU bits. Line 1697.
3. `internalArgs1` (`int32_t`): WGM/XCC bits, if `version >= 1`. Line 1701.
4. `numWorkGroups` (`uint32_t`): total WG count, if `version >= 1`. Line 1702.
5. `argsPtr` (`void const*`): pointer to device workspace holding per-problem arg blobs
   (when `argType==HBM`). Each blob is a flat `singleCallArgs` output for one group,
   written into `inputs.ws[0 .. requiredHostWorkspaceSizePerProblem * N)`. Line 2052.
6. `Synchronizer` (`void const*`): `inputs.grouped[0].Synchronizer`. Line 2080.
7. `Workspace` (`void const*`): `inputs.ws + requiredHostWorkspaceSizePerProblem * N`.
   Lines 2081–2083.

The device workspace must be allocated by the caller with size
`requiredHostWorkspaceSizePerProblem * N + accumulation_scratch`. Each per-problem blob is
produced by calling `build_kernel_args` for that group's problem parameters and writing the
bytes into the correct workspace offset before kernel launch.

**6.1 — Grouped GEMM workspace size (deferred to M10)**
The `grouped_gemm_workspace_size` binding requires `ContractionSolution` and
`ContractionProblemGemm` C++ types exposed only in M10. It is added to `tensilelite_runtime`
in M10 task 10.1. During M6 testing, workspace sizes are obtained using the following
approach:
1. Add a temporary `printf` at the **end** of `requiredHostSizeGroupedGemmSingle`
   (`ContractionSolution.cpp:3702–3721`), immediately before the `return` statement, printing
   the computed size: `fprintf(stderr, "WORKSPACE_SIZE=%zu\n", h_args.size());`.
2. Rebuild `tensilelite-client-common` (`cmake --build . --target tensilelite-client-common`).
3. For each test problem (solution_name, M, N, K, group_count), run:
   `./tensilelite-client --config-file <grouped_yaml> --device 0 2>&1 | grep WORKSPACE_SIZE`
   and record the printed value.
4. Remove the `printf` and rebuild before committing.
5. Acceptance criterion: grep the committed diff to confirm no `WORKSPACE_SIZE` or `fprintf`
   remains in `ContractionSolution.cpp`.

Store the recorded values in a hardcoded dict in the test file:
```python
_TEST_WORKSPACE_SIZES = {
    ("solution_name", 256, 256, 256, 4): <recorded_bytes>,
    # ...
}
```
Document the exact C++ client command used to obtain each value in a comment adjacent to the dict.

**6.2 — Grouped GEMM reference**
`gemm_grouped(groups) -> list[np.ndarray]`: loop over groups, call `gemm()` for each.

**6.3 — Grouped GEMM argument builder**
`build_grouped_gemm_args(groups: list[dict], workspace_sizes: list[int]) -> (top_level_bytes, workspace_layout)`:
- Accepts `workspace_sizes[i]` as the pre-determined byte count for group `i`'s arg blob
  (obtained from the hardcoded dict during testing, or from `grouped_gemm_workspace_size`
  post-M10).
- Builds each per-group blob via `build_kernel_args` for that group's problem parameters.
- For kernels with `globalAccumulation == 3` (MBSK): raise `NotImplementedError` (appends
  `dstD`, `Synchronizer`, `GSUSync` trailing slots at `ContractionSolution.cpp:2000–2013`
  — not yet implemented).
- Returns the top-level kernel argument bytes and a workspace layout descriptor.

**6.4 — Sparse GEMM metadata in `Tensile/client/sparse.py`**
`compress_2_4(A) -> (compressed, metadata)`: dense → 2:4 sparse with AMD metadata layout.
`decompress_2_4(compressed, metadata, shape) -> np.ndarray`: inverse.
Unit tests: round-trip for several shapes; separately, decode hand-constructed metadata
bytes against known dense values (not round-trip alone).

**6.5 — StreamK=4 and StreamK=5 argument branches in `build_kernel_args`**
Read `ContractionSolution.cpp:778–908` for the StreamK=4 and SK=5 argument extensions.
Port the relevant conditional branches.

**6.6 — Write `test_gemm_grouped.py` and `test_gemm_sparse.py`**
- Grouped: 2-group, 4-group, 8-group cases; varying shapes.
- Sparse: sizes (256,256,256), (512,512,512); dtypes fp16, bf16.
- Metadata round-trip test (pure Python).
- Metadata decode-from-known-bytes test (pure Python, hand-computed ground truth).
- StreamK=4 and SK=5: at least one problem size each; cross-validate against C++ client.
- Poison-input tests for grouped and sparse.

### Acceptance criteria
- Dispatch path determination documented and agreed by reviewer before implementation.
- Grouped GEMM device workspace allocation verified correct by reviewer reading
  `ContractionSolution.cpp:2024–2061` alongside `build_grouped_gemm_args`.
- Metadata decode-from-known-bytes test (not just round-trip) passes.
- Reviewer confirms no `WORKSPACE_SIZE` print or `fprintf` debug output remains in
  `ContractionSolution.cpp` (grep the diff).
- All new tests pass on gfx950. No regressions.

---
