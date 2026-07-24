> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 1 — `build_kernel_args`: Standard GEMM + float32/float16/bfloat16 + Strided-Batched

**Executed by:** fresh implementor agent
**Reviewed before:** Milestone 2 begins

### Goal

Port the core of `ContractionSolution.cpp:singleCallArgs` to Python for standard and
strided-batched GEMM kernels. Add numpy references and correctness tests for fp32, fp16,
and bf16.

### Tasks

**1.1 — Port `singleCallArgs` for the standard subset**
In `Tensile/client/gemm_args.py`, implement `build_kernel_args` for the following flag
combination: `stridedBatched` ∈ {True, False}, `groupedGemm=False`, `streamK=0`, `GSU=1`,
`globalAccumulation=0`, `useInitialStrides=False`, `internalArgsSupport.version` ≤ 2,
`sparse=False`, `mxBlockA=0`, `mxBlockB=0`, `expertSchedulingMode=0`.

Read `ContractionSolution.cpp:548–1130` directly. Port each conditional branch in order.
The solution dict from `yaml_solution_builder.py` contains all flag values needed to
traverse the tree. Return the argument bytes as a `bytes` object matching exactly what the
C++ function would produce.

Also port `kernelArgs` (`ContractionSolution.cpp:1547`) for the packed bit-field suffix.

**1.2 — Poison-input test**
Before any correctness test: set `strideA[0]` (the leading dimension of A in the argument
vector) to `correct_value + M`, where M is the row dimension. This guarantees every row of
A reads from the wrong device offset, producing detectable corruption for any M>0, N>0, K>0.
Assert that the GPU output differs from the numpy reference by more than `10 * RTOL` on at
least 50% of elements. This proves the argument vector actually drives the computation — a
passing poison test is a prerequisite for trusting the correctness tests.

Also verify `NotImplementedError` fires for: `GSU > 1 && streamK == 0` (workspace D/C
pointer path); `globalAccumulation == 3` (MBSK); `useSFC=True` (different `internalArg1`
packing); `internalArgsSupport.version > 2`.

**1.3 — numpy references for fp16 and bf16 in `reference.py`**
- `gemm_fp16`: upcast to float32, matmul, downcast. Tolerance `RTOL_FP16`.
- `gemm_bf16`: via `ml_dtypes.bfloat16`, upcast to float32, matmul. Tolerance `RTOL_BF16`.
- Unit tests against hand-computed reference values (no GPU).

**1.4 — Write `Tensile/client/tests/test_gemm_standard.py`**
- `@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])`
- `@pytest.mark.parametrize("strided_batched", [True, False])`
- Problem sizes come from `Tensile/client/tests/yaml/gemm_standard.yaml`. This file must be
  created by the M1 implementor as a standard Tensile BenchmarkProblems YAML with a
  `ProblemSizes` block listing square (256, 512, 1024, 2048, 4096) and non-square shapes, and
  a `BenchmarkFinalParameters` section containing one or more standard GEMM solution configs.
  The YAML schema matches `epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml` (present on the
  implementation branch — imported from `gemm_rms` by M0 task 0.0) — use that file as a
  template, removing PartialRMS-specific epilogue fields. Solutions are enumerated via
  `enumerate_all_solutions(yaml_path)` from M0 task 0.7.
- **Kernel filter:** Skip any solution where `solution_dict["WorkGroupMapping"] == 0`
  (auto-WGM; requires `calculateAutoWGM` not yet bound). Also skip solutions where
  `solution_dict.get("StaggerU", 0) != solution_dict.get("StaggerU", 0)` (auto-StaggerU).
  Use `pytest.skip` with a message naming the skipped solution so the skip is visible in
  test output, not silent. This replaces the incorrect `SupportCustomWGM=False` filter from
  prior plan versions — `SupportCustomWGM` is always `True` for all generated solutions.
- Include `batch_count=4` for the batched cases.
- For each (dtype, strided_batched, size, solution): `build_kernel_args` → `KernelRunner`
  → compare D against `reference.gemm` within tolerance.
- `HAVE_DEPS` and `requires_gfx950` guards.

**1.5 — Cross-validate against C++ client**
Run C++ client on the same YAML+sizes, compare CSV. Document all discrepancies. This is a
one-time validation step, not a permanent CI dependency.

### Acceptance criteria
- Poison-input test fails (GPU output wrong) before correctness tests run.
- All `test_gemm_standard.py` tests pass on gfx950.
- Reviewer reads `gemm_args.py` and `ContractionSolution.cpp:548–700` side-by-side and
  confirms the first 20 argument slots match for both stridedBatched=True and False.
- No regressions.

---
