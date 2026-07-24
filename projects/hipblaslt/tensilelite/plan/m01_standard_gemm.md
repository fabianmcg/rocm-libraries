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
Before any correctness test, corrupt a stride in the argument vector so the kernel reads A
from wrong offsets. **Use `strideA[1]` (the leading-dimension stride), NOT `strideA[0]`:**
for column-major layouts `strideA[0] == 1`, so perturbing it by `+M` gives an offset of
`1 + M` that can coincidentally land on a valid (if wrong) row boundary for some shapes,
weakening the signal. `strideA[1]` is always `>= M`, giving a larger, unambiguous corruption
signal regardless of layout. Set `strideA[1] = correct_value + M`, where M is the row
dimension. This guarantees the kernel reads A from the wrong device offset, producing
detectable corruption for any M>0, N>0, K>0.
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
- Problem sizes come from `Tensile/client/tests/yaml/gemm_standard.yaml`. **Do not write this
  YAML from scratch.** Start from an existing standard-GEMM characterization YAML under
  `Tensile/Tests/unit/characterization/` and adapt it — copy a suitable base and strip any
  epilogue-specific fields. Suitable bases (BenchmarkProblems-format, containing `ProblemSizes`)
  live under `Tensile/Tests/unit/characterization/_codegen/data/test_data/_designed/<arch>/`,
  e.g. `gfx942/asmaddr2_fp32.yaml` (fp32) or `gfx942/asmaddr2_bf16_srvw.yaml` (bf16);
  `epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml` (imported by M0 task 0.0) is an alternative
  template for the overall structure. The adapted file must have a `ProblemSizes` block listing
  square (256, 512, 1024, 2048, 4096) and non-square shapes and a `BenchmarkFinalParameters`
  (or equivalent solution) section with one or more standard GEMM solution configs. Find a
  suitable base first, confirm it parses via `enumerate_all_solutions`, then adapt. Solutions
  are enumerated via `enumerate_all_solutions(yaml_path)` from M0 task 0.7.
- **Kernel filter:** Skip any solution where `solution_dict["WorkGroupMapping"] == 0`
  (auto-WGM; requires `calculateAutoWGM`, not bound until M10). Also skip any solution that
  would require auto-StaggerU: skip when `StaggerU == 0` **and** the solution's
  `InternalSupportParams` reports `SupportCustomStaggerU == True` (a StaggerU of 0 with
  custom-StaggerU support means the kernel expects a runtime-computed StaggerU via
  `calculateAutoStaggerU`, not bound until M10). The correct YAML key is `SupportCustomStaggerU`
  (inside the `InternalSupportParams` sub-block — see M0 task 0.8), **NOT** `SupportStaggerU`.
  Concretely:
  ```python
  isp = solution_dict.get("InternalSupportParams", {})
  if solution_dict.get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
      pytest.skip(f"auto-StaggerU not supported: {solution_id}")
  ```
  Use `pytest.skip` with a message naming the skipped solution so the skip is visible in test
  output, not silent. This replaces the incorrect `SupportCustomWGM=False` filter from prior
  plan versions — `SupportCustomWGM` is always `True` for all generated solutions.
- Include `batch_count=4` for the batched cases.
- For each (dtype, strided_batched, size, solution): `build_kernel_args` → `KernelRunner`
  → compare D against `reference.gemm` within tolerance.
- `HAVE_DEPS` and `requires_gfx950` guards.

**1.5 — Cross-validate against C++ client**
Run C++ client on the same YAML+sizes, compare CSV. This is a one-time validation step, not a
permanent CI dependency. **The outcome is not optional:**
- Record every discrepancy (per problem: Python GFLOPS, C++ GFLOPS, delta%) in
  `Tensile/client/tests/fixtures/m1_cross_validate_notes.txt`, committed with the milestone.
- **Blocker rule:** if GFLOPS diverge by more than **±5%** for any problem, M1 is not
  complete — the divergence must be explained (documented as thermal/measurement noise with a
  re-run, or root-caused to an argument-construction bug and fixed) in that notes file before
  M2 may begin. Any tensor-value mismatch is always a blocker.

### Acceptance criteria
- Poison-input test fails (GPU output wrong) before correctness tests run.
- All `test_gemm_standard.py` tests pass on gfx950.
- Reviewer reads `gemm_args.py` and `ContractionSolution.cpp:548–700` side-by-side and
  confirms the first 20 argument slots match for both stridedBatched=True and False.
- No regressions.

---
