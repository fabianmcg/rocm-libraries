> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 5 — Fused Epilogues: Bias, Activations, Scales, AmaxD, E Tensor

**Executed by:** fresh implementor agent
**Reviewed before:** Milestone 6 begins

### Goal

Port the epilogue argument slots from `ContractionSolution.cpp:singleCallArgs` and add
numpy references for all fused epilogue types.

### Tasks

**5.1 — Epilogue references in `reference.py`**
- `apply_bias(D, bias, bias_source)`: `"row"` → `D += bias[None,:]`, `"col"` → `D += bias[:,None]`, `"matrix"` → `D += bias`.
- `apply_activation(D, name)`: Relu, Gelu, DGelu, Silu, Swish, Sigmoid, Tanh. Match
  `Reference.cpp:746–853`. Unit test each against 5 hand-chosen input values.
- `apply_scale_ab(A, B, scale_a, scale_b)`: scalar or vector before matmul.
- `apply_scale_cd(C, D, scale_c, scale_d)`: elementwise after matmul.
- `apply_scale_alpha_vec(D, scale_vec, factor_dim)`: per-row (dim=0) or per-column (dim=1).
- `compute_amax_d(D) -> float`: `float(np.max(np.abs(D)))`.
- `compute_e_tensor(D) -> np.ndarray`: copy of D before output cast.

**5.2 — Epilogue argument slots in `build_kernel_args`**
Read `ContractionSolution.cpp:singleCallArgs` for each epilogue flag's argument extension:
- Bias: `useBias=True` adds bias pointer + `biasSource` enum + `biasSrc` tensor type.
- Activation: `activationType` adds enum + optional α/β parameters.
- ScaleAB: adds scalar or vector pointer per tensor.
- ScaleCD: adds ScaleC and ScaleD pointers.
- ScaleAlphaVec: adds pointer + `factorDim`.
- AmaxD: adds output pointer for the amax scalar.
- E tensor: adds pointer and strides.

**5.3 — Write `test_gemm_epilogues.py`**
One test class per epilogue, parametrized over (fp32, bf16) and two problem sizes:
- `TestBias`: row, col, matrix bias.
- `TestActivations`: one test per activation name.
- `TestScaleAB`: scalar and vector scale.
- `TestScaleCD`.
- `TestScaleAlphaVec`: factor_dim=0 and 1.
- `TestAmaxD`: assert scalar output matches `np.max(np.abs(D_ref))`.
- `TestETensor`: assert E matches D before output cast.
- Multi-epilogue combinations: bias+Relu, bias+ScaleAB+Gelu, ScaleAB+ScaleCD+AmaxD.
- Poison-input test for at least bias and ScaleAB.

### Acceptance criteria
- Each activation function unit-tested in pure Python against hand-chosen input/output pairs
  from `Reference.cpp`.
- Reviewer confirms epilogue argument slots in `build_kernel_args` match
  `ContractionSolution.cpp` by reading both files for each epilogue type.
- All `test_gemm_epilogues.py` tests pass on gfx950.
- No regressions.

---
