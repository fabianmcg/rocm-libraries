> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 2 — Int8/Int32 and XFloat32 (TF32)

**Executed by:** fresh implementor agent
**Reviewed before:** Milestone 3 begins

### Goal

Add argument construction, references, and tests for int8 accumulation and XFloat32.

### Tasks

**2.1 — Int8/Int32 reference**
- `gemm_int8(A, B, ...) -> np.ndarray`: widen to int32, matmul, then for int8 output apply
  `np.clip(np.round(D), -128, 127).astype(np.int8)`.
- Boundary test: exhaustively test `std::nearbyint` (round-half-to-even) equivalence for the
  set {-129, -128.5, -128, -0.5, 0, 0.5, 127, 127.5, 128} — values at and around the
  rounding/saturation boundary. Additionally test at least these two specific values that
  exercise round-half-to-even in the float32 accumulator path: **`2.5` (must round to `2`, the
  even neighbor, not `3`) and `3.5` (must round to `4`, the even neighbor, not `3`)**. For the
  bf16-accumulator branch (`SaturateCast` with `Accumulator == BFloat16`,
  `client/src/Reference.cpp:424–431`), test **`256.5`**: it is representable in float32 but not
  in bfloat16 (the bf16 step near 256 is 2.0), so casting the accumulator to bf16 first yields
  `256.0`, and `nearbyint` then gives `256` — not the naive `257`. The Python reference must
  reproduce this. Matches `client/src/Reference.cpp:419–443` (verify line numbers against the
  current source).

**2.2 — XFloat32 reference**
- `to_xf32(arr) -> np.ndarray`: `arr.view(np.uint32) & 0xFFFFE000`, reinterpret as
  float32. Matches `DataTypes_XFloat32.hpp:100–110`.
- `gemm_xf32(A, B, ...) -> np.ndarray`: apply `to_xf32` to each operand, accumulate in
  float32.
- Unit test: for (16, 16, 16), compare against C++ client golden output within `RTOL_FP32`.

**2.3 — Extend `build_kernel_args`** for int8 and XFloat32 dtype flag branches.

The dtype-specific argument slots that change for int8 vs fp32 (for the reviewer's 5-slot
verification per `review_protocol.md`) are driven by the tensor data types. In C++
`singleCallArgs` these are read from the `TensorDescriptor`s (`problem.a().dataType()`,
`problem.b().dataType()`, `problem.c()/d().dataType()`) — the reviewer sees them as `aType`,
`bType`, `cType`, `dType` (and `computeType`). In Python `build_kernel_args` they come from the
solution dict's nested `ProblemType` block. Mapping (verified against
`Tensile/SolutionStructs/Problem.py:416–422,719–731`):

| C++ (`singleCallArgs`) | Python key under `solution_dict["ProblemType"]` |
|---|---|
| `aType` | `DataTypeA` (falls back to `DataType`) |
| `bType` | `DataTypeB` (falls back to `DataType`) |
| `cType` | `DestDataType` |
| `dType` | `DestDataType` |
| `computeType` | `ComputeDataType` |

Values are the `rocisa.enum.DataTypeEnum` integer codes (e.g. Int8=8, Int32=6, XFloat32=10 —
see the enum table in M3 task 3.1). The argument vector is otherwise structurally identical for
int8 and fp32 in the supported subset (GSU=1, no epilogue, no StreamK). The 5 slots to
verify side-by-side with `ContractionSolution.cpp` are: (1) D pointer, (2) C pointer,
(3) A pointer, (4) B pointer, (5) the `computeType` / accumulation-type field in the packed
bit-fields (`kernelArgs`, `ContractionSolution.cpp:1547–1720`). For int8→int32, confirm the
accumulation type field encodes int32; for int8→int8, confirm the output-cast type is present.
Also verify that `SaturateCast` semantics (rounding then clipping) are not in `kernelArgs`
but are the kernel's internal responsibility — the argument vector carries only type codes.

**2.4 — Write `test_gemm_int_xf32.py`**
- Int8→Int32: sizes (256,256,256), (512,512,512).
- Int8→Int8 with saturation: boundary values.
- XFloat32→Float32: sizes (256,256,256), (1024,1024,1024).
- Poison-input test for each dtype.
- Cross-validate against C++ client golden files.

### Acceptance criteria
- Boundary test covers all listed values and passes.
- Poison-input tests fail as expected.
- All `test_gemm_int_xf32.py` tests pass on gfx950.
- No regressions.

---
