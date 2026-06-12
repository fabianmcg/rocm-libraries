# Fused Epilogue Extensions: RMSNorm and Composable Epilogue Chains

This document specifies the hipBLASLt API extensions for fused epilogues, starting with
RMSNorm, and the composable epilogue-chain mechanism that lets RMSNorm compose with
residual add, AMax capture, and FP8 requantization. It also reserves the design points
required so that Gated Linear Units (SwiGLU/GeGLU/ReGLU) can be added later without
reworking the API.

## 1. Motivation and scope

The public hipBLASLt matmul computes:

```
D = Activation(alpha * op(A) * op(B) + beta * op(C) + bias)
```

Today the epilogue is selected by a single combinatorial enum `hipblasLtEpilogue_t`
(e.g. `HIPBLASLT_EPILOGUE_RELU_AUX_BIAS`). That model does not scale to ordered chains of
post-GEMM operations such as `GEMM -> residual -> RMSNorm -> FP8 requant`, because every
combination would require a distinct enum value.

This design defines:

1. RMSNorm as the first fused, composable epilogue stage.
2. An explicit, ordered epilogue-chain descriptor so that allowed and disallowed stage
   sequences can be expressed and validated at API-call time, before any kernel selection
   or launch.
3. Reserved extension points for adjacent epilogue components and additional epilogue families
   without requiring new combinatorial enum values.

## 2. RMSNorm definition

For each row `x` of the post-GEMM tensor (optionally after a residual add), RMSNorm is:

```
y = x * rsqrt(mean(x^2) + eps) * gamma
```

- Normalization axis: per row of `D`, over the `N` (feature) dimension.
- Accumulation precision: FP32 for the sum of squares and the reciprocal sqrt, regardless
  of the storage type of `D`.
- `eps`: user-specified small constant added inside the rsqrt for numerical stability.
- `gamma`: required per-feature scale vector of length `N` (the column count of `D`). This
  corresponds to the RMSNorm affine scale.
- Unlike LayerNorm, RMSNorm does not subtract the mean and has no `beta`. It also does not
  emit `mean`/`invvar` side outputs (the standalone `hipblasltExtLayerNorm` does; the fused
  RMSNorm epilogue writes only the normalized result into `D`).

## 3. Composable-stage model

RMSNorm is modeled as a discrete epilogue *stage*, configured by its own descriptor
attributes, rather than as another value baked into the combinatorial `hipblasLtEpilogue_t`
enum. Its stage-specific inputs are `gamma` and `eps`. Other epilogue components, such as
residual add, AMax, and FP8 requant, can use the same descriptor mechanism with their own
stage-specific attributes instead of new enum combinations.

### 3.1 Stage taxonomy

Epilogue stages can be classified by how they affect the main output tensor and whether they
also produce side outputs. Compatibility and ordering rules are defined per supported chain
family:

| Category               | Main output effect                         | Stages                                  |
|------------------------|--------------------------------------------|-----------------------------------------|
| Shape-preserving       | `[M,N] -> [M,N]`                            | bias, activation, residual add, RMSNorm |
| Side-output reduction  | `[M,N] -> [M,N]` plus scalar/row side output | AMax capture                            |
| Type-changing          | `[M,N] -> [M,N]` with different storage type | FP8 requant                             |
| Shape-changing         | `[M,2N] -> [M,N]`                           | GLU / SwiGLU (split-activate-gate)      |

RMSNorm performs an internal row reduction to compute the reciprocal RMS scale, but its
main output remains `[M,N]`. AMax capture is different: it records a reduction result as a
side output while leaving the main tensor available to later stages.

The supported chain family in this design is the RMSNorm path. Shape-changing stages such as
GLU are reserved as an orthogonal extension point and are not part of the RMSNorm ordering
rule. If added later, they need their own compatibility checks because the GEMM output shape
differs from the final `D` shape.

### 3.2 RMSNorm chain order and the legality rule

The library defines a supported order for the RMSNorm chain:

```
bias -> residual add -> RMSNorm -> AMax -> FP8 requant
```

A user-specified RMSNorm chain is **legal** if and only if it is an order-preserving
subsequence of this supported order, with each stage appearing at most once. This one rule
produces the allowed/disallowed examples:

- Allowed: `GEMM + residual + RMSNorm + FP8 requant`
  (subsequence of the supported RMSNorm order)
- Disallowed: `GEMM + residual + FP8 requant + RMSNorm`
  (requant precedes RMSNorm, violating the supported RMSNorm order)

## 4. API surface

The composable chain is built through an opaque, handle-based builder rather than baking
each combination into the flat `hipblasLtEpilogue_t` enum or carrying an `int32_t[]` token
array on the matmul descriptor. The builder makes the order explicit, keeps stage parameters
attached to the stage that owns them, and gives a single attachment point on the matmul
descriptor.

### 4.1 Fuseable-epilogue enum

```c
typedef enum {
  HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD = 0, // reserved component
  HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM      = 1, // RMSNorm
  HIPBLASLT_FUSEABLE_EPILOGUE_AMAX         = 2, // reserved component
  HIPBLASLT_FUSEABLE_EPILOGUE_FP8_REQUANT  = 3, // reserved component
  HIPBLASLT_FUSEABLE_EPILOGUE_SWIGLU       = 4, // reserved epilogue family
} hipblasLtFuseableEpilogue_t;
```

These enum *values* are stable identifiers and are deliberately independent of pipeline
order; ordering for the RMSNorm chain (section 3.2) is enforced by an internal rank map, not
by the numeric value. New components are added by appending enum values without renumbering.
`HIPBLASLT_FUSEABLE_EPILOGUE_SWIGLU` is reserved for a separate epilogue family and is not a
legal member of the RMSNorm chain.

### 4.2 Builder handle and functions

```c
typedef struct hipblasLtFusedEpilogueDescriptor* hipblasLtFusedEpilogueDescriptor_t;

hipblasStatus_t hipblasLtFusedEpilogueCreate(hipblasLtFusedEpilogueDescriptor_t* desc);
hipblasStatus_t hipblasLtFusedEpilogueAdd(hipblasLtFusedEpilogueDescriptor_t desc,
                                          hipblasLtFuseableEpilogue_t        epilogue);
hipblasStatus_t hipblasLtFusedEpilogueSetAttribute(hipblasLtFusedEpilogueDescriptor_t desc,
                                                   hipblasLtFusedEpilogueAttribute_t  attr,
                                                   const void*                        value,
                                                   size_t                             sizeInBytes);
hipblasStatus_t hipblasLtFusedEpilogueDestroy(hipblasLtFusedEpilogueDescriptor_t desc);
```

`Add` accumulates stages in call order. The chain is attached to a matmul descriptor with a
single new attribute:

| Attribute                            | Type                                 | Meaning                       |
|--------------------------------------|--------------------------------------|-------------------------------|
| `HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE` | `hipblasLtFusedEpilogueDescriptor_t` | Attach a built fused-epilogue chain |

The descriptor stores a non-owning pointer to the handle; the caller owns the handle and
destroys it after the matmul(s) that use it.

### 4.3 Stage-specific attributes

Stage parameters are set on the handle (not on the matmul descriptor), so they travel with
the stage that consumes them:

| Attribute                              | Type    | Stage   | Meaning                              |
|----------------------------------------|---------|---------|--------------------------------------|
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA` | `void*` | RMSNorm | Device pointer to gamma, length `N` |
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS`   | `float` | RMSNorm | Epsilon inside the rsqrt             |

Residual-add, AMax, and FP8-requant stage parameters (residual pointer/layout, `AMAX_D` /
`D_SCALE` reuse) are not RMSNorm attributes; they require their own stage-specific
attributes.

### 4.4 Usage sketch

```c
hipblasLtFusedEpilogueDescriptor_t fused;
hipblasLtFusedEpilogueCreate(&fused);
hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM);
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA,
                                   &gamma, sizeof(gamma));
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS,
                                   &eps, sizeof(eps));
hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &fused, sizeof(fused));
// ... hipblasLtMatmul(...) ...
hipblasLtFusedEpilogueDestroy(fused);
```

### 4.5 Datatype requirements

RMSNorm supports `D` storage in FP16, BF16, or FP32. The `gamma` vector uses the compute
type or the `D` storage type, and RMSNorm accumulation follows the FP32 rule in section 2.

## 5. Error conditions and return codes

| Condition                                                        | Return code                       | Detected at            |
|------------------------------------------------------------------|-----------------------------------|------------------------|
| Unrecognized or unsupported `hipblasLtFuseableEpilogue_t` passed to `Add` | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time             |
| `Add` breaks the supported RMSNorm order (out-of-order stage)    | `HIPBLAS_STATUS_INVALID_VALUE`    | `Add` time             |
| Duplicate stage added                                            | `HIPBLAS_STATUS_INVALID_VALUE`    | `Add` time             |
| Unknown attribute passed to `SetAttribute`                       | `HIPBLAS_STATUS_INVALID_VALUE`    | `SetAttribute` time    |
| RMSNorm stage present but `gamma` null or `eps` unset            | `HIPBLAS_STATUS_INVALID_VALUE`    | attach (`SetAttribute` of `FUSED_EPILOGUE`) |
| Attached fused epilogue without a matching kernel implementation | `HIPBLAS_STATUS_NOT_SUPPORTED`    | `hipblasLtMatmul`      |

The key requirement: an illegal *ordering* for the supported RMSNorm chain is rejected at the
API-call level, incrementally as stages are added, before any codegen or kernel launch.
RMSNorm completeness (`gamma` and `eps`) is validated when the handle is attached to the
matmul descriptor (the C-API wrapper can see the handle struct).

## 6. Interaction with existing descriptors and preferences

- A new opaque builder object (`hipblasLtFusedEpilogueDescriptor_t`) is introduced; it is
  attached to the existing `hipblasLtMatmulDesc_t` via a single attribute. The matmul
  descriptor stores only a non-owning pointer.
- `hipblasLtMatmulPreference_t` is unchanged. Workspace and SM-count hints apply as usual.
- The C++ extension API (`hipblaslt_ext::GemmEpilogue` / `GemmInputs`) is outside this
  document; the C handle API is the defined surface.

## 7. SwiGLU/GLU extensibility

GLU/SwiGLU is a separate epilogue family, not part of the RMSNorm chain order in section 3.2.
It can reuse the same fused-epilogue descriptor handle, `Add`/`SetAttribute` builder flow,
matmul-descriptor attachment point, and stage-specific attribute model. It still requires
its own validation because it is shape-changing: the GEMM produces `[M, 2N]` for the packed
gate/up projection, while the GLU stage emits the final `[M, N]` tensor.
