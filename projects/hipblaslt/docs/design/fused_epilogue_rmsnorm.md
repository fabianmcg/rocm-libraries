# Fused Epilogue Extensions: RMSNorm and Composable Epilogue Chains

This document specifies the hipBLASLt API extensions for fused epilogues, starting with
RMSNorm, and the composable epilogue-chain mechanism that lets RMSNorm compose with
optional residual add, AMax capture, and FP8 requantization. It also reserves the design points
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
2. Two RMSNorm flows built on a shared partial-reduction producer: a full RMSNorm flow that
   applies the per-row scale immediately, and a decomposed `GEMM -> residual -> RMSNorm ->
   GEMM` flow that defers the scale into the consuming GEMM's epilogue.
3. An explicit, ordered epilogue-chain descriptor so that allowed and disallowed stage
   sequences can be expressed and validated at API-call time, before any kernel selection
   or launch.
4. Reserved extension points for adjacent epilogue components and additional epilogue families
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

When a residual-add stage precedes RMSNorm, the chain has two relevant values:

```
z = gemm_out + residual
y = RMSNorm(z, gamma, eps)
```

The residual-add stage produces the intermediate sum `z`; RMSNorm then consumes `z` and
produces the normalized value `y`. The normalized value `y` remains the main output written
to `D`, while `z` is the updated residual stream that transformer blocks carry forward. The
fused epilogue therefore needs an explicit write-back rule for `z`, independent of the main
`D` output.

## 3. Cross-tile RMSNorm via partial reduction

Because RMSNorm reduces over the full `N` feature dimension of each row, computing it inside a
single GEMM epilogue only works when a whole output row lives in one workgroup, i.e. when the
N-direction macro tile (`MacroTile1`) covers all of `N`. In the transformer shapes this fusion
targets, `N` is much larger than `MacroTile1`, so each row is split across
`ceil(N / MacroTile1)` workgroups and no single workgroup holds the whole row needed to
compute `mean(x^2)`. RMSNorm therefore cannot be a single tile-local epilogue in general; it
is realized as a producer epilogue that emits tile-local partial statistics, plus a
lightweight cross-tile reduction.

This section adopts the GEMM-epilogue reparameterization idea from CODA ([arXiv:2605.19269](https://arxiv.org/abs/2605.19269)):
emit tile-local partial statistics in the first GEMM's epilogue, combine them with a lightweight
reduction, and either apply the per-row scale immediately (full RMSNorm) or defer it into the
epilogue of a *consuming* GEMM (`GEMM -> residual -> RMSNorm -> GEMM`, for example attention
out-projection -> residual -> RMSNorm -> MLP gate/up projection).

### 3.1 Algebraic basis

The RMSNorm reciprocal scale, also commonly called `rstd`, is `r = rsqrt(mean(x^2) + eps)`,
a single scalar per row. Because the following projection is linear and `r` is shared across
the row, the scale commutes with the second GEMM:

```
y = (r ⊙ ((x @ W0 + z) ⊙ gamma)) @ W1
  = r ⊙ (((x @ W0 + z) ⊙ gamma) @ W1)
```

Here `@` denotes matrix multiplication and `⊙` denotes elementwise multiplication with
broadcasting: `gamma` is broadcast across rows, and `r` is broadcast across columns. Since
`r` has one value per row, it can be applied before or after the second GEMM. This lets the
normalization reduction be split out of the critical dependency between the two GEMMs.

### 3.2 Kernel organization for the two flows

Both flows share the same GEMM producer. They differ in what happens after the cross-tile
reduction: the full flow applies the scale immediately, while the decomposed flow returns
the per-row scale for a subsequent GEMM to apply.

**Kernel 1 (producer, shared by both flows).** The GEMM plus its epilogue:

```
h0   = x @ W0
h1   = h0 + z              # residual add (tile-local)
h2   = h1 ⊙ gamma         # RMSNorm weight (tile-local)
r_hat = partialRMS(h1)     # tile-local partial sum-of-squares
```

The producer emits two independent results from `h1`:

- `h2 = h1 ⊙ gamma`, the downstream value. The `gamma` multiply is tile-local.
- `r_hat = partialRMS(h1)`, a tile-local partial sum-of-squares over `h1`. This is a raw
  statistic, not a partially normalized output, and it is taken before the `gamma` multiply.

The mean (division by `d`) and the `rsqrt` are deliberately left out of the producer: they can
only be computed once the partials from all tiles in a row are combined, so they happen after
the cross-tile reduction.

**Full RMSNorm flow (2 kernels).** Kernel 2 reduces the partials and applies the row-wise
scale, optionally fusing quantization:

```
C = h2 ⊙ rsqrt(reduce(r_hat) / d + eps)   # reduce + apply (+ optional quant)
```

The result `C` is `RMSNorm(GEMM)`. This flow is exposed as a single `RMSNorm` stage and hides
both kernels behind one `hipblasLtMatmul` call.

**Decomposed flow feeding a GEMM (3 kernels).** Kernel 2 reduces the partials and returns the
per-row scale (it does not apply it); Kernel 3 is the second GEMM whose epilogue applies the
deferred scale:

```
Kernel 2:  r = rsqrt(reduce(r_hat) / d + eps)   # reduce + return r (no apply)
Kernel 3:  h3 = h2 @ W1;   y = r ⊙ h3           # GEMM2 + CODA scale-apply epilogue
```

Kernel 1 and the GEMM in Kernel 3 are both GEMM kernels produced by the single codegen option,
selecting different epilogue options. The reduction (Kernel 2) is a small custom kernel; it does
slightly different work in the two flows (apply vs return). The per-row scale and all
tiling-dependent metadata are carried between calls in an opaque descriptor, so the caller never
has to compute or observe tile-shape details such as the partial-buffer column count.

### 3.3 Choosing a flow

The caller selects the flow explicitly through the epilogue stages it adds, based on its own
use case; the library never substitutes one flow for the other based on a shape heuristic.

- Add a single `RMSNorm` stage for the full flow when the normalized result is the final
  output of interest. The library realizes it as producer + reduce-and-apply internally.
- Add `partial RMSNorm stats` on the first GEMM and `RMSNorm scale-apply` on the second GEMM
  for the decomposed flow when the RMSNorm output directly feeds another matmul. This removes
  the standalone RMSNorm pass and defers the scale into the consuming GEMM's epilogue.

Both flows run the same producer plus a cross-tile reduction, so launch counts are comparable. The
decomposed flow writes only the per-row scale `[M]` and folds the scale-apply into the consuming
GEMM's epilogue, avoiding the write and re-read of the full `[M, N]` normalized tensor that the
full flow materializes. That bandwidth saving grows with `M` and `N`.

### 3.4 Reduction backend (internal)

The "combine partials -> rstd" step is the same class of cross-workgroup reduction that
Stream-K already performs to combine partial output tiles, so its workspace, flag/synchronizer
handshake, and device-scope fences can be reused. The combine is realized as a custom kernel
and its backend (for example atomic accumulation, an in-kernel tree fixup, or a separate
reduction kernel) is an internal implementation choice, not a user-selected option.

### 3.5 Numerics

Applying `r` after the projection (decomposed flow) changes rounding relative to applying it
before. Partial sums of squares, the reciprocal square root, and the row-wise scale `r`/`rstd`
are kept in FP32 regardless of storage type: the epilogue multiplies by the FP32 `rstd` and only
then casts the result to the requested output type. CODA paper reports this delayed-scale
reparameterization on BF16 Llama-style layers against an FP32 reference and does not observe a
numerical regression. hipBLASLt should still validate both flows against an FP32 reference 
for the target model shapes and dtypes before enabling them by default.

## 4. Composable-stage model

RMSNorm is modeled as a discrete epilogue *stage*, configured by its own descriptor
attributes, rather than as another value baked into the combinatorial `hipblasLtEpilogue_t`
enum. Its stage-specific inputs are `gamma` and `eps`. Other epilogue components, such as
residual add, AMax, and FP8 requant, can use the same descriptor mechanism with their own
stage-specific attributes instead of new enum combinations. The decomposed cross-tile flow of
section 3 is expressed with the same mechanism, using two additional stages plus an opaque
handoff descriptor.

### 4.1 Stage taxonomy

Epilogue stages can be classified by how they affect the main output tensor and whether they
also produce side outputs. Compatibility and ordering rules are defined per supported chain
family:

| Category              | Main output effect                                           | Stages                                  |
|-----------------------|--------------------------------------------------------------|-----------------------------------------|
| Shape-preserving      | `[M,N] -> [M,N]`                                             | bias, activation, residual add, RMSNorm |
| Side-output reduction | `[M,N] -> [M,N]` plus per-row side output                    | AMax capture, partial RMSNorm stats     |
| Deferred scale        | `[M,N] -> [M,N]` scaled by a per-row vector produced earlier | RMSNorm scale-apply                     |
| Type-changing         | `[M,N] -> [M,N]` with different storage type                 | FP8 requant                             |
| Shape-changing        | `[M,2N] -> [M,N]`                                            | GLU / SwiGLU (split-activate-gate)      |

RMSNorm (single-stage) performs an internal row reduction to compute the reciprocal RMS scale,
but its main output remains `[M,N]`. AMax capture records a reduction result as a side output
while leaving the main tensor available to later stages.

The decomposed flow of section 3 adds two stages in this taxonomy:

- **Partial RMSNorm stats** is a side-output reduction producer: its main output is the
  tile-local value `h1 ⊙ gamma` written by GEMM1, and its side output is the per-row RMSNorm
  scale, produced by the internal cross-tile reduction and stored in an opaque handoff
  descriptor (section 5.2).
- **RMSNorm scale-apply** is a deferred-scale consumer: it multiplies the second GEMM's
  accumulator by the per-row scale carried in the handoff descriptor. It performs no reduction
  of its own.

The supported chain families in this design are the full RMSNorm path and the decomposed
RMSNorm path. Shape-changing stages such as GLU are reserved as an orthogonal extension point
and are not part of either ordering rule.

### 4.2 RMSNorm chain order and the legality rule

The library defines a supported order for the single-call RMSNorm chain:

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

### 4.3 Decomposed cross-call chains

The decomposed flow is not a single chain on one matmul call; it is two chains on two matmul
calls, linked by an opaque RMSNorm handoff descriptor (section 5.2):

- **Producer chain (GEMM1):** `bias -> residual add -> partial RMSNorm stats`. The supported
  order places the partial-stats stage after residual add, since it reduces the post-residual
  value before `gamma` scaling.
- **Consumer chain (GEMM2):** `RMSNorm scale-apply`. The GEMM2 epilogue only applies the
  deferred per-row scale to complete the normalization on GEMM2's output; it adds no further
  stages.

In this design, FP8 activation quantization for the consuming GEMM is input preparation for
that GEMM, so it lives in GEMM2's prologue / mainloop input path, not in either decomposed
epilogue chain above. Quantization of a full-RMSNorm output, by contrast, fuses into the
apply step of the full flow.

The section 4.2 legality rule applies within each chain independently; the
producer-before-consumer ordering is a data dependency through the handoff descriptor, not a
within-chain rank. A single chain uses either `HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM` or the
decomposed producer/consumer stages, never both.

## 5. API surface

The composable chain is built through an opaque, handle-based builder rather than baking
each combination into the flat `hipblasLtEpilogue_t` enum or carrying an `int32_t[]` token
array on the matmul descriptor. The builder makes the order explicit, keeps stage parameters
attached to the stage that owns them, and gives a single attachment point on the matmul
descriptor.

### 5.1 Fuseable-epilogue enum

```c
typedef enum {
  HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD          = 0, // residual add component
  HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM               = 1, // full RMSNorm (library realizes internally)
  HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS = 2, // GEMM1: producer, emits per-row RMSNorm stats
  HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY   = 3, // GEMM2: apply deferred per-row RMSNorm scale
  HIPBLASLT_FUSEABLE_EPILOGUE_AMAX                  = 4, // reserved component
  HIPBLASLT_FUSEABLE_EPILOGUE_FP8_REQUANT           = 5, // reserved component
  HIPBLASLT_FUSEABLE_EPILOGUE_SWIGLU                = 6, // reserved epilogue family
} hipblasLtFuseableEpilogue_t;
```

These enum values are independent of pipeline order; chain ordering (sections 4.2 and 4.3) is
enforced by an internal rank map, not by the numeric value.
`HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS` and
`HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY` are the decomposed-flow stages of section 3.
They are attached to two different matmul descriptors and are linked by the opaque RMSNorm
handoff descriptor.

Bias is intentionally absent from this enum. It continues to be configured through the
existing matmul descriptor attributes (`HIPBLASLT_MATMUL_DESC_BIAS_POINTER` and
`HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`). When a fused-epilogue chain is attached, the
library applies bias first (if set on the matmul descriptor), then enters the fused chain.
The ordering in sections 4.2 and 4.3 reflects this: bias precedes residual add logically but
is not a member of the builder-managed chain.

### 5.2 Builder handle and functions

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

The decomposed flow additionally uses an opaque **RMSNorm handoff descriptor**. Like
`hipblasLtFusedEpilogueDescriptor_t`, it is an opaque handle, but it is library-populated rather
than user-configured: the caller only creates it, passes it into the producer and consumer calls,
and destroys it, never setting or reading its fields directly.

```c
typedef struct hipblasLtFusedEpilogueRMSNormDescriptor* hipblasLtFusedEpilogueRMSNormDescriptor_t;

hipblasStatus_t hipblasLtFusedEpilogueRMSNormDescriptorCreate(hipblasLtFusedEpilogueRMSNormDescriptor_t* desc);
hipblasStatus_t hipblasLtFusedEpilogueRMSNormDescriptorDestroy(hipblasLtFusedEpilogueRMSNormDescriptor_t desc);
```

The producer call (`partial RMSNorm stats`) writes the finalized per-row scale and all
tiling-dependent metadata into this descriptor; the consumer call (`RMSNorm scale-apply`)
reads them back. The fields it carries -- the per-row scale buffer, the partial-buffer column
count `ceil(N / MacroTile1)`, `1/d`, and the `h2` shape -- are internal and are not part of
the public API.

`Add` accumulates stages in call order. The chain is attached to a matmul descriptor with a
single new attribute:

| Attribute                              | Type                                 | Meaning                             |
|----------------------------------------|--------------------------------------|-------------------------------------|
| `HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE` | `hipblasLtFusedEpilogueDescriptor_t` | Attach a built fused-epilogue chain |

The matmul descriptor stores a non-owning pointer to the fused-epilogue handle; the caller
owns the handle. The handle must remain valid for as long as any matmul descriptor references
it, until that descriptor is destroyed or the fused-epilogue attribute is replaced or set to
`NULL`. For each `hipblasLtMatmul` call, a fused-kernel implementation must copy any needed
scalar values and device pointers from the host handle into the launch parameters before
returning; it must not retain the host handle for asynchronous device execution. Once no
descriptor references the handle and all `hipblasLtMatmul` calls that used it have returned,
the handle may be destroyed without waiting for the launched kernels to complete on the
device. A single handle may be shared across multiple matmul descriptors, but must not be
mutated while any referencing matmul call is in-flight on another thread.

For the decomposed flow, the producer chain (GEMM1) and consumer chain (GEMM2) are separate
fused-epilogue handles attached to separate matmul descriptors. The caller is responsible for
issuing the producer matmul before the consumer matmul, and for keeping the RMSNorm handoff
descriptor alive across both.

### 5.3 Stage-specific attributes

Stage parameters are set on the handle (not on the matmul descriptor), so they travel with
the stage that consumes them:

| Attribute                                           | Type                                        | Stage                       | Meaning                                                                                        |
|-----------------------------------------------------|---------------------------------------------|-----------------------------|------------------------------------------------------------------------------------------------|
| `HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER`         | `void*`                                     | residual add                | Non-null device pointer to the residual input tensor                                           |
| `HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER`  | `void*`                                     | residual add                | Optional device pointer that receives the updated residual stream (`NULL` or unset = in-place) |
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA`            | `void*`                                     | RMSNorm / partial stats     | Non-null device pointer to gamma, length `N`                                                   |
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS`              | `float`                                     | RMSNorm / partial stats     | Epsilon inside the rsqrt                                                                       |
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS`            | `hipblasLtFusedEpilogueRMSNormDescriptor_t` | partial stats / scale apply | Opaque handoff object; producer writes it, consumer reads it (same object on both handles)     |

The residual input tensor has the same logical shape `[M,N]`, layout, data type, batch
count, and batch stride as `D` unless a future extension adds an explicit residual layout
descriptor. This keeps the initial API narrow and matches the transformer residual-stream
case targeted by the first RMSNorm fusion.

`HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER` is optional. If the attribute is never
set on the handle, or if it is explicitly set to `NULL`, the residual input pointer is also
the write-back target and the updated residual stream is written in place. This lets callers
clear a previously configured separate output pointer without destroying the descriptor. If
it is set to a valid non-null device pointer, the updated residual stream is written there
instead. The output pointer may alias the residual input pointer for explicit in-place
operation. It must not alias `D` when a later stage such as RMSNorm writes a different main
output value to `D`; that invalid alias is rejected by kernel-specific validation once fused
kernels are wired in. The same restriction applies when a later single-call stage such as FP8
requant changes the main output storage value written to `D`.

For the decomposed flow, `gamma` and `eps` are set on the producer handle: `gamma` is applied
tile-locally in Kernel 1, and `eps` (together with `1/d`, which the library derives from the
problem shape) is consumed by the internal reduction that finalizes the per-row scale. The
handoff descriptor is set on both the producer and consumer handles and must refer to the same
object. The per-row scale and the partial sum-of-squares scratch are carried inside the
handoff descriptor and the matmul preference workspace; they are not caller-supplied buffers
and the caller never computes their shapes.

AMax and FP8-requant stage parameters (`AMAX_D` / `D_SCALE` reuse) are not RMSNorm or
residual-add attributes; they require their own stage-specific attributes.

### 5.4 Usage sketch

Full RMSNorm, exposed as a single `hipblasLtMatmul` call (the library runs the producer and
the reduce-and-apply internally):

```c
hipblasLtFusedEpilogueDescriptor_t fused;
hipblasLtFusedEpilogueCreate(&fused);
hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD);
hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM);
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER,
                                   &residual, sizeof(residual));
// Optional: omit this attribute to update `residual` in place.
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                   &residual_out, sizeof(residual_out));
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA,
                                   &gamma, sizeof(gamma));
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS,
                                   &eps, sizeof(eps));
hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &fused, sizeof(fused));
// ... hipblasLtMatmul(...) ...
// Internally launches GEMM producer + auxiliary reduce-and-apply kernel.
hipblasLtFusedEpilogueDestroy(fused);
```

Decomposed flow for `GEMM -> residual -> RMSNorm -> GEMM`, using the opaque handoff descriptor:

```c
// Handoff object shared by the producer and consumer matmul calls.
hipblasLtFusedEpilogueRMSNormDescriptor_t stats;
hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats);

// GEMM1 producer: residual add + gamma + partial RMSNorm stats.
hipblasLtFusedEpilogueDescriptor_t prod;
hipblasLtFusedEpilogueCreate(&prod);
hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD);
hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS);
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER,
                                   &residual, sizeof(residual));
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA,
                                   &gamma, sizeof(gamma));
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS,
                                   &eps, sizeof(eps));
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS,
                                   &stats, sizeof(stats));
hipblasLtMatmulDescSetAttribute(matmulDesc1, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &prod, sizeof(prod));
// ... hipblasLtMatmul(GEMM1) ...
// Internally launches GEMM1 producer + auxiliary reduce-and-return kernel.
// After the queued work completes, `stats` carries the per-row scale.

// GEMM2 consumer: apply the deferred per-row scale from `stats` in the epilogue.
hipblasLtFusedEpilogueDescriptor_t cons;
hipblasLtFusedEpilogueCreate(&cons);
hipblasLtFusedEpilogueAdd(cons, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY);
hipblasLtFusedEpilogueSetAttribute(cons, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS,
                                   &stats, sizeof(stats)); // same object as the producer
hipblasLtMatmulDescSetAttribute(matmulDesc2, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &cons, sizeof(cons));
// ... hipblasLtMatmul(GEMM2) ...
// launches GEMM2; its epilogue reads `stats`

hipblasLtFusedEpilogueDestroy(prod);
hipblasLtFusedEpilogueDestroy(cons);
hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
```

### 5.5 Datatype requirements

The initial release targets BF16/FP16 storage for the RMSNorm input and normalized output. The
residual input and residual-output write-back use the same storage type as the residual-add
sum. The `gamma` vector must use the same storage type as the RMSNorm input/pre-requant value,
not necessarily the final `D` storage type when a later single-call FP8-requant stage changes
the main output type. Regardless of the BF16/FP16 storage type, RMSNorm accumulation uses FP32
for the sum of squares and the reciprocal square root.

For the decomposed flow, the internal partial-stats scratch and the per-row scale carried in
the handoff descriptor are FP32, independent of the GEMM storage types, so the cross-tile
combine and the deferred scale match the full-flow accumulation precision.

### 5.6 Strided-batched semantics

When `batch_count > 1` in a strided-batched GEMM:

- `gamma` is broadcast across batches (a single vector of length `N` shared by all batches).
- `eps` is a scalar and is always shared across batches.
- The residual input tensor follows the same batch stride as `D`. If
  `HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER` is set, the residual output also uses
  the same batch stride as `D`.
- For the decomposed flow, the per-row scale is indexed per `(row, batch)` inside the handoff
  descriptor. The per-batch row count `M` and the batch count are derived from the matmul
  descriptor, so the producer and consumer stay consistent without any caller-supplied layout.

## 6. Error conditions and return codes

| Condition                                                                 | Return code                    | Detected at                                 |
|---------------------------------------------------------------------------|--------------------------------|---------------------------------------------|
| Unrecognized or unsupported `hipblasLtFuseableEpilogue_t` passed to `Add` | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| `Add` breaks the supported RMSNorm order (out-of-order stage)             | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Duplicate stage added                                                     | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Full and decomposed RMSNorm stages mixed in one chain                     | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Unknown attribute passed to `SetAttribute`                                | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time                         |
| Required pointer attribute explicitly set to `NULL`                       | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time                         |
| Residual-add stage present but residual pointer unset                     | `HIPBLAS_STATUS_INVALID_VALUE` | attach (`SetAttribute` of `FUSED_EPILOGUE`) |
| RMSNorm stage present but `gamma` or `eps` unset                          | `HIPBLAS_STATUS_INVALID_VALUE` | attach (`SetAttribute` of `FUSED_EPILOGUE`) |
| Partial-stats or scale-apply stage present but stats descriptor unset     | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| Scale-apply consumes a stats descriptor not populated by a producer       | `HIPBLAS_STATUS_INVALID_VALUE` | `hipblasLtMatmul`                           |
| Attached fused epilogue without a matching kernel implementation          | `HIPBLAS_STATUS_NOT_SUPPORTED` | `hipblasLtMatmul`                           |

The key requirement: an illegal *ordering* for a supported RMSNorm chain is rejected at the
API-call level, incrementally as stages are added, before any codegen or kernel launch.
Residual completeness (`residual`), RMSNorm completeness (`gamma` and `eps`), and
decomposed-flow completeness (a set stats descriptor) are validated when the handle is
attached to the matmul descriptor via `HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE`. This means
required stage-specific attributes must be set on the handle before attachment and must not be
cleared or mutated after attachment. The matmul descriptor stores a non-owning pointer to the
handle, so later optional-attribute updates are visible to `hipblasLtMatmul`, but callers
should not rely on late updates to satisfy attach-time required-attribute validation. For the
decomposed flow, the producer-before-consumer dependency on the handoff descriptor can only be
fully validated at `hipblasLtMatmul` time.

## 7. Interaction with existing descriptors and preferences

- A new opaque builder object (`hipblasLtFusedEpilogueDescriptor_t`) is introduced; it is
  attached to the existing `hipblasLtMatmulDesc_t` via a single attribute. The matmul
  descriptor stores only a non-owning pointer. The decomposed flow adds one more opaque handle
  type (`hipblasLtFusedEpilogueRMSNormDescriptor_t`) that is only passed across calls, never
  inspected by the caller.
- `hipblasLtMatmulPreference_t` is unchanged. Workspace and SM-count hints apply as usual.
  The decomposed flow has no caller-allocated normalization buffers: the per-row scale rides
  in the handoff descriptor, and the partial-stats scratch plus any synchronizer/flag buffer
  are internal and drawn from the matmul preference workspace, sized via the workspace-size
  query.
- On the codegen side, the GEMM kernels for both flows come from a single TensileLite option
  (`FusedEpilogues = 1`) that bundles the optional-residual, RMSNorm-producer, and
  RMSNorm-scale-apply epilogue options and selects among them at runtime. The cross-tile
  reduction is a separate custom kernel. This keeps the number of generated GEMM kernels
  bounded (no combinatorial explosion) at the cost of larger kernels.
- The C++ extension API (`hipblaslt_ext::GemmEpilogue` / `GemmInputs`) is outside this
  document; the C handle API is the defined surface.

## 8. SwiGLU/GLU extensibility

GLU/SwiGLU is a separate epilogue family, not part of the RMSNorm chain order in section 4.2.
It can reuse the same fused-epilogue descriptor handle, `Add`/`SetAttribute` builder flow,
matmul-descriptor attachment point, and stage-specific attribute model. It still requires
its own validation because it is shape-changing: the GEMM produces `[M, 2N]` for the packed
gate/up projection, while the GLU stage emits the final `[M, N]` tensor.
