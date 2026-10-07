# MegaFusedEpilogue: Tile Math and Register Layout

<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

This document describes, at the level of tile math and register layout, what the
**MegaFusedEpilogue** component computes. It is scoped tightly to that one component: the
generic hipBLASLt epilogue chain (bias, per-row/per-column alpha scaling, beta·C, activation)
is a separate, unrelated code path and is **not** part of MegaFusedEpilogue's own computation.
MegaFusedEpilogue does not read or apply bias, and it does not apply any per-row/per-column
scale vector analogous to bias. It only ever touches: the raw matrix-accumulator tile, one
residual input tensor, one per-row weight vector ("gamma"), and (only for one destination
type) the output's global scale factor and a block quantizer.

## 1. What MegaFusedEpilogue computes

MegaFusedEpilogue runs once per output tile, directly on the raw matrix-multiply accumulator,
before any of the generic epilogue stages. For every element of the tile — identified by its
row position and column position — it performs, in this order:

1. **Residual add.** `H = Acc + Residual`, where `Acc` is the raw accumulator value and
   `Residual` is the matching element of a dedicated residual input tensor (same logical shape
   as the output tile; not related to the generic epilogue's beta-scaled `C` operand).
2. **Sum-of-squares accumulation.** For each column, the component adds `H²` into a
   running per-column scalar accumulator. This happens **before** the next step (i.e. the square
   of the pre-scaled residual sum, not of the final scaled value).
3. **Residual-stream output.** `H` itself (not yet scaled) is written out to a separate output
   tensor, independent of the primary output. This is the updated running total that a
   surrounding model would carry forward as its next residual stream.
4. **Per-row scale.** The accumulator is overwritten in place: `Acc := H * gamma[row]`,
   where `gamma` is a vector of length equal to the full row axis, broadcast identically
   across every column and, when the problem is batched, across every batch. This is
   the per-row weight of a norm layer, applied *before* that norm's own division step —
   the division/normalization itself is explicitly deferred (see §3).
5. **Block-amax fold, only for one destination type.** When (and only when) the output tile's
   destination element type is 8-bit floating point, the component additionally folds
   `|H * gamma|` into a running per-quantization-block maximum-magnitude accumulator, used to
   derive that block's quantization scale later in the same pass.

Steps 1-4 are unconditional whenever MegaFusedEpilogue is active; step 5 is conditional on the
8-bit-float destination case described next.

**Everything this component does *not* do:** it does not add a bias vector, it does not apply
any per-row/per-column "alpha vector" scaling, it does not evaluate an activation function, and
(outside of the 8-bit-float case below) it does not apply the GEMM's global `alpha`/`beta`
scalars. All of those belong exclusively to the generic epilogue pipeline, which — for every
destination type except the one case below — runs afterward on the mutated accumulator exactly
as if it were the original raw matrix-multiply result.

### The 8-bit-float destination case

When the destination element type is an 8-bit float, MegaFusedEpilogue additionally takes over
responsibility for applying the GEMM's global `alpha` scalar and for quantizing its own output,
and the generic pipeline's `beta` path is disabled entirely for this configuration (beta is
required to be zero). Concretely, after the per-block maximum magnitude (step 5) has been
reduced to completion across the whole quantization block (see §2.2), the component: folds
`alpha` into that per-block maximum, derives a per-block power-of-two quantization multiplier
from it, re-reads each accumulator element, multiplies it by `alpha * (that multiplier)`, writes
the result back to the accumulator, and stores one quantization-scale byte per block to a
dedicated scale tensor. For every other destination type, `alpha`/`beta` are left entirely to
the generic pipeline, which then operates on `H * gamma` as if it were the raw accumulator.

### Pseudocode: the per-tile computation

The steps above, restated as sequential pseudocode over logical tile indices rather than as a
parallel execution schedule (the schedule — which lane and register holds which element — is
covered separately in §4). `r` ranges over the tile's row positions, `c` over its column
positions; `block(r)` is the quantization block a given row belongs to.

```
for each column c in the tile:
    for each row r in the tile:
        H[r, c]             = Acc[r, c] + Residual[r, c]     # 1. residual add
        ssqPartial[c]       += H[r, c]^2                     # 2. sum-of-squares over H, pre-gamma
        ResidualStream[r,c] = H[r, c]                        # 3. residual-stream output, pre-gamma
        Acc[r, c]           = H[r, c] * gamma[r]             # 4. per-row scale, overwrites Acc

        if destination element type is 8-bit float:
            blockAmax[block(r)] = max(blockAmax[block(r)], |Acc[r, c]|)   # 5. block-amax fold

# ssqPartial[c] is partial: it only covers this tile's slice of the row axis (see §3).

if destination element type is 8-bit float:
    for each quantization block b:
        quantMult[b]   = powerOfTwo(alpha * blockAmax[b])     # reciprocal-direction scale: nearest power of two that rescales this block's magnitude down into fp8 range, not an approximation of the magnitude itself.
        scaleTensor[b] = quantMult[b]                         # one scale byte stored per block
        for each (r, c) with block(r) == b:
            Acc[r, c] = Acc[r, c] * alpha * quantMult[b]
```

## 2. Row-axis reduction: computing one workgroup's partial statistic

The per-column sum of squares needed by a norm layer must be summed over the *entire* row
axis, which is wider than one tile: a single column's full row vector is normally split
across many workgroups along that axis, each holding only a slice of it. MegaFusedEpilogue
therefore only ever produces a **partial** sum — complete over its own workgroup's slice of the
row axis, not the whole row axis — and writes that partial out to a scratch buffer for
a later stage to finish combining.

Producing that partial, for one column, proceeds in four steps:

1. **Lane-local accumulation (free).** Each lane of the GPU wavefront owns a fixed, disjoint set
   of rows for the whole tile (its exact assignment is derived in §4). It simply
   keeps adding `H²` for every row it owns into one running register — no cross-lane
   communication needed, since the rows are disjoint by construction.
2. **Intra-wave butterfly.** One MFMA instruction's output is spread across several
   *row-groups* of lanes within the same wavefront, each row-group owning a different
   disjoint band of rows but the same set of columns. An XOR-butterfly over
   `log2(row-groups-per-wavefront)` rounds — each round exchanging partial sums with the lane
   whose row-group index differs in one bit, via a lane-shuffle, then adding — combines all
   row-groups' partials with no shared memory and no barrier. After this, every lane holding
   a given column's partial sum has the complete sum over its *wavefront's* share of the row
   axis.
3. **Cross-wave combine (only if more than one wave covers the row axis).** When a workgroup
   assigns more than one wave to cover different row-axis slices (as opposed to only
   different column slices), those sibling waves cannot shuffle directly between each other; each
   wave instead stores its partial to a per-wave slot in shared (workgroup-local) memory, a
   barrier ensures all slots are written, and then every wave reads and accumulates all
   sibling waves' slots. There is no analogous step across waves that instead split the *column*
   axis, since those waves already own disjoint columns that never need merging for a per-column
   statistic.
4. **Predicated write to global scratch.** Exactly one lane per column (the one holding the
   now-complete, workgroup-wide partial — i.e. row-group 0 of the wave owning row-offset
   0) writes that scalar to a global scratch buffer, indexed by (column, workgroup's row-axis
   tile index). This buffer holds one partial sum per column per row-axis tile; completing the
   reduction across *all* such tiles for one column is left to the stage described next.

## 3. What a later stage does with that output (not part of MegaFusedEpilogue itself)

MegaFusedEpilogue never computes the norm's actual divisor or applies it — by design, since no
single workgroup ever sees the complete per-column sum. A separate, much smaller kernel, launched
after every workgroup that is producing output for the same GEMM has finished, performs the
remaining two steps:

- **Finish the reduction:** sum the scratch buffer's partials across every row-axis tile
  belonging to one column (there are `ceil(row_extent / macro_tile_row_extent)` of them),
  using a cross-lane reduction within one small kernel launch.
- **Compute and apply the scale:** `scale = 1/sqrt(eps + mean)`, where `mean` is that completed
  sum divided by the total row count, and `eps` is a small stabilizing constant. This
  `scale` is then multiplied, in place, into the primary output tile that MegaFusedEpilogue
  already wrote back to memory — i.e. the per-row-scaled value `H * gamma` from step 4 (not
  the separate, pre-gamma residual-stream tensor from step 3) — either directly materializing
  the final normalized result, or — in a variant where normalization is decomposed from a
  following linear layer — returned as a per-column scalar to be folded into a *different* GEMM's
  own epilogue later (valid because a single scalar per column commutes through a subsequent
  linear projection along the row axis).

As pseudocode, continuing the per-column partial sums MegaFusedEpilogue left behind (`numTiles`
is `ceil(row_extent / macro_tile_row_extent)`, one row-axis tile per workgroup):

```
# Finishing step -- a separate, later kernel; not part of MegaFusedEpilogue itself.
for each column c:
    sum[c]  = sum over tile in [0, numTiles) of ssqPartial[c, tile]   # finish the reduction
    mean[c] = sum[c] / row_extent
    scale[c] = 1 / sqrt(eps + mean[c])

    for each row r:
        # Acc here is the per-row-scaled value (H * gamma) that MegaFusedEpilogue's step 4
        # already wrote to the primary output tile; the finishing kernel re-reads and overwrites
        # it in place. H itself was only a loop-local value in section 1 and is not available here.
        normalized[r, c] = Acc[r, c] * scale[c]
    # or, in the decomposed variant, scale[c] alone is forwarded to a later GEMM's epilogue.
```

This finishing stage, and any further requantization it may also perform for an 8-bit-float
destination, is explicitly **not** part of what MegaFusedEpilogue itself computes; it is listed
here only because MegaFusedEpilogue's partial-sum output is otherwise meaningless on its own.

## 4. Register and lane layout

This section gives the exact rule for where one tile element lives: which lane holds it, and
which register (and offset within that register's row-run) holds it.

### 4.1 Parameters

- `MI_M`, `MI_N`: the M and N dimensions of one native matrix-instruction (MFMA) output tile.
- `W`: the wavefront size, always 64 for this component.
- `MT_0`, `MT_1`: the macro tile's M (row) and N (column) extents.
- `G_M`, `G_N`: the number of waves a workgroup replicates along the M and N directions
  respectively (the "wave group").
- `V_M`, `V_N`: output-side sub-tiling factors along M and N — each a power of two, 1 in the
  component's native/default configuration, greater than 1 only under an alternate
  wave-interleaved layout described below.

Derived per-wave tile-step counts (the "wave tile": how many instruction-sized steps one wave
covers along each axis):
```
T_M = (MT_0 / (MI_M * G_M)) / V_M
T_N = (MT_1 / (MI_N * G_N))
```

### 4.2 Lane assignment (per wavefront)

Within one wavefront, each lane's identity decomposes as:
```
c = lane mod MI_N              -- this lane's column within one instruction tile
g = floor(lane / MI_N)         -- this lane's row-group index; there are W / MI_N row-groups
```
Every lane in row-group `g` owns a contiguous run of
```
R = V_M * (MI_M * MI_N / W)
```
rows, offset by `g * R` from the start of that instruction tile's row range — i.e.
row-group 0 owns the first `R` rows, row-group 1 the next `R`, and so on, covering all
`MI_M` (times `V_M`) rows of one wave-tile step between the `W / MI_N` row-groups. The
same lane owns this same row-offset and column for every one of the `T_M x T_N` wave-tile
steps the wave executes.

### 4.3 Register placement

Each of the `T_M x T_N` wave-tile steps (indexed `m in [0, T_M)`, `n in [0, T_N)`) is backed by
its own contiguous bank of `R` accumulator registers per lane (physically, either the
hardware's matrix-accumulator register file or an ordinary vector-register mirror of it,
depending on a per-tile placement decision made when registers are allocated — the epilogue
itself is indifferent to which, reading/writing through whichever interface that tile uses).
Within that bank, local row index `k in [0, R)` is exactly register slot `k`. So: tile
element `(m, n, k)` lives in lane `g*MI_N + c`, in the `k`-th register of the `(m, n)` wave-tile
step's register bank — a direct, O(1) address, not a permutation.

When `V_M` or `V_N` exceed 1 (the alternate wave-interleaved layout, used by a second kernel
family that this component also supports), a lane's row-run additionally concatenates `V_M`
consecutive instruction-blocks back to back, and output columns are grouped `V_N` at a time
before the per-wave-tile-step index advances — the lane/row-group formulas above are
unchanged, only the row-run length `R` and the column grouping change.

### 4.4 Wave-to-tile arrangement

Two conventions exist for how the `G_M x G_N` waves of one workgroup divide up the macro tile,
selected per kernel family:

- **Block-contiguous:** each wave owns one contiguous block of `T_M` (resp. `T_N`) wave-tile
  steps; consecutive steps within a wave are `MI_M` (resp. `MI_N`) rows/columns apart, and
  different waves' blocks are stacked end-to-end (wave index `i`'s row origin is offset by
  `i * T_M * MI_M`).
- **Interleaved:** consecutive wave-tile steps within one wave are `G_M * V_M * MI_M` (resp.
  `G_N * V_N * MI_N`) apart, with different waves' steps interleaved at `V_M * MI_M` (resp.
  `V_N * MI_N`) granularity instead of being stacked in blocks.

Both conventions are handled by the same lane/register formulas above; only the step stride and
the wave's row/column origin differ.

## 5. Mapping to the global output matrix

The GEMM's output indices are conventionally labeled by position in the index-numbering order:
the free index belonging only to the left (first) operand is index 0, the free index belonging
only to the right (second) operand is index 1, a shared batch index (when present) is index 2,
and the remaining shared contraction/summation index is whichever position is left (index 2 if
unbatched, index 3 if batched). MegaFusedEpilogue's "row" axis is the left operand's free
index (M, index 0) and its "column" axis is the right operand's free index (N, index 1); the
per-row weight vector is indexed solely along the former and broadcast along the latter (and
across batches, when the problem is batched).

The problem/kernel naming convention assigns each index a letter by that same position (`I`,
`J`, `K`, `L`, ... in index-number order), then spells out each operand's own index order using
those letters, lower-cased when the stride runs in the natural (non-mirrored) direction. For the
operand-transpose combination MegaFusedEpilogue is actually deployed with in this codebase (left
operand transposed, right operand not transposed, batched) -- confirmed by the kernel/problem
names actually generated for kernels that enable it -- that works out to: `I` = the left
operand's free index (row/M), `J` = the right operand's free index (column/N), `K` = the
batch index, and `L` = the contraction/summation index. The letters are positional labels, not
fixed semantic names -- which physical axis plays `I` versus `L` depends on each operand's own
index order and transpose setting, not on the letter itself -- but for this component's
deployment the mapping is exactly row=`I`, column=`J`, batch=`K`, contraction=`L`.

Global position of tile element `(m, n, k)` (row-group `g`, lane column `c`):
```
row_position    = workgroup_row_origin + g*R + m*rowStride + k
column_position = workgroup_column_origin + n*columnStride + c
```
where `workgroup_row_origin = workgroupIndex_M * MT_0 (+ wave row origin if G_M > 1)` and
`workgroup_column_origin = workgroupIndex_N * MT_1 (+ wave column origin if G_N > 1)` — i.e. the
standard tiling convention where the workgroup's position in the output grid scales directly by
the macro tile's extents, and multiple workgroups' tiles cover the full output matrix by tiling
both axes independently.

The output (and the residual/residual-stream/scratch-partial tensors, which all share the same
tiling) are addressed with the row index (M axis) as the contiguous (fastest-varying)
dimension and the column index (N axis) as the strided dimension — each element's linear
offset is `column_position * row_extent + row_position`, times the element size. This is
confirmed directly by the component's own address arithmetic for its side tensors, not merely
assumed.

Elements are stored **column-major**: contiguous along the row axis (M), strided along the
column axis (N), with the (padded) row-axis extent as the leading dimension. This is the
conventional storage order for A, B, C, and D throughout this GEMM library, confirmed by the
problem's index-assignment convention: the M-axis (row) free index is always the lower-numbered,
fastest-moving index in C/D, with the N-axis (column) free index strided by the M extent.

Batch handling: MegaFusedEpilogue's own address arithmetic contains no batch-specific term; when
the problem is batched, the batch index is resolved upstream by the kernel's buffer base
addresses before this component ever runs, consistent with how the surrounding kernel generally
handles batching for every buffer it touches.

## 6. When this component is active

MegaFusedEpilogue is an optional, gated component: it requires a 16x16 matrix instruction, a
compatible accumulator-tile view (either the component's native layout or the alternate
interleaved layout), one specific GPU target generation, a restricted set of destination element
types (bf16, f16, or an 8-bit float), and the accumulator tile physically resident in the
hardware's matrix-accumulator register file. It also requires that any split-reduction
(multi-pass accumulation) or persistent-kernel-across-tiles scheme in use resolve to a complete,
fully-summed tile before this component ever runs — it is never reachable with a partial,
un-reduced accumulator. Residual add, the sum-of-squares accumulation, and the residual-stream
output are all unconditional once the component is active; only the block-amax fold and the
alpha/quantization step (§1) are conditional on the 8-bit-float destination case.

Two points were not confirmed from the validators read for this report and are left as open:
whether the generic pipeline's activation function can be requested together with
MegaFusedEpilogue, and whether the generic pipeline's beta·C path is meaningful in combination
with MegaFusedEpilogue for non-8-bit-float destinations (it is not explicitly forbidden there,
unlike the 8-bit-float case where it is required to be off).
