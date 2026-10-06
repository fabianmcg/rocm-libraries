# MegaFusedEpilogue: Tile Math and Register Layout

<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

This document describes, at the level of tile math and register layout, what the
**MegaFusedEpilogue** component computes. It is scoped tightly to that one component: the
generic hipBLASLt epilogue chain (bias, per-row/per-column alpha scaling, beta·C, activation)you 
is a separate, unrelated code path and is **not** part of MegaFusedEpilogue's own computation.
MegaFusedEpilogue does not read or apply bias, and it does not apply any per-row/per-column
scale vector analogous to bias. It only ever touches: the raw matrix-accumulator tile, one
residual input tensor, one per-column weight vector ("gamma"), and (only for one destination
type) the output's global scale factor and a block quantizer.

## 1. What MegaFusedEpilogue computes

MegaFusedEpilogue runs once per output tile, directly on the raw matrix-multiply accumulator,
before any of the generic epilogue stages. For every element of the tile — identified by its
row position and column position — it performs, in this order:

1. **Residual add.** `H = Acc + Residual`, where `Acc` is the raw accumulator value and
   `Residual` is the matching element of a dedicated residual input tensor (same logical shape
   as the output tile; not related to the generic epilogue's beta-scaled `C` operand).
2. **Sum-of-squares accumulation.** For each row, the component adds `H²` into a
   running per-row scalar accumulator. This happens **before** the next step (i.e. the square
   of the pre-scaled residual sum, not of the final scaled value).
3. **Residual-stream output.** `H` itself (not yet scaled) is written out to a separate output
   tensor, independent of the primary output. This is the updated running total that a
   surrounding model would carry forward as its next residual stream.
4. **Per-column scale.** The accumulator is overwritten in place: `Acc := H * gamma[column]`,
   where `gamma` is a vector of length equal to the full column axis, broadcast identically
   across every row and, when the problem is batched, across every batch. This is
   the per-column weight of a norm layer, applied *before* that norm's own division step —
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
covered separately in §4). `c` ranges over the tile's column positions, `r` over its row
positions; `block(c)` is the quantization block a given column belongs to.

```
for each row r in the tile:
    for each column c in the tile:
        H[c, r]             = Acc[c, r] + Residual[c, r]     # 1. residual add
        ssqPartial[r]       += H[c, r]^2                     # 2. sum-of-squares over H, pre-gamma
        ResidualStream[c,r] = H[c, r]                        # 3. residual-stream output, pre-gamma
        Acc[c, r]           = H[c, r] * gamma[c]             # 4. per-column scale, overwrites Acc

        if destination element type is 8-bit float:
            blockAmax[block(c)] = max(blockAmax[block(c)], |Acc[c, r]|)   # 5. block-amax fold

# ssqPartial[r] is partial: it only covers this tile's slice of the column axis (see §3).

if destination element type is 8-bit float:
    for each quantization block b:
        quantMult[b]   = powerOfTwo(alpha * blockAmax[b])     # reciprocal-direction scale: nearest power of two that rescales this block's magnitude down into fp8 range, not an approximation of the magnitude itself.
        scaleTensor[b] = quantMult[b]                         # one scale byte stored per block
        for each (c, r) with block(c) == b:
            Acc[c, r] = Acc[c, r] * alpha * quantMult[b]
```

## 2. Column-axis reduction: computing one workgroup's partial statistic

The per-row sum of squares needed by a norm layer must be summed over the *entire* column
axis, which is wider than one tile: a single row's full column vector is normally split
across many workgroups along that axis, each holding only a slice of it. MegaFusedEpilogue
therefore only ever produces a **partial** sum — complete over its own workgroup's slice of the
column axis, not the whole column axis — and writes that partial out to a scratch buffer for
a later stage to finish combining.

Producing that partial, for one row, proceeds in four steps:

1. **Lane-local accumulation (free).** Each lane of the GPU wavefront owns a fixed, disjoint set
   of columns for the whole tile (its exact assignment is derived in §4). It simply
   keeps adding `H²` for every column it owns into one running register — no cross-lane
   communication needed, since the columns are disjoint by construction.
2. **Intra-wave butterfly.** One MFMA instruction's output is spread across several
   *column-groups* of lanes within the same wavefront, each column-group owning a different
   disjoint band of columns but the same set of rows. An XOR-butterfly over
   `log2(column-groups-per-wavefront)` rounds — each round exchanging partial sums with the lane
   whose column-group index differs in one bit, via a lane-shuffle, then adding — combines all
   column-groups' partials with no shared memory and no barrier. After this, every lane holding
   a given row's partial sum has the complete sum over its *wavefront's* share of the column
   axis.
3. **Cross-wave combine (only if more than one wave covers the column axis).** When a workgroup
   assigns more than one wave to cover different column-axis slices (as opposed to only
   different row slices), those sibling waves cannot shuffle directly between each other; each
   wave instead stores its partial to a per-wave slot in shared (workgroup-local) memory, a
   barrier ensures all slots are written, and then every wave reads and accumulates all
   sibling waves' slots. There is no analogous step across waves that instead split the *row*
   axis, since those waves already own disjoint rows that never need merging for a per-row
   statistic.
4. **Predicated write to global scratch.** Exactly one lane per row (the one holding the
   now-complete, workgroup-wide partial — i.e. column-group 0 of the wave owning column-offset
   0) writes that scalar to a global scratch buffer, indexed by (row, workgroup's column-axis
   tile index). This buffer holds one partial sum per row per column-axis tile; completing the
   reduction across *all* such tiles for one row is left to the stage described next.

## 3. What a later stage does with that output (not part of MegaFusedEpilogue itself)

MegaFusedEpilogue never computes the norm's actual divisor or applies it — by design, since no
single workgroup ever sees the complete per-row sum. A separate, much smaller kernel, launched
after every workgroup that is producing output for the same GEMM has finished, performs the
remaining two steps:

- **Finish the reduction:** sum the scratch buffer's partials across every column-axis tile
  belonging to one row (there are `ceil(column_extent / macro_tile_column_extent)` of them),
  using a cross-lane reduction within one small kernel launch.
- **Compute and apply the scale:** `scale = 1/sqrt(eps + mean)`, where `mean` is that completed
  sum divided by the total column count, and `eps` is a small stabilizing constant. This
  `scale` is then multiplied, in place, into the primary output tile that MegaFusedEpilogue
  already wrote back to memory — i.e. the per-column-scaled value `H * gamma` from step 4 (not
  the separate, pre-gamma residual-stream tensor from step 3) — either directly materializing
  the final normalized result, or — in a variant where normalization is decomposed from a
  following linear layer — returned as a per-row scalar to be folded into a *different* GEMM's
  own epilogue later (valid because a single scalar per row commutes through a subsequent
  linear projection along the column axis).

As pseudocode, continuing the per-row partial sums MegaFusedEpilogue left behind (`numTiles`
is `ceil(column_extent / macro_tile_column_extent)`, one column-axis tile per workgroup):

```
# Finishing step -- a separate, later kernel; not part of MegaFusedEpilogue itself.
for each row r:
    sum[r]  = sum over tile in [0, numTiles) of ssqPartial[r, tile]   # finish the reduction
    mean[r] = sum[r] / column_extent
    scale[r] = 1 / sqrt(eps + mean[r])

    for each column c:
        # Acc here is the per-column-scaled value (H * gamma) that MegaFusedEpilogue's step 4
        # already wrote to the primary output tile; the finishing kernel re-reads and overwrites
        # it in place. H itself was only a loop-local value in section 1 and is not available here.
        normalized[c, r] = Acc[c, r] * scale[r]
    # or, in the decomposed variant, scale[r] alone is forwarded to a later GEMM's epilogue.
```

This finishing stage, and any further requantization it may also perform for an 8-bit-float
destination, is explicitly **not** part of what MegaFusedEpilogue itself computes; it is listed
here only because MegaFusedEpilogue's partial-sum output is otherwise meaningless on its own.

## 4. Register and lane layout

This section gives the exact rule for where one tile element lives: which lane holds it, and
which register (and offset within that register's column-run) holds it.

### 4.1 Parameters

- `MI_M`, `MI_N`: the M and N dimensions of one native matrix-instruction (MFMA) output tile.
- `W`: the wavefront size, always 64 for this component.
- `MT_0`, `MT_1`: the macro tile's M (column) and N (row) extents.
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
r = lane mod MI_N              -- this lane's row within one instruction tile
g = floor(lane / MI_N)         -- this lane's column-group index; there are W / MI_N column-groups
```
Every lane in column-group `g` owns a contiguous run of
```
R = V_M * (MI_M * MI_N / W)
```
columns, offset by `g * R` from the start of that instruction tile's column range — i.e.
column-group 0 owns the first `R` columns, column-group 1 the next `R`, and so on, covering all
`MI_M` (times `V_M`) columns of one wave-tile step between the `W / MI_N` column-groups. The
same lane owns this same column-offset and row for every one of the `T_M x T_N` wave-tile
steps the wave executes.

### 4.3 Register placement

Each of the `T_M x T_N` wave-tile steps (indexed `m in [0, T_M)`, `n in [0, T_N)`) is backed by
its own contiguous bank of `R` accumulator registers per lane (physically, either the
hardware's matrix-accumulator register file or an ordinary vector-register mirror of it,
depending on a per-tile placement decision made when registers are allocated — the epilogue
itself is indifferent to which, reading/writing through whichever interface that tile uses).
Within that bank, local column index `k in [0, R)` is exactly register slot `k`. So: tile
element `(m, n, k)` lives in lane `g*MI_N + r`, in the `k`-th register of the `(m, n)` wave-tile
step's register bank — a direct, O(1) address, not a permutation.

When `V_M` or `V_N` exceed 1 (the alternate wave-interleaved layout, used by a second kernel
family that this component also supports), a lane's column-run additionally concatenates `V_M`
consecutive instruction-blocks back to back, and output rows are grouped `V_N` at a time
before the per-wave-tile-step index advances — the lane/column-group formulas above are
unchanged, only the column-run length `R` and the row grouping change.

### 4.4 Wave-to-tile arrangement

Two conventions exist for how the `G_M x G_N` waves of one workgroup divide up the macro tile,
selected per kernel family:

- **Block-contiguous:** each wave owns one contiguous block of `T_M` (resp. `T_N`) wave-tile
  steps; consecutive steps within a wave are `MI_M` (resp. `MI_N`) columns/rows apart, and
  different waves' blocks are stacked end-to-end (wave index `i`'s column origin is offset by
  `i * T_M * MI_M`).
- **Interleaved:** consecutive wave-tile steps within one wave are `G_M * V_M * MI_M` (resp.
  `G_N * V_N * MI_N`) apart, with different waves' steps interleaved at `V_M * MI_M` (resp.
  `V_N * MI_N`) granularity instead of being stacked in blocks.

Both conventions are handled by the same lane/register formulas above; only the step stride and
the wave's column/row origin differ.

## 5. Mapping to the global output matrix

The GEMM's output indices are conventionally labeled by position in the index-numbering order:
the free index belonging only to the left (first) operand is index 0, the free index belonging
only to the right (second) operand is index 1, a shared batch index (when present) is index 2,
and the remaining shared contraction/summation index is whichever position is left (index 2 if
unbatched, index 3 if batched). MegaFusedEpilogue's "column" axis is the left operand's free
index (M, index 0) and its "row" axis is the right operand's free index (N, index 1); the
per-column weight vector is indexed solely along the former and broadcast along the latter (and
across batches, when the problem is batched).

The problem/kernel naming convention assigns each index a letter by that same position (`I`,
`J`, `K`, `L`, ... in index-number order), then spells out each operand's own index order using
those letters, lower-cased when the stride runs in the natural (non-mirrored) direction. For the
operand-transpose combination MegaFusedEpilogue is actually deployed with in this codebase (left
operand transposed, right operand not transposed, batched) -- confirmed by the kernel/problem
names actually generated for kernels that enable it -- that works out to: `I` = the left
operand's free index (column/M), `J` = the right operand's free index (row/N), `K` = the
batch index, and `L` = the contraction/summation index. The letters are positional labels, not
fixed semantic names -- which physical axis plays `I` versus `L` depends on each operand's own
index order and transpose setting, not on the letter itself -- but for this component's
deployment the mapping is exactly column=`I`, row=`J`, batch=`K`, contraction=`L`.

Global position of tile element `(m, n, k)` (column-group `g`, lane row `r`):
```
column_position = workgroup_column_origin + g*R + m*columnStride + k
row_position     = workgroup_row_origin + n*rowStride + r
```
where `workgroup_column_origin = workgroupIndex_M * MT_0 (+ wave column origin if G_M > 1)` and
`workgroup_row_origin = workgroupIndex_N * MT_1 (+ wave row origin if G_N > 1)` — i.e. the
standard tiling convention where the workgroup's position in the output grid scales directly by
the macro tile's extents, and multiple workgroups' tiles cover the full output matrix by tiling
both axes independently.

The output (and the residual/residual-stream/scratch-partial tensors, which all share the same
tiling) are addressed with the column index as the contiguous (fastest-varying) dimension and
the row index as the strided dimension — i.e. row-major storage with the column-axis extent as
the leading dimension. This is confirmed directly by the component's own address arithmetic for
its side tensors (each element's linear offset is `row_position * column_extent +
column_position`, times the element size), not merely assumed.

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
