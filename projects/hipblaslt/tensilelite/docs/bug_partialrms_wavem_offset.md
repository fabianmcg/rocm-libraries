# Bug report: missing waveM M-row offset in PartialRMS and RstdScale epilogue emitters

## Overview

Two related correctness bugs existed in the Subtile epilogue emitters for the
fused GEMM+RMSNorm pipeline on gfx950.  Both share the same root cause: when
`MIWaveGroup[0] > 1` (multiple waves tiling the M dimension of an output tile),
neither the PartialRMS K1 emitter nor the RstdScale K3 emitter accounted for a
wave's position along the M axis when computing global memory addresses.  All
prior tests used `MIWaveGroup[0] = 1`, so the bug was latent until the kernel
configuration was changed to improve performance.

A third, independent bug — an assembler encoding error — was also discovered and
fixed during the same work.

---

## Background: kernel geometry

The Subtile kernel tiles the output matrix with a work group of
`MIWaveGroup[0] × MIWaveGroup[1]` waves.

```
MacroTile0 = MatrixInstM × MIWaveTile[0] × MIWaveGroup[0]  (M dimension)
MacroTile1 = MatrixInstN × MIWaveTile[1] × MIWaveGroup[1]  (N dimension)
```

Each wave owns a sub-tile of size `(mma_m × mfma_m) × (mma_n × mfma_n)`:

```
mma_m = (MacroTile0 / MatrixInstM) / MIWaveGroup[0]
mma_n = (MacroTile1 / MatrixInstN) / MIWaveGroup[1]
```

For `MatrixInstM = MatrixInstN = 16`, `MIWaveTile = [4, 4]`:

| wg0_waves | wg1_waves | MacroTile0 | MacroTile1 | mma_m | mma_n |
|-----------|-----------|------------|------------|-------|-------|
| 1         | 1         | 64         | 64         | 4     | 4     |
| 1         | 2         | 64         | 128        | 4     | 4     |
| 2         | 1         | 128        | 64         | 4     | 4     |
| 4         | 1         | 256        | 64         | 4     | 4     |
| 4         | 2         | 256        | 128        | 4     | 4     |

The wave identity within the work group is derived from the hardware `Serial`
register:

```
waveId = Serial / WavefrontSize
waveM  = waveId % MIWaveGroup[0]   (wave's index along M; 0..wg_m-1)
waveN  = waveId / MIWaveGroup[0]   (wave's index along N; 0..wg_n-1)
```

Wave `waveM` owns M-rows in the global tile:

```
row_start = WorkGroup0 * MacroTile0 + waveM * mma_m * MatrixInstM
row_end   = row_start + mma_m * MatrixInstM
```

---

## Bug 1 — PartialRMS: wrong partialBuf row address in `_writePartials`

### File

`Tensile/Components/Subtile/SubtilePartialRMSEmit.py`, method `_writePartials`

### Description

After the within-wave butterfly reduction, each writing lane stores its
per-row partial sum-of-squares to `partialBuf[global_row, tile_col]`.  The
row address was computed as:

```
global_row = WorkGroup0 * MacroTile0 + m*mfma_m + k + row_group*rows_per_lane
```

This is correct when `MIWaveGroup[0] = 1` (one wave per M-strip), but wrong
when `MIWaveGroup[0] > 1` because the term `waveM * mma_m * mfma_m` is missing.

### Consequence

With `wg_m = 2` and `MacroTile0 = 128`:

- Wave 0 (`waveM = 0`) should write rows `[wg_row_base + 0, wg_row_base + 64)`.
- Wave 1 (`waveM = 1`) should write rows `[wg_row_base + 64, wg_row_base + 128)`.

Without the offset both waves compute identical addresses and write to rows
`[wg_row_base, wg_row_base + 64)`.  Wave 1 overwrites wave 0's correct
values with wrong values (wrong `m` index into the wave's own accumulators),
and the upper half of `partialBuf` is never written, remaining zero.

### Root cause

`waveM` was computed inside `_crossWaveReduce` for LDS addressing but was
not propagated to (or recomputed in) `_writePartials`.  Because
`_crossWaveReduce` is only called when `wg_n > 1`, the mechanism that would
have surfaced the missing `wg_m` offset was never triggered in isolation.

### Fix

Before the write loop, derive `waveM` from `Serial` and add its M-row offset
to `wg_row_base`:

```python
if self.wg_m > 1:
    # wave_id = Serial / wave_size
    vectorStaticDivide(wave_id, "Serial", self.wave_size, ...)
    # waveM = wave_id & (wg_m - 1)  [wg_m is a power of two]
    VAndB32(wave_m, wave_id, self.wg_m - 1)
    # waveM_off = waveM * (mma_m * mfma_m)
    VMulLOU32(wave_m, wave_m, self.mma_m * self.mfma_m)
    # wg_row_base += waveM_off
    VAddU32(wg_row_base, wg_row_base, wave_m)
```

When `wg_m = 1` the block is skipped entirely (no cost, no register
pressure).

---

## Bug 2 — RstdScale: wrong rstdBuf row address in `_loadRstd`

### File

`Tensile/Components/Subtile/SubtileRstdScaleEmit.py`, method `_loadRstd`
(row base established in `_setup`)

### Description

`_setup` computed the per-wave row base as an SGPR:

```
row_base_sgpr = WorkGroup0 * MacroTile0   (scalar, same for all waves)
```

`_loadRstd` then loaded:

```
rstdBuf[row_base + m*mfma_m + g*rows_per_lane + k]
```

Again the `waveM * mma_m * mfma_m` term was absent, so all waves loaded
rstd values from the same rows (wave 0's rows).  Wave 1's accumulator
elements were then scaled by the wrong rstd values, producing corrupted D
output.

### Fix

After computing `row_base_sgpr`, derive `waveM` from `Serial` and add the
M-row offset into a VGPR `wave_row_base` used for all rstdBuf loads:

```python
VMovB32(wave_row_base, row_base_sgpr)          # SGPR → VGPR
if self.wg_m > 1:
    vectorStaticDivide(wave_id, "Serial", self.wave_size, ...)
    VAndB32(wave_m, wave_id, self.wg_m - 1)
    VMulLOU32(wave_m, wave_m, self.mma_m * self.mfma_m)
    VAddU32(wave_row_base, wave_row_base, wave_m)
```

All subsequent address computations use `wave_row_base` (VGPR) rather than
`row_base_sgpr` (SGPR).

---

## Bug 3 — Assembler encoding error: `VMulLOU32` with large literal operand

### File

`Tensile/Components/Subtile/SubtilePartialRMSEmit.py`, method `_writePartials`

### Description

The original code for computing `wg_row_base = WorkGroup0 * MacroTile0` was:

```python
VMulLOU32(dst=vgpr(wg_row_base), src0=self.macro_tile0, src1=sgpr("WorkGroup0"))
```

`v_mul_lo_u32` is a VALU (VOP3) instruction.  VALU instructions allow a
32-bit literal in at most one source slot, and that slot must be `src0` of a
VOP2 encoding — VOP3 (two non-immediate sources) has no literal slot.  With
`MacroTile0 = 64` the value happens to fall in the AMDGPU inline constant
range (0–64), which assembles silently.  With `MacroTile0 = 256` the
assembler correctly rejects it:

```
error: literal operands are not supported
v_mul_lo_u32 v28, 256, s[sgprWorkGroup0]
```

### Fix

Move the integer constant into a VGPR first, then multiply register × register:

```python
VMovB32(dst=vgpr(mt0_vgpr), src=self.macro_tile0)   # literal → VGPR
VMulLOU32(dst=vgpr(wg_row_base), src0=vgpr(mt0_vgpr), src1=sgpr("WorkGroup0"))
```

The temporary VGPR is checked in immediately after the multiply.

---

## Bug 4 — GREmit: wrong wave partition for B tensor when `wg_m > 1`

### File

`Tensile/Components/Subtile/SubtileGREmit.py`, function
`_grComputeRowPartition_legacy`

### Description

When `loadRatioGR == 2.0` (each wave loads 2× its own K-strip cooperatively),
the legacy partitioning code set:

```python
localRow    = waveId   # row within the cooperative load group
partitionRow = 0       # no partition separation
```

This works when `MIWaveGroup[0] = 1`, where `waveId` directly encodes the
wave's position along M.  With `MIWaveGroup[0] > 1`, `waveId` encodes both the
M-position (`waveM = waveId % wg_m`) and the N-position
(`waveN = waveId / wg_m`).  For tensor B, the cooperative global read covers
the full N-tile so `localRow` should reflect the wave's M-position only, and
`partitionRow` should reflect `waveN` to separate the two B LDS regions.

Without the fix, waves with different `waveN` values compute the same
`(localRow, partitionRow)` pair and load from / write to the same LDS
addresses, causing B data corruption.

### Fix

For tensor B with `wg_n > 1`, decompose `waveId`:

```python
wg_m = kernel["MIWaveGroup"][0]
VAndB32(localRow,     wg_m - 1,              waveId)   # waveM = waveId % wg_m
VLShiftRightB32(partitionRow, log2(wg_m),   waveId)   # waveN = waveId / wg_m
```

For tensor A and the `wg_n == 1` case the original assignment is unchanged.

---

## Bug 5 — LREmit: missing per-N-wave LDS offset for B tensor when `wg_m > 1`

### File

`Tensile/Components/Subtile/SubtileLREmit.py`, function
`_applyWavePartitionLROffset`

### Description

When `loadRatioGR >= 2.0`, the function returned early without applying any
wave-specific LDS read offset.  This is correct for tensor A and for single-N-wave
configurations, but wrong for tensor B with multiple N-waves (`wg_n > 1`).

With `wg_n > 1`, each N-wave reads a distinct N-column strip from LDS.  The
strip boundaries are separated by `sInterval = partitionOffset * subIterKBytes`
bytes.  Without adding `waveN * sInterval` to the local-read offset, all
N-waves read from the same LDS addresses — the B data one wave loaded is
overread by another, producing wrong accumulator values.

### Fix

After computing `waveN = waveId / wg_m`, accumulate the per-N-wave stride into
each shared LDS read-offset VGPR:

```python
waveN * sInterval  →  add to each tileInfo.sharedVgprLROffset[i]
```

The correction is guarded on `tc == 'B' and wg_n > 1` so A and single-N-wave
cases are unaffected.

---

## Why the test suite did not catch these bugs

All three existing test files (`test_gemm_partial_rms.py`,
`test_gemm_rstd_scale.py`, `test_gemm_rmsnorm_gemm_pipeline.py`) parametrized
only on `wg_n` (`MIWaveGroup[1]`), while `wg0_waves` was fixed at 1 inside the
`build_k1_solution` / `build_k3_solution` helper functions.  Every executed
configuration therefore had `MIWaveGroup[0] = 1` and `MacroTile0 = 64`:

- Bug 1 and Bug 2: `waveM = 0` always → missing offset is zero → no error.
- Bug 3: `MacroTile0 = 64` → integer falls in inline-literal range → no
  assembler error.

The bugs only surfaced when `wg0_waves` was raised from 1 to 4 (giving
`MacroTile0 = 256`) as a performance improvement, which exposed all three
failures simultaneously.

---

## Performance context

The original kernel configuration:

```python
mi9 = [16, 16, 32, 1, 1, 4, 4, 1, wg_n]   # wg0_waves=1, MacroTile0=64
```

produced a 64×64 tile — far too small for gfx950.  With 4096×4096×4096
workloads, ~4096 work groups are launched, each doing minimal arithmetic per
wave.  The target configuration:

```python
mi9 = [16, 16, 32, 1, 1, 4, 4, 4, wg_n]   # wg0_waves=4, MacroTile0=256
```

gives a 256×128 tile (with `wg_n = 2`), 32× more output elements per work
group, and is expected to sustain > 900 TFLOPS on gfx950 for bf16 GEMM.

---

## Test coverage improvement

To prevent regression, the fixture parametrization was changed from a single
`wg_n` axis to explicit `(wg0_waves, wg1_waves)` pairs covering all
tile configurations of interest:

```python
_WG_CONFIGS = [
    (1, 1),   # MT0=64,  MT1=64   — baseline (inline-literal range)
    (1, 2),   # MT0=64,  MT1=128  — wg_n cross-wave LDS
    (2, 1),   # MT0=128, MT1=64   — first value outside inline-literal range
    (4, 1),   # MT0=256, MT1=64   — triggered the original VMulLOU32 bug
    (4, 2),   # MT0=256, MT1=128  — default high-perf config
]
```

Solutions are now built directly from `(wg0_waves, wg1_waves)` rather than
delegating to example-script helpers, so the test coverage is independent of
whatever defaults those helpers choose.  M shapes are generated at runtime from
`_m_shapes_for_mt0(MT0)` so boundary coverage (MT0-1, MT0+1, 2×MT0-1, etc.)
adapts automatically for any tile size.

**Result: 540 tests, all passing.**

---

## Files changed

| File | Change |
|------|--------|
| `Tensile/Components/Subtile/SubtilePartialRMSEmit.py` | Fix VMulLOU32 literal encoding (Bug 3); add waveM M-row offset in `_writePartials` (Bug 1) |
| `Tensile/Components/Subtile/SubtileRstdScaleEmit.py` | Add waveM M-row offset in `_loadRstd` (Bug 2) |
| `Tensile/Components/Subtile/SubtileGREmit.py` | Decompose `waveId` into `(waveM, waveN)` for B tensor when `wg_n > 1` (Bug 4) |
| `Tensile/Components/Subtile/SubtileLREmit.py` | Add per-N-wave LDS read offset for B tensor when `wg_n > 1` (Bug 5) |
| `tensile_gemm_rmsnorm_gemm_example.py` | Raise `wg0_waves` 1→4; `PrefetchGlobalRead` 1→2; default `wg_n` 1→2 |
| `tensile_gemm_rmsnorm_gemm_benchmark.py` | Default `wg_n` 1→2 |
| `Tensile/Tests/unit/test_gemm_partial_rms.py` | Generalize fixture to `(wg0_waves, wg1_waves)` axis; MT0-relative M shapes |
| `Tensile/Tests/unit/test_gemm_rstd_scale.py` | Same generalization |
| `Tensile/Tests/unit/test_gemm_rmsnorm_gemm_pipeline.py` | Same generalization |
