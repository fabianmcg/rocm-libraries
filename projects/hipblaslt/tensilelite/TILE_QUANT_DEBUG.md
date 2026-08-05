# TileQuant D_fp8 Output Bug — Debug State

## Status

`quantScale` output is **correct**. `D_fp8` output is **wrong**. 70/70 GPU unit tests fail on the D comparison.

## Symptom

```
D[0, 0:4]  ≈ 5.0          (wrong — unscaled)
D[0, 4:8]  ≈ 288–416      (correct — properly scaled)
ref                ≈ 288–320 for all columns
```

Raw GEMM `h1T` values are ~0.14–0.19 for all columns (similar magnitude).
Expected `quantMult ≈ 2056` (for amax ≈ 0.218).
GPU D[0,0] = 5.0 → implies effective `quantMult ≈ 34`, i.e. `amax ≈ 13` in the writing lane.

## What is confirmed working

- **QuantScale** matches reference exactly → `_laneTileAmax` + `_butterflyReduce` + `_computePerTileScale` + `_writeScale` are all correct.
- **Butterfly reduce** has 6 rounds (XOR 1,2,4,8,16,32) with `s_waitcnt lgkmcnt(0)` before the final `v_max_f32` batch. Correct per ISA spec.
- **AGPR indexing**: `vgprTiles[n*mmaM+m].regList.indices[k] = (n*4+m)*4+k`. The store's `codeAccVgprRead` reads `acc(acc2arch[destIdx])` = same sequential AGPR. Identity mapping confirmed analytically for VW=1.
- **Hook timing**: TileQuant emitter fires at KernelWriter.py ~5242, before `globalWriteElements` (~12xxx). All TileQuant AGPR writes finish by line 7455. `codeAccVgprRead` begins at line 8256.
- **kernarg slot**: QuantScale is at slot 27 (not 28) because `UseBeta=False` removes the beta slot. Test and `buildSubtileArgs` both use slot 27 now.
- **No exec manipulation** in TileQuant section (0 `s_and_saveexec` calls between lines 1989–7460, confirmed by grep).
- **`v37` (quantMult[0]) not overwritten** between `_computePerTileScale` (line 5257) and `_applyScaleInPlace` (line 5420).

## Assembly timeline (MT64x128, Q=[16,16])

| Line range | Action |
|---|---|
| 1989–2515 | `_applyAlphaInPlace`: reads acc0..127 into v102, multiplies by `sgpr("Alpha")`, writes back |
| 2516–4848 | `_laneTileAmax`: reads acc0..127 into v1, accumulates abs into v5..v36 (per quant-tile) |
| 4848–5217 | `_butterflyReduce`: 6 rounds of `ds_bpermute_b32` + `v_max_f32` |
| 5217 | `s_waitcnt lgkmcnt(0)` |
| 5218–5249 | Final `v_max_f32` round for all 32 tiles |
| 5252–5413 | `_computePerTileScale`: `v_rcp_f32 + v_mul_f32` → v37..v68 (quantMult), v69..v100 (scaleDequant) |
| 5415–7455 | `_applyScaleInPlace`: reads acc0..127 into v1, multiplies by lane's quantMult, writes back |
| 7457–8179 | `_writeScale`: `buffer_store_dword` for QuantScale (with exec save/restore per tile) |
| 8256–12142 | `codeAccVgprRead`: reads acc0..127 into v[16..155] (vgprValuC=4, so v[4+12..4+139]) |
| 8256–12626 | Store loop: fp8 clamp + `v_cvt_pk_fp8_f32` + `buffer_store_byte` |

`.amdhsa_accum_offset 176` → acc0=v[176], acc1=v[177], ...  
`vgprValuC = 4` → staging VGPRs v[16..155], completely separate from AGPRs.

## Split pattern analysis

The wrong/correct split is at D column 4 within the same MFMA tile (n=0, qj=0):
- Lanes 0..3 (col=0..3): wrong (~5)
- Lanes 4..7 (col=4..7): correct (~350)

All are in rowGroup=0, same quant-tile (qi=0, qj=0), should use the same `v37`.
After 4 col-butterfly rounds (XOR 1,2,4,8), all 16 lanes in rowGroup=0 should have the same `v5`.
After 2 rowGroup rounds (XOR 16,32), all 64 lanes should have the same `v5`.
Since quantScale is correct, v5 IS correct in the writing lane (lane 0 = col=0, rowGroup=0).
Yet D[0,0] implies `quantMult ≈ 34` in lane 0, while D[0,4] is correct.

This is contradictory: lane 0 writes correct quantScale but wrong D.

## Hypothesis

`_applyScaleInPlace` either:
1. Is not running for some accumulator registers (e.g., acc0..acc3 are skipped), OR
2. Runs but its AGPR writes are not visible to `codeAccVgprRead` (hazard?), OR
3. `quantMult[0]` (v37) has different per-lane values despite the butterfly being "correct"

The most likely explanation not yet ruled out:
- The `_writeScale` section (lines 7457–8179) manipulates `v102` and `v103` for address computation. These overlap with VGPRs used by `_applyScaleInPlace` for the `qi` computation. If `_writeScale` runs partially interleaved with `_applyScaleInPlace` (it does not — they are sequential), there would be a conflict. But they ARE sequential, so no conflict.
- There could be a gfx950-specific `v_accvgpr_write_b32` → `v_accvgpr_read_b32` hazard requiring more than `s_nop 1`. The store's `codeAccVgprRead` does NOT insert any nop between TileQuant's last write (line 7455) and the first read (line 8256) — but there are >700 instructions of intervening code, so timing is not an issue.

## Next debugging step

**Experiment 1**: Modify `_applyScaleInPlace` to multiply by a literal constant (e.g., 1000.0) instead of `quantMult`. If D becomes fp8_max=448 everywhere, the AGPR writes ARE reaching the store, and the bug is in `quantMult` value. If D is still ~5, the store is not reading TileQuant's AGPR writes.

**Experiment 2**: Skip TileQuant entirely (early return in `emit()`) and verify D = raw unscaled fp8 ≈ 0.002..0.25.

The `compileSolution()` function re-generates and recompiles assembly from scratch each call, so modifying the Python emitter immediately affects the test output.

## Key files

| File | What changed |
|---|---|
| `Tensile/Components/Subtile/SubtileTileQuantEmit.py` | Main emitter — `_applyScaleInPlace` is the suspect |
| `Tensile/KernelWriterAssembly.py` | Sets `applyAlpha=False, betas=[False]` for TileQuant store path |
| `epilogues/unittests/test_gemm_tile_quant.py` | 70 test cases; reads QuantScale from slot 27 |
| `epilogues/tensilelite/partialrms_helpers.py` | `buildSubtileArgs` has `hasBeta=False` param; TileQuant uses it |
| `epilogues/yaml/gemm_tile_quant_k1.yaml` | `UseBeta: False` (was True — was silently rejecting all solutions) |
