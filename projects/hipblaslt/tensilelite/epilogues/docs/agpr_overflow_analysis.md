# AGPR Overflow Analysis: PartialRMS Epilogue on MT320×320

## Corrected summary

The PartialRMS emitter correctly uses `v_accvgpr_read_b32` / `v_accvgpr_write_b32` to move
data between the input D-tile accumulators and VGPR temporaries. It allocates nothing from
the AGPR pool itself — all its own temporaries come from `vgprPool` and `sgprPool`.

The bug is in how it addresses the **input** accumulator tiles: it computes a flat linear
index `accVgprBase + (n*mma_m + m)*rows_per_lane + k` and passes that directly to
`accvgpr(a)`, assuming every tile lives at a contiguous AGPR offset from the base. This
assumption breaks when the allocator places some tiles in the VGPR pool instead of the AGPR
pool, which happens whenever the tile count exceeds the hardware AGPR limit (256 on gfx950).

For MT320×320 (400 accumulators total), tiles 0–63 are in `acc[0..255]` and tiles 64–99
spill to the VGPR pool. The emitter computes indices 256–399 for those spilled tiles and emits
`v_accvgpr_read_b32 v1, acc256`, which the assembler rejects because `acc256` is not a valid
AGPR index.

---

## How the allocator assigns tiles

`allocVgprTileRegisters_legacy` (`Tensile/Components/Subtile/Kernel.py`) fills the AGPR pool
up to the hardware limit, then spills to VGPRs:

```python
maxAgpr = regCaps["PhysicalMaxVgpr"] - regCaps["MaxVgpr"]  # 256 on gfx950
for i in range(numMMATiles):
    if isDTile and agprPool.size() < maxAgpr:
        pool = agprPool      # tile lives at acc[agpr_idx]
    else:
        pool = vgprPool      # tile lives at v[vgpr_idx]
    vgprTiles.append(RegisterTileInfo(pool, regType))
```

Each `vgprTiles[i]` records the pool and the actual register index. The MFMA instruction
emitter already consults this correctly:

```python
dIsVgpr = (vgprTileD.regList.pool == writer.vgprPool)
dAccAlias = vgpr if (dIsVgpr or kernel["MIArchVgpr"]) else accvgpr
```

---

## What the PartialRMS emitter does instead

`KernelWriter.py` extracts only the first tile's base index and passes it as a flat integer:

```python
dtile_agpr_base = dtileInfo.vgprTiles[0].regList.indices[0] if dtileInfo.vgprTiles else 0
module.add(pRmsEmitter.emit(dtile_agpr_base))
```

The emitter's `_acc_idx` then computes every accumulator address as a contiguous offset from
that base:

```python
def _acc_idx(self, base: int, m: int, n: int, k: int) -> int:
    return base + (n * self.mma_m + m) * self.rows_per_lane + k
```

This is only correct when every tile was allocated contiguously in the AGPR pool. When some
tiles spilled to VGPRs their actual register indices are independent of the AGPR base, and the
formula produces wrong indices for them.

---

## Two separate assembler errors

### Error 1 — `acc256..acc399` out of range

The flat formula produces indices 256–399 for the spilled tiles, which the assembler rejects.

### Error 2 — `s[92:93]` / `s[96:97]` invalid register alignment

The emitter allocates the gamma SRD and partialBuf SRD with `sgprPool.checkOutAligned(4, 4)`.
With the additional SGPR pressure from the combined `UseBias + UseScaleAlphaVec + Activation +
PartialRMS` problem type, these pairs land on odd-aligned indices. `s_mov_b64` requires an
even-aligned destination. This is a separate issue from the AGPR overflow.

---

## Fix for Error 1

Pass `dtileInfo.vgprTiles` into the emitter instead of a flat integer base. Replace `_acc_idx`
with a per-tile lookup that reads the actual register index and picks the correct register kind:

```python
def _acc_reg(self, vgprTiles, m: int, n: int, k: int):
    tile = vgprTiles[n * self.mma_m + m]
    reg_idx = tile.regList.indices[k]
    if tile.regList.pool == self.writer.vgprPool:
        return vgpr(reg_idx)
    return accvgpr(reg_idx)
```

Every `accvgpr(self._acc_idx(base, m, n, k))` call is replaced with
`self._acc_reg(vgprTiles, m, n, k)`. The emitter still emits `v_accvgpr_read_b32` when the
tile is in the AGPR pool and `v_mov_b32` (or reads directly as a VGPR) when it is not — it
just does so based on the allocator's decision rather than a flat assumption.

With this fix, MT320×320 and any other tile whose accumulator count exceeds 256 will compile
correctly, because the emitter delegates pool and index decisions to the allocator rather than
recomputing them independently.
