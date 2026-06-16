# Subtile Primer

A developer primer for the **Subtile** code-generation path in TensileLite
(`Tensile/Components/Subtile/`).

---

## 1. What is Subtile?

Subtile is an alternative **MFMA-mainloop code generator** for TensileLite. It
emits the performance-critical inner GEMM loop (global reads → LDS → local reads
→ matrix-multiply-accumulate) using a *subtile* abstraction. The macro tile is
first decomposed into **MMA-instruction tiles** (the 16×16×K rectangles a single
MFMA/WMMA/MXFMA instruction consumes), and those MMA tiles are then grouped into
**subtiles** — the cooperative unit that a wave-group loads in one round and that
the scheduler reasons about when interleaving loads with math.

It is enabled per-solution by the `UseSubtileImpl` flag and is restricted to
**gfx950 and gfx1250**; the flag is silently cleared on any other ISA
(`Solution.py:826-830`). Its purpose is to support the newest data paths —
MX block-scaled FP4/FP6/FP8 (`MXMFMA`), bf16/fp16, and WMMA on gfx1250 — with a
cleaner, schedule-driven mainloop than the legacy `KernelWriter.kernelBody`
path. On gfx950, MX block-scaled GEMM *requires* Subtile (`Solution.py:832-833`):
the legacy path has no MX implementation there.

The design philosophy is a separation between **frozen geometry** (pure shape
math, no register state, no emit) and **mutable runtime state** (register pools,
offset VGPRs, instruction emission). Emit logic is selected by **tag dispatch**
(`functools.singledispatch` over empty marker types) so new tile shapes can be
added without touching existing code paths.

---

## 2. How Tensile calls Subtile

The top-level dispatch is a single branch in the kernel-source generator
(`KernelWriter.py:9942`):

```python
if not kernel["UseSubtileImpl"]:
    (error, kb) = self.kernelBody(kernel, tPA, tPB)          # legacy generator
else:
    (error, kb) = self.kernelBodySubtile(kernel, tPA, tPB)   # Subtile generator
```

`kernelBodySubtile` (`KernelWriter.py:4487`) is still a `KernelWriter` method —
Subtile does **not** own the whole kernel. It builds the prologue (function
signature, `defineAndResources`, StreamK `preLoop`, persistent-loop open,
`setupNewTile`, TDM descriptor setup, `graAddresses`, tile-assignment via
`graTileAssignment`/`lraTileAssignment`, D-tile allocation, accumulator zeroing)
and then hands the inner loop to Subtile's orchestrator in `Kernel.py`:

```python
preLoop(writer, kernel)    # PGR/PLR prefetch scaffold (Kernel.py:1197)
mainLoop(writer, kernel)   # builds scheduler, emits main + tail loops (Kernel.py:1233)
```

`mainLoop` constructs a `LogicalScheduler`, searches M/N partition candidates
until the predicted VGPR footprint fits `regCaps["MaxVgpr"]`, allocates tiles,
populates instructions, and emits the main/NGLL/NLL/tail loops, then wraps the
tail loop with the runtime `K % DepthU` counter setup borrowed from the legacy
`KernelWriter` (`Kernel.py:1318-1344`).

Geometry selection happens earlier, in solution validation
(`Solution.py:857-877`): from dtype + `TLU` + wavefront size a string key
(`_ABTilePair{A,B}`) is stored on the kernel, which Subtile later resolves
through `selectABGeometry` (`Kernel.py:362`).

---

## 3. Architecture overview

Subtile is layered into four concerns spread across the files:

| Layer | Files | Role |
|-------|-------|------|
| **Geometry** (frozen) | `SubtileGeometry.py` | Pure data: MMA lane layouts, tile/subtile shapes, grid math, tag sentinels. No emit, no registers. |
| **Runtime tiles** | `Kernel.py` | `TileInfo` + `ABGRTile`/`ABLRTile` bind geometry to a kernel config and own register state. Pre-defined geometry instances, selection maps, MFMA emit, orchestrators. |
| **Emit (tag dispatch)** | `SubtileGREmit.py`, `SubtileLREmit.py`, `SubtileScaleEmit.py` | `@singledispatch` over tags → concrete `buffer_load`/`ds_write`/`ds_read`/MXFMA instructions and offset math. |
| **Scheduling** | `LogicalScheduler.py`, `InstructionScheduler.py`, `InstructionEmitter.py` | Build a logical dependency graph of ops, lower ops to instructions, then interleave non-MFMA instructions between MFMAs. |

The data flow at generation time:

```
Solution.py (pick _ABTilePair) 
      │
      ▼
kernelBodySubtile  ──► TileInfo(geometry, tc, writer, kernel)   [Kernel.py]
      │                     ▲ geometry from SubtileGeometry.py
      ▼
preLoop / mainLoop  ──► LogicalScheduler.build()                [LogicalScheduler.py]
                              │  (passes: place_LRs → … → emit)
                              ▼
                        InstructionEmitter.populate()           [InstructionEmitter.py]
                              │  (calls GR/LR/Scale emit funcs)
                              ▼
                        instructionSchedule()                   [InstructionScheduler.py]
                              │  (interleave into MFMA slots)
                              ▼
                        rocisa Module → assembly text
```

---

## 4. Per-file reference

### 4.1 `__init__.py`

Empty package marker (one blank line). Subtile sub-modules are imported by their
fully-qualified paths (e.g. `from Tensile.Components.Subtile.Kernel import …`),
and `KernelWriter.py:50` does `from .Components.Subtile.Kernel import *`, so no
re-exports are needed here.

### 4.2 `SubtileGeometry.py` — frozen shape/layout definitions

**Contents.** All immutable, config-free description of *shape*. Nothing here
emits instructions or touches register pools.

- `RegList` (`:28`) — a typed register-index list that knows its kind
  (`Sgpr/Vgpr/Accvgpr`) and how to wrap an index into the matching rocisa
  container via `ref()`. `alloc/append/dealloc` manage pool checkout/checkin.
  This is the one stateful helper here and is reused by `Kernel.py`'s
  `RegisterTileInfo`.
- `LoadShape` (`:90`) — `(m, k)` elements per load/store instruction; makes the
  contiguous direction explicit (row-major ⇒ `k>1, m==1`; column/TLU ⇒ `m>1`).
- `MMALayout` (`:107`) / `MMAScaleLayout` (`:169`) — ISA lane geometry: `instM`,
  `blocks`, `vgprs`/lane, `waveSize`; derives `contiguousLanes`, `kGroups`,
  `elementsPerLaneNonK`, and helpers `inputBytesPerLane`, `tileSizeBytes`,
  `regsPerTile`. Pre-defined: `MFMA_16x16_1B_4K_4V` (bf16/fp4), `..._8V` (fp8),
  `MFMA_16x16_1B_4N_4V` (f32 C/D), `MFMA_SCALE_16x16_1B_MX32_8V` (mxfp scale).
- `TileGeometry` (ABC) → `ABInputGeometry` → `ABGRGeometry` / `ABLRGeometry`.
  The A/B operand geometry is **split into GR and LR** because the two may use
  different subtile shapes. `ABGRGeometry` describes the cooperative global-read
  footprint (`subtileShape`, `subtileCount`, `subtileStride` — a CuTe-style
  strided-strip layout) and provides grid queries (`globalMMATileGrid`,
  `globalSubtileGrid`), byte counts (`subtileSizeBytes`, `bytesPerLoad`),
  `localGRGranularity`, `for_kernel` (materializes derived counts from
  wave-group/macro-tile), and `subtileForMmaTile` (maps an MMA tile → its
  subtile and sibling tiles). `ABLRGeometry` owns the per-MFMA `ds_read` subtile
  shape.
- `ABTilePair` (`:447`) — bundles one GR + one LR geometry for an A/B operand;
  delegates common dtype properties to `gr`. This is the object passed to
  `TileInfo`.
- `CDTileGeometry` (`:485`, abstract) — output (C/D) tile partitioning with
  wave-group split in M *and* N; abstract `emitStoreD`/`emitLoadC`.
- `MXScaleInputGeometry` → `MXScaleGRGeometry` / `MXScaleLRGeometry` /
  `MXScaleTilePair` — scale-factor operands with compressed K (one scale per
  `mxBlock` data elements).
- **Tag sentinels** (`:682-708`) — empty frozen dataclasses `GRTag_1x1/1x2/2x2/
  TLU1`, `LRTag_1x1/1x2/TLU1`. They carry no data; they are the `singledispatch`
  keys (analogous to C++ tag-dispatch types).

**Interactions.** Consumed by `Kernel.py` (instances + `TileInfo`). The tag
types are imported by `SubtileGREmit.py`/`SubtileLREmit.py` to register emit
implementations. Nothing in this file imports the emit or scheduler files —
geometry is the dependency root.

### 4.3 `Kernel.py` — runtime tiles, geometry instances, MFMA emit, orchestrators

**Contents.** The bridge between frozen geometry and mutable codegen.

- `ABGRTile` / `ABLRTile` (`:131`, `:207`) — mutable wrappers holding a frozen
  geometry config + offset-register state. Their `emit*`/`alloc*` methods
  forward to the GR/LR emit modules using `self.config.tag` as the dispatch key
  (e.g. `_emitGlobalRead(self.config.tag, self, ti, writer, kernel)`).
- Pre-defined geometry instances: `AB_B16`, `AB_B16_W32`, `AB_B4`, `AB_B8`,
  `AB_B4_2x2`, `AB_B16_2x2`, `AB_B16_TLU1(_16x1)`, the MX scale pairs
  `MXSA/MXSB_B4/B8`, and outputs `CD_F32`/`CD_F32_W32` (`:279-338`).
- Selection: `AB_GEOMETRY_MAP` + `selectABGeometry` (`:351-365`),
  `selectMXScaleGeometry` (`:340`), `selectDGeometry` (`:368`).
- `TileInfo` (`:379`) — the central runtime object. From geometry + tensor
  component + writer + kernel it computes every instantiated grid
  (`globalMMATileGrid`, `localMMATileGrid`, `globalSubtileGrid`,
  `localSubtileGrid`), load ratios (`loadRatioGR/LR`, `numGRPerSubtile`,
  `numGRTotal`), byte strides, and runs `_check_dim` consistency assertions that
  each macro-tile dimension is covered exactly. It exposes index-mapping helpers
  (`grLoadIndexForSubtile`, `lrTileIndexForSubtile`, `globalMmaTilesForSubtile`,
  `waveMmaTilesForSubtile`) and register accessors, and dispatches `emit*` to its
  `gr`/`lr` runtime tiles (or directly to geometry for MX/CD).
- `RegisterTileInfo` (`:880`) — wraps a `RegList` for one MMA-tile slot.
- MMA emit: `emitMfmaInstruction` (`:1039`) emits one MFMA/MXMFMA;
  `emitMfmaCode` (`:1107`) triple-loops the local MMA grid emitting the full
  `C += A*B`; `_selectF8F6F4InstType` (`:975`) picks the F8/F6/F4 (incl. mixed)
  instruction variant; `initVgprTilesToZero`/`_zeroRegRange` (`:937`/`:901`) zero
  accumulators via MFMA/WMMA.
- Orchestrators: `preLoop` (`:1197`) emits the PGR/PLR prefetch scaffold;
  `mainLoop` (`:1233`) drives the scheduler end-to-end (see §2).

**Interactions.** Imports everything from `SubtileGeometry.py`,
`SubtileGREmit.py`, `SubtileLREmit.py`, `SubtileScaleEmit.py`, and
`LogicalScheduler.py`. It is imported wholesale by `KernelWriter.py` (`*`) and
its `emitMfmaInstruction`/emit funcs are called back by `InstructionEmitter.py`.

### 4.4 `SubtileGREmit.py` — global-read emit (tag-dispatched)

**Contents.** `@singledispatch` bases (`_emitGlobalReadOffset`,
`_emitGlobalRead`, `_emitLocalWrite`, `_allocGROffsetRegisters`,
`_emitDTLInit`, `_emitGRLDSBufferSwap`, `_emitGRPtrUpdate`) plus per-tag
implementations registered for `GRTag_1x1/1x2/2x2` (the row-major `_TLU0`
family). Offset math helpers: `_grComputeOffset`, `_grComputeSubtileOffsets`,
`_grComputeRowPartition`, `_grComputeAllOffsets`. Driver/entry functions used by
`KernelWriter` and the scheduler: `graTileAssignment`, `graInitPointer`,
`emitSingleBufferLoad` (one `buffer_load`), `emitSubtileBufferLoad`,
`globalReadDoSubtile`, `globalReadLDSBufferSwap`, `globalReadPtrUpdates`,
`globalReadDTLInitCommonSgpr`. **TDM** (`tensor_load_to_lds`) path:
`tdmGlobalOffsetSubtile`, `initTDMDescriptorSubtile`,
`tdmApplyStreamKOffsetSubtile`. A `_legacy` suffixed parallel set exists for the
in-progress migration.

**Interactions.** Imports tags from `SubtileGeometry.py` and
`emitScaleGRLDSSwap` from `SubtileScaleEmit.py`. Called by `ABGRTile`
(via dispatch) and by `InstructionEmitter.emit_gr`/`emit_gr_inc`.

### 4.5 `SubtileLREmit.py` — local-read emit (tag-dispatched)

**Contents.** Mirror of the GR file for LDS reads. Dispatch bases
(`_emitLocalReadOffset`, `_emitLocalRead`, `_allocLROffsetRegisters`,
`_emitLRDTLInit`, `_emitLRLDSBufferSwap`) with `LRTag_1x1/1x2` implementations.
Helpers `_computeLROffset`, `_applyWavePartitionLROffset`, `_setExecMask`.
Entry points: `lraTileAssignment`, `emitSingleDsRead` (one `ds_read`),
`emitSubtileDsRead`, `localReadDoSubtile`, `localReadDTLInitCommonSwapVgpr`,
`localReadLDSBufferSwap`, `localReadResetOffsetsSubtile`. Includes
`VPermlane16SwapB32` usage for wave32/WMMA register swaps and `_legacy` variants.

**Interactions.** Imports tags from `SubtileGeometry.py` and
`emitScaleLRLDSSwap` from `SubtileScaleEmit.py`. Called by `ABLRTile` and by
`InstructionEmitter.emit_lr`/`emit_lr_inc`.

### 4.6 `SubtileScaleEmit.py` — MX scale-factor emit

**Contents.** Plain functions (no tag dispatch — scale geometry is fixed) for
the MXSA/MXSB operands: GR (`emitScaleGROffset`, `emitScaleGRLoad`,
`emitScaleGRPtrUpdate`, `emitScaleGRLDSSwap`, `globalReadDoScaleSubtile`,
`globalReadScalePtrUpdates`, `globalReadScaleSwizzledDTLInitCommonSgpr`,
`graTileAssignmentScaleSwizzled`) and LR (`emitScaleLROffset`, `emitScaleLRLoad`,
`emitScaleLRLDSSwap`, `lraTileAssignmentScaleSwizzled`, `emitSubtileScaleDsRead`,
`localReadDoScaleSubtile`). The access pattern is simpler: one `buffer_load` per
wave covers the whole scale tile; `ds_read_b32` per scale group. Several
functions are currently **stubs** (e.g. `emitScaleGROffset` early-returns before
its body).

**Interactions.** Imported by `Kernel.py` (orchestrators + dispatch table), by
`SubtileGREmit.py`/`SubtileLREmit.py` (LDS-swap helpers), and by
`InstructionEmitter.py` (`globalReadDoScaleSubtile`, `globalReadScalePtrUpdates`).

### 4.7 `LogicalScheduler.py` — the logical schedule builder

**Contents.** The largest and most central scheduling module. It builds a
schedule keyed on MMA-tile indices through a fixed, topologically-ordered pass
pipeline (`Pass` enum + `_PASS_PIPELINE`, `:37-73`):

```
place_LRs → assign_vgpr_tiles → place_GRs → annotate_deps
→ remove_unnecessary_gr_deps → remove_unnecessary_lr_deps → remove_cross_deps
→ insert_gr_lr_inc → group_lr_gr → remove_unnecessary_wait_lr_sync → emit → build
```

Key data types: `ReadGranularity` (`:113`, `(mn, k)` per-op load granularity),
`SchedulerConfig` (`:131`, all knobs incl. partition sizes and `pgr`),
`MFMATileRange`, and the op classes `MFMAPlacement`/`LRPlacement`/`GRPlacement`
plus `BaseOp` subclasses (`WaitGROp`, `WaitLROp`, `SyncOp`, `MaskKOp`,
`LRIncOp`, `GRIncOp`, `SkipOp`, `InlineModuleOp`) and `EmittedModule` (carrying
a `before` dependency link consumed downstream).

Relevant methods: `place_LRs`/`place_GRs` (lay reads onto subIterK slots),
`assign_vgpr_tiles` (per-tensor free-list tile assignment + VGPR peaks),
`annotate_deps` and the `remove_*_deps`/`remove_cross_deps` passes (prune
redundant RAW/WAR edges covered by MFMA syncs), `insert_gr_lr_inc` (SRD/LDS
increment preOps at MT transitions), `group_lr_gr` (serialize into paths),
`emit`/`build` (produce `EmittedModule` chains). Loop-variant builders:
`build_preloop`, `build_ngll` (No-Global-Load Loop, drops GR(n+2)), `build_nll`
(No-Load Loop), `build_tailloop_pgr0`. Top-level drivers called by
`Kernel.mainLoop`: `get_partition_candidates` (M/N split options),
`getNumVgpr` (VGPR budget estimate = max of mainloop vs tail peak),
`allocVgprTiles` (physical checkout, 4-VGPR aligned blocks),
`populate_instructions` (constructs an `InstructionEmitter` and fills every loop
variant), `emitMainAndExitLoops` (owns all control flow: skip-to-tail branch,
preloop, mainloop with per-unroll copies, NGLL/NLL exit chains), `emitTailLoop`.

**Interactions.** Imports only `rocisa`. It imports `Kernel.RegisterTileInfo` and
`InstructionEmitter` lazily inside methods to avoid a circular import. Driven
entirely by `Kernel.mainLoop`.

### 4.8 `InstructionEmitter.py` — lower logical ops to instructions

**Contents.** `InstructionEmitter` (`:51`) holds a dispatch table mapping each
`opType` (`mfma`, `lr`, `gr`, `wait_gr`, `wait_lr`, `sync`, `lr_inc`, `gr_inc`,
`skip`, `mask_k`, `inline`) to an `emit_*` method. `emit_mfma` calls back into
`Kernel.emitMfmaInstruction`; `emit_gr` calls `emitSingleBufferLoad`; `emit_lr`
calls `emitSingleDsRead`; scale ops call into `SubtileScaleEmit`. The tail-loop
K-masking is handled by `emit_mask_k_init`/`emit_mask_k`/`emit_mask_k_done`
(per-lane K-remainder predication). `populate` walks an `EmittedModule` list and
fills each module's `.instructions`. `SWaitCntEx` (`:33`) is a `SWaitCnt`
subclass carrying an `adjustVmcnt` flag honored by the instruction scheduler's
post-pass.

**Interactions.** Imports `Kernel.emitMfmaInstruction`, GR/LR/Scale emit
functions, and `rocisa`. Instantiated by `LogicalScheduler.populate_instructions`.

### 4.9 `InstructionScheduler.py` — interleave non-MFMA into MFMA slots

**Contents.** The final low-level pass. `instructionSchedule(emittedModules)`
(`:364`) preserves MFMA order and places non-MFMA instructions into **2 slots
between each adjacent MFMA pair**. `_SlotPlacer` (`:16`) is the placement engine;
`_SchedulingRules` (`:157`) supplies pluggable callbacks: validators
(`oneDsReadPerInterval`, `minGapDsReadBeforeWait`, `minGapDsReadToWait`,
`noM0WithBufferLoad`), an adjuster (`spreadBufferLoads`), and a placement hook
(`trackPlacement`). Helpers `extractPathsFromBeforeDeps` (turn `before` links
into linear chains, split out pre-MFMA paths), `_classifyPaths` (wait_gr paths
first), `_flattenPath`. A post-pass fixes each `SWaitCnt.vlcnt` to account for
buffer-loads the scheduler moved ahead of it.

**Interactions.** Imports only `rocisa`. Operates on the `EmittedModule` output
of the scheduler; invoked from the loop-emit path. Independent of the geometry
and emit files.

---

## 5. Memory handling

- **Global read (GR).** Default `buffer_load` (B32/B64/B128, typically 128-bit
  per lane); when `enableTDM{A,B}` is set, **TDM** `tensor_load_to_lds` streams
  directly into LDS (`SubtileGREmit.py`). Loads are cooperative across the
  wave-group; `loadRatioGR`/`subtileCount`/`subtileStride` describe how many
  subtiles one load round covers and the strided strip layout in M. Per-lane
  byte offsets live in shared VGPRs (`sharedVgprGROffset`); per-subtile-row
  `soffset`s live in SGPRs (`localSubtilesRegister`). Pointers advance with
  `globalReadPtrUpdates`; LDS double-buffering swaps via `emitGRLDSBufferSwap`.
- **LDS write/read.** GR results are written to LDS, then `ds_read` (B32..B128)
  feeds MFMA operand VGPRs. `lraTileAssignment` computes per-lane LR offsets;
  swap VGPRs (`sharedVgprLROffsetSwap`) implement ping-pong LDS buffers. On
  wave32/WMMA, `VPermlane16SwapB32` reshuffles lanes for the WMMA register
  layout.
- **Scale factors.** Simpler: one `buffer_load` per wave covers the whole scale
  tile; `ds_read_b32` per scale group (2 M-adjacent subtiles).

---

## 6. Register allocation

All registers come from the writer's pools (`vgprPool`, `sgprPool`, `agprPool`)
via the typed `RegList` (`SubtileGeometry.py:28`) and `RegisterTileInfo`
(`Kernel.py:880`). Offset registers are checked out by `allocOffsetRegisters`
(tag-dispatched per shape). MMA operand/accumulator tiles are sized as
`ceil(mmaTileRegCount * lrGranularity.k * lrGranularity.mn)` and allocated by the
scheduler's `allocVgprTiles` (`LogicalScheduler.py:2617`, preferred) or the
`allocVgprTileRegisters_legacy` path (`Kernel.py:685`). The mainloop **budgets
before committing**: `getNumVgpr` returns `max(mainloop_peak, tail_peak)` and
`mainLoop` retries partition candidates until `vgprUsed + numVgpr <= MaxVgpr`.
D-tile accumulators prefer **AGPRs** and **spill to VGPRs** when AGPR capacity is
exceeded; `emitMfmaInstruction` aliases `vgpr()`/`accvgpr()` per operand pool
accordingly (`Kernel.py:1054-1057`). On gfx1250 (`MIArchVgpr`/WMMA), accumulators
are plain VGPRs.

---

## 7. GEMM / MFMA emission

`emitMfmaCode`/`emitMfmaInstruction` (`Kernel.py:1107`/`:1039`) iterate the local
MMA-tile grid, one matrix instruction per `(mma0, mma1, mmak)`:

```python
for mmak in range(localMMATileGrid_K):       # K accumulation
  for mma1 in range(N_tiles):
    for mma0 in range(M_tiles):
      C[mma0, mma1] += A[mma0, mmak] * B[mmak, mma1]   # MFMA or MXMFMA
```

- `MatrixInstK == 128` → **MX path**: `MXMFMAInstruction` with per-tile scale
  VGPRs (`mxsa`/`mxsb`) and `op_sel` selection. Plain FP8 (no MX scale) uses a
  unit scale VGPR pre-set to `0x7f7f7f7f` (E8M0 = 1.0) once before the loop
  (`Kernel.py:1304-1309`); the scheduler drops stray non-MFMA instructions from
  the MFMA module, so it can't be initialized inline.
- `MatrixInstK == 32` → **bf16 path**: `MFMAInstruction` (`INST_BF16`/`INST_F32`).
- `_selectF8F6F4InstType` (`:975`) chooses the exact F8/F6/F4 variant, including
  mixed-input combinations, from `DataTypeA/B`.

MatrixInst is constrained to **16×16** (`Solution.py:897`); `SourceSwap=0`,
`VectorWidth=1`, `BufferStore=1` are forced for the Subtile path.

---

## 8. GEMM algorithms available (and not)

**Available:**

- **Data-parallel GEMM with software pipelining.** `PrefetchGlobalRead` 0/1/2 and
  `PrefetchLocalRead`, realized as preloop + mainloop + **NGLL** (No-Global-Load
  Loop) + **NLL** (No-Load Loop) + tail loop. The control flow lives in
  `LogicalScheduler.emitMainAndExitLoops` (`:2444`); loop bodies and per-unroll
  copies are pre-built by `build_*` and lowered by `populate_instructions`.
- **Stream-K** — supported and in fact **mandatory** for Subtile:
  `UseSubtileImpl` rejects `StreamK==0` (no GSU) and requires **StreamK=3 or 4**
  (DP-before-SK mode) (`Solution.py:901-904`). Stream-K K-offsets are applied to
  TDM addresses via `tdmApplyStreamKOffsetSubtile`.
- **MX block-scaled GEMM** (FP4/FP6/FP8 with MXSA/MXSB scale tensors).

**Not available** (explicitly rejected in `Solution.py`): GSU (Global Split-U),
`ScheduleIterAlg` 1/2, `UseCustomMainLoopSchedule`, `DebugStreamK`, non-16×16
MFMA, `SourceSwap`, 64-bit shadow limit, and any non-gfx950/gfx1250 ISA.
Additionally several store/scale emit paths are still **stubs**
(e.g. `emitScaleGROffset` early-returns; `CDTile_1x1.emitStoreD/emitLoadC` are
`pass`), and a `_legacy` migration shadow still exists in the emit files.

---

## 9. Independence from other codegen paths

Subtile is **not fully independent**. It re-implements only the performance-
critical *inner loop*; it reuses the surrounding `KernelWriter` machinery:
function signature, `defineAndResources`, persistent loop, `Component.StreamK`,
`Component.PersistentLoop`, `setupNewTile`, `graAddresses`,
`calculateLoopNumIter`, `computeTailLoopSrdLimit`, `closeLoop`, and the entire
global-write/epilogue path. `KernelWriter.py` carries many
`if kernel["UseSubtileImpl"]` branches — e.g. TDM descriptor and SGPR-pool
handling around `:2572-2888` and the whole `kernelBodySubtile` body
(`:4485-4672`). The reasons are deliberate: Subtile is an **incremental
migration** that replaces the validated legacy mainloop only where new hardware
demands it, while continuing to lean on the shared, already-validated prologue,
StreamK component, register pools, and solution validation that every TensileLite
kernel depends on.

---

## 10. Adding a new tile shape

1. Define a new tag (empty frozen dataclass) in `SubtileGeometry.py`.
2. Add geometry instances using that tag + register them in the maps in
   `Kernel.py` (`AB_GEOMETRY_MAP`, etc.).
3. Register `@_emit*.register(NewTag)` implementations in `SubtileGREmit.py` /
   `SubtileLREmit.py`.
4. Wire selection in `Solution.py` (`_ABTilePair{tc}`), respecting the
   `UseSubtileImpl` constraints.
