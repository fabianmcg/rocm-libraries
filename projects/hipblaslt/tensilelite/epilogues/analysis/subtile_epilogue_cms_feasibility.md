# Feasibility: Making Subtile Fused Epilogues Available Under the CMS Main-Loop Path

**Date:** 2026-10-01
**Status:** Investigation report (explorer pass), revised after adversarial review (static
corrections), then **empirically tested on gfx950** (see final section) — the headline
conclusion did not survive empirical testing and is superseded by that section.
**Scope:** The original pass was read-only/static. A follow-up pass built and ran real
kernels (probe YAML, relaxed validator gates, a spike adapter) to test the claims; those
experiments were reverted from the source tree afterward (see final section for exactly
what was tested and what artifacts remain). All line numbers refer to the working tree at
the time of writing and may drift.

---

## Scope note

This is a read-only investigation of TensileLite as of the current working tree
(`/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite`). All line numbers refer to
files as currently checked out. Where the investigation infers rather than directly observes
a fact, it is called out explicitly below.

---

## 1. What "Subtile" actually provides that CMS kernels currently lack

Subtile is really two separable things bundled under one name:

**(A) A main-loop scheduler/geometry model** (`SubtileGeometry.py`, `Kernel.py`,
`LogicalScheduler.py`, `InstructionScheduler.py`, `ClusterBarrier.py`, `WaitAluInsertion.py`,
`SubtileGREmit.py`, `SubtileLREmit.py`, `SubtileScaleEmit.py`).

- `SubtileGeometry.py` defines a data-type-independent MMA lane-layout model (`MMALayout`,
  `MMAScaleLayout`, `SubtileGeometry.py:108-166`) and frozen tile-geometry dataclasses
  (`ABGRGeometry`, `ABLRGeometry`, `ABTilePair`, `CDTileGeometry`, `MXScale*Geometry`,
  `SubtileGeometry.py:241-680`) that describe how a macro-tile is decomposed into "subtiles"
  — groups of MMA tiles loaded/stored together, with explicit
  `subtileShape`/`subtileCount`/`subtileStride` parameters distinguishing global-read (GR)
  layout from local-read (LR) layout (`SubtileGeometry.py:280-364`).
- `Kernel.py`'s `TileInfo` (`Kernel.py:394-895`) binds these frozen geometries to a live
  kernel config and writer, producing register-pool-backed "tile" objects
  (`RegisterTileInfo`, `Kernel.py:901-920`) for A/B/D/MXSA/MXSB. The main loop
  (`mainLoop()`, `Kernel.py:1361-1558`) builds an `MFMASchedulerConfig`/`LogicalScheduler`
  (`Kernel.py:1398-1429`), which performs multi-pass logical scheduling of GR/LR/MFMA
  placement (`LogicalScheduler.py:1-24` enumerates the passes: `place_LRs`,
  `assign_vgpr_tiles`, `place_GRs`, `annotate_deps`, dependency-pruning passes, `group`,
  `emit`).
- `InstructionScheduler.py`, `ClusterBarrier.py`, `WaitAluInsertion.py` implement
  lower-level instruction ordering, barrier pairing, and `s_wait_alu`/hazard handling for
  that scheduler's output.
- This machinery replaces the entire CMS timeline-validator mechanism
  (`Tensile/Components/CMSValidator.py`) and the curated catalog lookup
  (`Tensile/Components/CustomSchedule.py`, `hasCustomSchedule`, `CustomSchedule.py:518-544`)
  with its own from-scratch scheduling pass over a parametric (not catalog-based) tile
  geometry.
- This part genuinely has **no CMS equivalent** for Subtile's parametric tile/cooperative-load
  model (e.g. GR cooperative loading with `loadRatioGR`, `Kernel.py:220-256` /
  `Solution.py:220-256`), and it is this piece that `Solution.py:3733-3735` says is
  incompatible with CMS ("UseSubtileImpl has its own main loop scheduler; CMS is not
  compatible").

**(B) A fused epilogue** (`SubtileMegaFusedEmit.py`), invoked from the *shared*,
non-Subtile-specific file `Tensile/KernelWriterAssembly.py` via `emitSubtileFusedEpilogue`
(`KernelWriterAssembly.py:15937-15949`), itself called from the common
`globalWriteElements()` entry point (`KernelWriterAssembly.py:16650-16651`), which both CMS
and non-CMS kernels funnel through.

**Key finding:** Part (B), the fused epilogue, does **not** import or call into any of part
(A)'s scheduler modules. `SubtileMegaFusedEmit.py`'s import block
(`SubtileMegaFusedEmit.py:13-91`) pulls only `rocisa` instruction/container primitives,
`math`, `struct`, and `Tensile.Common.DataType` — there is no reference to
`LogicalScheduler`, `InstructionScheduler`, `ClusterBarrier`, `WaitAluInsertion`, or even
`SubtileGeometry`/`SubtileGREmit`/`SubtileLREmit` anywhere in that file. It drives code
generation directly against the generic `writer` object's register pools
(`self.writer.vgprPool`, `.sgprPool`, `.agprPool`, `.allocTmpSgpr`, `self.writer._syncThreads`),
exactly the same primitives any `KernelWriterAssembly` instance (CMS or not) exposes.

This decoupling is the single most important structural fact for this investigation:
**the fused epilogue's mechanism is not inherently tied to the Subtile main-loop scheduler.**
Its only hard couplings are (i) a Subtile-specific accumulator register representation, and
(ii) Solution.py's validator gating — both discussed in §2.

---

## 2. Where epilogue logic is coupled to Subtile-specific state vs. independent

### 2a. `RMSEpilogueGeometry` (`SubtileMegaFusedEmit.py:132-234`) is generic MFMA-output geometry, not Subtile geometry

`RMSEpilogueGeometry.__init__` (`SubtileMegaFusedEmit.py:135-141`) derives everything from
plain kernel-dict fields that are universal Tensile solution parameters, not Subtile
constructs:
- `MatrixInstM`/`MatrixInstN`/`WavefrontSize`/`MIWaveGroup`/`MacroTile0`/`MacroTile1`
  (`SubtileMegaFusedEmit.py:149-162`) — these are standard parameters used by every
  MFMA-based kernel including CMS ones.
- `mmaM = (MacroTile0 // mfmaM) // wgM` and `mmaN = (MacroTile1 // mfmaN) // wgN`
  (`SubtileMegaFusedEmit.py:157-158`) is exactly the standard `MIWaveTile` formula
  (MacroTile / (MatrixInst × MIWaveGroup)) used throughout TensileLite, not something
  Subtile invented.
- The lane/row math in `emit()` (`SubtileMegaFusedEmit.py:1316-1333`:
  `rowGroup = laneId >> log2(mfmaN)`, `rowGroupOff = rowGroup * rowsPerLane`) is the
  standard MFMA 16x16 output lane layout, and the row/col addressing uses
  `WorkGroup0`/`WorkGroup1`/`Serial` SGPRs/VGPRs (`SubtileMegaFusedEmit.py:1351-1430`) —
  again standard kernel-wide quantities available to every kernel.

So the *geometry derivation* is portable in principle: nothing in `RMSEpilogueGeometry`
reads a `TileInfo`/`SubtileGeometry` object.

### 2b. The hard coupling: `self.states.d.tileInfo.vgprTiles`

The emitter is handed a `vgprTiles` list (`KernelWriterAssembly.py:15942-15947`,
`self.states.d.tileInfo.vgprTiles`), and indexes it as
`vgprTiles[n * self.geom.mmaM + m]` (`SubtileMegaFusedEmit.py:465`), then reads/writes
individual rows via `tile.regList.indices[ki]`, branching on
`tile.regList.pool == self.writer.vgprPool` vs. AGPR (`SubtileMegaFusedEmit.py:466-473`).

This `tileInfo` attribute is **only ever populated when `kernel["UseSubtileImpl"]` is
true**: `KernelWriter.py:7581-7589` calls `initSubTileInfo('D')` (and A/B/MXSA/MXSB) only
inside `if kernel["UseSubtileImpl"]:`. The field itself is declared generically on the base
(non-Subtile) `MatrixInfo` dataclass shared by all kernels —
`tileInfo: object = field(init=False)  # TileInfo for all tile types`
(`KernelWriter.py:109-117`) — but for a CMS/non-Subtile kernel it is simply never assigned,
so accessing it would raise `AttributeError`. `emitSubtileFusedEpilogue` guards this with an
early return on `not kernel.get("UseSubtileImpl")` (`KernelWriterAssembly.py:15940-15941`)
specifically to avoid that.

CMS kernels instead address their D/C accumulator through a **formula-based** scheme,
`mapAcctoArchRegs` / `accToArchMapper` (`Tensile/KernelWriterModules.py:183-305`), which
computes a permutation `acc2arch` from `MIWaveTile`, `VectorWidthA/B` (output vector
width), `SourceSwap`, `MatrixInstBM/BN` (`KernelWriterModules.py:159-170, 187-204`) and then
addresses either a flat AGPR range (`accvgpr(srcIdx)`) or a flat VGPR range
(`vgpr("ValuC+%u"%srcIdx)`) — there is no list of discrete per-tile Python objects, just
index arithmetic against a contiguous register block.

**Important nuance (confidence: high, directly observed):** Subtile forces
`VectorWidthA=1`, `VectorWidthB=1`, `SourceSwap=False` (`Solution.py:1478-1480`), which is
exactly the condition under which `accToArchMapper`'s permutation degenerates toward the
simple sequential/`n*mmaM+m` ordering the epilogue assumes. General CMS-eligible
configurations are *not* constrained this way a priori — `accToArchMapper`
(`KernelWriterModules.py:197-202`) produces a non-trivial permutation whenever
`SourceSwap=True` or output `VectorWidth>1`. No explicit evidence was found in the CMS
registry (`CustomSchedule.py`) that any of its 13 registered tile configs use
`SourceSwap=True` or output-vector-width > 1 (the probe harness varies
`VectorWidthA/VectorWidthB` only in the context of A/B operand vector widths used for
layout-detection, `CustomSchedule.py:722-726, 753-775`, not necessarily for output
`VectorWidth`), so **whether real CMS-registered kernels actually hit the non-identity
`acc2arch` case is unresolved from the evidence gathered — flagged as uncertain** and would
need to be checked against actual generated CMS solutions before any port.

Also worth noting: a precedent for bridging these two representations already exists.
`mapAcctoArchRegs` has a `spilledVgprBase` parameter whose comment says exactly this: "For
subtile kernels, D-tile accumulators that overflow the accvgpr pool are placed in arch
vgprs allocated from the vgpr pool... mapAcctoArchRegs needs to know the base address of
those vgprs" (`KernelWriterAssembly.py:9067-9081`, `KernelWriterModules.py:256-266`). I.e.,
the shared "Acc register -> C Vgpr register" copy code (used by every kernel, CMS included)
has *already* been special-cased once to understand Subtile's register-pool-based tile
allocation. This is direct evidence that bridging the two addressing schemes is a known,
previously-solved kind of problem in this codebase, not an open research question — but it
is also evidence that no generic abstraction yet exists; each new consumer (here,
`mapAcctoArchRegs`) has had to add its own `if kernel.get("UseSubtileImpl")` special case.

### 2c. Other couplings, all confirmed to be *validator*-level (not mechanism-level)

- `MIArchVgpr=False` requirement (`Solution.py:314-319`, `_validateSubtileEpiloguePrereqs`)
  — **not actually Subtile-specific as a mechanism**: CMS/standard kernels fully support
  `MIArchVgpr=False` (AGPR accumulation) today; `GlobalWriteBatch.py:290, 1337, 1349, 1364`
  and `KernelWriterAssembly.py:6622, 9086, 9359, 9368, 9438, 17178` all branch generically
  on `MIArchVgpr`. So this constraint, as stated, is not an obstacle to reuse by CMS kernels
  — CMS kernels can already be built with `MIArchVgpr=False`.
- `ISA == gfx950` requirement, both for `_validateSubtileEpiloguePrereqs`
  (`Solution.py:306-308`) and for `hasCustomSchedule`/CMS eligibility (`CustomSchedule.py:527`)
  — both paths already target the *same* hardware generation. No ISA mismatch.
- `MatrixInst 16x16` requirement for RMSEpilogue (`Solution.py:377-378`) vs. CMS's registered
  catalog, which uses `matrix_inst=[16, 16, 32, 1]` (bf16) or `[16, 16, 128, 1]` (fp8) for
  nearly all entries (`CustomSchedule.py:676, 967, 1029, 1167, 1305, 1358, 1504, 1633, 1843,
  1974, 2078, 2197, 2270, 2421, 2546`). **Correction (adversarial review):** one entry,
  `CustomSchedule.py:4494` (inside the `_get_schedule_128x128x32_TF32_plr1` TF32 schedule
  function), uses `matrix_inst=[32, 32, 16, 1]` — the original "all `16,16,...`" claim was
  false. That entry is a TF32 config, already excluded by the dtype whitelist below, so the
  practical conclusion (MI16x16 world) is unaffected, but the blanket statement was wrong as
  written.
- Kernarg/SGPR plumbing for `RMSEpilogue` (RMSNormGamma/PartialBuf/ResidualBuf/
  AddressResidualOut/MXScale) is gated purely on `kernel["RMSEpilogue"]`, in the **shared**,
  non-Subtile files `Tensile/Components/Signature.py:443-484` and
  `Tensile/KernelWriter.py:10899-10925` — **not** on `UseSubtileImpl`. This plumbing already
  works generically.
- `StreamK` constraint: `_validateSubtileEpiloguePrereqs` requires `StreamK==0` or
  `StreamKForceDPOnly==1` (`Solution.py:326-331`), reasoned about via the Subtile/StreamK
  deferred-store interaction. Whether CMS + StreamK is itself a supported/exercised
  combination is **not established by the evidence gathered** (no explicit CMS+StreamK
  mutual-exclusion found in `Solution.py`, but also no confirmation it's validated/tested)
  — flagged as uncertain.

### 2d. Scheduling/barrier hooks: not used by the epilogue

Confirmed by direct inspection (§1): the epilogue emits its own `s_waitcnt`/barrier
sequencing manually (e.g. `self.writer._syncThreads(...)`, `SubtileMegaFusedEmit.py:610-611`;
manual `SWaitCnt(vlcnt=0)`/`SWaitCnt(kmcnt=0)`, `SubtileMegaFusedEmit.py:1314-1315`) rather
than going through `LogicalScheduler`/`InstructionScheduler`/`ClusterBarrier`. So **no new
scheduler-hook infrastructure is required on the CMS side** merely to run the epilogue —
this is category (a)/(b) work, not category (c).

---

## 3. How CMS kernels currently implement their own epilogue/post-loop store path

CMS and Subtile kernels **already share the same post-loop store entry point**.
`notLocalSplitUGlobalWrite()` (`KernelWriterAssembly.py:15216-15237`) is the single
top-level post-loop function; it computes element/vector-width batching via
`notLocalFullTileElements`, then always calls `self.globalWriteElements(...)`
(`KernelWriterAssembly.py:15228-15232`), regardless of CMS/non-CMS/Subtile.
`globalWriteElements` (`KernelWriterAssembly.py:15951` onward) is the shared machinery that
handles beta-C, alpha, bias, activation, dtype conversion, bounds/edge handling, StreamK
partials, GSU reduction, etc. via `Component.GlobalWriteBatch`/`GlobalWriteBatchWriter`
(`Tensile/Components/GlobalWriteBatch.py`).

Concretely:
- `GlobalWriteBatchWriter` (`GlobalWriteBatch.py:163-...`) drives per-element store code
  generation (`_prolog`, `_epilog`, `_emitAdd`, `_applyAlpha` at `GlobalWriteBatch.py:4214`,
  `_addSumAlphaWithCBeta` at `GlobalWriteBatch.py:4323`, activation/bias loads at
  `_emitElt0EpilogueLoads` `GlobalWriteBatch.py:705`), addressing the C/D accumulator
  through `MIRegPerOut`-based formulas (`GlobalWriteBatch.py:435`) — the same
  `mapAcctoArchRegs`/`accToArchMapper` style as described in §2b.
- This same file already contains Subtile-aware special cases bolted on:
  `_emitSubtileOobGuard`/`_finalizeSubtileOobGuards` (`GlobalWriteBatch.py:3417, 3505`),
  `_emitSubtilePackedPermute` (`GlobalWriteBatch.py:2527`),
  `_emit16bitSubtilePairedStore`/`_emit16bitSubtileScalarStore`
  (`GlobalWriteBatch.py:3633, 3760`). This confirms UseSubtileImpl kernels *already* go
  through this shared store path for their *non-fused* (plain beta/activation/bias) output,
  with extra Subtile-only branches layered in — it is not a parallel, fully-separate store
  mechanism.
- For the RMSEpilogue (fused) case specifically: `notLocalSplitUGlobalWrite` sets
  `ownsEpilogue = kernel["RMSEpilogue"] and DestDataType.isFloat8()`
  (`KernelWriterAssembly.py:15223-15226`) and, only in that MXFP8 sub-case, disables the
  normal alpha-apply and forces `betas=[False]` because "The MXFP8 epilogue... applies
  alpha and handles beta=0 itself" (comment at `KernelWriterAssembly.py:15223`). Otherwise,
  the standard `globalWriteElements` path still runs in full.
- `emitSubtileFusedEpilogue` is invoked from inside `globalWriteElements`, right after the
  StreamK role-C branch (`KernelWriterAssembly.py:16644-16651`), i.e. *before* the normal
  per-element store loop that follows later in the same function. Based on
  `_amaxAndWriteAcc`'s explicit comment "write VGPR back to accumulator register file"
  (`SubtileMegaFusedEmit.py:464`) and code (writes `H*gamma` back into the exact same
  `vgprTiles[...].regList` registers that originally held the raw MFMA accumulation,
  `SubtileMegaFusedEmit.py:465-473`), the design is: **the fused epilogue mutates the
  accumulator in place, then falls through to the ordinary shared store/convert/activation/
  bias code path to actually produce the final global D write.** (This fall-through
  relationship is inferred from code structure and comments, not directly traced at
  runtime, so confidence here is high-but-not-certain.)

This is a materially favorable finding for feasibility: the "standard CMS epilogue" is not
a fundamentally different code path that would need to be bypassed or duplicated — it is
the same shared function the fused epilogue already partially rides on top of for Subtile
kernels today.

---

## 4. Validation/gating machinery

- **Primary exclusivity gate:** `Solution.py:3733-3735`:
  ```python
  # UseSubtileImpl has its own main loop scheduler; CMS is not compatible.
  if state["UseSubtileImpl"] and state["UseCustomMainLoopSchedule"] == 1:
      reject(state, printRejectionReason, "UseCustomMainLoopSchedule=1 is incompatible with UseSubtileImpl")
  ```
  This sits right after CMS itself is resolved (`state["UseCustomMainLoopSchedule"] = 1 if
  hasCMS else 0`, `Solution.py:3728`) and is a pure boolean AND-exclusion; it carries no
  epilogue-specific logic at all. In principle this single line is what would need
  loosening to let a kernel be both `UseCustomMainLoopSchedule=1` and run the fused
  epilogue — **but only if the fused epilogue is first decoupled from
  `UseSubtileImpl`-gated state**, since nearly every other RMSEpilogue validator (next
  bullet) is written in terms of `UseSubtileImpl`, not in terms of "has a
  fused-epilogue-compatible accumulator."

- **RMSEpilogue-specific gate:** `_validateRMSEpilogue` (`Solution.py:368-453`) calls
  `_validateSubtileEpiloguePrereqs` (`Solution.py:295-332`), whose very first check is:
  ```python
  if not state["UseSubtileImpl"]:
      reject(state, printRejectionReason, "%s requires UseSubtileImpl" % epilogueName)
  ```
  (`Solution.py:303-305`). This is the direct, explicit block preventing RMSEpilogue on
  non-Subtile (hence CMS) kernels today. It is one `if`, independent of the CMS-exclusivity
  line above — i.e. there are *two* separate guards that both currently encode
  "RMSEpilogue/MegaFused only runs under Subtile," and *both* would need to change (or be
  made conditional on a new, accurate capability check) to unlock CMS+RMSEpilogue.
  **Correction (empirical pass, see final section): this is wrong — there is a *third*
  literal `UseSubtileImpl` gate, `KernelWriterAssembly.py:15940`, an early-return inside
  `emitSubtileFusedEpilogue` itself. Relaxing only the two `Solution.py` gates produces a
  kernel that builds cleanly but silently never runs the epilogue — this third gate was
  found only by actually relaxing the first two and observing the result, not by reading.**

- **Interacting parameters already covered in validators** that any port would have to
  re-derive compatibility against: `MIArchVgpr` (`Solution.py:314-319`), `ISA==gfx950`
  (`Solution.py:306-308`), dtype whitelist bf16/f16/fp8/bfp8 (`Solution.py:309-313`),
  `StreamK`/`StreamKForceDPOnly` (`Solution.py:326-331`), `PrefetchAcrossPersistent`
  exclusion (`Solution.py:388-391`), `MacroTile0/1` multiple-of-64 and `MIWaveGroup[0]`
  power-of-two constraints for the cross-wave reduction LDS scratch (`Solution.py:413-441`),
  `_GlobalAccumulation` exclusions for `MultipleBuffer(SingleKernel)`/`AdaptiveGemmGSUA`
  (`Solution.py:399-409`), `GroupedGemm` exclusion (`Solution.py:410-412`), and the
  MXFP8-specific `HighPrecisionAccumulate`/`UseBeta=False` requirements
  (`Solution.py:446-453`). None of these are inherently about the *main-loop scheduler*;
  they are about epilogue arithmetic/kernarg-layout constraints, so in principle they'd
  carry over to a CMS variant largely unchanged — but each would need re-auditing against
  CMS's narrower, catalog-based kernel space (§5).

**Retraction (empirical pass, see final section):** §2c above separately claimed the
RMSEpilogue kernarg/SRD plumbing "already works generically" because it's gated only on
`kernel["RMSEpilogue"]`, not `UseSubtileImpl`. This is **false in the shadow-init case**.
`SrdResidualOut` is only defined in the non-shadow-init store-SGPR branch
(`KernelWriterAssembly.py:8997-9009`); CMS kernels take the shadow-init path
(`doShadowInit=2`, enabled whenever `not ForceDisableShadowInit and not UseSubtileImpl and
PGR`, `KernelWriter.py:7696-7699`), where `SrdResidualOut` is never defined, producing a
`KeyError: 'SrdResidualOut'`. This is a second real coupling the static report missed
entirely, found only by actually building a kernel.

- `CMSValidator.py` itself (2499 lines) is entirely about main-loop GR/LR/MFMA/Pack/Wait/
  Barrier timeline validation (class list at `CMSValidator.py:38-2434`:
  `SchedulePosition`, `ValidatorInstruction` subclasses `LocalRead`/`MFMA`/`Pack`/
  `GlobalRead`/`SWait`/`Barrier`, `Timeline`, `apply_*` passes, `isValid`). It has no
  epilogue-related logic and would not need to change for this port; it only needs to keep
  validating the main loop it already validates.

---

## 5. Concrete list of obstacles

**(a) Low effort — validator/gating changes:**
1. `Solution.py:3734-3735` — the blanket `UseSubtileImpl AND UseCustomMainLoopSchedule`
   exclusion. Trivial to special-case ("unless RMSEpilogue is targeting the
   CMS-compatible path"), but only safe to relax once (b)/(c) below are actually
   addressed; relaxing this line alone does nothing useful on its own.
2. `Solution.py:303-305` inside `_validateSubtileEpiloguePrereqs` — the literal "requires
   UseSubtileImpl" check. Needs to become "requires UseSubtileImpl OR
   (UseCustomMainLoopSchedule and a verified-compatible accumulator layout)".
3. Kernarg/Signature plumbing (`Signature.py:443-484`, `KernelWriter.py:10899-10925`) —
   **already gated only on `kernel["RMSEpilogue"]`**, not `UseSubtileImpl`. No change
   needed here; this part is already portable (evidence, not inference).

**(b) Medium effort — re-deriving epilogue geometry/addressing for CMS's data layout, no
new mechanism:**
4. Building a `vgprTiles`-equivalent accumulator view for CMS kernels. The information
   needed (which physical AGPR/VGPR holds the accumulator value for MFMA output tile
   `(m, n)`, row `ki`) is already computable via `accToArchMapper`/`mapAcctoArchRegs`
   (`KernelWriterModules.py:183-305`) — this is a "no new mechanism" case *as long as* the
   target CMS configuration has an identity (or otherwise invertible/known) `acc2arch`
   permutation. Concretely this means either (i) writing an adapter that constructs a flat
   tile list from `acc2arch`+`MIRegPerOut`, handling the AGPR/VGPR spill boundary exactly
   as `mapAcctoArchRegs`'s `spilledVgprBase` logic already does
   (`KernelWriterModules.py:256-274`), or (ii) rewriting `_amaxAndWriteAcc`/
   `_issueUnitLoads`/etc. in `SubtileMegaFusedEmit.py` to index through
   `accToArchMapper` directly instead of through `vgprTiles`.
5. Re-validating/re-deriving the `MIWaveGroup` cross-wave reduction LDS-scratch math
   (`Solution.py:429-441`) and MXFP8 quant-tile derivation (`SubtileMegaFusedEmit.py:187-234`)
   against whichever specific CMS tile shapes (from the catalog, `CustomSchedule.py`) are
   targeted — the formulas themselves (`MacroTile0 % 64`, `MIWaveGroup` power-of-two,
   `tilesPerBlockM`) are generic, but must be checked per CMS-catalog tile shape since CMS
   is a fixed, narrow catalog of `(MT0,MT1,DU,PGR,PLR,...)` tuples (e.g.
   `TileConfig(256, 96, 64, 2, 1, 1, False, 0, 0)` at `CustomSchedule.py:673`), not an open
   parametric space like Subtile. **Correction (adversarial review):** the catalog is **39
   entries** (`grep -c "tile_config=TileConfig" Tensile/Components/CustomSchedule.py` → 39),
   not the 13 originally cited — the original count was taken from an earlier, much smaller
   version of this file. Of the 39, 14 are TF32-only schedule functions
   (`_get_schedule_*_TF32*`, e.g. `CustomSchedule.py:3437,3503,3833,4113,4320,4408,4497,
   4682,4813,5130,5223,5323,5415,5430`) and are already excluded by the dtype whitelist
   (`Solution.py:309-313`: bf16/f16/fp8/bfp8 only), leaving an effective candidate set of
   roughly **25** bf16/fp8 MI16x16 configs to re-validate — not independently re-verified
   entry-by-entry beyond this grep, so treat 25 as an estimate, not an exact count.
6. Confirming/handling the `accToArchMapper` permutation in the non-identity case
   (SourceSwap=True or output VectorWidth>1) — unresolved from evidence gathered whether
   this actually arises for CMS-registered configs (flagged uncertain in §2b, and the
   adversarial review did not resolve this either — still open).
7. The storeBranches/StreamK role-C integration point
   (`KernelWriterAssembly.py:16642-16651`) where `emitSubtileFusedEpilogue` is currently
   hard-wired right after `skComponent.storeBranches`. If CMS+StreamK is a real target
   combination, this placement (and the `gsuLimitIdx == gsuLimit - 1` final-iteration
   gating at `KernelWriterAssembly.py:16650`) needs re-verification against CMS's
   StreamK/GSU control flow, which this investigation did not trace in detail.

**(c) High effort — new infrastructure CMS would need that Subtile provides "for free":**
8. None identified with high confidence. The main candidate — a scheduler hook to
   interleave the epilogue with the loop — is **not actually used by the existing
   MegaFused emitter** (§1, §2d: it's a self-contained post-loop emission using generic
   pool/barrier primitives already available to CMS kernels). So no concrete evidence of a
   category-(c) obstacle was found; this is a positive finding that should be treated with
   some caution since it rests on the absence of evidence for a scheduler dependency (true
   for the *current* MegaFused implementation) rather than a guarantee that no future
   epilogue variant would need one.

**(d) Fundamental architectural mismatch:**
9. None identified. The two "camps" (Subtile and CMS) already share the entire post-loop
   store file (`KernelWriterAssembly.py`/`GlobalWriteBatch.py`) and the entire
   kernarg/Signature layer; the actual divergence is confined to the main loop (out of
   scope for this specific epilogue-reuse question) and to one data structure
   (`TileInfo.vgprTiles` vs. `accToArchMapper`-style index arithmetic) that already has a
   documented bridging precedent (`spilledVgprBase`, `KernelWriterAssembly.py:9067-9081`).

---

## 6. Overall feasibility assessment

This is **not** "essentially a rewrite," and the premise that the fused epilogues are "only
reachable through the Subtile main-loop scheduler" turns out, on direct inspection, to be
**not quite accurate as a mechanism-level claim**: the fused epilogue is invoked from and
partially falls through to the exact same shared
`KernelWriterAssembly.globalWriteElements`/`GlobalWriteBatch` code that every kernel (CMS
included) already uses, and it has zero code dependency on the Subtile main-loop scheduler
modules (`LogicalScheduler.py`, `InstructionScheduler.py`, `ClusterBarrier.py`,
`WaitAluInsertion.py`). What *is* accurate, and is the real obstacle, is narrower: the
epilogue is coupled to **one specific accumulator-addressing abstraction**
(`TileInfo.vgprTiles`) that is only populated when `UseSubtileImpl=1`
(`KernelWriter.py:7581-7589`), plus **two validator lines** that explicitly require
`UseSubtileImpl` (`Solution.py:303-305`, `Solution.py:3734-3735`).

Given that, this is best characterized as **a well-scoped, medium-effort engineering task,
not a significant redesign** — provided the goal is "make MegaFused runnable under a CMS
main loop for some subset of the existing CMS tile catalog." The core work is:
1. Writing a `vgprTiles`-equivalent adapter over `accToArchMapper`/`mapAcctoArchRegs` for
   CMS kernels (§5.4), including handling the AGPR/VGPR spill boundary the same way the
   existing `spilledVgprBase` special-case does.
2. Relaxing the two Solution.py gates (§5.1-5.2) to depend on "has a compatible
   accumulator layout" instead of literally `UseSubtileImpl`.
3. Re-validating the handful of geometric/kernarg constraints (§5.5) against CMS's
   specific, narrow tile catalog (39 entries, ~25 after excluding TF32; not an open
   parametric space), and resolving the open question about `SourceSwap`/output-
   `VectorWidth` permutations (§5.6) if any CMS-registered config actually uses them
   non-trivially.

What would make this a *larger* effort than "medium" — and could not be fully ruled out
from static reading alone — is if real end-to-end testing surfaces interactions this
investigation didn't trace: CMS+StreamK co-existence (unconfirmed either way, §2c/§5.7),
whether any CMS catalog entry actually produces a non-identity `acc2arch` permutation
(§2b/§5.6), and whatever GPU-level correctness subtleties only show up in practice (the
Subtile side of this code is itself full of recently-fixed, narrowly-scoped bugs, e.g. the
GR K-partition bug guarded at `Solution.py:220-256`, suggesting this general area of the
codebase is delicate and under active hardening).

**Bottom line:** feasible as a near-to-medium-term port focused on bridging one data
structure and loosening two validator checks, not a ground-up redesign of either side —
but confidence in "medium effort" rather than "medium-high effort" is capped by the
unresolved questions in §2b/§2c/§5.6/§5.7, which would need to be answered (ideally by
running the solution-generation and codegen pipeline, which this investigation did not do)
before committing to an effort estimate with high confidence.

---

## Adversarial review

A second agent independently re-checked every file:line citation and central claim in this
report against the live codebase, without trusting the original's confidence labels.
**Verdict: changes requested, both now applied above.**

**Errors found and corrected:**
- The CMS tile catalog was cited as having 13 entries in §5.5/§5.6/§6; the actual catalog
  (`Tensile/Components/CustomSchedule.py`) has **39** registered tile configs
  (`grep -c "tile_config=TileConfig"` → 39). The original figure was apparently taken from
  an earlier, much smaller version of that file. This directly affected the stated
  per-catalog-entry validation effort in §5.5/§5.6/§6 and has been corrected above, along
  with a refinement: 14 of the 39 entries are TF32-only and already excluded by the dtype
  whitelist, leaving an effective candidate set of roughly 25 entries.
- §4's claim that all CMS registry entries use `matrix_inst=[16,16,...]` is false: one entry
  (`CustomSchedule.py:4494`, inside a TF32 schedule function) uses `[32, 32, 16, 1]`.
  Corrected above; the practical conclusion is unaffected since that entry is a TF32
  config already excluded on dtype grounds.

**Claims independently re-verified as CONFIRMED (not just re-asserted):**
- `SubtileMegaFusedEmit.py` has zero import/call dependency on `LogicalScheduler.py`,
  `InstructionScheduler.py`, `ClusterBarrier.py`, `WaitAluInsertion.py`, `SubtileGeometry.py`,
  `SubtileGREmit.py`, `SubtileLREmit.py` — verified independently via import inspection and
  full-file grep, not just re-reading the original claim.
- `emitSubtileFusedEpilogue` is invoked from the shared `globalWriteElements` path that CMS
  kernels already use (not a Subtile-only call site).
- `TileInfo.vgprTiles` is populated only when `UseSubtileImpl=True`
  (`KernelWriter.py:7581`), confirmed as the one hard coupling — no other hidden
  Subtile-only state reachable from the epilogue's call site was found.
- The `spilledVgprBase` bridging precedent is exactly as described.
- The two `UseSubtileImpl`-literal validator gates (`Solution.py:303-305`,
  `Solution.py:3734-3735`) are the only such gates — confirmed by exhaustive search, no
  third gate missed.
- No CMS+StreamK mutual-exclusion exists anywhere in `Tensile/` (confirmed by grep); the
  report's choice to leave this as an open/untested question (rather than claiming it's
  safe) was judged appropriate, not overcautious.

**Still open / unresolved** (adversarial pass did not close these either):
- Whether any CMS-registered config actually produces a non-identity `acc2arch` permutation
  (SourceSwap=True or output VectorWidth>1) — unresolved by either pass; would need to be
  checked against generated CMS solutions, not static reading.
- Whether CMS+StreamK is an actually-tested/supported combination in practice (absence of a
  validator exclusion is not the same as confirmed support).

**Independent bottom-line from the reviewer:** the report's core structural findings all
hold up under adversarial re-verification, and the bottom-line conclusion — "a well-scoped,
medium-effort engineering task, not a significant redesign" — is structurally sound and is
**affirmed**, conditional on the corrections above (which don't change the conclusion, only
the precision of the supporting numbers) and on the two still-open questions being resolved
before committing to an effort estimate with high confidence.

**Note: this verdict was reached by static reading plus adversarial re-reading, neither of
which built or ran a kernel. The empirical pass below supersedes it — the headline
conclusion did not survive actually building and running real kernels on gfx950.**

---

## Empirical validation (built and ran real kernels on gfx950)

A third pass stopped trusting static analysis entirely and tested the claims directly:
built a probe YAML, ran it against the stock validators, relaxed the cited gates and
rebuilt, and — once the predicted crash was reproduced — went one step further with a
minimal spike adapter to see whether the bridge the report proposed actually works end to
end. All experiments were reverted from the source tree afterward; only two untracked
artifacts remain: this file and `epilogues/YAMLs/cms_rms_epilogue_probe.yaml` (a CMS-tile
+ RMSEpilogue probe config, documented in its own header, reproducing the stock-validator
rejection as committed — not yet reviewed for whether it's worth keeping as a permanent
test fixture).

### Open question §2b (acc2arch permutation): RESOLVED — and the news is bad

All 23 unique non-TF32 `(dtype, MT0, MT1, MI, MIWG)` combos in the CMS registry were run
through the real `accToArchMapper` under three accumulator regimes:

| regime | identity | non-identity |
|---|---|---|
| `SourceSwap=False, VectorWidth=1` (the regime Subtile forces, `Solution.py:1478-1480`) | 23/23 | 0 |
| `SourceSwap=False, VectorWidth=auto` (default bf16/fp8 tuning) | 8 | **15** |
| `SourceSwap=True` (any VectorWidth) | 0 | **23** |

The CMS registry's matcher (`CustomSchedule.py:777-814`) keys only on tile/GRVW/LRVW/MI/
MIWG — it does not constrain `SourceSwap` or output `VectorWidth` at all. So "is acc2arch
identity" is not a property of a catalog entry; it's a property of tuning choices made
independently of CMS eligibility. Under realistic/default tuning, **most CMS bf16 configs
are already non-identity** even without `SourceSwap`. The simple case the report hoped for
is reachable only by deliberately re-pinning the Subtile regime's constraints onto a CMS
kernel, which is itself an extra restriction a real port would have to justify or accept.

### "Two validator gates block this": REFUTED — there are three, and a probe confirms the exact chain

- Building `cms_rms_epilogue_probe.yaml` against the stock validators reproduces the
  predicted rejection exactly: `reject: RMSEpilogue requires UseSubtileImpl`, firing at
  `Solution.py:303-305` before CMS even resolves.
- Relaxing *only* the two cited `Solution.py` gates (303-305, 3734-3735) does **not**
  reproduce the predicted `vgprTiles` crash. Instead the kernel **builds cleanly** with the
  fused epilogue silently skipped — because of a third, previously-uncited gate,
  `KernelWriterAssembly.py:15940`, an early-return inside `emitSubtileFusedEpilogue` itself.
- Only after also bypassing that third gate does the predicted crash appear, verbatim:
  `AttributeError: 'MatrixInfo' object has no attribute 'tileInfo'`
  (`KernelWriterAssembly.py:15943`, i.e. `self.states.d.tileInfo.vgprTiles`) — so the
  report's core *mechanism* prediction was right, but its count of "what's gating it" was
  incomplete by one, and that omission is exactly the kind of thing static reading alone
  reliably misses (an early-return nested inside the function body, not a validator).
- Two more validator interactions surfaced only by actually trying to build a runnable
  config: `RMSEpilogue` needs an enabled accumulation mode (`GSU=1` default hits the
  `MultipleBuffer` reject at `Solution.py:405`; `GSU=0+StreamK=0` hits the generic "Either
  GSU or StreamK must be enabled" at `Solution.py:2426`) — resolved for the probe with
  `GlobalSplitUAlgorithm: SingleBuffer`.

### CMS + StreamK: CONFIRMED working at the codegen level (previously untested)

A standalone bf16 GEMM matching the CMS 256x256x64 catalog entry, with
`StreamK=3 + StreamKForceDPOnly=1` and no RMSEpilogue, built successfully end to end
(solution name contains `_CMS_..._SK3_SKFDPO1...`). No validator excludes the combination.
Caveat: this confirms codegen only, not GPU numerical correctness. Notably, the repo's own
StreamK characterization fixture (`Tensile/Tests/unit/.../gfx950/MX_StreamK.yaml`) pairs
`StreamK:3` only with `UseCustomMainLoopSchedule:0` — so this combination was genuinely
unexercised anywhere in the repo before this probe, not just "unconfirmed in the report."

### Stretch spike — "one hard coupling, medium-effort bridge": REFUTED as understated

A minimal identity-case `vgprTiles` shim over `accToArchMapper` was wired into
`emitSubtileFusedEpilogue` for the probe config (physical acc register for tile
`t = n*mmaM + m`, row `ki` → `OutputsPerMFMA1B*t + ki`, AGPR pool, `MIRegPerOut=1`).
Outcome:
1. The shim **does get past the `vgprTiles` coupling** — confirming that specific coupling
   is bridgeable the way the report described.
2. It then hit the `SrdResidualOut`/shadow-init coupling described above (not anticipated
   by the report at all) — worked around with `ForceDisableShadowInit: [True]`.
3. With both fixes applied, the CMS+RMSEpilogue kernel **built, assembled, and ran on the
   GPU** (46.5us, ~5765 GFLOPS) — **but failed correctness validation**: `partialBuf` (the
   RMS cross-wave reduction) mismatched on 379/512 rows (right order of magnitude, wrong
   value — e.g. one row: CPU 4.24e6 vs. GPU 3.81e6, suggesting a permutation/indexing
   mismatch rather than random corruption), and `residualOut`/`d` also mismatched.

So even in the easiest case the report could imagine (identity permutation, single
coupling), bridging the data structure is **necessary but not sufficient** — the epilogue's
lane-layout and cross-wave-reduction assumptions, built and tuned against the Subtile main
loop's register/lane conventions, do not transfer correctly to a CMS kernel's accumulator
layout. This numerical-correctness gap is invisible to static code reading and is the
actual dominant cost of this port, not the `vgprTiles` bridge the original report focused
on.

### Revised bottom line (supersedes §6 above)

The report's structural claims (no scheduler dependency; shared store path; `vgprTiles` is
*a* real coupling and is bridgeable) all held up and were confirmed by actually running
code, not just reading it. But the headline effort conclusion — "well-scoped, medium-effort,
not a redesign" — **does not survive empirical testing** and should be revised:

- "Two validator gates" → **three** literal `UseSubtileImpl` gates block this
  (`Solution.py:303-305`, `Solution.py:3734-3735`, `KernelWriterAssembly.py:15940`), plus
  two GSU/StreamK-interaction rejects that only surface when actually trying to build a
  runnable config.
- The §2c claim that kernarg/SRD plumbing "already works generically" is **retracted**:
  `SrdResidualOut` is absent on the shadow-init path CMS kernels take by default.
- The §2b hope that the accumulator permutation might often be identity is **narrowed**:
  it's identity only when `SourceSwap=False` and output `VectorWidth=1` are deliberately
  pinned — not a property of the CMS catalog, and not true under default tuning for most
  configs.
- A previously-unknown, and likely the single largest, cost item is added: **numerical
  correctness of the epilogue's lane/reduction layout under CMS's accumulator mapping is
  broken even in the easiest (identity-permutation) case**, and fixing it is open-ended
  register/lane-layout work, not a known bridge.

**Revised effort grade: medium-high, not medium.** The critical path is no longer "bridge
`vgprTiles`" (confirmed easy) — it's "make the epilogue's cross-wave RMS reduction and
per-lane addressing correct under CMS's `accToArchMapper` layout for non-trivial
permutations," which is unscoped by this investigation and would need dedicated design
work, not just an adapter.
