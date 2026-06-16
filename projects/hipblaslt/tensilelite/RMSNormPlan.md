<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# Implementation Plan: Fused GEMM + RMSNorm Epilogue (TensileLite, gfx950)

**Jira ticket**: AIHPBLAS-3856 — Implement RMSNorm Epilogue in TensileLite for MI350 (gfx950)
**Planning model**: Opus. **Implementation model**: Sonnet.
**Target**: gfx950 (MI350), bf16 only, Subtile code path.

---

## General instructions (read first, follow exactly)

- The virtual environment is located in `~/.tensile/`, make sure to be inside before executing code. The build is in the `build_tmp` directory.
- Make sure to keep a markdown log on the top directory of your steps and reasoning (keep it brief and to the point), the name of this file should be `IMPL-MEMORY.md`, and from time to time read it to make sure progress is being made. Time stamp every new entry you add.
- You are a top engineer in the TensileLite working on delivering the highest performance kernels for AMDGPU, while also maintaining the highest software quality possible, and following best GPU, C++, and python practices. Your motto should be, there's elegance and robustness in simplicity.
- Always make sure your code is readable, don't create overly long functions, don't use unreadable hacks and don't take shortcuts.
- If you are unsure about something, ask the user.
- You are not done until the code compiles, and the test passes producing numerically valid results.

---

## 1. Jira ticket summary

Fuse a **row-wise RMSNorm** into the GEMM epilogue so the GEMM output never round-trips through global memory before normalization. Today the workflow is: GEMM kernel writes `D` to HBM → a standalone RMSNorm kernel reads `D` back, normalizes, writes again. This plan eliminates the intermediate HBM write/read by computing RMSNorm in-register/LDS inside the GEMM kernel's existing global-write epilogue, on gfx950 only, for bf16, through the Subtile code path.

**RMSNorm math** (applied per output row `x` of length `N`):

```
sum_sq = sum_j x_j^2            # reduction over the full row
rstd   = 1 / sqrt(sum_sq / N + eps)
y_j    = x_j * rstd * gamma_j   # gamma is a per-column weight vector, shape (N,)
```

`eps` is a small scalar (e.g. `1e-5`); `gamma` is a learned weight of shape `(N,)`.

---

## 2. Architecture overview — what must happen technically

The Subtile inner loop (`kernelBodySubtile()` in `Tensile/KernelWriter.py`, line ~4873) produces, at the start of the post-loop region (line ~5064 onward), a complete fp32 D-tile accumulator distributed across the wave's lanes. The standard global-write epilogue (`notLocalSplitUGlobalWrite`, line ~5105) is **reused unchanged** by Subtile — Subtile only replaces the GEMM main loop, not the epilogue. RMSNorm therefore inserts itself **between** the accumulator completion and the global write.

The fused epilogue does, in order, after the accumulators are final (post `endSummation`, before `notLocalSplitUGlobalWrite`):

1. **In-register square-and-sum**: each lane squares its owned accumulator elements and sums them (partial per-lane sum-of-squares).
2. **Within-wave butterfly reduction** via `ds_bpermute_b32` (reuse the lane-exchange pattern from `Tensile/Components/GlobalWriteBatch.py:_emitSubtilePackedPermute`, line ~1571): combine partial sums across the lanes that hold the same output row.
3. **LDS partial-sum write**: each row's wave-level partial sum-of-squares is written to LDS.
4. **Cross-tile LDS tree reduction** (a plain halving-stride tree built from `DSLoadB32`/`VAddF32`/`DSStoreB32`/`SBarrier` — see the addressing spec in Phase 4.4): combine partial sums from all waves that contribute columns of the same row, producing the full `sum_sq` per row in LDS.
5. **Compute `rstd`**: `rstd = v_rsq_f32(sum_sq / N + eps)` per row (one value per row, broadcast to all lanes owning that row).
6. **Load `gamma`**: `buffer_load` `gamma[col]` for the columns each lane owns.
7. **Scale**: multiply each fp32 accumulator element by `rstd * gamma[col]`, in place, overwriting the accumulator VGPRs.
8. **Global write**: the existing epilogue converts fp32 → bf16 and stores `D` — no change required, it consumes the now-normalized accumulators.

**Row-containment invariant** (Section 5): each workgroup must own *complete* output rows so the row reduction completes within one WG with no cross-WG communication. This is enforced in `Solution.py` validation.

**StreamK-completeness invariant** (Section 4a): Subtile requires `StreamK ∈ {3,4}` (Solution.py:917), which splits the K dimension across workgroups. Under K-splitting, the accumulator at the RMSNorm hook is only a **partial** sum for that output tile; the cross-WG combine happens later in the `deferredFixupModule` (KernelWriter.py:5140), which runs *after* the store. RMSNorm must therefore only run on tiles whose accumulator is already complete at the hook. This milestone enforces that by requiring `StreamKForceDPOnly=1` (valid only with StreamK=3, see Solution.py:179), which makes every launched WG compute a full data-parallel tile (no K-split, no fixup). See Section 4a.

---

## 3. Implementation phases (ordered; each builds on the previous)

> Work the phases in order. After each phase, append a timestamped entry to `IMPL-MEMORY.md`. Do not advance to the codegen phases (4–5) until parameter/validation/signature phases (1–3) parse cleanly with `python -c "import Tensile"`.

### Phase 1 — Parameter registration

**Read first**: `Tensile/Common/ValidParameters.py` (line 447, the `UseSubtileImpl` entry), `Tensile/Common/GlobalParameters.py` (line 464, the `{"UseSubtileImpl": [False]}` entry in `defaultSolution`).

**Edit `Tensile/Common/ValidParameters.py`**: add, alphabetically near the other booleans:
```python
"RMSNorm": [False, True],
```

**Edit `Tensile/Common/GlobalParameters.py`**: `defaultSolution` is a **list of single-key dicts**, not a flat dict. Add an entry in that format near line 464 (next to `{"UseSubtileImpl": [False]}`):
```python
{"RMSNorm": [False]},
```

**Acceptance**: `python -c "from Tensile.Common.ValidParameters import validParameters; assert 'RMSNorm' in validParameters"` succeeds inside the venv.

`RMSNorm` is a **Solution** parameter, not a `ProblemType` parameter — it changes codegen, not the problem taxonomy. Keep it out of `Contractions.py`.

---

### Phase 2 — Solution validation

**Read first**: `Tensile/SolutionStructs/Solution.py` — the `UseSubtileImpl` validation block at lines ~840–916 (this is your model for ISA-gated rejections), the `reject(state, printRejectionReason, ...)` helper, and `MacroTile1` derivation at line ~705.

**Edit `Tensile/SolutionStructs/Solution.py`**: add a self-contained validator function near `_validateStreamKForceDPOnly` (line ~179), then call it from `assignDerivedParameters` *after* `UseSubtileImpl` is resolved (after line ~844, so `state["UseSubtileImpl"]` reflects the ISA gate) and *after* `MacroTile1` is derived (line ~705).

```python
def _validateRMSNorm(state, printRejectionReason):
    if not state["RMSNorm"]:
        return
    # RMSNorm fuses a row reduction into the Subtile epilogue.
    if not state["UseSubtileImpl"]:
        reject(state, printRejectionReason, "RMSNorm requires UseSubtileImpl")
        return
    if state["ISA"] != (9, 5, 0):
        reject(state, printRejectionReason, "RMSNorm is only implemented on gfx950")
        return
    # bf16 in/out only for this milestone.
    dt = state["ProblemType"]["DataType"]
    if not dt.isBFloat16():
        reject(state, printRejectionReason, "RMSNorm currently supports bf16 data type only")
        return
    # StreamK-completeness: the accumulator must be a COMPLETE tile at the
    # RMSNorm hook. Subtile forces StreamK in {3,4}, which K-splits across WGs
    # and combines partials in deferredFixupModule AFTER the store. Requiring
    # StreamKForceDPOnly=1 makes every WG compute a full data-parallel tile
    # (no K-split, no fixup), so the accumulator is final at the hook. See
    # plan Section 4a.
    if not state["StreamKForceDPOnly"]:
        reject(state, printRejectionReason, "RMSNorm requires StreamKForceDPOnly=1 (complete tiles, no K-split fixup)")
        return
    # Row-containment: each WG must span full output rows (see plan Section 5).
    # The free-1 tile (columns of D, i.e. N) must cover N completely, so the
    # in-WG reduction sees the whole row. We enforce that the macro-tile's
    # N extent equals the problem N; expressed as a constraint the host must
    # honor, validated again at runtime in the example script.
    if state["MacroTile1"] <= 0:
        reject(state, printRejectionReason, "RMSNorm requires a positive MacroTile1")
        return
```

Notes:
- Use the `DataType.isBFloat16()` helper (see `Tensile/Common/DataType.py`); do not string-compare.
- The strict "MacroTile1 == N" identity cannot be checked at solution-derivation time because `N` is a runtime problem size, not a solution parameter. Solution.py validates the *shape capability* (Subtile + gfx950 + bf16 + StreamKForceDPOnly + positive tile); the **example/validation script (Phase 6)** is responsible for choosing `M, N` such that `N == MacroTile1` (or `N` divides evenly with `MacroTile0`-rows-per-WG containment). Document this contract in a comment at the call site.
- `state["StreamKForceDPOnly"]` is the user-supplied value here; its own consistency check (`_validateStreamKForceDPOnly`, requires StreamK=3) runs later at Solution.py:1567. Reading the raw boolean at the call site is correct.

**Call site** (after line ~844):
```python
_validateRMSNorm(state, printRejectionReason)
if not state["Valid"]:
    return
```

**Acceptance**: a solution dict with `RMSNorm=True, UseSubtileImpl=True, ISA=(9,5,0), DataType=bf16, StreamK=3, StreamKForceDPOnly=1` keeps `Valid=True`; with `ISA=(9,4,2)`, `DataType=fp32`, or `StreamKForceDPOnly=0` it is rejected with the stated reason.

---

### Phase 3 — Kernel signature extension (gamma pointer + eps scalar)

**Read first**: `Tensile/Components/Signature.py` lines 255–311 — the `bias` global-buffer arg (line 256) is your model for `gamma`; the activation `by_value` args (lines 278–281) are your model for `eps`. Note how `userArgumentsInfo.totalSize` is accumulated (lines 300–309).

**Edit `Tensile/Components/Signature.py`**: after the bias block (after line 262), before the `factorDim` block:
```python
if kernel["RMSNorm"]:
    # Per-column weight vector gamma (shape N) consumed by the fused epilogue.
    signature.addArg("RMSNormGamma", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
    # Scalar epsilon added before rsqrt. fp32 regardless of IO type.
    signature.addArg("RMSNormEps", SVK.SIG_VALUE, "f32")
```

Add the size to the total (mirror the `biasSize` accumulation pattern). Introduce a `rmsNormSize` local initialized to `0`, set it to `8 + 4` (8B pointer + 4B f32) when `kernel["RMSNorm"]`, and add it into the `userArgumentsInfo.totalSize` sum at line ~300. Keep argument ordering deterministic — append after bias/E/activation so existing argument offsets do not shift for non-RMSNorm kernels.

`cptValueType` is the compute-type string already resolved earlier in `__call__`; for bf16 GEMM the gamma elements are read as bf16 then up-converted, so `gamma` may instead be declared with the IO value type. **Decision**: declare `RMSNormGamma` as bf16 (`getSrcValueType`-style) so the host passes a bf16 array matching the GEMM IO type, and up-convert in the kernel. Confirm against `getSrcValueType` (line 65) and pick the bf16 value-type string used for the bias/IO path.

**Acceptance**: generating any non-RMSNorm kernel produces a byte-identical signature to before this change (diff the emitted `.amdgpu_metadata` args block). An RMSNorm kernel's metadata lists `RMSNormGamma` (global_buffer) and `RMSNormEps` (by_value, f32).

---

### Phase 4 — RMSNorm epilogue codegen (the main assembly work)

**New file**: `Tensile/Components/Subtile/SubtileRMSNormEmit.py`. Register it in `tensile.egg-info/SOURCES.txt` only if SOURCES.txt is hand-maintained for the build (it is regenerated by setuptools — verify; if regenerated, no edit needed, just ensure the file is under the package and imported).

**Read first, in this order**:
1. `Tensile/Components/GlobalWriteBatch.py` lines 1571–1623 (`_emitSubtilePackedPermute`) — the `DSBPermuteB32` lane-exchange idiom and the partner-lane byte-address convention (`partner_lane * 4`). This is the template for the within-wave butterfly (step 2).
2. `Tensile/Activation.py` lines 555–575 (the `VExpF32`/`SNop` block) — the canonical transcendental-wait idiom: emit `SNop(waitState=0)` after a transcendental op when `ti.getArchCaps()["TransOpWait"]`. Mirror this for `VRsqF32` in step 5.
3. `Tensile/KernelWriter.py` lines 5064–5112 — the post-loop region where the hook lands, and how `self.states.c.startVgprValu` aliases the D-tile accumulator VGPRs. Lines 5114–5160 show the `deferredFixupModule` (StreamK partial combine) that runs *after* the store — the reason for the StreamKForceDPOnly constraint (Section 4a).
4. `rocisa.instruction` for the exact instruction classes: `DSBPermuteB32`, `DSStoreB32`, `DSLoadB32`, `VMulF32`, `VFmaF32`/`VMacF32`, `VAddF32`, `VRsqF32` (rsqrt, confirmed present in the rocisa opcode table), `BufferLoadB16`/`BufferLoadB32`, `SBarrier`, `SNop`, `SWaitCnt`, `VCvtF32toBF16`/pack helpers.

The LDS tree reduction (step 4) is a plain halving-stride loop of `DSLoadB32` ×2 / `VAddF32` / `DSStoreB32` / `SBarrier` — no external template needed; the full addressing spec is in 4.4. Build it directly from the rocisa instruction classes.

**Module structure** — keep functions short and single-purpose. Suggested API:

```python
# SubtileRMSNormEmit.py
class SubtileRMSNormEmitter:
    def __init__(self, writer, kernel): ...

    def emit(self, accVgprBase, numAccVgpr) -> Module:
        """Top-level: in-place normalize the fp32 D-tile accumulators.
        accVgprBase/numAccVgpr describe the contiguous fp32 accumulator range
        (self.states.c.startVgprValu .. +numVgprValu).
        """
        module = Module("RMSNorm epilogue")
        module.add(self._squareAndLaneSum(...))      # step 1
        module.add(self._butterflyReduce(...))       # step 2
        module.add(self._writePartialToLds(...))     # step 3
        module.add(self._ldsTreeReduce(...))         # step 4
        module.add(self._computeRstd(...))           # step 5
        module.add(self._loadGamma(...))             # step 6
        module.add(self._applyScale(...))            # step 7
        return module
```

#### 4.1 — Square and per-lane partial sum (step 1)
Each lane owns `numAccVgpr` fp32 accumulator elements distributed across the D-tile per the MFMA 16x16 output layout. **Critical**: these `numAccVgpr` elements are *not* all in the same output row — with `MacroTile0 > 16` (multiple MFMA row-blocks) and `MacroTile1 == N` spanning multiple column-blocks, a lane owns elements belonging to **several distinct rows**. The square-and-sum must be grouped **per row**, producing one partial per row the lane touches, not a single scalar over all `numAccVgpr`.

First derive the accumulator→(row, col) mapping from the MFMA 16x16 layout via `SubtileGeometry.py` (see step 6 / Section 4 for the layout source). Then, for each row group `r` the lane owns, accumulate the squares of just that group's elements into `partial[r]`:
```
# for each row group r the lane owns:
v_mul_f32 partial[r], acc[k0], acc[k0]                  # first element of row r
for each subsequent acc[k] in row group r:
    v_fma_f32 partial[r], acc[k], acc[k], partial[r]    # partial[r] += acc[k]^2
```
Use `VFmaF32` to fuse square+accumulate (one issue per element). Do **not** mutate `acc[k]` — the original values are needed in step 7. If the lane owns `R` row groups, this yields `R` partials carried through steps 2–5 (one reduction lane-group and one rstd per row). Keep `R` small and explicit; in the common contained config `R` is the number of MFMA row-blocks in `MacroTile0`.

#### 4.2 — Within-wave butterfly reduction (step 2)
The MFMA 16x16 layout (Subtile requires MatrixInst 16x16 — see `Solution.py:912`) spreads one output row across a set of lanes. Reduce `partial` across exactly the lanes that share a row.

Reuse the `ds_bpermute` idiom from `_emitSubtilePackedPermute`:
- Precompute a partner-lane **byte** address VGPR = `(laneId XOR stride) * 4` for each butterfly step (`stride` halving from the row's lane-span down to 1).
- For each step: `DSBPermuteB32 tmp, permAddr, partial`; `SWaitCnt(dscnt=0)`; `VAddF32 partial, partial, tmp`.
- The number of steps is `log2(lanesPerRow)`. Derive `lanesPerRow` from the MFMA output layout (16x16 → 16 lanes hold the 16 columns of one row-block; verify against `SubtileGeometry.py` which selects the MFMA layout). After this loop, every lane that owns a piece of a given row holds that row's **wave-level** partial sum-of-squares.

> If any butterfly step could read a lane outside the valid row span, guard with an exec-mask save/restore around the `ds_bpermute` exchange (save exec to a scratch SGPR pair, set the active-lane mask, run the exchange, restore exec). For a full 16x16 block all 16 lanes are valid, so the guard is unnecessary — confirm the block is full (no edge tile) before omitting it.

#### 4.3 — Write partial sum to LDS (step 3)
One representative lane per (row, wave) writes its wave-level partial sum to a reserved LDS region. LDS layout: `[rowsPerTile][wavesPerRowSpan]` fp32 slots. Compute the LDS byte offset as `(rowIndexInTile * wavesPerRowSpan + waveColGroup) * 4`. Use `DSStoreB32`; follow with `SWaitCnt(dscnt=0)` then `SBarrier()` so all waves' partials are visible before the tree reduction reads them.

Reserve the LDS bytes: extend the kernel's `group_segment_fixed_size`. Subtile already computes LDS usage; add `rowsPerTile * wavesPerRowSpan * 4` bytes guarded by `kernel["RMSNorm"]`. Find where Subtile sizes LDS (search `group_segment` / `ldsNumBytes` near the signature/rodata emission) and add the RMSNorm region without overlapping the GEMM LDS buffers (place it at the top of the post-loop LDS window, which is free once the main loop's double buffers are no longer live).

#### 4.4 — Cross-tile LDS tree reduction (step 4)
If a single wave already spans the full row (i.e. `wavesPerRowSpan == 1` because `MacroTile1 == N` and one wave covers all N columns), **skip** this step — step 2 already produced the complete `sum_sq`. Otherwise reduce across the `wavesPerRowSpan` LDS slots per row with a halving-stride tree built directly from rocisa instructions:

```
stride = wavesPerRowSpan // 2
while stride >= 1:
    # active lanes (one per row, slot index < stride) do:
    DSLoadB32  vL, addr = base(row, slot)
    DSLoadB32  vR, addr = base(row, slot + stride)
    SWaitCnt(dscnt=0)
    VAddF32    vL, vL, vR
    DSStoreB32 addr = base(row, slot), vL
    SWaitCnt(dscnt=0)
    SBarrier()                 # all writes visible before next halving
    stride //= 2
```

`base(row, slot) = (row * wavesPerRowSpan + slot) * 4` bytes. After the tree, slot `[row][0]` holds the full `sum_sq` for that row. Guard the active-lane set with an exec mask if the partial-write lanes don't already isolate one lane per (row, slot).

> **Prefer the row-contained single-wave case.** The Phase 2 row-containment constraint and the Phase 6 script choose `N == MacroTile1`, which in the common Subtile config keeps the whole row inside one wave → step 4 is a no-op. Implement step 4 for completeness but keep it behind the `wavesPerRowSpan > 1` check so the fast path stays clean.

#### 4.5 — Compute rstd (step 5)
Each lane loads its row's `sum_sq` (from step 2 register, or `DSLoadB32` from `[row][0]` if step 4 ran):
```
v_mul_f32  t, sum_sq, (1.0/N)        # mean of squares; 1/N is a compile-time f32 immediate
v_add_f32  t, t, eps                 # eps from sgpr("RMSNormEps")
v_rsq_f32  rstd, t                   # rstd = rsqrt(mean + eps)
```
`N` here is the row length = problem `N` = `MacroTile1` under the containment constraint, so `1.0/N` is a **compile-time immediate** the codegen bakes in. If `N` is only known at runtime, load it from the size SGPR and compute the reciprocal with `VRcpF32`; prefer the immediate path under the containment contract. Honor `TransOpWait` after `v_rsq_f32`: emit `SNop(waitState=0)` when `self.states.archCaps["TransOpWait"]` (or `getArchCaps()["TransOpWait"]`) is set — the same idiom used after `VExpF32` in `Tensile/Activation.py:560`. Compute one `rstd` per row group `r` (Section 4.1).

#### 4.6 — Load gamma (step 6)
Each lane owns specific output columns `col` (its D-tile column indices). Issue one `buffer_load` per owned column from the `RMSNormGamma` SRD at byte offset `col * bpe(bf16)=col*2`. Build the gamma SRD from the `RMSNormGamma` kernel arg pointer (mirror how the epilogue builds the `D`/`bias` SRD; reuse `globalWriteWorkGroupInit` machinery if a gamma SRD slot is added there). For a 16x16 block with the standard column mapping, the lane's column index derives from `laneId % 16` plus the tile's column base (`MacroTile1` offset of the WG); derive the exact mapping from `SubtileGeometry.py` rather than assuming. Up-convert bf16 gamma → fp32 with the bf16→f32 conversion the store path already uses (`v_cvt_f32_bf16` / the codebase's bf16→f32 pack helper). `SWaitCnt(vmcnt=0)` before use.

#### 4.7 — Apply scale (step 7)
For each owned accumulator element `acc[k]` at column `col_k`:
```
v_mul_f32 s, rstd, gamma[col_k]      # combined per-element scale
v_mul_f32 acc[k], acc[k], s          # normalize in place
```
Overwrite the accumulator VGPRs in place. The existing global-write epilogue (`notLocalSplitUGlobalWrite`) then converts fp32 → bf16 and stores — **no change needed there**.

**Register hygiene**: use `self.vgprPool.checkOut`/`checkIn` and `self.allocTmpSgpr` (the standard KernelWriter register-management helpers used throughout the epilogue code). Check every temporary back in. Do not leak the gamma/rstd/partial scratch into the store path's register budget.

**Acceptance for Phase 4**: the module imports without error; `SubtileRMSNormEmitter(...).emit(...)` returns a `Module` whose `str()` contains `ds_bpermute_b32`, `v_rsq_f32`, and `buffer_load` for gamma. Full numeric validation happens in Phase 6.

---

### Phase 5 — Wire into `kernelBodySubtile`

**Read first**: `Tensile/KernelWriter.py` lines 5064–5112 (post-loop region).

**Edit `Tensile/KernelWriter.py`**: insert the RMSNorm epilogue **after** `endSummation` and `globalWriteWorkGroupInit` (so SRDs/work-group offsets are set) but **before** `notLocalSplitUGlobalWriteIndices`/`notLocalSplitUGlobalWrite` (line ~5101–5105), so the accumulators are normalized before the store consumes them:

```python
if kernel["RMSNorm"]:
    from .Components.Subtile.SubtileRMSNormEmit import SubtileRMSNormEmitter
    module.addComment1("RMSNorm: fused row-normalization epilogue")
    rmsEmitter = SubtileRMSNormEmitter(self, kernel)
    module.add(rmsEmitter.emit(self.states.c.startVgprValu, self.states.c.numVgprValu))
```

Place this immediately after line ~5091 (`globalWriteWorkGroupInit`) and before line ~5100 (`not-LocalSplitU: global write indices`). Import at module top instead of inline if it does not create a circular import (test both; inline import is acceptable to avoid cycles, matching other lazy imports in this file).

**Acceptance**: generating an RMSNorm Subtile kernel (via the Phase 6 script with `RMSNorm=True`) emits the epilogue between the accumulator and the store; generating with `RMSNorm=False` produces byte-identical assembly to the pre-change baseline.

---

### Phase 6 — Validation script

**New file**: `tensile_rmsnorm_example.py` (top of `tensilelite/`).

**Read first, in full**: `tensile_gemm_example.py` (the `setup_tensile`, solution-construction, `getSourceFileString`, `compile_asm_to_hsaco`, `execute_hsaco`, and `verify_fn` flow) and `tensile_amdgpu_exec_integration.md` (argument layout, column-major storage, grid/block computation).

Build on the GEMM example:
1. Construct the solution with `RMSNorm=True, UseSubtileImpl=True, StreamK=3, StreamKForceDPOnly=1, ISA=[9,5,0], DataType=bf16`, MFMA 16x16, and choose `M, N, K` so that `N == MacroTile1` (row containment). Assert this in the script. `StreamKForceDPOnly=1` is required (Section 4a) so every WG computes a complete tile — without it the accumulator at the RMSNorm hook is a partial K-sum and results are wrong.
2. Extend the argument list (Section 6) with the `RMSNormGamma` pointer and `RMSNormEps` scalar in the slots Phase 3 appended (after bias/E/activation). Allocate `gamma` as a bf16 `(N,)` array, column-major-compatible.
3. numpy reference (Section 7):
   ```python
   d_gemm = (np.asarray(a, np.float32) @ np.asarray(b, np.float32).T)  # fp32 GEMM
   ss     = np.mean(d_gemm.astype(np.float32)**2, axis=1, keepdims=True)
   rstd   = 1.0 / np.sqrt(ss + eps)
   d_ref  = (d_gemm * rstd * gamma[np.newaxis, :].astype(np.float32))
   d_ref_bf16 = d_ref.astype(bf16_view)  # round to bf16 to match kernel output dtype
   ```
   Reduce in fp32 (matches the kernel's fp32 accumulator reduction).
4. `verify_fn` compares GPU `D` (bf16) against `d_ref_bf16` with bf16-appropriate tolerance (Section 7).
5. CLI flags mirroring the GEMM example: `--M --N --K --eps --chip gfx950 --iterations`. Validate `N == MacroTile1` and reject otherwise with a clear message.

**Acceptance**: `python tensile_rmsnorm_example.py --chip gfx950` prints `verification: PASSED` on gfx950 hardware.

---

## 4. Detailed technical spec for Phase 4 (assembly codegen)

State entering the hook (post `endSummation`, pre store):
- The fp32 D-tile accumulators are live in VGPRs starting at `self.states.c.startVgprValu`, count `self.states.c.numVgprValu` (KernelWriter.py lines 5071–5078). For bf16 GEMM the accumulators are fp32 (MFMA accumulates in fp32).
- Each lane owns a fixed subset of the D-tile per the MFMA 16x16 output layout (Subtile forces 16x16, `Solution.py:912`). A row of D is spread across a known set of lanes (the "lanesPerRow" group).
- `gamma` SRD and `eps` SGPR are available from the Phase 3 signature args (build the gamma SRD in `globalWriteWorkGroupInit` alongside the D/C SRDs, or in the emitter from the raw pointer).

Sequence (one row's worth, replicated across all rows the tile owns):

| Step | Op | Instructions | Notes |
|------|-----|-------------|-------|
| 1 | per-lane Σx² | `VMulF32` + `VFmaF32`×(numAcc-1) → `partial` | do not clobber `acc[k]` |
| 2 | butterfly | `DSBPermuteB32`+`SWaitCnt(dscnt=0)`+`VAddF32`, ×log2(lanesPerRow) | partner addr = `(lane XOR s)*4` |
| 3 | LDS write | `DSStoreB32` + `SWaitCnt(dscnt=0)` + `SBarrier` | one writer per (row,wave) |
| 4 | LDS tree | `DSLoadB32`×2 + `VAddF32` + `DSStoreB32` + `SBarrier`, ×log2(wavesPerRowSpan) | skip if span==1 |
| 5 | rstd | `VMulF32`(×1/N) + `VAddF32`(eps) + `VRsqF32` (+`SNop` if TransOpWait) | 1/N immediate when N==MT1 |
| 6 | gamma | `BufferLoadB16` per col + bf16→f32 cvt + `SWaitCnt(vmcnt=0)` | offset `col*2` |
| 7 | scale | `VMulF32`(rstd,gamma)→s, `VMulF32`(acc,s) per element | in place |

Reuse points (do not reinvent):
- `ds_bpermute` partner-address + issue + waitcnt: `GlobalWriteBatch.py:_emitSubtilePackedPermute` (1571).
- LDS tree reduce: build directly from `DSLoadB32`/`VAddF32`/`DSStoreB32`/`SBarrier` per the loop in 4.4 — no external template.
- `rsqrt`/transcendental wait handling: emit `SNop(waitState=0)` after `VRsqF32` when `archCaps["TransOpWait"]`, mirroring the `VExpF32`+`SNop` block in `Tensile/Activation.py:560`.
- gamma SRD construction: mirror how the existing global-write epilogue builds the D/C SRDs (set base from the kernel-arg pointer, set the size/const fields). See `globalWriteWorkGroupInit` and the D/C SRD setup it performs; add a gamma SRD slot there (Phase 4.6).

---

## 4a. StreamK-completeness constraint (complete accumulators at the hook)

Subtile forces `StreamK ∈ {3,4}` (Solution.py:917). StreamK divides total work `(#MTs × #LoopIters)` across WGs by splitting **both** the MacroTile grid *and* the K (summation) dimension. When K is split, several WGs each compute a **partial** sum for the same output tile and write it to a workspace; a later **fixup** step adds the partials to produce the final tile. In the Subtile post-loop, that fixup is emitted as `deferredFixupModule` and appended at KernelWriter.py:5140 — i.e. **after** `notLocalSplitUGlobalWrite` (the store).

Consequence for RMSNorm: at the hook site (after `endSummation`, before the store) the accumulator for a K-split tile is **only a partial sum**. Running RMSNorm there would normalize an incomplete row — numerically wrong. The reduction `sum_sq` would be computed over a partial product, and the fixup that completes the tile happens downstream of both the normalization *and* the store.

**Constraint for this milestone**: require `StreamKForceDPOnly = 1`. This option (valid only with `StreamK = 3`; see `_validateStreamKForceDPOnly`, Solution.py:179) forces every launched WG to compute a **full data-parallel tile** — no K-split, no partials, no fixup path taken. Every tile's accumulator is therefore final at the hook, and RMSNorm is correct. Phase 2 rejects `RMSNorm=True` with `StreamKForceDPOnly=0`.

Lifting this constraint later (supporting genuine K-split StreamK) would require either: (a) running RMSNorm *inside* the fixup path on the completing WG after partials are summed, or (b) a separate post-fixup normalization pass — both out of scope here. Document the constraint in the Solution.py comment, the example docstring, and the integration YAML.

---

## 5. Row-containment constraint (valid tile dimensions)

The in-WG reduction is only complete if every output row a WG touches is **fully owned by that WG**. With column-major D (`M×N`, stride-1 along M), a "row" of the RMSNorm is a fixed `i` across all `j∈[0,N)`. RMSNorm reduces over `j` (the N dimension). Therefore the WG must own all N columns of every row it touches.

In Subtile/Tensile tiling: `MacroTile0` is the M (free-0) extent per WG, `MacroTile1` is the N (free-1) extent per WG. To contain full rows, **`MacroTile1` must equal `N`** — each WG spans the entire row width and a strip of `MacroTile0` rows. Equivalent statements:
- Valid: `MacroTile1 == N`, any `MacroTile0` (WG owns `MacroTile0` complete rows).
- Invalid: `MacroTile1 < N` (a row is split across WGs along N → cross-WG reduction needed, not supported).

Solution.py (Phase 2) validates the *capability* (Subtile + gfx950 + bf16 + StreamKForceDPOnly + positive MT1) because `N` is a runtime size. The **example and integration YAML choose sizes with `N == MacroTile1`**, and the example script asserts it at runtime. Document this contract everywhere the constraint is relevant (Solution.py comment, example docstring, YAML comment).

The reduction strategy degrades gracefully: when `N == MacroTile1` and a single wave covers all N columns, step 4 (LDS cross-wave tree) is a no-op and the whole reduction is the in-wave butterfly (step 2) — the fast path.

---

## 6. Argument layout extension (RMSNorm kernel)

Extends the table in `tensile_amdgpu_exec_integration.md` (slots 0–21). RMSNorm appends its args **after** the standard GEMM args and any bias/E/activation args (so non-RMSNorm offsets are unchanged). For a plain bf16 GEMM + RMSNorm with no bias/E/activation, the appended slots are:

| Slot | Name | Type | Description |
|------|------|------|-------------|
| 22 | RMSNormGamma | ptr | per-column weight vector, bf16, shape `(N,)` |
| 23 | RMSNormEps | float32 | epsilon added before rsqrt (e.g. `1e-5`) |

The exact slot numbers depend on which optional args (bias, E, activation) precede them — RMSNorm is always last in the optional group. The Phase 6 script must compute these offsets from the actual signature, not hardcode 22/23, if it enables any other optional feature. For the minimal RMSNorm-only kernel, 22/23 hold.

`gamma` storage: shape `(N,)` bf16. With `N == MacroTile1`, gamma byte offset for column `col` is `col * 2`. The host allocates `gamma = np.random.randn(N).astype(bf16)`.

---

## 7. Testing approach (numpy reference vs GPU; tolerances)

Reference (fp32 reduction to match the kernel):
```python
d_gemm = np.asarray(a, np.float32) @ np.asarray(b, np.float32).T   # fp32
ss     = np.mean(d_gemm**2, axis=1, keepdims=True)                 # per-row mean of squares
rstd   = 1.0 / np.sqrt(ss + eps)
d_ref  = d_gemm * rstd * gamma_f32[np.newaxis, :]
```
Round `d_ref` to bf16 the same way the kernel rounds its fp32→bf16 store (round-to-nearest-even) before comparing.

Tolerance considerations:
- **bf16 output**: ~3 decimal digits (8-bit mantissa). Use relative tolerance `rtol≈2e-2`, absolute `atol≈2e-2`, or compare in bf16 ULPs.
- **Non-associative reduction**: the GPU butterfly+tree sum order differs from numpy's; for large `N` this adds error. Reduce the reference in fp32 (as above) — the kernel also reduces in fp32, so order is the dominant discrepancy, bounded by `O(N) * eps_fp32` which is negligible vs bf16 output rounding. The bf16 output rounding dominates the tolerance.
- **eps placement**: ensure reference and kernel both add eps to the *mean* of squares (`ss + eps`), not to `sum + eps`. Match the kernel's `1/N` scaling exactly.
- Report max abs/rel error and a pass/fail at the chosen tolerance; print a few mismatched elements on failure for debugging.

---

## 8. Key references (read before each phase)

| Phase | Files to read |
|-------|---------------|
| 1 | `Tensile/Common/ValidParameters.py` (447), `Tensile/Common/GlobalParameters.py` (464 `defaultSolution`, list-of-dicts format) |
| 2 | `Tensile/SolutionStructs/Solution.py` (840–916 UseSubtileImpl gate, 179 `_validateStreamKForceDPOnly`, 705 MacroTile1), `Tensile/Common/DataType.py` (`isBFloat16`) |
| 3 | `Tensile/Components/Signature.py` (255–311 bias/activation args, 65 `getSrcValueType`, 300–309 totalSize) |
| 4 | `Tensile/Components/GlobalWriteBatch.py` (1571 `_emitSubtilePackedPermute` — butterfly), `Tensile/Activation.py` (555–575 `VExpF32`+`SNop` — TransOpWait idiom for `VRsqF32`), `Tensile/KernelWriter.py` (5064–5112 post-loop, 5114–5160 deferred StreamK fixup), `Tensile/Components/Subtile/SubtileGeometry.py` (MFMA layout / lanesPerRow / acc→(row,col) mapping), `rocisa.instruction` (DSLoadB32/DSStoreB32/VAddF32/SBarrier for the LDS tree) |
| 5 | `Tensile/KernelWriter.py` (4873 `kernelBodySubtile`, 5091–5105 hook site) |
| 6 | `tensile_gemm_example.py`, `tensile_amdgpu_exec_integration.md` |

---

## 9. Definition of done

- `RMSNorm` parameter registered, defaulted, and validated (gfx950 + Subtile + bf16 + StreamKForceDPOnly + row-containment capability).
- Signature emits `RMSNormGamma` + `RMSNormEps`; non-RMSNorm kernels byte-identical.
- `SubtileRMSNormEmit.py` emits the 7-step epilogue, wired into `kernelBodySubtile` before the store.
- `tensile_rmsnorm_example.py` prints `verification: PASSED` against the numpy reference within bf16 tolerance on gfx950 hardware.
- `IMPL-MEMORY.md` kept current with timestamped entries per phase.
- New files carry the SPDX header; no overly long functions; all temp registers checked back in.
