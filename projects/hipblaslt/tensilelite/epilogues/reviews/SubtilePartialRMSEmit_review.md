# Code Review: `SubtilePartialRMSEmit.py`

**Reviewed:** 2026-07-20  
**Model:** Claude Opus  
**File:** `Tensile/Components/Subtile/SubtilePartialRMSEmit.py`

---

## Critical

### Finding 1 — Missing `s_nop` after every `v_accvgpr_read`

- **Location:** `_squareAndLaneSumFree0` (lines 585–602), `_applyGammaFree0` (1031–1036), `_addResidualFree0` loop (559–561)
- **Issue:** Every `VAccvgprReadB32(accTmp, …)` is immediately followed by a VALU instruction consuming `accTmp` with no wait state. gfx950 requires `s_nop 1` between the read and its consumer (the mainline store path does this — see `ShiftVectorComponents.py:497-498`). The `SWaitCnt(waitAll=True)` at line 184 only drains the MFMA pipeline and does not cover this hazard.
- **Evidence:**
  ```python
  VAccvgprReadB32(vgpr(accTmp), accvgpr(first), …)
  VMulF32(dst=vgpr(pidx), src0=vgpr(accTmp), src1=vgpr(accTmp), …)   # no s_nop
  ```
- **Fix:** Emit `SNop(waitState=1)` between each `VAccvgprReadB32` and the first instruction that reads its destination VGPR.

---

## Major

### Finding 2 — `BufferStoreB32` wait uses `vlcnt=0` instead of `vscnt=0`

- **Location:** `_writePartialsFree0`, line 901
- **Issue:** On gfx950 (`SeparateVMcnt`), stores decrement `vscnt` (the store counter), not `vlcnt` (the load counter). `SWaitCnt(vlcnt=0)` after the `BufferStoreB32` calls never actually waits for those stores. The correct idiom (per `GlobalWriteBatch.py:1968,1997`) is `SWaitCnt(vscnt=0)`.
- **Evidence:**
  ```python
  module.add(SWaitCnt(vlcnt=0, comment="wait partialBuf stores"))  # wrong counter
  ```
- **Fix:** Replace with `SWaitCnt(vscnt=0, …)`. Load waits at lines 556 and 1023 correctly use `vlcnt=0`.

### Finding 3 — `MacroTile1` assumed power-of-two, but validation never enforces it

- **Location:** `_setup`, lines 353–363
- **Issue:** `wg1Shift = int(math.log2(self.macro_tile1 * 2))` followed by `SLShiftLeftB32`. `_validatePartialRMS` enforces MT0 is a power of two (`Solution.py:267`) but never MT1. A non-pow2 MT1 silently truncates the shift, corrupting every gamma/token address for `WorkGroup1 > 0`.
- **Fix:** Add a MT1 power-of-two check in `_validatePartialRMS`, or replace the shift with a multiply.

### Finding 4 — partialBuf column/stride contract contradicted between emitter docstring and `Solution.py`

- **Location:** Emitter docstring (lines 15–21) + `_writePartialsFree0` (883–892) vs. `Solution.py:226-230`
- **Issue:** The emitter writes `partialBuf[token, WorkGroup0]` with `n_d = ceil(SizesFree0 / MT0)` columns. `Solution.py`'s docstring describes the layout as indexed by `WorkGroup1` with `N_tiles_N = ceil(N_hidden / MT1)`. If the host follows `Solution.py`, the strides mismatch the kernel.
- **Fix:** Update `Solution.py` to match the emitter's actual behavior (`WG0`, `n_d = ceil(SizesFree0/MT0)`).

---

## Medium

### Finding 5 — Σx² includes free0 padding rows when `N_hidden % MT0 != 0`

- **Location:** `_squareAndLaneSumFree0` (lines 581–604)
- **Issue:** The square/sum loop over all `(m, k)` rows is unconditional — no free0 range mask. The residual path correctly clamps OOB positions (lines 534–546), but this path does not. If padding-row accumulators are nonzero, they inflate Σx².
- **Fix:** Confirm the GEMM zero-pads free0 edge accumulators at the tail WG, or mask `globalRow >= SizesFree0` out of the square accumulation.

---

## Minor / Style

### Finding 6 — LDS reservation invariant in `Solution.py` is stale

- **Location:** `Solution.py:5354-5360`
- **Issue:** The `max()` formula uses `mma_m * rows_per_lane` but the actual scratch footprint scales with `mma_n` (`numPartials`). When `mma_n > mma_m * rows_per_lane` the stated invariant is false (still within MaxLDS, no overflow, but the proof is wrong).

### Finding 7 — Dead field `self.numRows`

- **Location:** `__init__`, line 120
- **Issue:** `self.numRows = self.mma_m * self.rows_per_lane` is assigned and never read.
- **Fix:** Remove.

### Finding 8 — Unnecessary VGPR materialization for literals > 64

- **Location:** `_loopResidualFree0` (506–531), `_writePartialsFree0` (871–881), `_applyGammaFree0` (994–1006)
- **Issue:** When `nOff > 64` or `mBase > 64`, a `VMovB32` materializes a literal into a scratch VGPR before `VAddU32`. The immediate can be passed directly to `VAddU32` as a 32-bit literal; the threshold of 64 is also off (64 is itself a valid inline constant).

### Finding 9 — Wave-partition boilerplate duplicated three times

- **Location:** `_addResidualFree0` (415–454), `_writePartialsFree0` (808–836), `_applyGammaFree0` (925–984)
- **Issue:** `laneId`, `waveId`, `waveM`, `rowGroupOff`, and `wgRowBase` are computed near-verbatim in all three methods.
- **Fix:** Factor into a shared helper returning the common offset VGPRs.

---

## Items verified correct

- XOR-butterfly `ds_bpermute` addressing (`partnerLane * 4`) with `dscnt=0` waits.
- Cross-wave LDS reduction: `readBaseWave = waveId XOR waveM`, `strideW/laneSlotBytes`, and both `_syncThreads` barriers are correct.
- Residual byte-offset `(token * SizesFree0 + nhidden_pos) * 2`, SRD sizing, and OOB clamping.
- `n_d` ceil-division, write predicate `rowGroup==0 && waveM==0`, `SAndSaveExecB64`/`SMovB64(EXEC)` save/restore.
- Gamma applied after Σx² and residual added before squaring — mathematically correct for RMSNorm.
- VGPR/SGPR checkout/checkin pairing in `emit()` is balanced.

---

## Verdict

**Needs fixes before merge.** Two blocking bugs: Finding 1 (missing AGPR-read `s_nop` — a known gfx950 hazard enforced everywhere else in the codebase) and Finding 2 (wrong wait counter on stores). Findings 3–5 are correctness/robustness gaps that will produce wrong results on non-pow2 MT1 or non-MT0-divisible N\_hidden. Findings 6–9 are cleanup.
