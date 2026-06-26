# Implementation Plan: 3-Kernel GEMM + RMSNorm + GEMM Fusion (gfx950, bf16)

Status: APPROVED — ready for implementation.

Target: gfx950 (MI350), bf16 inputs/outputs, fp32 accumulation, TensileLite Subtile
kernels + one hand-written GCN aux kernel, validated with `amdgpu-exec`.

Validation GPU: **gfx950**. All `amdgpu-exec` launches and pytest runs target the local
gfx950 device. The chip string passed to `amdgpu_exec.compile_asm_to_hsaco` and
`setup_tensile` must use the detected chip (`amdgpu_exec.get_chip()`).

## Execution model

Each phase is implemented by a **fresh implementor agent** given only this plan file and
the relevant source files as context. The orchestrator (main session) hands off one phase
at a time, waits for the agent to report all tests passing, then commits before starting
the next agent. No phase starts until the previous phase's commit is on the branch.

Workflow per phase:
1. Spawn a fresh `implementor` agent with the phase's section of this plan.
2. Agent implements, runs the phase's test suite on gfx950, and iterates until all tests
   pass.
3. Agent reports back: "all tests pass, ready to commit."
4. Orchestrator reviews the diff and creates a git commit on branch `users/fmc/swiglu`
   with a short factual message (e.g. `tensilelite: add GEMM+partial-RMS epilogue K1`).
5. Next phase begins.

Hard rule: **never start the next phase's agent before the current phase's commit exists.**

This plan is grounded in the existing fused-epilogue art:
- `Tensile/Components/Subtile/SubtileRMSNormEmit.py` (the complete single-GEMM RMSNorm
  epilogue we split in two).
- `Tensile/Components/Subtile/SubtileSwiGLUEmit.py` (AGPR read/modify/write pattern).
- `tensile_rmsnorm_example.py` (Solution build, StreamK=3 ForceDPOnly arg layout, verify).
- `Tensile/Tests/unit/test_rmsnorm_epilogue.py` (session-scoped kernel fixture pattern).
- Integration surface confirmed in the codebase: solution-parameter registration
  (`Common/GlobalParameters.py`, `Common/ValidParameters.py`, `Common/RequiredParameters.py`),
  validation (`SolutionStructs/Solution.py::_validateRMSNorm`), kernarg signature
  (`Components/Signature.py` lines 320-327), epilogue hook
  (`KernelWriter.py` lines 5096-5110), and the contiguous store-sgpr kernarg load path
  (`KernelWriterAssembly.py` ~8220-8228).

## MMA tile geometry — no hardcoded constants

All emitters must derive the MMA geometry from `kernel` fields at construction time.
The current `SubtileRMSNormEmit.py` and `SubtileSwiGLUEmit.py` use module-level
constants `_MFMA_M = 16` and `_ROWS_PER_LANE = 4`, which silently assume the 16x16x32
instruction and a 64-lane wavefront. The new emitters must NOT repeat this mistake.

In every new emitter constructor, compute:
```python
mfma_m      = kernel["MatrixInstM"]   # M-rows per MMA tile
mfma_n      = kernel["MatrixInstN"]   # explicit even when equal to mfma_m; use it for N-column math
wave_size   = kernel["WavefrontSize"]
rows_per_lane = (mfma_m * mfma_n) // wave_size   # fp32 acc outputs per lane per MMA tile
```

Then replace every literal `16` with `self.mfma_n` (for N-column lane arithmetic) or
`self.mfma_m` (for M-row tile counts), and every literal `4` with `self.rows_per_lane`.
Downstream formulas that change:
- `mma_m = (MT0 // mfma_m) // wg_m`   — tiles along M per wave
- `mma_n = (MT1 // mfma_n) // wg_n`   — tiles along N per wave
- `num_rows = mma_m * rows_per_lane`   — output rows per lane
- `acc_idx(base, m, n, k) = base + (n*mma_m + m)*rows_per_lane + k`
- `partial_idx(m, k) = m*rows_per_lane + k`
- butterfly strides: `log2(mfma_n)` stages, strides `mfma_n//2, mfma_n//4, ..., 1`.
  The parameterized expression is `[mfma_n >> i for i in range(1, mfma_n.bit_length())]`.
  For mfma_n=16 this yields `[8, 4, 2, 1]` — 4 stages, not `mfma_n//2=8` stages.
  Never use `mfma_n//2` as the stage count; it is the first stride, not the count.
- lane N-column: `lane_id % mfma_n`     (replaces `lane_id % 16`)
- lane row-group: `lane_id // mfma_n`   (replaces `lane_id // 16`)
- gamma byte stride per MMA N-tile: `mfma_n * sizeof(bf16)`
- partialBuf write predicate: `lane_id % mfma_n == 0`. After the butterfly + crossWave
  reduction ALL lanes hold identical `partials[0..num_rows-1]` values. The predicate
  selects `wave_size // mfma_n` representative lanes (e.g. lanes 0, 16, 32, 48 for
  wave64/mfma_n=16). Each writing lane writes ALL `num_rows` values to the same global
  addresses; the writes are redundant but correct (same values, no race). The global
  byte offset for partial_idx(m, k) is simply `(m_tile_start + m*rows_per_lane + k)*4`.
  There is NO `lane_id // mfma_n` term — that offset does not exist.

The `_validatePartialRMS` and `_validateRstdScale` validators in `Solution.py` impose
no constraints on MMA tile geometry or wave size. All geometry is derived from kernel
fields at emitter construction time; correctness for a given configuration is the
responsibility of the upstream validators, not these epilogues.

---

## 1. Overview and Algorithm Recap

CODA fuses the RMSNorm between two GEMMs by exploiting that the RMS reciprocal-std
`r` is a per-row scalar, so `r * (h2 @ W1^T) == (r * h2) @ W1^T`. We never run a
standalone RMSNorm kernel. The pipeline:

```
GEMM1 (Kernel 1):   h0[m,n] = sum_k A[k,m] * W0[k,n]        (bf16 in, fp32 acc)
Epilogue1:          h1 = h0 + z                              (residual; optional, phase-gated)
                    partial_sq[m] = sum_n h1[m,n]^2          -> WRITE to partialBuf (fp32)
                    h2 = h1 * gamma[n]                        -> WRITE to D as bf16
Aux  (Kernel 2):    rstd[m] = rsqrt(partial_sq[m]/N_hidden + eps)   -> rstdBuf (fp32)
GEMM2 (Kernel 3):   h3[m,j] = sum_n h2[m,n] * W1[j,n]        (bf16 in, fp32 acc)
Epilogue2:          y[m,j] = rstd[m] * h3[m,j]               -> WRITE to D as bf16
```

Mathematical equivalence to standard RMSNorm-between-GEMMs:
`y = (RMSNorm(h1) * gamma) @ W1^T` where `RMSNorm(h1)[m,n] = h1[m,n] * rstd[m]`.
Because `h2 = h1 * gamma` (gamma applied in K1) and `rstd[m]` is applied in K3,
`y[m,j] = rstd[m] * sum_n (h1[m,n]*gamma[n]) * W1[j,n]
        = sum_n (rstd[m]*h1[m,n]*gamma[n]) * W1[j,n]` — exactly RMSNorm-scaled.

Row-containment invariant (same as existing RMSNorm kernel): for K1 the GEMM1
N (== N_hidden) must equal MacroTile1 so each WG owns complete output rows and can
reduce sum-of-squares within the workgroup. For K3 the GEMM2 K dimension is N_hidden;
the GEMM2 N (== N_out) must equal MacroTile1 (K3 epilogue only needs the per-row
scalar, so this is a plain row-containment-for-output requirement, not a reduction one).

StreamKForceDPOnly=1 guarantees one WG per output tile (no K-split fixup, accumulator
final at the hook). For K1 this means each WG owns exactly rows
`[WorkGroup0*MT0, WorkGroup0*MT0 + MT0)` with no overlap, so it can WRITE (not atomic-add)
its rows' partials directly into partialBuf.

K1's epilogue needs two categories of logic from the existing RMSNorm work:
the within-wave butterfly reduction and the cross-wave LDS reduction — these are
copied verbatim into the new file (with literals replaced by derived parameters).
Everything else — the partial-write to partialBuf, the gamma-only scale, the rstd
computation — is written fresh. K3's epilogue (per-row scalar broadcast) has no
overlap with the existing emitters and is written entirely from scratch.

---

## 2. Phase-by-Phase Breakdown

Hard rule: phases are strictly sequential. Do NOT begin Kernel 2 until Kernel 1's
amdgpu-exec validation passes on every test shape; likewise K3 after K2, and the
pipeline phase last.

### Tolerances (all phases)
- bf16 outputs (D from K1, y from K3): `rtol = atol = 2e-2`.
- fp32 values (partialBuf from K1, rstdBuf from K2): `rtol = atol = 1e-4`.
- Comparisons always upcast bf16 -> fp32 first; never compare bf16 arrays directly.
- NaN-safe: flag any non-finite GPU element as a failure.

---

## PHASE 1 — Kernel 1: GEMM + Partial-RMS epilogue

#### 1.1 What to implement

New solution parameter `PartialRMS` (bool, default False), mirroring how `RMSNorm`
and `SwiGLU` are registered. It selects a new epilogue emitter that:
1. Computes the full per-row sum-of-squares of `h1 = h0 (+ z)` using the existing
   3-stage reduction logic (squareAndLaneSum -> butterflyReduce -> crossWaveReduce
   when wg_n>1), reading the fp32 accumulator from AGPRs.
2. Writes the per-row sum-of-squares to `partialBuf` (fp32, global) at the WG's row
   offset (one fp32 per output row).
3. Applies gamma in-place to the accumulator AGPRs (so D = h1 * gamma). This is new
   code: load gamma bf16 per MMA N-tile via buffer load, convert to fp32, multiply
   each accumulator element — WITHOUT any rstd multiply. The existing bf16 global-write
   path then stores D as bf16.

Residual `z` is PHASE-GATED OFF initially: the first cut of K1 implements
`h1 = h0` (no residual). Add `z` only after the no-residual path validates, as a
follow-up sub-step within Phase 1 (see 1.7). Reason: residual requires an extra
global-load of z[m,n] per tile element and changes the reference; keeping it separate
isolates the partial-write logic from the residual-add logic during bring-up.

#### 1.2 Files to create / modify

Create:
- `Tensile/Components/Subtile/SubtilePartialRMSEmit.py`
  - Class `SubtilePartialRMSEmitter(writer, kernel)`. Constructor derives all geometry
    from kernel fields — NO module-level constants:
    ```python
    self.mfma_m       = kernel["MatrixInstM"]
    self.mfma_n       = kernel["MatrixInstN"]
    self.wave_size    = kernel["WavefrontSize"]
    self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.wave_size
    wg = kernel["MIWaveGroup"]
    self.wg_m, self.wg_n = wg[0], wg[1]
    self.mma_m  = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
    self.mma_n  = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
    self.num_rows = self.mma_m * self.rows_per_lane
    ```
  - All functions are written in `SubtilePartialRMSEmit.py` directly. Two are copied
    from the existing RMSNorm emitter (with hardcoded literals replaced by derived
    parameters); the rest are written from scratch:

    **Copied and parameterized** (source of truth is the new file, not a call into
    `SubtileRMSNormEmit`):
    - `_butterflyReduce`: within-wave butterfly sum across `mfma_n` column-sharing
      lanes. `log2(mfma_n)` stages with strides
      `[mfma_n >> i for i in range(1, mfma_n.bit_length())]` — for mfma_n=16 this is
      4 stages with strides 8,4,2,1. Uses `DSBPermuteB32` + `VAddF32` per partial row;
      `num_rows` drives the inner loop. Do NOT use `mfma_n//2` as the stage count.
    - `_crossWaveReduce`: LDS-backed sum across `wg_n` sibling waves. Address
      arithmetic uses `wave_size * num_rows * 4` for slot stride and
      `num_rows * 4` for the per-lane row stride; `sizeof(float) = 4` is the only
      literal remaining.

    **Written from scratch** (no equivalent in `SubtileRMSNormEmit`):
    - `_acc_idx(base, m, n, k)`: `base + (n*mma_m + m)*rows_per_lane + k`.
    - `_partial_idx(m, k)`: `m*rows_per_lane + k`.
    - `_setup`: build gamma SRD from `sgpr("RMSNormGamma", 2)` and partialBuf SRD from
      `sgpr("PartialBuf", 2)` — these names must match exactly the strings added to
      `numStoreSgprNames` in `KernelWriter.py`. Derive `lane_id`, `col_byte`
      (`lane_id % mfma_n`, scaled to bytes), and the wave column base for wg_n>1.
      Gamma byte offset per MMA N-tile: `n * mfma_n * 2`.
    - `_squareAndLaneSum`: read AGPRs into temp VGPRs, compute per-row Σx² (square
      first element, FMA the rest). Inner loop: `for k in range(rows_per_lane)`.
    - `_writePartials(partials, partialSrd, rowBaseSgpr)`: predicated write —
      `lane_id % mfma_n == 0` selects `wave_size // mfma_n` representative lanes (e.g.
      lanes 0, 16, 32, 48 for wave64/mfma_n=16). After the reduction ALL lanes hold
      identical `partials[0..num_rows-1]`, so each writing lane writes ALL `num_rows`
      values to the same global addresses — redundant but correct. The global byte
      offset for `partial_idx(m, k)` is `(m_tile_start + m*rows_per_lane + k) * 4`.
      There is no `lane_id // mfma_n` term.
    - `_applyGammaOnly(accVgprBase, partials, gamma_srd, gamma_tmp, acc_tmp, col_byte)`:
      for each MMA N-tile n: `BufferLoadD16B16` gamma, `SWaitCnt(vlcnt=0)`,
      `VCvtBF16toFP32`, then for each (m, k): read AGPR, `acc *= gamma`, write AGPR
      back. No rstd multiply. Inner loop uses `rows_per_lane`. The `SWaitCnt` before
      the convert is mandatory.

  - `emit(accVgprBase)` orchestrates: `s_waitcnt waitAll` → `_setup` →
    `_squareAndLaneSum` → `_butterflyReduce` → (`_crossWaveReduce` if wg_n>1) →
    `_writePartials` → `_applyGammaOnly`. VGPR/SGPR check-out/check-in follows the
    same pattern as the other emitters.

Modify:
- `Tensile/Common/GlobalParameters.py` (~line 498): add `{"PartialRMS": [False]},`.
- `Tensile/Common/ValidParameters.py` (~line 450): add `"PartialRMS": [False, True],`.
- `Tensile/Common/RequiredParameters.py` (~line 163): add `'PartialRMS',`.
- `Tensile/SolutionStructs/Solution.py`: add `_validatePartialRMS(state, ...)` modeled
  on `_validateRMSNorm` (lines 215-306): same `_validateSubtileEpiloguePrereqs` gate
  (UseSubtileImpl, gfx950, bf16, StreamKForceDPOnly, MIArchVgpr=False), MacroTile1>0,
  the wg_n>1 cross-wave LDS budget check, mutual exclusion with RMSNorm and SwiGLU.
  Also reject if `state["ProblemType"].get("OutputAmaxD")` or
  `state.get("_GlobalAccumulation") == "MultipleBufferSingleKernel"` or
  `state.get("AdaptiveGemmGSUA") == 1` — these insert extra args into the same kernarg
  range and corrupt the `PartialBuf` offset (same reason as in `_validateRMSNorm`).
  Also reject `GroupedGemm=True` — the grouped-GEMM `UserArgs` loop uses `totalSize`
  from `userArgumentsInfo` for pointer arithmetic; the `PartialRMS` args must be
  included in that total, and the interaction is untested. Rejection is simpler and
  matches the RMSNorm precedent.
  Call it from the same place the others are called (~line 1135).
  In `assignDerivedParameters` (~line 5306, alongside the RMSNorm LDS bump block): add
  an analogous block that computes `partialRMSLdsBytes` for wg_n>1 and bumps
  `state["LdsNumBytes"] = max(state["LdsNumBytes"], partialRMSLdsBytes)`, ensuring the
  cross-wave scratch fits regardless of main-loop LDS size.
- `Tensile/Components/Signature.py` (after the RMSNorm block, ~line 327): when
  `kernel["PartialRMS"]`, add args `RMSNormGamma` (bf16 global buffer, reuse the exact
  `getSrcValueType(kernel, True)` type), `RMSNormEps` is NOT needed for K1 (no rstd),
  and add `PartialBuf` (fp32 global buffer, "generic"). Keep the append ORDER fixed and
  documented; the contiguous store-sgpr load path
  (`KernelWriterAssembly.py` ~8220-8228) loads these by append order.
- `Tensile/KernelWriter.py` (~line 5096, alongside the RMSNorm/SwiGLU hooks): add
  ```
  if kernel["PartialRMS"]:
      from .Components.Subtile.SubtilePartialRMSEmit import SubtilePartialRMSEmitter
      emitter = SubtilePartialRMSEmitter(self, kernel)
      dtile_agpr_base = dtileInfo.vgprTiles[0].regList.indices[0] if dtileInfo.vgprTiles else 0
      module.add(emitter.emit(dtile_agpr_base))
  ```
- `Tensile/KernelWriter.py` (~line 9716, the `storeSgprLoad` / `numStoreSgprNames`
  accumulation block alongside the RMSNorm block): register the new SGPR names so
  `defineMultiSgprIndex` allocates them and `sgpr("RMSNormGamma")` / `sgpr("PartialBuf")`
  resolve at assembly-generation time:
  ```
  if kernel["PartialRMS"]:
      self.states.numStoreSgprNames.append("RMSNormGamma")
      self.states.numStoreSgprNameSizes.append(self.states.rpga)   # 2 SGPRs (64-bit ptr)
      self.states.numStoreSgprNames.append("PartialBuf")
      self.states.numStoreSgprNameSizes.append(self.states.rpga)   # 2 SGPRs (64-bit ptr)
      storeSgprLoad += self.states.rpga * 2
  ```
  Without this block, `sgpr("PartialBuf")` raises `KeyError` at assembly-generation time.

Create (host-side, no TensileLite changes):
- `tensile_gemm_rmsnorm_gemm_example.py` — for Phase 1 it builds and runs ONLY K1
  (the file grows across phases). Provide `build_k1_solution`, `generate_asm`,
  `compute_sk3_dp_args` (copy from `tensile_rmsnorm_example.py`), and `run_k1`.

#### 1.3 Data structures and argument layout

partialBuf layout (final, used by K1 write and K2 read):
- One fp32 per output row, indexed by GLOBAL row m.
- Size: `M_padded = ceil(M / MT0) * MT0` floats (edge tiles zero/garbage-padded;
  K2 guards with the real M).
- WG with `WorkGroup0 = t` writes rows `[t*MT0, t*MT0 + MT0)`. After the butterfly +
  crossWave reduction ALL lanes hold identical `partials[0..num_rows-1]`. The predicate
  `lane_id % mfma_n == 0` selects `wave_size // mfma_n` representative lanes; each
  writes ALL `num_rows` values to the same addresses (redundant but correct — identical
  values, no race). The global byte offset for `partial_idx(m, k)` is
  `(m_tile_start + m*rows_per_lane + k) * 4`. There is no `lane_id // mfma_n` term.
  `mfma_n` and `rows_per_lane` are read from `self.mfma_n`/`self.rows_per_lane` in
  the emitter; they must not be hardcoded.
- `_writePartials` does NOT need a bounds guard for edge tiles: K2's M-guard prevents
  reading OOB entries, and writing garbage to `partialBuf[M..M_padded)` is intentional.
- `m_tile_start = WorkGroup0 * MT0` — available as an SGPR; the kernel already computes
  `SubtileMGuard` from `WorkGroup0 * mt0` (`KernelWriterAssembly.py` ~14642), so the
  same `WorkGroup0 * MT0` base is computed for the partial-write offset.

K1 kernarg layout (extends the RMSNorm layout from `tensile_rmsnorm_example.py`
lines 342-408). Slots 0-29 are IDENTICAL (GemmInfo, kernel_info0/1, numWG, M, N=N_hidden,
batch, K, D, C, A=A, B=W0, WS, Flags, strides x8, alpha, beta, StreamK args x6).
Appended custom args (must match Signature append order):
- slot 30: `RMSNormGamma` (ptr, bf16, len N_hidden)
- slot 31: `PartialBuf` (ptr, fp32, len M_padded)  [InOutArray]
(No `RMSNormEps` for K1.)

#### 1.4 amdgpu-exec launch config

- `block_dim = (solution["NumThreads"], 1, 1)`
- `grid_dim  = (numWG, 1, 1)` where `numWG = ceil(M/MT0) * ceil(N_hidden/MT1)`
  and `N_hidden == MT1` (single N-tile column; row containment).
- alpha=1.0, beta=0.0 (enforced; hook runs before alpha/beta apply).

#### 1.5 Validation (numpy reference)

```
a_ref  = bf16(A).astype(f32)            # K x M  (TransposeA=True, TN layout)
w0_ref = bf16(W0).astype(f32)           # K x N_hidden
h0     = a_ref.T @ w0_ref               # M x N_hidden, fp32
h1     = h0                             # phase-1a: no residual
gamma  = bf16(gamma).astype(f32)
D_ref  = bf16(h1 * gamma[None, :])      # M x N_hidden bf16 (K1 D output)
sumsq_ref[m] = sum_n h1[m,n]^2          # per-row, fp32 (NOTE: sum, not mean)
```
IMPORTANT decision: store the raw SUM of squares (not the mean) in partialBuf, and let
K2 divide by N_hidden. This keeps K1 ignorant of N_hidden semantics and matches the
existing RMSNorm reduction which sums then multiplies by 1/N. Document this contract
loudly at the top of both K1 and K2 files.

Checks:
- D (slot 8) vs `D_ref`, bf16 tol 2e-2.
- partialBuf[:M] (slot 31) vs `sumsq_ref`, fp32 tol 1e-4. Ignore padded rows [M:M_padded].

#### 1.6 Test shapes (Phase 1)

Reuse the `_SHAPES`/`_WG_N` matrix from `test_rmsnorm_epilogue.py` (it already covers
full/edge M, sub/full/partial/multi/prime K, and wg_n in {1,2}). N is pinned to
MacroTile1. Add a dedicated pytest `Tensile/Tests/unit/test_gemm_partial_rms.py` with
a session-scoped `k1_kernel` fixture (params = wg_n) that builds+compiles once, and a
parametrized `test_k1_shape` asserting BOTH D and partialBuf within tolerance.

#### 1.7 Residual sub-step (Phase 1b, after 1a passes)

Add optional residual `z` (M x N_hidden bf16): new kernarg `ResidualZ` (ptr, bf16),
gated by a `PartialRMSResidual` bool param. Epilogue loads z[m,n] per accumulator
element (same addressing as the gamma load but 2D), adds to the accumulator BEFORE the
square-and-sum and before gamma. Reference: `h1 = h0 + bf16(z).astype(f32)`. Extend
the pytest with a residual-on parametrization. Keep 1a (no-residual) tests green.

#### 1.8 Definition of done (Phase 1)

All of the following must pass on **gfx950** before the commit gate:
- `Tensile/Tests/unit/test_gemm_partial_rms.py` passes all shapes x wg_n {1,2} for D
  and partialBuf (residual off). Residual-on variant passes if 1b is in scope.
- `tensile_gemm_rmsnorm_gemm_example.py --phase k1` prints `verification: PASSED` for a
  representative shape (e.g. M=2048, N_hidden=64, K=4096).
- `tox -e unit -- Tensile/Tests/unit/test_rmsnorm_epilogue.py` and the SwiGLU suite
  still pass (no regression to shared code).

**Commit gate**: once all tests pass the implementor agent reports completion; the
orchestrator commits with message `tensilelite: add GEMM+partial-RMS epilogue (K1)`.
Phase 2 agent does not start until this commit exists.

---

## PHASE 2 — Kernel 2: Auxiliary reduction (hand-written GCN assembly)

#### 2.1 What to implement

A standalone elementwise AMDHSA kernel (NOT through TensileLite) computing, per row:
```
rstd[m] = rsqrt(partialBuf[m] / N_hidden + eps)   for m in [0, M)
```
One thread per row. `grid = (ceil(M/256), 1, 1)`, `block = (256, 1, 1)`.
Threads with global id >= M do nothing (bounds guard).

The kernel is emitted as a raw GCN assembly STRING and compiled with
`amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)`. It does not depend on any
TensileLite solution object.

#### 2.2 Files to create / modify

Create:
- A new Python helper inside `tensile_gemm_rmsnorm_gemm_example.py`:
  `build_aux_reduction_asm(chip, N_hidden)` returning `(asm_str, kernel_name, hsaco)`.
  `N_hidden` is a required parameter because `1.0/N_hidden` is embedded as an f32
  immediate — a distinct HSACO must be compiled for each N_hidden value. Use an
  `@lru_cache` keyed on `(chip, N_hidden)` so the compilation happens once per pair.
  Plus `run_k2(hsaco, kernel_name, M, N_hidden, eps, ...)`. Rationale: the aux kernel
  is tiny; embedding the asm string avoids a separate .s file.

#### 2.3 Argument layout and kernel structure

Kernarg struct (packed, 8-byte aligned pointers):
- offset 0:  `partialBuf` (global ptr, fp32, 8B)
- offset 8:  `rstdBuf`    (global ptr, fp32, 8B)
- offset 16: `M`          (u32)
- offset 20: `eps`        (f32)

`N_hidden` is NOT a runtime arg. `1.0 / N_hidden` is computed in Python at the time
the assembly string is generated (same approach as `SubtileRMSNormEmit._setup` line 177:
`struct.unpack('I', struct.pack('f', 1.0 / N_hidden))[0]`) and embedded as an f32
immediate in a `v_mov_b32` instruction. This is exact for power-of-two N_hidden; for
non-power-of-two values the error is at most 1 f32 ULP (well within the fp32 tol 1e-4).
No device-side `v_rcp_f32` is needed.

Required AMDHSA metadata: `.amdhsa_kernel` block with `.amdhsa_next_free_vgpr`,
`.amdhsa_next_free_sgpr`, `.amdhsa_user_sgpr_kernarg_segment_ptr 1`,
`.amdhsa_system_sgpr_workgroup_id_x 1`, group/private segment sizes = 0; plus the
`.amdgpu_metadata` YAML (kernel name, `.args` list with `.offset/.size/.value_kind`,
`.kernarg_segment_size 32` (24 bytes of args rounded up to the required 16-byte multiple), `.wavefront_size 64`,
`.max_flat_workgroup_size 256`).

Instruction sequence (one thread):
1. Load kernargs: `s_load_dwordx2 s[partialBase:partialBase+1], s[kernarg], 0x0`;
   `s_load_dwordx2 s[rstdBase:rstdBase+1], s[kernarg], 0x8`;
   `s_load_dword s_M, s[kernarg], 0x10`; `s_load_dword s_eps, s[kernarg], 0x14`;
   `s_waitcnt lgkmcnt(0)`.
2. tid = `WorkGroup0 * 256 + v0`. Compute: `s_mul_i32 s_base, s_wg_id, 256`;
   `v_add_u32 v_tid, s_base, v0`.
3. Bounds: `v_cmp_lt_u32 vcc, v_tid, s_M`; `s_and_saveexec_b64 exec, vcc`; branch
   to end if exec == 0.
4. Byte offset: `v_lshlrev_b32 v_off, 2, v_tid`  (v_off = tid * 4).
5. `global_load_dword v_p, v_off, s[partialBase:partialBase+1]`; `s_waitcnt vmcnt(0)`.
   The `global_load_dword` instruction takes `(dst, vaddr_offset, saddr_base_pair)` —
   the SGPR pair is the base address and the VGPR is the 32-bit signed byte offset.
6. `v_mov_b32 v_invN, 0x{inv_n_bits:08x}` (f32 immediate, `1.0/N_hidden` computed in
   Python: `struct.unpack('I', struct.pack('f', 1.0 / N_hidden))[0]`).
   `v_mul_f32 v_p, v_p, v_invN`.
7. `v_mov_b32 v_eps, s_eps`; `v_add_f32 v_p, v_p, v_eps`;
   `v_rsq_f32 v_p, v_p`; `s_nop 0` (1 TransOpWait cycle on gfx950).
8. `global_store_dword v_off, v_p, s[rstdBase:rstdBase+1]`; `s_endpgm`.

#### 2.4 amdgpu-exec launch config

- args: `[InputArray(partialBuf_f32), InOutArray(rstdBuf_f32), u32(M), f32(eps)]`
- `grid_dim = (ceil(M/256), 1, 1)`, `block_dim = (256, 1, 1)`.

#### 2.5 Validation

```
rstd_ref[m] = 1.0 / sqrt(sumsq_ref[m] / N_hidden + eps)    # fp32
```
Two test modes:
- Unit: feed a synthetic `partialBuf` (e.g. random positive fp32, plus edge values
  0, large, tiny) and compare `rstdBuf[:M]` vs `rstd_ref`, fp32 tol 1e-4.
- Chained: feed the actual partialBuf produced by K1 for a few shapes and compare
  against the K1 reference's `sumsq_ref`.

#### 2.6 Test shapes (Phase 2)

M in {1, 16, 255, 256, 257, 1024, 2048, 4093 (prime), 65535}, N_hidden in {64, 4096},
eps in {1e-5, 1e-6}. New pytest `Tensile/Tests/unit/test_aux_reduction.py`. Because
`1.0/N_hidden` is a compile-time immediate, a separate HSACO is compiled per N_hidden
value; the session fixture is parametrized over N_hidden (and uses `lru_cache` or
equivalent to compile each HSACO only once across tests sharing the same N_hidden).

#### 2.7 Definition of done (Phase 2)

All of the following must pass on **gfx950** before the commit gate:
- `test_aux_reduction.py` passes all M/N_hidden/eps combos, fp32 tol 1e-4, including
  the bounds-guard (tid>=M untouched/garbage-tolerant) and M not a multiple of 256.
- `run_k2` in the example script prints PASSED on a representative shape.
- K1 test suite still passes (no regression).

**Commit gate**: once all tests pass the implementor agent reports completion; the
orchestrator commits with message `tensilelite: add aux RMS reduction kernel (K2)`.
Phase 3 agent does not start until this commit exists.

---

## PHASE 3 — Kernel 3: GEMM + Delayed-RMSNorm-scale epilogue

#### 3.1 What to implement

New solution parameter `RstdScale` (bool, default False) selecting an epilogue that,
for each output row in the WG's tile, loads the per-row scalar `rstd[row]` from rstdBuf
and multiplies every N-column accumulator of that row by it. No reduction, no LDS, no
butterfly — simplest of the three emitters.

GEMM2 operands: A = h2 (M x N_hidden, the bf16 D produced by K1), B = W1
(N_out x N_hidden). With the same TN layout convention as K1 (TransposeA=True,
TransposeB=False), GEMM2 contracts over N_hidden (= GEMM2 "K"), output is M x N_out,
and N_out must equal MacroTile1 (row containment for the output tile).

#### 3.2 Files to create / modify

Create:
- `Tensile/Components/Subtile/SubtileRstdScaleEmit.py`
  - Class `SubtileRstdScaleEmitter(writer, kernel)`. Constructor derives geometry the
    same way as `SubtilePartialRMSEmitter` — from `kernel["MatrixInstM"]`,
    `kernel["MatrixInstN"]`, `kernel["WavefrontSize"]`, `kernel["MIWaveGroup"]`,
    `kernel["MacroTile0"]`, `kernel["MacroTile1"]`. No module-level constants.
  - `_setup`: build the rstdBuf SRD from `sgpr("RstdBuf", 2)` — must match the name
    added to `numStoreSgprNames`. Compute `m_tile_start = WorkGroup0 * MT0` as
    an SGPR. The rstdBuf byte offset for `partial_idx(m, k)` is
    `(m_tile_start + m*rows_per_lane + k) * 4`. There is no `lane_id // mfma_n` term
    — every lane loads the same row's rstd for the same (m, k), because `partial_idx`
    already spans all `num_rows` rows the wave owns regardless of which N-column the
    lane covers.
  - `_loadRstd(rstd_vgpr_array)`: for each of the `num_rows` rows (`for m, k`), issue
    `buffer_load_dword` at byte offset `(m_tile_start + m*rows_per_lane + k) * 4`;
    collect into `rstd_vgpr_array[partial_idx(m,k)]`. `s_waitcnt vmcnt(0)` after all loads.
  - `_applyScale(accVgprBase, rstd_vgpr_array)`: for each (m, n, k), read AGPR,
    `v_mul_f32 acc, acc, rstd[_partial_idx(m,k)]`, write AGPR back. Written from
    scratch; inner loop uses `rows_per_lane`, no literal 4.
  - `emit(accVgprBase)`: `s_waitcnt waitAll` -> `_setup` -> `_loadRstd` -> `_applyScale`.
- Append to `tensile_gemm_rmsnorm_gemm_example.py`: `build_k3_solution`, `run_k3`.

Modify (same five wiring points as Phase 1, for the `RstdScale` param):
- `Common/GlobalParameters.py`, `Common/ValidParameters.py`, `Common/RequiredParameters.py`.
- `SolutionStructs/Solution.py`: `_validateRstdScale` — `_validateSubtileEpiloguePrereqs`
  + MacroTile1>0 + mutual exclusion with RMSNorm/SwiGLU/PartialRMS. Also reject
  `OutputAmaxD`, `_GlobalAccumulation == "MultipleBufferSingleKernel"`, and
  `AdaptiveGemmGSUA == 1` (same kernarg-layout-conflict reason as `_validateRMSNorm`).
  No cross-wave LDS needed; the scale is per-row local, so wg_n>1 just means each wave
  scales its own N-slice with the same per-row rstd independently.
- `Components/Signature.py`: when `kernel["RstdScale"]`, append `RstdBuf`
  (fp32 global buffer, "generic"). One arg only.
- `KernelWriter.py` (~line 5096): add the `if kernel["RstdScale"]:` epilogue hook next
  to the others.
- `KernelWriter.py` (~line 9716, `numStoreSgprNames` block): register `"RstdBuf"` so
  `sgpr("RstdBuf")` resolves at assembly-generation time:
  ```
  if kernel["RstdScale"]:
      self.states.numStoreSgprNames.append("RstdBuf")
      self.states.numStoreSgprNameSizes.append(self.states.rpga)   # 2 SGPRs (64-bit ptr)
      storeSgprLoad += self.states.rpga
  ```

#### 3.3 Data structures and argument layout

rstdBuf: fp32, length M_padded (produced by K2; K3 indexes by global row m, same
mapping as K1's partial write). Each wave/lane loads rstd[row] for the rows it owns.

K3 kernarg layout: slots 0-29 identical to K1 (GEMM scaffolding + StreamK), with
A = h2 (the K1 D output), B = W1, N = N_out, K_contraction = N_hidden. Appended:
- slot 30: `RstdBuf` (ptr, fp32, len M_padded) [InputArray]

#### 3.4 amdgpu-exec launch config

- `grid_dim = (ceil(M/MT0) * ceil(N_out/MT1), 1, 1)`, `N_out == MT1`.
- `block_dim = (NumThreads,1,1)`. alpha=1, beta=0.

#### 3.5 Validation

```
h2_ref = bf16(h1 * gamma).astype(f32)          # same bf16 rounding as K1 D output
w1_ref = bf16(W1).astype(f32)                  # N_out x N_hidden
h3     = h2_ref @ w1_ref.T                      # M x N_out, fp32
y_ref  = bf16(h3 * rstd_ref[:, None])           # M x N_out bf16
```
Check y (slot 8) vs `y_ref`, bf16 tol 2e-2. Drive rstd from the K2/numpy reference.

#### 3.6 Test shapes (Phase 3)

Reuse the M/K matrix; here "K" is N_hidden (GEMM2 contraction) and N is N_out=MT1.
Vary N_hidden across {64, 128, 256, 4096}. New pytest
`Tensile/Tests/unit/test_gemm_rstd_scale.py`, session-scoped fixture over wg_n.

#### 3.7 Definition of done (Phase 3)

All of the following must pass on **gfx950** before the commit gate:
- `test_gemm_rstd_scale.py` passes all shapes x wg_n {1,2}, bf16 tol 2e-2.
- `run_k3` prints PASSED on a representative shape.
- No regression in RMSNorm/SwiGLU/K1/K2 suites.

**Commit gate**: once all tests pass the implementor agent reports completion; the
orchestrator commits with message `tensilelite: add GEMM+delayed-rstd-scale epilogue (K3)`.
Phase 4 agent does not start until this commit exists.

---

## PHASE 4 — End-to-end pipeline + benchmark

#### 4.1 What to implement

- `tensile_gemm_rmsnorm_gemm_example.py` gains a `--phase pipeline` mode that, with
  matched device buffers, runs K1 -> K2 -> K3 in sequence:
  - Allocate partialBuf (M_padded f32) and rstdBuf (M_padded f32) ONCE and thread them
    through: K1 writes partialBuf + D(=h2); K2 reads partialBuf, writes rstdBuf;
    K3 reads h2 (= K1's D) and rstdBuf, writes y.
  - Use `amdgpu_exec.InOutArray` for buffers that persist across launches. Device
    buffers live for the lifetime of the memory manager, so partialBuf and rstdBuf
    allocated before the first launch carry their contents into K2 and K3 unchanged.
- `tensile_gemm_rmsnorm_gemm_benchmark.py`: time each kernel (best-of-N via the
  returned `times_ns`) and the summed pipeline; report TFLOPS:
  - K1 GEMM flops = 2*M*N_hidden*K.
  - K3 GEMM flops = 2*M*N_out*N_hidden.
  - K2 is memory-bound; report GB/s instead of TFLOPS.
  - Pipeline TFLOPS = (K1+K3 flops) / total_time.
  - Baseline comparison: run K1-as-plain-GEMM + a standalone RMSNorm (reuse the
    existing `tensile_rmsnorm_example` kernel) + K3-as-plain-GEMM, and report the
    speedup of the fused 3-kernel path over the unfused baseline.

#### 4.2 End-to-end reference

```
h0    = bf16(A).T @ bf16(W0)                       # fp32
h1    = h0 (+ bf16(z) if residual)
sumsq = sum_n h1^2                                  # per row
rstd  = 1/sqrt(sumsq/N_hidden + eps)
h2    = bf16(h1 * gamma)
h3    = bf16(h2).astype(f32) @ bf16(W1).T
y     = bf16(h3 * rstd[:,None])
```
Compare pipeline y vs `y` above, bf16 tol 2e-2. Also assert the intermediate
partialBuf and rstdBuf match (fp32 tol 1e-4) for at least one shape to localize any
drift.

#### 4.3 Test shapes (Phase 4)

A focused set spanning LLM-like shapes: (M, N_hidden, N_out) in
{(64,64,64), (2048,4096,4096), (1000,256,64 edge-M), (513,128,128)} x wg_n {1,2}.
New pytest `Tensile/Tests/unit/test_gemm_rmsnorm_gemm_pipeline.py`.

#### 4.4 Definition of done (Phase 4)

All of the following must pass on **gfx950** before the commit gate:
- Pipeline pytest passes for all listed shapes; intermediate buffers validated on >=1.
- Benchmark script runs and prints per-kernel + pipeline TFLOPS/GB/s and the
  fused-vs-unfused speedup without correctness failures.
- Full regression: `tox -e unit -- Tensile/Tests/unit` (RMSNorm + SwiGLU + all new
  suites + pipeline) green on gfx950.

**Commit gate**: once all tests pass the implementor agent reports completion; the
orchestrator commits with message `tensilelite: add end-to-end GEMM+RMSNorm+GEMM pipeline and benchmark`.

---

## 3. Key Design Decisions and Rationale

1. Three independent solution params (`PartialRMS`, `RstdScale`) rather than overloading
   `RMSNorm`. Rationale: each selects a distinct epilogue; mutual-exclusion validation
   keeps the param space unambiguous and matches the existing RMSNorm/SwiGLU precedent.
2. Self-contained emitter files with no cross-file calls. `SubtilePartialRMSEmit.py`
   copies only `_butterflyReduce` and `_crossWaveReduce` from the RMSNorm work (with
   literals parameterized); all other functions are written from scratch.
   `SubtileRstdScaleEmit.py` is written entirely fresh. Rationale: matches the
   file-per-epilogue convention (SubtileRMSNormEmit, SubtileSwiGLUEmit); each file is
   independently reviewable and can diverge without risk of breaking the other epilogues.
3. partialBuf stores SUM of squares, not the mean; K2 divides by N_hidden. Rationale:
   K1 stays agnostic of N semantics and the copied butterfly/crossWave reduction
   accumulates a plain sum; the divide lives in one place (K2).
4. Direct write (not atomic) to partialBuf/one-fp32-per-row, justified by
   StreamKForceDPOnly=1 (one WG per tile, disjoint row ranges). One representative lane
   per row writes to avoid duplicate stores. Rationale: matches constraint #4; avoids
   atomic-add cost and ordering concerns.
5. K2 is hand-written GCN, embedded as a string in the pipeline script. Rationale: a
   1-thread-per-row rsqrt is trivial and does not belong in the TensileLite codegen
   pipeline; keeping it inline removes a separate-file build step.
6. K3 applies rstd per-row only (no gamma, no reduction) — the cheapest epilogue.
   The CODA scalar-commute identity makes this exact, not an approximation.
7. Residual `z` is deferred to Phase 1b. Rationale: isolates partial-write correctness
   from residual-add correctness during bring-up.
8. Reuse the existing StreamK=3 ForceDPOnly arg-computation and the proven kernarg
   ordering from `tensile_rmsnorm_example.py` to minimize host-side risk.

---

## 4. Open Questions / Risks

1. amdgpu-exec buffer persistence across `execute_hsaco` calls (Phase 4). RESOLVED:
   device buffers persist for the lifetime of the memory manager, so partialBuf and
   rstdBuf allocated once will carry their contents from K1 → K2 → K3 without any
   host round-trip. Allocate them as `InOutArray` before the first kernel launch and
   pass the same Python objects to all three `execute_hsaco` calls. Correctness relies
   on `execute_hsaco` blocking until the GPU kernel completes before returning, providing
   the global memory ordering barrier needed between K1's partialBuf stores and K2's
   reads. Do not introduce async or multi-stream launch paths without verifying this
   ordering is preserved.
2. Partial-write predication: the epilogue hook fires after the main summation loop
   completes, so the accumulator is always final at that point. After the butterfly +
   crossWave reduction all `mfma_n` column-lanes hold identical row partials by
   construction. The `lane_id % mfma_n == 0` predicate is therefore always correct.
   Not a risk.
3. Kernarg ordering vs the contiguous store-sgpr load path. New args are appended after
   the RMSNorm slot; the loader reads by append order
   (`KernelWriterAssembly.py` ~8220-8228). Any reordering breaks offsets silently.
   Mitigation: keep append order fixed, document it in both Signature.py and the host
   script, assert the host arg list length matches. Medium risk.
4. K1 and K3 are independent Solutions, each with their own MT1 (N_hidden and N_out
   respectively). N_hidden ≠ N_out is the normal case; the pipeline script simply
   builds two solution objects. Not a risk.
5. wg_n>1 for K3: each wave independently loads rstd[row] for its rows; no cross-wave
   work needed since rstd is per-row not per-column. Confirm the row->lane mapping is
   identical to K1's so the same global row index is used. Low risk.
6. Edge-tile padded rows in partialBuf/rstdBuf contain garbage; K2's M-guard prevents
   reading them as valid, and K3 only loads rows it owns within [0,M) via the same
   guard. Verify K3 does not load rstd for OOB rows in edge tiles (SubtileMGuard).
   Low-medium risk.
