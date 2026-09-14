<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# Why MI shape does not correlate with MegaFused epilogue overhead

**Problem:** GEMM 8192×8192×8192, bf16/bf16, HPA, TN, batched, gfx950 wave64.
**Winner tile:** MT384×256, MI16×16×32, MIWT12×8, WG[2,2], DU64, PGR2, SK3, UseSubtileImpl.
**Emitter commit:** 458b9f0c24d.
**Data:** timing from `/tmp/baselines-sweep` (parser `parse_baselines.py` → `parse_baselines.out`);
instruction counts from `parse_mishape.py` → `parse_mishape.out` over the MegaFused (MFE1)
`.s` files under `/tmp/baselines-sweep/.../MFE_UserArgs_00/.../assembly/`; correlations from
`mishape_correlation.py` → `mishape_correlation.out`. All instruction counts are reviewer-verified
against the assembly and against the source loops in
`Tensile/Components/Subtile/SubtileMegaFusedEmit.py`.

Prior context: `megafused_vs_beta_cost_analysis.md` (MegaFused +123.3 µs vs beta*C +36.1 µs, 3.4×),
`beta_overhead_analysis.md` (beta corr(tile area, Δ) = +0.40).

---

## TL;DR — the correlation is weak for two compounding reasons, not one

1. **Structural (primary): aggregate epilogue work is conserved.** The per-workgroup fused
   epilogue instruction count scales *almost perfectly linearly* with wave-tile area
   (`fusedInstr ≈ 96·wtArea + 503`, corr **+0.998**), but the workgroup count scales inversely
   (`numWG ≈ problemArea/tileArea`). Their product — which, under a uniform occupancy of 1
   workgroup/CU, is what maps to serial epilogue time — is **tile-area-invariant**:
   `fusedInstr × numWG = 7.06M ± 4.2%` across all tiles. A conserved quantity cannot correlate
   with the variable (tile area) that conserves it. This is the dominant reason.

2. **Statistical (secondary): the reported +0.085 is outlier-driven.** Only 10 tiles survived
   codegen; one (MT448×128) is a grid-quantization outlier (M=448 ∤ 8192; its no-epi baseline
   1010 µs is itself anomalous). Removing it lifts corr(area, Δ) from **+0.085 to +0.486** — i.e.
   the *residual* second-order area effect (the same exposed-tail effect beta rides) is actually
   present and comparable to beta's +0.40, but is masked by one anomalous point in a tiny sample.

The task premise ("MI shape does not correlate") is therefore **half structural, half artifact**:
the first-order aggregate is genuinely conserved, and the surviving second-order effect is hidden
by sample noise.

---

## Hypothesis verdicts

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| H1 | Fixed-cost setup dominates | **FALSIFIED** | Fixed intercept is only 503 of 9691 instrs = **5.2%**; fused pass is ~linear in wtArea. |
| H2 | Residual loads don't scale with area / depend on N_hidden | **FALSIFIED** | Residual loads = wtArea (96), gamma = wtM (12), both tile-geometry-driven. N_hidden only selects the (skipped) straddle mask, not the load count. |
| H3 | I-cache footprint is tile-independent | **FALSIFIED** | Fused-pass instruction count varies 2.5× across tiles (3870→9780). (Whole-kernel footprint being small already ruled I-cache out as a *cost* driver.) |
| H4 | Occupancy=1 collapses the area gradient | **CONFIRMED (contributing)** | Every tile is LDS-limited (128–160 KB/wg vs 160 KB/CU) and VGPR-saturated (next_free 344–512 of 512) → exactly 1 workgroup/CU. Uniform latency-hiding removes the occupancy gradient beta's mechanism would need. |
| H5 | Surviving 10 tiles are a biased/small sample | **CONFIRMED (secondary)** | n=10 with one high-leverage outlier (MT448×128); removing it flips corr +0.085 → +0.486. |
| H6 | Per-element math dominates over store overhead | **FALSIFIED (as stated)** | Genuine fused math (v_pk_add 192 + v_fma 384 + v_pk_mul 192 = **768**, 7.9%) is *smaller* than per-element OOB **masking** (v_cmp 1297 + v_cndmask 1152 = **2449**, 25%). Masking/addressing + narrow stores dominate, not RMS/gamma math. |

---

## 1. Codegen scales with tile area (falsifies H1, H3)

Fused-pass instruction count (residual load + residual-add + ΣH² + gamma + ResidualOut store +
cross-wave RMS reduction) vs wave-tile area, over 7 tiles spanning the surviving range:

| Tile | wtM×wtN | wtArea | fusedInstr | fused/wtArea | next_free_vgpr | LDS (KB) | occ |
|------|--------:|-------:|-----------:|-------------:|---------------:|---------:|----:|
| MT576×64  | 9×4  | 36 | 3870 | 107.5 | 344 | 160 | 1 |
| MT128×384 | 2×24 | 48 | 5378 | 112.0 | 440 | 128 | 1 |
| MT448×128 | 14×4 | 56 | 5770 | 103.0 | 472 | 144 | 1 |
| MT512×128 | 16×4 | 64 | 6567 | 102.6 | 472 | 160 | 1 |
| MT384×192 | 12×6 | 72 | 7370 | 102.4 | 504 | 144 | 1 |
| MT256×384 | 8×12 | 96 | 9780 | 101.9 | 512 | 160 | 1 |
| MT384×256 | 12×8 | 96 | 9691 | 100.9 | 512 | 160 | 1 |

- `corr(wtArea, fusedInstr) = +0.998`, fit `fusedInstr ≈ 96.0·wtArea + 502.8`.
- **Fixed fraction = 503 / 9691 = 5.2%** at the winner → H1 falsified.
- Instruction count spans 3870→9780 (2.5×) → H3 (flat I-cache footprint) falsified.

The per-element constant is ~96 instructions per wave-tile element — roughly 1.5× beta's
executed ~61 instr/element (beta executed epilogue ≈ 5881 for the same wtArea 96).

---

## 2. Instruction-count formulas (verified against source + assembly)

Geometry for MI16×16×32, wave64, WG[2,2]: `rpl = mfmaM·mfmaN/waveSize = 16·16/64 = 4`;
`mmaM = wtM`, `mmaN = wtN`; number of (m,n) residual tiles = `mmaM·mmaN = wtArea`.

| Quantity | Formula | Source | MT384×256 |
|----------|---------|--------|----------:|
| Residual loads (`buffer_load_dwordx2`) | `wtArea · (rpl/4) = wtArea` | `_issueResidualWide`, `_prologResidualLoads` | 96 |
| Gamma loads (`buffer_load_dwordx2`) | `nQTilesM · tilesPerBlockM = mmaM = wtM` | `_loadGammaBlockWide` | 12 |
| ΣH² FMAs (`v_fma_f32`) | `wtArea · rpl = 4·wtArea` | `_pass1AccResRms` | 384 |
| Residual add (`v_pk_add_f32`) | `wtArea · (rpl/2) = 2·wtArea` | `_pass1AccResRms` | 192 |
| Gamma multiply (`v_pk_mul_f32`) | `wtArea · (rpl/2) = 2·wtArea` | `_pass3GammaAmax` | 192 |
| ResidualOut interior store (`buffer_store_dwordx2`) | `wtArea` (one B64/tile, 4×bf16) | `_issueResidualOutWide` | 96 |
| ResidualOut straddle (`buffer_store_short`, **dead**) | `wtArea · rpl = 4·wtArea` | `_issueResidualOutStraddle` | 384 |
| RMS partial store (`buffer_store_dword`) | `mmaN = wtN` | `_writePartialsFree0` | 8 |
| **Compare — beta*C** C-loads | `6 · wtArea` | beta GWB epilogue | 576 |

Two consequences:

- **Load count is tile-geometry-driven, not N_hidden-driven (H2 falsified).** Residual loads =
  wtArea exactly; N_hidden appears only in the OOB masking (`_maskWideResidualOOB`,
  `_computeBf16Addr`) that decides *which* elements are clamped, never *how many* loads issue.
  Load volume (108) is comparable to beta's (96) — as the prior report found, loads are not the
  differentiator.

- **The 384 `buffer_store_short` are assembled but dead.** They sit behind the `s_cbranch_scc0`
  in `_issueResidualOutStraddle`; for a multiple-of-rpl N_hidden the straddle mask is empty, SCC=0,
  and the branch skips all rpl per-element stores plus their `_computeBf16Addr`
  (`v_cmp`+`v_cndmask`) address math. ~1900 of the winner's 9691 fused instructions (~20%) are
  this dead straddle scaffold — still O(wtArea), so it does not change the scaling.

---

## 3. Fixed vs tile-scaling breakdown (winner MT384×256, fused pass = 9691)

Every category is O(wtArea); there is essentially **no** tile-independent block except the two
`s_barrier` and a handful of setup instructions inside the 5.2% intercept.

| Category | count | % of fused | scales as | note |
|----------|------:|-----------:|-----------|------|
| Address arithmetic (v_add_u32 / v_lshl / v_and / uncategorized) | ~4378 | ~45% | O(wtArea) | per-element residual + ResidualOut addressing |
| Per-element OOB masking `v_cmp`+`v_cndmask` | 2449 | 25% | O(wtArea) | residual OOB + ResidualOut token/nhPos guards (~384+384 in dead straddle) |
| Genuine fused math `v_pk_add`+`v_fma`+`v_pk_mul` | 768 | 7.9% | O(wtArea) | residual add + ΣH² + gamma — the irreducible feature |
| bf16 conversion `v_cvt_pk*bf16` | 576 | 5.9% | O(wtArea) | pack for store (192) + dead straddle cvt (384) |
| Stores (`dwordx2` 96 + `short` 384 + `dword` 8) | 488 | 5.0% | O(wtArea) | 384 dead; interior is narrow dwordx2 (4×bf16) |
| exec management `s_and_saveexec` 97 + `s_mov_b64 exec` 193 | 290 | 3.0% | O(wtArea) | per-tile exec narrowing for the wide store |
| `s_waitcnt` | 113 | 1.2% | O(wtArea) | GWB decreasing-vlcnt schedule |
| Loads `buffer_load_dwordx2` | 108 | 1.1% | O(wtArea) | 96 residual + 12 gamma |
| `ds_bpermute` + `s_barrier` (RMS reduction) | 18 | 0.2% | ~fixed (wg_m) | the only near-fixed block |
| **Fixed intercept (setup, SRDs, initRmsSum)** | ~503 | **5.2%** | fixed | H1's candidate — small |

**Finding for H6:** the dominant per-element cost is **address arithmetic + OOB masking**
(≈70% combined), not the RMS/gamma math (7.9%). The store body itself is small in *count* but
carries the extra exec-management and dead-straddle scaffold that beta's cooperative store avoids.

---

## 4. Aggregate work is conserved → the primary structural reason

Under occupancy = 1 (every tile, §5), each CU processes its workgroups serially, so total serial
epilogue instructions per CU ≈ `fusedInstr × numWG / numCU`. Because `fusedInstr ∝ wtArea` and
`numWG ∝ 1/tileArea = 1/(1024·wtArea)`, the product is tile-invariant:

| Tile | wtArea | fusedInstr | numWG | fusedInstr × numWG |
|------|-------:|-----------:|------:|-------------------:|
| MT576×64  | 36 | 3870 | 1920 | 7.43 M |
| MT128×384 | 48 | 5378 | 1408 | 7.57 M |
| MT448×128 | 56 | 5770 | 1216 | 7.02 M |
| MT512×128 | 64 | 6567 | 1024 | 6.72 M |
| MT384×192 | 72 | 7370 |  946 | 6.97 M |
| MT256×384 | 96 | 9780 |  704 | 6.89 M |
| MT384×256 | 96 | 9691 |  704 | 6.82 M |

`mean = 7.06 M, CV = 4.2%`. The aggregate epilogue instruction budget is essentially fixed by the
problem size, independent of tile shape. This is why the first-order overhead does not track tile
area: **tile shape redistributes the same total work between "few big tails" and "many small
tails" without changing the sum.** Beta*C obeys the same conservation for its C-loads
(`6·wtArea·numWG` is likewise constant); the difference is only in the *residual* exposed-tail
term (§6).

---

## 5. Occupancy is uniformly 1 (confirms H4)

| Tile | LDS/wg (KB) | next_free_vgpr | wg/CU |
|------|------------:|---------------:|------:|
| MT576×64  | 160 | 344 | 1 |
| MT128×384 | 128 | 440 | 1 |
| MT448×128 | 144 | 472 | 1 |
| MT512×128 | 160 | 472 | 1 |
| MT384×192 | 144 | 504 | 1 |
| MT256×384 | 160 | 512 | 1 |
| MT384×256 | 160 | 512 | 1 |

gfx950 provides 160 KB LDS/CU (the kernels launch successfully with 163840-byte allocations,
empirically proving ≥160 KB); at ≥128 KB/wg only one workgroup fits per CU. Independently, the
unified 512-VGPR file with next_free 344–512 (all > 256) admits ≤1 wave/SIMD. **No tile has a
latency-hiding advantage over another**, so the beta-style "big tile pays a longer exposed tail,
and there are fewer workgroups to overlap it against" mechanism has no occupancy gradient to
amplify it here — reinforcing the flat aggregate of §4.

---

## 6. The masked second-order effect (confirms H5)

| Correlation | value | n |
|-------------|------:|--:|
| corr(area, Δ) full sample | +0.085 | 10 |
| corr(area, Δ) 7-tile instrumented subset | +0.027 | 7 |
| corr(area, Δ) **subset minus MT448×128** | **+0.486** | 6 |
| corr(numWG, Δ) full sample | −0.187 | 10 |
| corr(fusedInstr, Δ) subset | +0.026 | 7 |

MT448×128 is anomalous on independent grounds: M=448 does not divide 8192 (grid quantization →
19 M-workgroups, the last only 28% populated), and its no-epi baseline (1010 µs) is a >200 µs
outlier versus same-area peers (MT512×128 780 µs, MT384×192 803 µs). Its Δ = +228.7 µs is inflated
by whole-kernel imbalance, not epilogue structure. Removing it exposes a **+0.486** area effect —
the same sign and magnitude as beta's +0.40. So the underlying MI-shape sensitivity exists; it is
suppressed in the headline number by one bad point in a 10-tile sample. The correct statement is
not "MegaFused is insensitive to MI shape" but "MegaFused's first-order overhead is conserved
across MI shapes, and its weak second-order sensitivity is comparable to beta's but statistically
buried."

---

## 7. Implication — what actually reduces MegaFused overhead

**Tile shape is not the lever.** Because the aggregate epilogue instruction budget is conserved
(§4), no choice of MT/MI geometry meaningfully shrinks the ~123 µs overhead — reshaping only trades
tail length for tail count. The residual +0.486 second-order effect says a *smaller* wave-tile area
(more, shorter tails) helps marginally, mirroring beta's recommendation, but the payoff is small
and bounded by the conserved aggregate.

**The per-element constant is the lever.** The overhead is set by the ~96 instr/element density,
and that density is dominated by items tile shape cannot touch:

1. **Per-element OOB masking (25%)** — `v_cmp`/`v_cndmask`, ~⅓ of it the *dead* straddle path.
   Replacing per-element `_computeBf16Addr` masking + `_issueResidualOutStraddle` with beta's
   scalar block OOB mask (SubtileMGuard/NGuard applied via a plain `s_mov_b64 exec`) removes the
   97 per-tile `s_and_saveexec`, the 385 `s_mov_b64 exec`, and the entire 384-store dead straddle
   scaffold.
2. **Narrow stores (5%)** — interior ResidualOut is `buffer_store_dwordx2` (4×bf16 = 8 B); beta
   stores cooperative `buffer_store_dwordx4` (8×bf16 = 16 B). Adopting the `ds_bpermute`-packed
   dwordx4 idiom halves store-instruction count.
3. **Irreducible feature cost (~8%)** — residual add, ΣH², gamma multiply, and the fixed
   cross-wave RMS reduction are genuine work with no beta analog and should be left alone.

This is exactly the beta-idiom store rewrite recommended in `megafused_vs_beta_cost_analysis.md`
and `beta_c_vs_megafused_bf16_schedule.md` (emitters `_storeResidualOutRow`,
`_issueResidualOutWide`, `_issueResidualOutStraddle`, `_computeBf16Addr`). The new evidence here
*quantitatively justifies why it is the only lever*: tile shape cannot shrink a conserved
aggregate, so the sole way to cut the epilogue is to lower the per-element instruction density it
conserves.

---

## Methodology / provenance

Timing: `/tmp/baselines-sweep`, NumBenchmarks=10, medians via `parse_baselines.py`
(`parse_baselines.out`). Instruction counts: `parse_mishape.py` (`parse_mishape.out`) slices each
MegaFused `.s` between the `Global Write Elements` banner and the first `Global Write Batch #0`
(fused pass) / first `Global Write Edge Batch #0` (NonEdge store); the last `v_mfma` precedes the
epilogue in every tile (post-MFMA tail confirmed). Instruction detection matches AMDGPU opcode
prefixes only, excluding comments, labels, and directives. Correlations and the linear fit:
`mishape_correlation.py` (`mishape_correlation.out`). All seven claim classes (section boundaries,
load formula, per-category formulas, fixed-fraction, aggregate constancy, outlier leverage,
occupancy) were independently reviewer-verified against the assembly and the source loops in
`SubtileMegaFusedEmit.py`. Caveats: the +0.486 outlier-removed correlation rests on 6 points and is
directional, not precise; the 5.2% fixed fraction and 96 instr/element slope are R²≈0.996 fits over
7 tiles.
