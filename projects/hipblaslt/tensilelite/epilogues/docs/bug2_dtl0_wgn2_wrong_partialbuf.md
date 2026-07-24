# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Bug Report: PartialRMS LDSB0 + DirectToLds=0 + MT1=256 produces wrong `partialBuf`

**Status:** Open — GPU kernel bug, root cause not yet isolated  
**Discovered:** 2026-07-17, comprehensive PartialRMS tuning sweep at 8192×8192×1×8192 on gfx950  
**Reproducer:** `epilogues/docs/repro_bug2_dtl0_wgn2.sh`

---

## Trigger conditions

All three conditions must be present simultaneously:

| Parameter | Required value |
|---|---|
| `1LDSBuffer` | `0` (double-buffered) |
| `DirectToLds` | `0` (buffer_load + local_write, no TDM) |
| `MIWaveGroup[1]` (`wg_n`) | `2` (i.e. `MT1=256`) |

Failing tiles confirmed: `MT64×256×64`, `MT64×256×128`, `MT256×256×64`.  
The same geometries with `DirectToLds=1` pass.

The bug is not sensitive to: `DepthU`, `PrefetchGlobalRead`, `StaggerU`, `StreamKAtomic`, `PreloadKernArgs`, or `ExpandPointerSwap`. Every parameter combination that satisfies the three conditions above fails.

---

## Symptom

The GPU kernel writes incorrect values into `partialBuf` (the per-row partial RMS accumulator). The CPU reference computes the correct result. Tensile validation reports `FAILED` for the affected kernels at problem size `8192×8192×1×8192`; small problem sizes pass.

---

## Root cause

Not yet isolated. The failure correlates strictly with `DirectToLds=0` — switching to `DirectToLds=1` on the same tile geometry and problem size passes. The `partialBuf` CPU reference loop in `Reference.cpp` is provably race-free (per-iteration local accumulators, unique write indices), so the error originates in the GPU kernel.

Exhaustive static analysis of `SubtilePartialRMSEmit.py` has ruled out: AGPR indexing, D-tile `localMMATileGrid`, MFMA schedule order, `waveN` computation, LR wave partitioning, XOR butterfly reduction, and token/colByte address computation.

Recommended next step: runtime assembly comparison between a failing (`DTLA0`) kernel and the passing (`DTLA1`) counterpart on the same tile size and DepthU.

### What this bug is NOT

- It is not the `1LDSBuffer=1` swap-mask bug (Bug 1) — `1LDSBuffer` is `0` here.
- It is not an OpenMP race in `Reference.cpp` — that was a separate UB (`omp_set_num_threads` inside a parallel region, commit `7bea35036d`) unrelated to this failure. Controlled runs at both 1 and 64 OMP threads show the same 24 failures.

---

## Reproduction

### Prerequisites

- gfx950 GPU (MI350X)
- `~/.tensile` Python venv with TensileLite and rocisa installed
- TensileLite client built under `build_tmp/` (`invoke build-client`)

### Steps

```bash
# From the tensilelite root:
source ~/.tensile/bin/activate
bash epilogues/docs/repro_bug2_dtl0_wgn2.sh
```

The script:
1. Deletes any prior output directory to ensure a fully clean run (no cached kernels).
2. Runs `Tensile/bin/Tensile epilogues/yaml/tune_prms_8192.yaml` — the full four-group sweep (Groups A–D, ~128 kernels total once invalid combos are pruned).
3. Counts `PASSED`/`FAILED` lines from the benchmark CSV output.
4. Exits `0` if exactly 24 failures are found, `1` if some other nonzero count (load-dependent shift), `2` if all pass.

### Expected output

```
Results: 104 passed, 24 failed
==> Bug 2 reproduced: exactly 24 DTLA0+MT1=256 failures.
```

Failing kernels all match the pattern `DTLA0` + one of `MT64x256`, `MT256x256`.

### Important: the full sweep is required

The bug is **order- or load-dependent**. Running the isolated per-DepthU YAMLs (`repro_bug2_dtl0_wgn2_du64.yaml`, `repro_bug2_dtl0_wgn2_du128.yaml`) in isolation may show all-pass — the failing set in those files passed when run standalone on this system. The failures appear reliably only when Groups A–D are benchmarked in sequence within the same Tensile invocation.

Under heavy concurrent GPU load the failing set can shift: 24 `DTLA0+MT1=256` failures on a quiet system, up to 40 `DTLA1+MT1=128` failures under load. This is system-level nondeterminism unrelated to the underlying kernel correctness bug.

---

## Affected kernels (quiet-system run, 24 total)

| Tile | DepthU | Variants |
|---|---|---|
| `MT64×256` | 64 | 8 (all `PGR`×`SU`×`PKA` combos) |
| `MT64×256` | 128 | 8 |
| `MT256×256` | 64 | 8 |

All are `LDSB0`, `DTLA0`, `SIA3`, `SK3`, `SKFDPO1`, `UseSubtileImpl=True`, `UsePartialRMS=True`.

---

## Code locations to investigate

| File | Relevance |
|---|---|
| `Tensile/Components/Subtile/SubtilePartialRMSEmit.py` | PartialRMS accumulation and `partialBuf` write logic |
| `Tensile/Components/Subtile/SubtileGREmit.py` | Global-read path for `DTLA0` (buffer_load + local_write) |
| `Tensile/Components/Subtile/SubtileLREmit.py` | Local-read path |
| `client/source/Reference.cpp` | CPU reference (verified correct, race-free) |
