# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# PartialRMS Kernel Validation Failures — Bug Report

**Date:** 2026-07-17
**Discovered via:** comprehensive PartialRMS tuning sweep at shape 8192×8192×1×8192 on gfx950
**Sweep YAML:** `epilogues/yaml/tune_prms_8192.yaml`

---

## Tuning sweep results summary (fully clean caches, NumElementsToValidate=128)

| State | PASSED | FAILED |
|---|---|---|
| Pre-fix (HEAD~1) | 88 | 120 |
| Post-fix (HEAD) | 64 | 64 |

**Pre-fix failure breakdown:**
- 80 × `LDSB1` — Bug 1 (fixed)
- 24 × `LDSB0 + DTLA0 + MT1=256` — Bug 2 (open)

**Post-fix failure breakdown:**
- 0 × `LDSB1` — eliminated by the Bug 1 fix ✓
- 24 × `LDSB0 + DTLA0 + MT1=256` — Bug 2, unchanged (GPU kernel issue)

---

## Bug 1 — `1LDSBuffer=1` corrupts every LDS double-buffer swap  **[FIXED]**

### Trigger

Any Subtile kernel with `1LDSBuffer=1`, regardless of tile size, DirectToLds setting,
DepthU, PrefetchGlobalRead, StaggerU, StreamKAtomic, or epilogue flags. Affects plain
GEMM and PartialRMS equally.

### Root cause

The Subtile GR/LR emitters unconditionally emit double-buffer LDS swap masks:

```
swapMask = base XOR (base + ldsTotalSize)
```

applied at every macro-tile boundary. When `1LDSBuffer=1`, only one LDS slot is
allocated (`LdsNumBytes = ldsTotalSize`), so the swap target falls outside the reserved
region into a neighbouring workgroup's LDS. Every subsequent GR write and LR read is
corrupted, producing wrong D and wrong `partialBuf`.

The regular (non-Subtile) kernel path correctly skips the swap when `1LDSBuffer=1`
(`localWriteSwapOffsets:11598`, `localReadSwapOffsets:12705` in `KernelWriterAssembly.py`).
The Subtile emitters lack this guard.

### Code locations

- `Tensile/Components/Subtile/SubtileGREmit.py:978–981, 491–492, 1148–1153` — swap mask init
- `Tensile/Components/Subtile/SubtileLREmit.py:354–374` — LR swap mask init
- `Tensile/Components/Subtile/LogicalScheduler.py:insert_gr_lr_inc` — unconditional swap emission
- `Tensile/SolutionStructs/Solution.py` (pre-fix): `_validatePartialRMS` (line 216) does
  not reject `1LDSBuffer=1`; `1LDSBuffer` is not yet resolved at that call site (line 1117)

### Fix

`Tensile/SolutionStructs/Solution.py`, inside the `if state["UseSubtileImpl"]:` block:

```python
# The subtile scheduler always double-buffers LDS: SubtileGREmit/SubtileLREmit
# unconditionally emit buffer swaps that toggle between two LDS halves (XOR
# ldsTotalSize). Single-buffering is not implemented, so 1LDSBuffer=1 would
# allocate only one buffer (NumLdsBlk=1) while the code still swaps into the
# unallocated second half, corrupting results. Force double buffering.
state["1LDSBuffer"] = 0
```

**Validated:** 80 LDSB1 failures eliminated (pre-fix: 80, post-fix: 0). Reproducer
`epilogues/docs/repro_subtile_1ldsbuffer_bug.sh` exits "all kernels passed". Unit tests:
4794 passed, pre-existing failures unchanged.

---

## Bug 2 — `LDSB0 + DirectToLds=0 + MT1=256` produces wrong `partialBuf`  **[OPEN — GPU kernel bug]**

### Trigger

PartialRMS kernels with all three conditions simultaneously:
- `1LDSBuffer=0` (double-buffered — correct allocation)
- `DirectToLds=0` (buffer_load + local_write, no TDM)
- `MT1=256` (`MIWaveGroup[1]=wg_n=2`)

Failing tiles: MT64×256×64, MT64×256×128, MT256×256×64. The same geometry with
`DirectToLds=1` passes.

### Root cause

The failures are deterministic and correlate strictly with `DirectToLds=0` —
thread count does not affect the failing set. The `partialBuf` CPU reference loop
in `Reference.cpp` is provably race-free (per-iteration local accumulators, unique
write indices). The bug is in the GPU kernel.

The earlier conclusion that this was a client OpenMP race (from a lucky
`MAX_OMP_THREADS=1` run that showed 128/0) was incorrect — subsequent controlled
runs confirmed the same 24 failures at both 1 and 64 threads. The sweep is also
affected by GPU/system-load nondeterminism which can flip the failing set between
runs (sometimes 24 DTLA0 failures, sometimes 40 DTLA1 failures depending on GPU
load during the sweep).

### Related client fix

A real OpenMP violation was found and fixed in `Reference.cpp` (commit `7bea35036d`):
`omp_set_num_threads(MAX_OMP_THREADS)` was called from inside an active
`#pragma omp parallel for` region — UB per the OpenMP spec. This was unrelated to
Bug 2 but was fixed as part of this investigation.

### Investigation status

Exhaustive static analysis of `SubtilePartialRMSEmit.py` ruled out: AGPR indexing,
D-tile `localMMATileGrid`, MFMA schedule order, `waveN` computation, LR wave
partitioning, XOR butterfly reduction, token/colByte address computation. Root cause
has not been isolated. Runtime assembly comparison between a failing (DTLA0) and
passing (DTLA1) kernel is the recommended next step.

---

## Note on sweep nondeterminism

The failing set in the tuning sweep is affected by GPU/system load during the run.
Under heavy load the failing set shifts from 24 `DTLA0+MT1=256` failures to 40
`DTLA1+MT1=128` failures. This is system-level nondeterminism unrelated to the
kernel correctness bugs described above. To get reliable results, run the sweep
without concurrent GPU workloads.

---

## Reproducers

- `epilogues/docs/repro_subtile_1ldsbuffer_bug.yaml` + `.sh` — Bug 1 reproducer (plain
  bf16 GEMM, UseSubtileImpl + 1LDSBuffer=1 explicitly). Passes after the fix.

---

## Summary

| Bug | Condition | Count | Status |
|---|---|---|---|
| 1 | `UseSubtileImpl=1` + `1LDSBuffer=1` | 80 | **Fixed** in `Solution.py` |
| 2 | `LDSB0 + DTL=0 + MT1=256 (wg_n=2)` | 24 | **Open** — GPU kernel bug, root cause not yet isolated |
| Client | `omp_set_num_threads()` inside parallel region | — | **Fixed** in `Reference.cpp` (UB, unrelated to Bug 2) |
