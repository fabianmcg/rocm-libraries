# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Bug Report: PAP=1 + PartialRMS/RstdScale causes GPU hard fault

**Status:** Fixed — validation guard added to `Solution.py`
**Discovered:** 2026-07-20, during investigation of Bug 2 linkage
**Reproducer:** `epilogues/docs/repro_bug3_pap1_dtl1_mt128.yaml`

---

## Trigger

Any Subtile PartialRMS or RstdScale kernel with `PrefetchAcrossPersistent=1`.
The combination `UseSubtileImpl=True + PartialRMS=True + PrefetchAcrossPersistent=1`
was never validated or rejected, and generates a kernel that faults at runtime.

Confirmed crashing config:
- `MT0=64 MT1=128`, `DirectToLds=1`, `DepthU=64`, `PGR=2`, `PAP=1`, `SU=0`, `PKA=False`

The fault is a **write-access to a read-only GPU memory page**, not a software
assertion. Fault address varies between runs (non-deterministic allocation),
ruling out a fixed-offset overwrite.

---

## Root cause

### What `PrefetchAcrossPersistent=1` does in the Subtile path

`prefetchAcrossPersistentSubtile` is injected by the Subtile scheduler at the NLL
(No-Load Loop) drain point of each persistent tile. It:

1. Checkpoints `WorkGroup0/1/2` to VGPRs.
2. Calls `skIndexToWG` to compute the **next** tile's `WorkGroup0/1/2`.
3. Calls `globalReadDTLInitCommonSgpr` which reinitializes `LocalWriteBaseAddrA/B`
   using the **next tile's** wave-row partition.
4. Calls `graAddresses` which overwrites `SrdA/SrdB` to point to the next tile's
   global A/B data.
5. Issues DTL `buffer_load_b128` (lds=True) writing next-tile A/B data directly
   into LDS at `m0 = LocalWriteBaseAddrA`.
6. Restores `WorkGroup0/1/2` and sets `SkPrefetchPrimed = 1`.

### The missing `papDtlSaveLdsBank` call

The non-Subtile PAP path calls `papDtlSaveLdsBank` after the DTL loads, encoding
which LDS buffer bank the PAP writes landed in. The Subtile PAP path
(`setupPrefetchAcrossPersistentSubtileLoads`) **does not** — only bit 0 of
`SkPrefetchPrimed` is set.

On the **next** persistent tile's preloop, `papDtlRestoreLdsBank` therefore does
nothing useful (it reads `SkPrefetchPrimed & LdsOffsetA_Blk = 0`), leaving the
LDS-bank alignment broken. The main loop reads stale/wrong data from LDS,
corrupting the AGPR accumulators.

### How corrupted AGPRs lead to the fault

The corrupted accumulators produce garbage `partials` values. In
`SubtilePartialRMSEmit._writePartialsFree0`, the `globalAddr` byte offset is
computed as `(token * n_d + WG0) * 4` where `token = colByte >> 1` is derived
from the corrupted accumulator. For certain input/tile combinations this offset
lands outside the `partialBuf` allocation, in a read-only mapped GPU page,
triggering the hardware memory protection fault.

A secondary mechanism: PAP fires on the last persistent tile even when
`StreamKIter >= StreamKIterEnd` should skip it. With garbage `WorkGroup0/1`
from `skIndexToWG` on a non-existent next tile, the subsequent PartialRMS
post-loop uses that garbage value as the `WG0` index in `partialBuf[token, WG0]`,
also producing an out-of-bounds store.

---

## Fix

**`Tensile/SolutionStructs/Solution.py` — three guard locations:**

### 1. `_validatePartialRMS` (primary guard)

After the `_validateSubtileEpiloguePrereqs` check:

```python
if state.get("PrefetchAcrossPersistent", 0):
    reject(state, printRejectionReason,
           "PartialRMS is not supported with PrefetchAcrossPersistent")
    return
```

### 2. `_validateRstdScale` (same gap, same fix)

```python
if state.get("PrefetchAcrossPersistent", 0):
    reject(state, printRejectionReason,
           "RstdScale is not supported with PrefetchAcrossPersistent")
    return
```

### 3. `if state["PrefetchAcrossPersistent"]:` block (defence in depth)

```python
if state.get("PartialRMS", False) or state.get("RstdScale", False):
    reject(state, printRejectionReason,
           "UseSubtileImpl=1 PrefetchAcrossPersistent not supported with PartialRMS/RstdScale epilogues")
```

This follows the same pattern as the Bug 1 fix (`state["1LDSBuffer"] = 0` for
UseSubtileImpl in `Solution.py`): reject the combination at validation time rather
than silently generating a broken kernel.

---

## Connection to Bug 2 and Bug 4

Removing PAP=1 from all groups in the sweep reproducer (`repro_bug2_24configs.yaml`)
caused the Bug 2 failures (DTLA0+MT1=256 wrong output) to disappear entirely across
5 consecutive clean-cache runs. This establishes that the PAP=1 kernels in the
warm-up groups were corrupting GPU state that then caused the Bug 2 kernels to
produce wrong output.

Similarly, the Bug 4 failures (DTLA1+MT1=128 wrong output, data-dependent) appeared
in a run that included PAP=1 kernels. Bug 4 may also be a downstream effect of
PAP=1 state corruption rather than an independent kernel bug.

Both Bug 2 and Bug 4 are **order-dependent** and do not reproduce in pytest
isolation (where each kernel runs in a quiescent GPU context). The pytest suites
(`test_bug2_failing_configs.py`, `test_bug4_dtl1_mt128.py`) serve as regression
gates for post-fix verification.

---

## Summary

| Item | Detail |
|---|---|
| Bug | PAP=1 + PartialRMS/RstdScale → GPU hard fault (write to read-only page) |
| Root cause | Subtile PAP path omits `papDtlSaveLdsBank`; breaks LDS-bank alignment on the next tile; corrupts AGPR accumulators; produces out-of-bounds `partialBuf` store |
| Fix | Reject `PrefetchAcrossPersistent=1` in `_validatePartialRMS` and `_validateRstdScale` in `Solution.py` |
| Side effect | Also root-causes Bug 2 and likely Bug 4 (both are downstream of PAP=1 state corruption) |
| Reproducer | `epilogues/docs/repro_bug3_pap1_dtl1_mt128.yaml` |
