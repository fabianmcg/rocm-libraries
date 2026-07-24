> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md), [`review_protocol.md`](review_protocol.md), and [`amdgpu_exec_reference.md`](amdgpu_exec_reference.md) before starting this milestone.

## Milestone 7 — Low-Level Harness: Rotating Buffers and I-Cache Simulation

**Executed by:** fresh implementor agent
**Reviewed before:** Milestones 8 and 10 begin (M7 unblocks both tracks)

### Goal

Extend `KernelRunner` with rotating buffer and I-cache module rotation. Extract
`getMinKernelSizeToGwEnd` and bind it via nanobind.

### Tasks

**7.0 — Extract `getMinKernelSizeToGwEnd` (prerequisite)**
Move the **entire `#if defined(__linux__)` block spanning `main.cpp:914–1025`** — including the
guard (`#if defined(__linux__)` at line 914), the explanatory comment (lines 915–926), the
function body `getMinKernelSizeToGwEnd` (lines 927–1024), and the closing `#endif` (line 1025)
— into `client/src/ElfUtils.cpp`, and declare `getMinKernelSizeToGwEnd` in
`client/include/ElfUtils.hpp`. (The function definition itself starts at line 927; do not leave
the guard and comment on lines 914–926 behind.) Both `ElfUtils.cpp` and the declaration in
`ElfUtils.hpp` must be wrapped in the same `#if defined(__linux__)` guard — otherwise `<elf.h>`
and the ELF types will fail to compile on non-Linux platforms. The nanobind binding for
`get_icache_module_copies` must also be conditionally compiled. Add the compilation unit to
`tensilelite-client-common` in `client/CMakeLists.txt`. Verify these line numbers against the
current `main.cpp` before extracting, and confirm `tensilelite-client` still builds and passes
existing tests.

**7.1 — Create `tensilelite_runtime` nanobind module with ELF binding**
Create the `tensilelite_runtime` nanobind module (alongside rocisa, following its CMake and
editable-install pattern). This is the **single** `tensilelite_runtime` target; M10 will
extend it by adding more bindings to the same CMake target and source file.

Initial binding:
```python
from tensilelite_runtime import get_icache_module_copies
n: int = get_icache_module_copies(co_path: str)
```
`get_icache_module_copies` wraps `getMinKernelSizeToGwEnd` from `ElfUtils.hpp`.

Note: `calculateAuto*` and `grouped_gemm_workspace_size` are NOT added here — they are
member functions of `ContractionSolution` (accessing mutable member caches) and require
`ContractionSolution` to be exposed as a Python type, which happens in M10. They are added
to this module as bound methods in M10.

Ensure the new module's source tree is **outside** rocisa's scanned source roots
(`rocisa/rocisa/__init__.py:141–152` scans both `_bi.SOURCE_ROOT` and
`_bi.STINKYTOFU_SOURCE_ROOT`). Verify by importing rocisa after adding the new module and
confirming no spurious staleness error fires.

**7.2 — Module rotation in `KernelRunner`**
`KernelRunner(hsaco_bytes, kernel_name, n_module_copies=1, co_path=None)`: loads
`n_module_copies` independent `GpuModule` instances. `run()` cycles through them per iteration.
Default of 1 is identical to the pre-rotation behavior — no performance regression for
correctness tests.
`n_module_copies="auto"` resolves the copy count via `get_icache_module_copies`, which requires
a **file path** to the `.co` (it parses the ELF symbol table). `KernelRunner`, however, is
constructed from `hsaco_bytes` (in-memory bytes), not a path — so `"auto"` needs the optional
`co_path: str = None` argument:
- If `n_module_copies == "auto"` and `co_path` is provided, call `get_icache_module_copies(co_path)`.
- If `n_module_copies == "auto"` and `co_path` is `None`, fall back to `n_module_copies = 1`
  and log a warning (the ELF probe cannot run without a file path).

**7.3 — Extend `BufferPool`** to integrate with `KernelRunner.run()`: output buffer slot
is advanced each iteration, matching `main.cpp:1266–1274`.

**7.4 — Benchmark timing helpers**
- `auto_scale_iters(flops, min_flops_per_sync, num_enqueues_per_sync=1, max_enqueues_per_sync) -> int`:
  replicates `BenchmarkTimer::numEnqueuesPerSync()` (`BenchmarkTimer.cpp:295–326`). When
  `min_flops_per_sync > 0`, `enqueuesByFlops = CeilDivide(min_flops_per_sync, max(flops, 1))`
  (else 0), then the result is
  `min(max(num_enqueues_per_sync, enqueuesByFlops), max_enqueues_per_sync)`. These three
  parameters map exactly to the three `BenchmarkTimer` constructor arguments
  (`BenchmarkTimer.cpp:52–54`): `num_enqueues_per_sync` ← `--num-enqueues-per-sync` (default 1,
  `main.cpp:293`), `max_enqueues_per_sync` ← `--max-enqueues-per-sync` (default -1,
  `main.cpp:294`), `min_flops_per_sync` ← `--min-flops-per-sync` (default 0, `main.cpp:297`).
  None have magic defaults beyond these — read them from `globalParameters` (`MinFlopsPerSync`
  and the enqueues-per-sync args) or the command line, not hardcoded constants. Verify line
  numbers against the current source.
- `BenchmarkResult.gflops(M, N, K)`: assert value is within plausible hardware range
  (100–2000 GFLOPS for gfx950 bf16) in the test suite — catches unit-conversion bugs.

**7.5 — Write `test_harness_rotation.py`**
- `TestBufferPool`: slot cycling in pure Python (mock GpuBuffer).
- `TestModuleRotation`: run bf16 GEMM with `n_module_copies=4`, verify output identical to
  `n_module_copies=1`.
- `TestTimingStats`: `p50_us <= p95_us`, `min_us <= p50_us`, `gflops > 0`.
- `TestIcacheCopyCount`: `get_icache_module_copies` on a compiled `.co` returns a positive
  integer.
- `TestGflopsPlausibility`: GFLOPS for a standard (1024,1024,1024) bf16 GEMM is in
  [100, 2000].

### Acceptance criteria
- `tensilelite-client` still builds and existing client tests pass after 7.0.
- `n_module_copies=1` produces identical GPU output to before.
- ELF binding returns a positive integer for a real `.co` file.
- Plausibility test passes. No regressions.

---
