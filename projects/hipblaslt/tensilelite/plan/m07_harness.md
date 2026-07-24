> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md), [`review_protocol.md`](review_protocol.md), and [`amdgpu_exec_reference.md`](amdgpu_exec_reference.md) before starting this milestone.

## Milestone 7 — Low-Level Harness: Rotating Buffers and I-Cache Simulation

**Executed by:** fresh implementor agent
**Reviewed before:** Milestones 8 and 10 begin (M7 unblocks both tracks)

### Goal

Extend `KernelRunner` with rotating buffer and I-cache module rotation. Extract
`getMinKernelSizeToGwEnd` and bind it via nanobind.

### Tasks

**7.0 — Extract `getMinKernelSizeToGwEnd` (prerequisite)**
Move `main.cpp:916–1027` into `client/src/ElfUtils.cpp` and declare in
`client/include/ElfUtils.hpp`. The entire block is inside `#if defined(__linux__)` (see
`main.cpp:916` and `main.cpp:1027`). Both `ElfUtils.cpp` and the declaration in
`ElfUtils.hpp` must be wrapped in the same `#if defined(__linux__)` guard — otherwise `<elf.h>` and the ELF types will fail to compile on non-Linux platforms. The nanobind binding for `get_icache_module_copies` must also be conditionally compiled. Add the compilation unit to `tensilelite-client-common` in `client/CMakeLists.txt`. Verify `tensilelite-client` still builds and passes existing tests.

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
`KernelRunner(hsaco_bytes, kernel_name, n_module_copies=1)`: loads `n_module_copies`
independent `GpuModule` instances. `run()` cycles through them per iteration. Default of 1
is identical to the pre-rotation behavior — no performance regression for correctness tests.
`n_module_copies="auto"` calls `get_icache_module_copies` on the compiled `.co` file.

**7.3 — Extend `BufferPool`** to integrate with `KernelRunner.run()`: output buffer slot
is advanced each iteration, matching `main.cpp:1266–1274`.

**7.4 — Benchmark timing helpers**
- `auto_scale_iters(flops, min_flops_per_sync) -> int`: replicates the algorithm in `BenchmarkTimer::numEnqueuesPerSync()` (`BenchmarkTimer.cpp:295–325`): `CeilDivide(min_flops_per_sync, max(flops, 1))`, clamped to `[m_numEnqueuesPerSync, m_maxEnqueuesPerSync]`. Note: `min_flops_per_sync` has no hardcoded default — read the value from `globalParameters["MinFlopsPerSync"]` or the command-line arg, not a magic constant. Line 54 stores the parsed arg; the algorithm is at lines 295–325.
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
