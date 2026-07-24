> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md), [`review_protocol.md`](review_protocol.md), and [`amdgpu_exec_reference.md`](amdgpu_exec_reference.md) before starting this milestone.

## Milestone 8 — Bounds Checking (Sentinel-Value Detection)

**Executed by:** fresh implementor agent (parallel with M10 after M7)
**Reviewed before:** Milestone 12 begins

### Goal

Add out-of-bounds write detection to `KernelRunner` using a sentinel-value strategy.

**Important:** The original design (mmap guard pages + SIGSEGV handler) is technically wrong for GPU memory. GPU out-of-bounds writes do not raise CPU `SIGSEGV` — they either surface as HIP errors at `hipDeviceSynchronize` or silently corrupt adjacent allocations depending on XNACK configuration. A CPU signal handler cannot intercept GPU memory faults. The approach below uses a sentinel value written at a known device offset past the valid output region, which is technically correct.

**Also important:** `amdgpu_exec` is installed as a binary wheel with no buildable source available on this system (`~/.tensile/lib/python3.12/site-packages/amdgpu_exec/`). Do NOT attempt to modify `amdgpu_exec`. Instead, implement bounds checking as a new standalone nanobind module `tensilelite_bounds` following the same CMake and editable-install pattern as `tensilelite_runtime` (created in M7).

### Tasks

**8.1 — Implement sentinel-based `BoundedBuffer` in a new `tensilelite_bounds` nanobind module**

Create a new standalone nanobind module `tensilelite_bounds` (alongside `tensilelite_runtime`, following M7's CMake pattern). Do NOT add to `amdgpu_exec`.

`BoundedBuffer(size_bytes: int, sentinel_slots: int = 1)`:
- Allocates `size_bytes + sentinel_slots * sizeof(uint32_t)` bytes via `hipMalloc`.
- Fills the trailing `sentinel_slots` uint32 slots with the sentinel value `0xDEADBEEF` via `hipMemset` after allocation.
- Exposes `ptr_value: int` — the device pointer to the start of the **valid** region (same size as `GpuBuffer(size_bytes)` would produce).
- Exposes `data_ptr: int` — same as `ptr_value`.
- Exposes `sentinel_ptr: int` — device pointer to the first sentinel slot (`ptr_value + size_bytes`).
- `check_sentinel(self) -> bool`: copies the sentinel slots back to host via `hipMemcpy` and returns `True` if all sentinel values are still `0xDEADBEEF`, `False` if any were overwritten.
- `free() -> None`: calls `hipFree`. Called by `__del__`.

**8.2 — Integrate into `KernelRunner`**

`KernelRunner.run(..., bounds_check: bool = False)`: when `True`, allocates output buffers as `BoundedBuffer` instead of `GpuBuffer`. After the last iteration, calls `buf.check_sentinel()` for each output buffer and raises `AssertionError("output buffer overrun detected at byte offset N")` if the sentinel was overwritten. The sentinel check happens after `hipDeviceSynchronize` (already implicit from `GpuEvent.synchronize()`).

**8.3 — Write `test_bounds_check.py`**

- `TestCorrectKernel`: run a bf16 GEMM with `bounds_check=True`, assert `check_sentinel()` returns `True` and GPU output matches numpy reference.
- `TestOverrunDetection`: construct a `BoundedBuffer` where `size_bytes` is deliberately smaller than what the kernel writes (achieved by launching with an inflated N that exceeds the allocated output size), assert `check_sentinel()` returns `False`. This proves the sentinel is at the right offset and the check fires.
- `TestSentinelIntegrity`: without any kernel launch, create a `BoundedBuffer`, immediately call `check_sentinel()`, assert it returns `True` (sentinel was not disturbed by allocation or memset).

### Acceptance criteria
- `BoundedBuffer` allocates and initializes correctly (sentinel integrity test passes without GPU kernel).
- Correct kernels leave the sentinel intact.
- Overrun detection test produces `check_sentinel() == False`, not a silent wrong result.
- `tensilelite_bounds` follows the same editable-install pattern as `tensilelite_runtime` (M7).
- No regressions.
