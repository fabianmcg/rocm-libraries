> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 11 — ROCprofiler-SDK Bindings

**Executed by:** fresh implementor agent (after M10)
**Reviewed before:** Milestone 12 begins

### Goal

Expose ROCprofiler-SDK hardware counter collection to Python via a nanobind module.
`LD_PRELOAD` is not required — `rocprofiler_force_configure` is called from `PyInit`.

### Tasks

**11.1 — New nanobind module `tensilelite_profiler`**
Built conditionally with `-DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK=ON` (the option defined in `client/CMakeLists.txt:77`).

The module's `PyInit_tensilelite_profiler` function:
1. Calls `rocprofiler_force_configure(&rocprofiler_configure)`, which triggers `tool_init_impl`
   synchronously.
2. Checks the **return code** of `rocprofiler_force_configure`. If it returns
   `ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED`, the configuration window has already
   closed (HSA was initialized by a HIP call before import), and `PyInit` raises
   `RuntimeError("import tensilelite_profiler before any HIP call")`.
   Do NOT use `rocprofiler_is_initialized()` as the guard — it reflects rocprofiler's own
   initialization state, not HIP's, and returns 0 in the exact failure case we need to catch. `rocprofiler_configure` is exported from this `.so` with default visibility.
   `tool_init_impl` creates the context and registers the dispatch/record callbacks — this is
   the existing `RocProfiler.cpp` implementation, reused as-is.

Python API (maps 1:1 to existing `RocProfiler` methods):
```python
tensilelite_profiler.initialize(device_idx: int, counter_names: list[str])
tensilelite_profiler.start()   # rocprofiler_start_context
tensilelite_profiler.stop()    # rocprofiler_stop_context
tensilelite_profiler.enable()  # resets std::promise, arms for next dispatch
tensilelite_profiler.disable()
tensilelite_profiler.fetch(solution_idx: int) -> str  # blocks on m_future.get(), GIL released
```

The `fetch()` binding releases the GIL before `m_future.get()` using
`nb::gil_scoped_release release;` — the `recordCallback` fires from a C++ HSA signal thread
and must not block on Python.

**11.2 — Import-ordering enforcement**
In `Tensile/client/tests/conftest.py` for the profiler test environment: import
`tensilelite_profiler` at session scope (via `autouse=True` session fixture) before any
fixture that touches a GPU. Add a `TENSILELITE_PROFILER_AVAILABLE` flag analogous to
`HAVE_DEPS`.

**11.3 — Integrate into `KernelRunner`**
`KernelRunner.run(..., rocprof_counters: list[str] = None)`: when non-empty, wraps each
iteration in `enable()` / `disable()` and calls `fetch()` after each, attaching results to
`BenchmarkResult.counters: dict[str, float]`.

**11.4 — Test marking**
No dedicated tox environment needed. `LD_PRELOAD` is not required — the module uses
`rocprofiler_force_configure` from `PyInit`. Add a `requires_rocprof` pytest marker
(analogous to `requires_gfx950`) that skips when `tensilelite_profiler` is not built. The
import-ordering constraint is enforced by importing the module in a session-scoped
`autouse=True` fixture in `conftest.py` before any GPU fixture runs.

Note: M13.4 instructs updating `AGENTS_reference.md` with `tox -e unit -k requires_rocprof`
as the command to run profiler tests — NOT `tox -e profiler` (no dedicated env exists).

**11.5 — Write `test_profiler.py`**
- `TestProfilerUnavailable`: when `tensilelite_profiler` is not built, all profiler calls
  degrade gracefully (no crash, no import error).
- `TestImportOrdering`: importing `tensilelite_profiler` after a HIP call raises
  `RuntimeError` with a clear message.
- `TestCounterCollection` (requires ROCprofiler-SDK): run a bf16 GEMM with
  `rocprof_counters=["SQ_WAVES"]`, assert the returned counter value is a positive integer.

### Acceptance criteria
- `TestImportOrdering` passes — the module correctly detects late import and raises.
- `TestCounterCollection` passes with at least one hardware counter on gfx950.
- Graceful degradation when `TENSILELITE_CLIENT_ENABLE_ROCPROFSDK=OFF`.
- Reviewer confirms GIL is released in `fetch()` binding.
- No regressions.
