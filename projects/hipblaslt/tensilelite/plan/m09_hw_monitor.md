> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 9 — Hardware Monitoring (pyamdsmi)

**Executed by:** fresh implementor agent (parallel with M10 after M7)
**Reviewed before:** Milestone 12 begins

### Goal

Add GPU clock, temperature, and fan monitoring during benchmark runs.

### Tasks

**9.1 — Implement `Tensile/client/hw_monitor.py`**
```python
class HardwareMonitor:
    def __init__(self, device_id=0, interval_ms=10): ...
    def __enter__(self): ...   # starts daemon thread
    def __exit__(self, ...): ...  # stops, computes averages
    avg_temp_edge: float
    avg_gpu_clock_mhz: float
    avg_soc_clock_mhz: float
    avg_mem_clock_mhz: float
```
Daemon thread polls the pyamdsmi GPU metrics API at `interval_ms`. Before implementing,
verify the correct function name for the installed pyamdsmi version:
```python
import pyamdsmi; print([x for x in dir(pyamdsmi) if "metric" in x.lower()])
```
Pin the pyamdsmi version in `tox.ini` deps alongside `amdgpu_exec`. If `amdsmi_get_gpu_metrics`
does not exist in the installed version, use `amdsmi_get_gpu_activity` or the equivalent
available function and document the substitution. A `threading.Event` signals the benchmark
window start/stop. No HIP API calls inside the thread. Matches `HardwareMonitor.cpp:38–60`.

**9.2 — Integrate into `KernelRunner`**
`KernelRunner.run(..., hw_monitor=False)`: when `True`, wraps the benchmark window in a
`HardwareMonitor` context and attaches `.hw` to the returned `BenchmarkResult`.

**9.3 — Write `test_hw_monitor.py`**
- With `pyamdsmi` available: run a bf16 GEMM with `hw_monitor=True`, assert
  `avg_gpu_clock_mhz > 0` and `avg_temp_edge > 0`.
- Without `pyamdsmi`: `HardwareMonitor.__enter__` logs a warning, is a no-op, no crash.

### Acceptance criteria
- Monitor produces plausible values during a real benchmark.
- Graceful no-op when `pyamdsmi` is absent. No regressions.
