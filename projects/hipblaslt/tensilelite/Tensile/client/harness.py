# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level GPU execution harness shared by all milestone tests.

Provides BufferPool, KernelRunner, and BenchmarkResult without importing
any GPU primitives at module level so that collection does not trigger HIP
initialization.
"""

from __future__ import annotations

import contextlib
import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Union

_log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Timing data returned by KernelRunner.run()."""

    timesNs: List[int]
    warmupN: int
    # Set to a HardwareMonitor instance when run() is called with hwMonitor=True.
    hw: Optional[object] = None

    @property
    def meanUs(self) -> float:
        return statistics.mean(self.timesNs) / 1_000.0

    @property
    def p50Us(self) -> float:
        return statistics.median(self.timesNs) / 1_000.0

    @property
    def p95Us(self) -> float:
        sorted_ns = sorted(self.timesNs)
        idx = max(0, int(len(sorted_ns) * 0.95) - 1)
        return sorted_ns[idx] / 1_000.0

    @property
    def minUs(self) -> float:
        return min(self.timesNs) / 1_000.0

    def gflops(self, M: int, N: int, K: int) -> float:
        """Compute GFLOPs from mean timing using 2*M*N*K FLOPs per iteration."""
        ops = 2 * M * N * K
        return ops / (self.meanUs * 1e-6) / 1e9


class BufferPool:
    """Round-robin pool of GpuBuffer instances.

    Pre-allocates nSlots independent device buffers of sizeBytes each.
    Cycling them across iterations prevents write-combining effects and
    avoids the kernel reading its own output.
    """

    def __init__(self, nSlots: int, sizeBytes: int, gpuBufferCls) -> None:
        self._slots = [gpuBufferCls(sizeBytes) for _ in range(nSlots)]
        self._idx = 0

    def next(self):
        buf = self._slots[self._idx]
        self._idx = (self._idx + 1) % len(self._slots)
        return buf

    def iterSlots(self):
        """Yield each buffer slot in allocation order."""
        return iter(self._slots)

    def freeAll(self) -> None:
        for buf in self._slots:
            buf.free()
        self._slots = []


class KernelRunner:
    """Wraps one or more GpuModule instances and a BufferPool for benchmarking.

    No GPU calls are made in __init__; GpuModule and GpuFunction objects are
    passed in by the caller after they have already loaded the HSACO.

    Module rotation (using several independent GpuModule copies of the same
    HSACO) prevents I-cache reuse from artificially inflating performance.
    """

    def __init__(self, functions: list, outputPool: "BufferPool | None" = None) -> None:
        """
        functions:  list of GpuFunction objects (one per module copy).
        outputPool: BufferPool for rotating output buffers, or None if the
                    caller manages output buffers inside argsFn.
        """
        if not functions:
            raise ValueError("functions must be non-empty")
        self._functions = functions
        self._outputPool = outputPool
        self._call_count = 0

    @classmethod
    def fromHsaco(
        cls,
        hsacoBytes: bytes,
        kernelName: str,
        nModuleCopies: Union[int, str] = 1,
        coPath: str = None,
    ) -> "KernelRunner":
        """Create a KernelRunner by loading nModuleCopies GpuModule instances.

        When nModuleCopies='auto' and coPath is provided, the copy count is
        determined from the ELF symbol table via get_icache_module_copies.
        When nModuleCopies='auto' and coPath is None, falls back to 1.
        """
        try:
            from amdgpu_exec import GpuModule
        except ImportError as exc:
            raise RuntimeError("amdgpu_exec is required for KernelRunner.fromHsaco") from exc

        if nModuleCopies == "auto":
            if coPath is not None:
                try:
                    from tensilelite_runtime import get_icache_module_copies
                    nModuleCopies = get_icache_module_copies(coPath)
                except ImportError:
                    _log.warning(
                        "tensilelite_runtime not available; falling back to nModuleCopies=1"
                    )
                    nModuleCopies = 1
            else:
                _log.warning(
                    "nModuleCopies='auto' with no coPath; falling back to nModuleCopies=1"
                )
                nModuleCopies = 1

        modules = [GpuModule(hsacoBytes) for _ in range(nModuleCopies)]
        functions = [m.get_function(kernelName) for m in modules]
        return cls(functions=functions, outputPool=None)

    def _runOneIter(self, argsFn, grid: tuple, block: tuple, GpuEvent) -> int:
        """Execute one kernel iteration, advance the call counter, return elapsed ns."""
        fn = self._functions[self._call_count % len(self._functions)]
        out_buf = self._outputPool.next() if self._outputPool is not None else None
        args = argsFn(out_buf)
        start = GpuEvent()
        stop = GpuEvent()
        start.record()
        fn.launch(grid, block, args)
        stop.record()
        stop.synchronize()
        self._call_count += 1
        return stop.elapsed_ns(start)

    def run(
        self,
        argsFn,
        grid: tuple,
        block: tuple,
        nWarmup: int,
        nIters: int,
        boundsCheck: bool = False,
        hwMonitor: bool = False,
    ) -> BenchmarkResult:
        """Launch the kernel and collect timing.

        argsFn:      callable(output_buf) -> list of kernel args. Called once per
                     iteration with the next output buffer from the pool (or None
                     if no outputPool was provided).
        grid:        (gridX, gridY, gridZ) tuple.
        block:       (blockX, blockY, blockZ) tuple.
        nWarmup:     number of iterations before timing begins.
        nIters:      number of timed iterations.
        boundsCheck: when True, call checkSentinel() on every output pool slot
                     after the final iteration. Raises AssertionError if any
                     sentinel was overwritten. Requires pool slots to be
                     BoundedBuffer instances.
        hwMonitor:   when True, wraps the benchmark window in a HardwareMonitor
                     context and attaches .hw to the returned BenchmarkResult.
        """
        try:
            from amdgpu_exec import GpuEvent
        except ImportError as exc:
            raise RuntimeError("amdgpu_exec is required for KernelRunner.run") from exc

        for _ in range(nWarmup):
            self._runOneIter(argsFn, grid, block, GpuEvent)

        monitor = None
        if hwMonitor:
            from Tensile.client.hw_monitor import HardwareMonitor as _HwMonitor
            monitor = _HwMonitor()

        timesNs = []
        ctx = monitor if monitor is not None else contextlib.nullcontext()
        with ctx:
            for _ in range(nIters):
                timesNs.append(self._runOneIter(argsFn, grid, block, GpuEvent))

        if boundsCheck and self._outputPool is not None:
            # The device is idle after the final synchronize(); safe to read sentinels now.
            for buf in self._outputPool.iterSlots():
                if hasattr(buf, "checkSentinel") and not buf.checkSentinel():
                    raise AssertionError("output buffer overrun detected")

        return BenchmarkResult(timesNs=timesNs, warmupN=nWarmup, hw=monitor)


def autoScaleIters(
    flops: int,
    minFlopsPerSync: int,
    numEnqueuesPerSync: int = 1,
    maxEnqueuesPerSync: int = -1,
) -> int:
    """Compute the number of enqueues per sync, matching BenchmarkTimer::numEnqueuesPerSync.

    When minFlopsPerSync > 0, enqueuesByFlops = ceil(minFlopsPerSync / max(flops, 1)).
    The result is clamp(max(numEnqueuesPerSync, enqueuesByFlops), maxEnqueuesPerSync).
    maxEnqueuesPerSync < 0 means no upper bound (matches the C++ -1 default).
    """
    enqueuesByFlops = 0
    if minFlopsPerSync > 0:
        enqueuesByFlops = math.ceil(minFlopsPerSync / max(flops, 1))
    result = max(numEnqueuesPerSync, enqueuesByFlops)
    if maxEnqueuesPerSync >= 0:
        result = min(result, maxEnqueuesPerSync)
    return result
