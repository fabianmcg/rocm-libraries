# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level GPU execution harness shared by all milestone tests.

Provides BufferPool, KernelRunner, and BenchmarkResult without importing
any GPU primitives at module level so that collection does not trigger HIP
initialization.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import List, Union

_log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Timing data returned by KernelRunner.run()."""

    timesNs: List[int]
    warmupN: int

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
    def from_hsaco(
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
            raise RuntimeError("amdgpu_exec is required for KernelRunner.from_hsaco") from exc

        if nModuleCopies == "auto":
            if coPath is not None:
                try:
                    from tensilelite_runtime import get_icache_module_copies
                    nModuleCopies = get_icache_module_copies(coPath)
                except ImportError:
                    _log.warning(
                        "tensilelite_runtime not available; falling back to nModuleCopies=1."
                    )
                    nModuleCopies = 1
            else:
                _log.warning(
                    "nModuleCopies='auto' with no coPath; falling back to nModuleCopies=1."
                )
                nModuleCopies = 1

        modules = [GpuModule(hsacoBytes) for _ in range(nModuleCopies)]
        functions = [m.get_function(kernelName) for m in modules]
        return cls(functions=functions, outputPool=None)

    def run(
        self,
        argsFn,
        grid: tuple,
        block: tuple,
        nWarmup: int,
        nIters: int,
    ) -> BenchmarkResult:
        """Launch the kernel and collect timing.

        argsFn:  callable(output_buf) -> list of kernel args. Called once per
                 iteration with the next output buffer from the pool (or None
                 if no outputPool was provided).
        grid:    (gridX, gridY, gridZ) tuple.
        block:   (blockX, blockY, blockZ) tuple.
        nWarmup: number of iterations before timing begins.
        nIters:  number of timed iterations.
        """
        try:
            from amdgpu_exec import GpuEvent
        except ImportError as exc:
            raise RuntimeError("amdgpu_exec is required for KernelRunner.run") from exc

        timesNs = []
        total = nWarmup + nIters
        for i in range(total):
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
            if i >= nWarmup:
                timesNs.append(stop.elapsed_ns(start))

        return BenchmarkResult(timesNs=timesNs, warmupN=nWarmup)


def auto_scale_iters(
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
