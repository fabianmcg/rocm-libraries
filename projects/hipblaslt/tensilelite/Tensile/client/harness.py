# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level GPU execution harness shared by all milestone tests.

Provides BufferPool, KernelRunner, and BenchmarkResult without importing
any GPU primitives at module level so that collection does not trigger HIP
initialization.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List


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

    def __init__(self, functions: list, outputPool: "BufferPool") -> None:
        """
        functions:  list of GpuFunction objects (one per module copy).
        outputPool: BufferPool for rotating output buffers.
        """
        if not functions:
            raise ValueError("functions must be non-empty")
        self._functions = functions
        self._outputPool = outputPool
        self._call_count = 0

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
                 iteration with the next output buffer from the pool.
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
            out_buf = self._outputPool.next()
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
