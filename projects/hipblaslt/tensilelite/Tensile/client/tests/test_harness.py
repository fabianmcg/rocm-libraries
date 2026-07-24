# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for Tensile/client/harness.py — no GPU required.

All tests use mock GpuModule/GpuBuffer/GpuEvent objects so that amdgpu_exec
does not need to be installed and HIP is never initialized.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from Tensile.client.harness import BenchmarkResult, BufferPool, KernelRunner


# ---------------------------------------------------------------------------
# BenchmarkResult tests.
# ---------------------------------------------------------------------------


def test_benchmark_result_mean_us():
    result = BenchmarkResult(times_ns=[1_000_000, 2_000_000, 3_000_000], warmup_n=0)
    assert abs(result.meanUs - 2000.0) < 1e-6


def test_benchmark_result_p50_us():
    result = BenchmarkResult(times_ns=[1_000_000, 2_000_000, 3_000_000], warmup_n=0)
    assert abs(result.p50Us - 2000.0) < 1e-6


def test_benchmark_result_p95_us():
    # With 4 elements, p95 index is max(0, int(4*0.95)-1) = max(0, 2) = 2
    result = BenchmarkResult(
        times_ns=[1_000_000, 2_000_000, 3_000_000, 4_000_000], warmup_n=1
    )
    assert result.p95Us == 3000.0


def test_benchmark_result_min_us():
    result = BenchmarkResult(times_ns=[5_000_000, 1_000_000, 3_000_000], warmup_n=0)
    assert abs(result.minUs - 1000.0) < 1e-6


def test_benchmark_result_gflops():
    # 1 GFLOP in 1 ms = 1 TFLOPS (note: gflops returns GFLOPs not TFLOPs)
    # 2*M*N*K ops, mean time in microseconds
    M, N, K = 1024, 1024, 1024
    ops = 2 * M * N * K  # 2 GFLOPs
    mean_us = 1_000_000.0  # 1 second
    times_ns = [int(mean_us * 1_000)]
    result = BenchmarkResult(times_ns=times_ns, warmup_n=0)
    expected_gflops = ops / (mean_us * 1e-6) / 1e9
    assert abs(result.gflops(M, N, K) - expected_gflops) < 1e-3


# ---------------------------------------------------------------------------
# BufferPool tests.
# ---------------------------------------------------------------------------


def test_buffer_pool_cycles():
    fake_cls = MagicMock(side_effect=lambda sz: MagicMock())
    pool = BufferPool(n_slots=3, size_bytes=1024, gpu_buffer_cls=fake_cls)
    assert fake_cls.call_count == 3

    bufs = [pool.next() for _ in range(6)]
    # First three calls should cycle: buf0, buf1, buf2, buf0, buf1, buf2
    assert bufs[0] is bufs[3]
    assert bufs[1] is bufs[4]
    assert bufs[2] is bufs[5]


def test_buffer_pool_free_all():
    mocks = [MagicMock() for _ in range(2)]
    idx = [0]

    def fake_cls(sz):
        m = mocks[idx[0]]
        idx[0] += 1
        return m

    pool = BufferPool(n_slots=2, size_bytes=512, gpu_buffer_cls=fake_cls)
    pool.freeAll()
    for m in mocks:
        m.free.assert_called_once()
    assert pool._slots == []


# ---------------------------------------------------------------------------
# KernelRunner tests.
# ---------------------------------------------------------------------------


def _makeRunner(n_fns=2, n_out=3):
    fns = [MagicMock() for _ in range(n_fns)]
    out_bufs = [MagicMock() for _ in range(n_out)]
    idx = [0]

    def fake_cls(sz):
        m = out_bufs[idx[0] % n_out]
        idx[0] += 1
        return m

    pool = BufferPool(n_slots=n_out, size_bytes=64, gpu_buffer_cls=fake_cls)
    runner = KernelRunner(functions=fns, output_pool=pool)
    return runner, fns, pool


def test_kernel_runner_requires_non_empty_functions():
    pool = BufferPool(n_slots=1, size_bytes=64, gpu_buffer_cls=MagicMock())
    with pytest.raises(ValueError, match="non-empty"):
        KernelRunner(functions=[], output_pool=pool)


def test_kernel_runner_run_with_mock_gpu():
    """run() should return correct timing with mocked GpuEvent."""
    runner, fns, pool = _makeRunner(n_fns=2, n_out=3)

    fake_event_instances = []

    class FakeEvent:
        def __init__(self):
            self._elapsed = 1_000_000  # 1 ms in ns
            fake_event_instances.append(self)

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_ns(self, other):
            return self._elapsed

    args_calls = []

    def args_fn(buf):
        args_calls.append(buf)
        return [buf]

    with patch.dict("sys.modules", {"amdgpu_exec": MagicMock(GpuEvent=FakeEvent)}):
        result = runner.run(
            args_fn=args_fn,
            grid=(1, 1, 1),
            block=(64, 1, 1),
            n_warmup=1,
            n_iters=3,
        )

    assert len(result.times_ns) == 3
    assert all(t == 1_000_000 for t in result.times_ns)
    assert len(args_calls) == 4  # 1 warmup + 3 iters
    assert result.warmup_n == 1


def test_kernel_runner_rotates_functions():
    """Each successive call should use the next function in rotation."""
    runner, fns, pool = _makeRunner(n_fns=2, n_out=4)

    class FakeEvent:
        def record(self): pass
        def synchronize(self): pass
        def elapsed_ns(self, other): return 500_000

    with patch.dict("sys.modules", {"amdgpu_exec": MagicMock(GpuEvent=FakeEvent)}):
        runner.run(
            args_fn=lambda buf: [buf],
            grid=(1, 1, 1),
            block=(64, 1, 1),
            n_warmup=0,
            n_iters=4,
        )

    # With 2 functions and 4 iterations: fn0, fn1, fn0, fn1
    assert fns[0].launch.call_count == 2
    assert fns[1].launch.call_count == 2


def test_kernel_runner_raises_without_amdgpu_exec():
    runner, fns, pool = _makeRunner()
    import sys
    original = sys.modules.get("amdgpu_exec")
    sys.modules["amdgpu_exec"] = None
    try:
        with pytest.raises((ImportError, RuntimeError)):
            runner.run(
                args_fn=lambda buf: [buf],
                grid=(1, 1, 1),
                block=(64, 1, 1),
                n_warmup=0,
                n_iters=1,
            )
    finally:
        if original is None:
            del sys.modules["amdgpu_exec"]
        else:
            sys.modules["amdgpu_exec"] = original
