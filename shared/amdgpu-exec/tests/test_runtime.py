# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for amdgpu_exec.runtime."""

import ctypes

import numpy as np
import pytest

from amdgpu_exec import (
    GpuBuffer,
    GpuEvent,
    GpuModule,
    GpuStream,
    InOutArray,
    InputArray,
    MemoryManager,
    OutputArray,
    compile_hsaco,
    create_kernel_args,
    execute_hsaco,
    occupancy_max_active_blocks_per_multiprocessor,
)
from amdgpu_exec._runtime_module import hip_get_device_count, hip_init

# ---------------------------------------------------------------------------
# Shared test kernel: increments every i32 element by 1.
# ---------------------------------------------------------------------------

_ADD_IR = """\
; ModuleID = 'add_kernel'
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @add(ptr addrspace(1) %a) {
entry:
  %val = load i32, ptr addrspace(1) %a, align 4
  %inc = add i32 %val, 1
  store i32 %inc, ptr addrspace(1) %a, align 4
  ret void
}
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gpu():
    """Skip the test if no AMD GPU is available."""
    try:
        hip_init()
        count = hip_get_device_count()
    except RuntimeError:
        count = 0
    if count == 0:
        pytest.skip("no AMD GPU available")


@pytest.fixture(scope="session")
def hsaco(gpu):
    """Compile the test kernel once per session."""
    return compile_hsaco(_ADD_IR)


# ---------------------------------------------------------------------------
# No-GPU tests — always run
# ---------------------------------------------------------------------------


class TestArgumentWrappers:
    def test_input_array_stores_array(self):
        a = np.zeros(4, dtype=np.int32)
        assert InputArray(a).array is a

    def test_output_array_stores_array(self):
        a = np.zeros(4, dtype=np.int32)
        assert OutputArray(a).array is a

    def test_inout_array_stores_array(self):
        a = np.zeros(4, dtype=np.int32)
        assert InOutArray(a).array is a

    def test_types_are_distinct(self):
        a = np.zeros(4, dtype=np.int32)
        assert type(InputArray(a)) is not type(OutputArray(a))
        assert type(InputArray(a)) is not type(InOutArray(a))


class TestCreateKernelArgs:
    def test_empty_returns_none_triple(self):
        params, kargs, kptrs = create_kernel_args()
        assert params is None
        assert kargs is None
        assert kptrs is None

    def test_int_packed_as_c_int32(self):
        params, kargs, kptrs = create_kernel_args(42)
        assert params is not None
        assert bool(params)
        assert kargs._field0 == 42

    def test_float_packed_as_c_float(self):
        params, kargs, kptrs = create_kernel_args(3.14)
        assert params is not None
        assert abs(kargs._field0 - 3.14) < 1e-5

    def test_numpy_integer_scalar(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            val = dtype(7)
            params, kargs, _ = create_kernel_args(val)
            assert params is not None
            assert kargs._field0 == 7

    def test_numpy_int64_preserves_width(self):
        # np.int64 must not be truncated to c_int32.
        val = np.int64(2**33)
        params, kargs, _ = create_kernel_args(val)
        assert params is not None
        assert kargs._field0 == 2**33

    def test_numpy_floating_scalar(self):
        for dtype in (np.float32, np.float64):
            val = dtype(1.5)
            params, kargs, _ = create_kernel_args(val)
            assert params is not None
            assert abs(kargs._field0 - 1.5) < 1e-4

    def test_ctypes_scalar(self):
        params, kargs, _ = create_kernel_args(ctypes.c_int32(99))
        assert params is not None
        assert kargs._field0 == 99

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            create_kernel_args("bad_argument")

    def test_ptr_addresses_within_struct_bounds(self):
        params, kargs, kptrs = create_kernel_args(1, 2, 3)
        struct_start = ctypes.addressof(kargs)
        struct_end = struct_start + ctypes.sizeof(kargs)
        for i in range(3):
            addr = kptrs[i]
            assert struct_start <= addr < struct_end

    def test_multiple_mixed_types(self):
        params, kargs, kptrs = create_kernel_args(10, 2.5, ctypes.c_int64(100))
        assert params is not None
        assert kargs._field0 == 10
        assert abs(kargs._field1 - 2.5) < 1e-5
        assert kargs._field2 == 100


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


class TestGpuBuffer:
    def test_alloc_and_free(self, gpu):
        buf = GpuBuffer(256)
        assert buf.size_bytes == 256
        assert bool(buf.ptr)
        buf.free()

    def test_free_is_idempotent(self, gpu):
        buf = GpuBuffer(64)
        buf.free()
        buf.free()  # Should not raise.

    def test_free_sets_freed_flag(self, gpu):
        buf = GpuBuffer(64)
        buf.free()
        assert buf._freed

    def test_ptr_value_is_int(self, gpu):
        buf = GpuBuffer(64)
        assert isinstance(buf.ptr_value, int)
        assert buf.ptr_value != 0
        buf.free()

    def test_copy_roundtrip(self, gpu):
        data = np.array([1, 2, 3, 4], dtype=np.int32)
        buf = GpuBuffer(data.nbytes)
        buf.copy_from_host(data)
        result = np.zeros_like(data)
        buf.copy_to_host(result)
        np.testing.assert_array_equal(result, data)
        buf.free()

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float16, np.float64])
    def test_copy_roundtrip_dtypes(self, gpu, dtype):
        data = np.arange(8, dtype=dtype)
        buf = GpuBuffer(data.nbytes)
        buf.copy_from_host(data)
        result = np.zeros_like(data)
        buf.copy_to_host(result)
        np.testing.assert_array_equal(result, data)
        buf.free()

    def test_type_error_on_non_ndarray(self, gpu):
        buf = GpuBuffer(64)
        with pytest.raises(TypeError):
            buf.copy_from_host([1, 2, 3])
        with pytest.raises(TypeError):
            buf.copy_to_host([1, 2, 3])
        buf.free()

    def test_size_error_on_oversize_array(self, gpu):
        buf = GpuBuffer(4)
        big = np.zeros(100, dtype=np.int32)
        with pytest.raises(ValueError):
            buf.copy_from_host(big)
        with pytest.raises(ValueError):
            buf.copy_to_host(big)
        buf.free()

    def test_memset_zeros_buffer(self, gpu):
        data = np.full(4, 0xFF, dtype=np.uint8)
        buf = GpuBuffer(data.nbytes)
        buf.copy_from_host(data)
        buf.memset(0)
        result = np.empty_like(data)
        buf.copy_to_host(result)
        np.testing.assert_array_equal(result, np.zeros(4, dtype=np.uint8))
        buf.free()

    def test_memset_fills_with_value(self, gpu):
        buf = GpuBuffer(4)
        buf.memset(0xAB)
        result = np.empty(4, dtype=np.uint8)
        buf.copy_to_host(result)
        np.testing.assert_array_equal(result, np.full(4, 0xAB, dtype=np.uint8))
        buf.free()


class TestGpuStream:
    def test_create_sync_destroy(self, gpu):
        stream = GpuStream()
        stream.synchronize()
        stream.destroy()

    def test_destroy_is_idempotent(self, gpu):
        stream = GpuStream()
        stream.destroy()
        stream.destroy()  # Should not raise.

    def test_destroy_sets_destroyed_flag(self, gpu):
        stream = GpuStream()
        stream.destroy()
        assert stream._destroyed


class TestGpuEvent:
    def test_create_destroy(self, gpu):
        event = GpuEvent()
        event.destroy()

    def test_destroy_is_idempotent(self, gpu):
        event = GpuEvent()
        event.destroy()
        event.destroy()  # Should not raise.

    def test_destroy_sets_destroyed_flag(self, gpu):
        event = GpuEvent()
        event.destroy()
        assert event._destroyed

    def test_record_and_synchronize(self, gpu):
        event = GpuEvent()
        event.record()
        event.synchronize()
        event.destroy()

    def test_elapsed_ns_nonnegative(self, gpu):
        start = GpuEvent()
        stop = GpuEvent()
        start.record()
        stop.record()
        stop.synchronize()
        ns = stop.elapsed_ns(start)
        assert isinstance(ns, int)
        assert ns >= 0
        start.destroy()
        stop.destroy()


class TestCreateKernelArgsGpu:
    def test_gpu_buffer_passes_device_ptr(self, gpu):
        buf = GpuBuffer(64)
        params, kargs, kptrs = create_kernel_args(buf)
        assert params is not None
        # The stored value should equal the raw device pointer.
        assert kargs._field0 == buf.ptr_value
        buf.free()

    def test_mixed_gpu_buffer_and_scalar(self, gpu):
        buf = GpuBuffer(64)
        params, kargs, kptrs = create_kernel_args(buf, ctypes.c_int32(5))
        assert kargs._field0 == buf.ptr_value
        assert kargs._field1 == 5
        buf.free()


class TestMemoryManager:
    def test_register_returns_gpu_buffer(self, gpu):
        mm = MemoryManager()
        a = np.zeros(4, dtype=np.int32)
        buf = mm.register(a)
        assert isinstance(buf, GpuBuffer)
        mm.release_all()

    def test_register_is_idempotent(self, gpu):
        mm = MemoryManager()
        a = np.zeros(4, dtype=np.int32)
        buf1 = mm.register(a)
        buf2 = mm.register(a)
        assert buf1 is buf2
        mm.release_all()

    def test_type_error_on_non_ndarray(self, gpu):
        mm = MemoryManager()
        with pytest.raises(TypeError):
            mm.register([1, 2, 3])

    def test_sync_to_and_from_gpu_roundtrip(self, gpu):
        mm = MemoryManager()
        a = np.array([10, 20, 30], dtype=np.float32)
        mm.register(a)
        a[:] = 0
        mm.sync_to_gpu(a)
        result = np.zeros_like(a)
        mm.get_buffer(a).copy_to_host(result)
        # a was zeroed before sync_to_gpu, so device should have zeros.
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))
        mm.release_all()

    def test_sync_from_gpu_updates_host(self, gpu):
        mm = MemoryManager()
        a = np.array([1, 2, 3], dtype=np.int32)
        mm.register(a, upload=True)
        # Overwrite host to verify sync_from_gpu brings back device data.
        a[:] = 0
        mm.sync_from_gpu(a)
        np.testing.assert_array_equal(a, [1, 2, 3])
        mm.release_all()

    def test_release_removes_entry(self, gpu):
        mm = MemoryManager()
        a = np.zeros(4, dtype=np.int32)
        mm.register(a)
        mm.release(a)
        with pytest.raises(KeyError):
            mm.get_buffer(a)

    def test_release_unregistered_raises(self, gpu):
        mm = MemoryManager()
        a = np.zeros(4, dtype=np.int32)
        with pytest.raises(KeyError):
            mm.release(a)

    def test_release_all_clears_all(self, gpu):
        mm = MemoryManager()
        a = np.zeros(4, dtype=np.int32)
        b = np.zeros(4, dtype=np.int32)
        mm.register(a)
        mm.register(b)
        mm.release_all()
        with pytest.raises(KeyError):
            mm.get_buffer(a)
        with pytest.raises(KeyError):
            mm.get_buffer(b)

    def test_multiple_arrays_independent_buffers(self, gpu):
        mm = MemoryManager()
        a = np.zeros(4, dtype=np.int32)
        b = np.ones(4, dtype=np.int32)
        buf_a = mm.register(a)
        buf_b = mm.register(b)
        assert buf_a is not buf_b
        mm.release_all()


class TestExecuteHsaco:
    def test_runs_kernel_and_increments(self, gpu, hsaco):
        a = np.array([41], dtype=np.int32)
        times = execute_hsaco(
            hsaco=hsaco,
            kernel_name="add",
            arguments=[InOutArray(a)],
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
        assert a[0] == 42
        assert len(times) == 1

    def test_timing_returns_nonnegative_ints(self, gpu, hsaco):
        a = np.array([0], dtype=np.int32)
        times = execute_hsaco(
            hsaco=hsaco,
            kernel_name="add",
            arguments=[InOutArray(a)],
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
            num_iterations=3,
        )
        assert len(times) == 3
        for t in times:
            assert isinstance(t, int)
            assert t >= 0

    def test_input_array_not_written_back(self, gpu, hsaco):
        a = np.array([10], dtype=np.int32)
        original = a.copy()
        execute_hsaco(
            hsaco=hsaco,
            kernel_name="add",
            arguments=[InputArray(a)],
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
        # InputArray: no D->H copy, host value unchanged.
        np.testing.assert_array_equal(a, original)

    def test_output_array_written_back(self, gpu, hsaco):
        # Seed the device buffer with a known value via MemoryManager, then run
        # with OutputArray (no H->D copy) to confirm the D->H copy path works.
        out = np.full(1, fill_value=-1, dtype=np.int32)
        mm = MemoryManager()
        mm.register(out, upload=False)
        mm.get_buffer(out).copy_from_host(np.array([99], dtype=np.int32))
        execute_hsaco(
            hsaco=hsaco,
            kernel_name="add",
            arguments=[OutputArray(out)],
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
            memory_manager=mm,
        )
        mm.release_all()
        # Device had 99, kernel increments to 100, D->H copies it back.
        assert out[0] == 100

    def test_file_path_accepted(self, gpu, hsaco, tmp_path):
        path = tmp_path / "kernel.hsaco"
        path.write_bytes(hsaco)
        a = np.array([0], dtype=np.int32)
        execute_hsaco(
            hsaco=str(path),
            kernel_name="add",
            arguments=[InOutArray(a)],
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
        assert a[0] == 1

    def test_verify_fn_called_after_first_iteration(self, gpu, hsaco):
        a = np.array([0], dtype=np.int32)
        called = []

        def verify(arguments):
            called.append(a[0])

        execute_hsaco(
            hsaco=hsaco,
            kernel_name="add",
            arguments=[InOutArray(a)],
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
            num_iterations=3,
            verify_fn=verify,
        )
        assert len(called) == 1
        assert called[0] == 1

    def test_large_grid_dispatch(self, gpu, hsaco):
        # Verify the kernel launches correctly with non-trivial grid/block dims.
        a = np.array([0], dtype=np.int32)
        times = execute_hsaco(
            hsaco=hsaco,
            kernel_name="add",
            arguments=[InOutArray(a)],
            grid_dim=(64, 1, 1),
            block_dim=(64, 1, 1),
            num_iterations=1,
        )
        assert len(times) == 1
        assert times[0] >= 0


@pytest.fixture(scope="class")
def gpu_fn(gpu, hsaco):
    """Load the test module and return the 'add' GpuFunction once per class."""
    module = GpuModule(hsaco)
    yield module.get_function("add")


class TestOccupancy:
    def test_returns_positive_int(self, gpu_fn):
        blocks = occupancy_max_active_blocks_per_multiprocessor(gpu_fn, block_size=64)
        assert isinstance(blocks, int)
        assert blocks > 0

    def test_larger_block_size_does_not_increase_blocks(self, gpu_fn):
        # Larger blocks use more resources, so active blocks per CU cannot grow.
        blocks_small = occupancy_max_active_blocks_per_multiprocessor(
            gpu_fn, block_size=64
        )
        blocks_large = occupancy_max_active_blocks_per_multiprocessor(
            gpu_fn, block_size=256
        )
        assert blocks_small >= blocks_large

    def test_dyn_shared_mem_zero_matches_default(self, gpu_fn):
        blocks_default = occupancy_max_active_blocks_per_multiprocessor(
            gpu_fn, block_size=64
        )
        blocks_zero = occupancy_max_active_blocks_per_multiprocessor(
            gpu_fn, block_size=64, dyn_shared_mem_per_blk=0
        )
        assert blocks_default == blocks_zero

    def test_large_shared_mem_reduces_blocks(self, gpu, gpu_fn):
        from amdgpu_exec._runtime_module import hip_get_device_props

        lds_per_cu = hip_get_device_props(0)["lds_per_cu"]
        # Requesting nearly all LDS per block leaves room for at most one block.
        blocks = occupancy_max_active_blocks_per_multiprocessor(
            gpu_fn, block_size=64, dyn_shared_mem_per_blk=lds_per_cu - 1
        )
        assert isinstance(blocks, int)
        assert 0 <= blocks <= 1

    def test_negative_shared_mem_raises(self, gpu_fn):
        with pytest.raises((ValueError, RuntimeError)):
            occupancy_max_active_blocks_per_multiprocessor(
                gpu_fn, block_size=64, dyn_shared_mem_per_blk=-1
            )
