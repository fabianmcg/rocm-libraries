# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# File adapted from https://github.com/iree-org/aster
"""HIP execution utilities for amdgpu-exec.

Provides RAII wrappers for GPU resources and a high-level kernel-execution
entry point. Adapted from aster/execution/core.py; the key difference is that
this project's C++ bindings exchange pointers as Ptr objects (plain uintptr_t
wrappers) rather than PyCapsule values, so no capsule construction is needed
here.

Usage::

    from amdgpu_exec.runtime import execute_hsaco, InputArray, OutputArray

    times_ns = execute_hsaco(
        hsaco=hsaco_bytes,
        kernel_name="my_kernel",
        arguments=[InputArray(A), InputArray(B), OutputArray(C)],
        grid_dim=(304, 1, 1),
        block_dim=(256, 1, 1),
        num_iterations=5,
    )
"""

import ctypes
import dataclasses
from typing import Any, Callable, List, Optional, Tuple

from amdgpu_exec._runtime_module import (
    Ptr,
    hip_clear_last_error,
    hip_event_create,
    hip_event_destroy,
    hip_event_elapsed_time,
    hip_event_record,
    hip_event_synchronize,
    hip_free,
    hip_get_device_props,
    hip_init,
    hip_malloc,
    hip_memcpy_device_to_host_async,
    hip_memcpy_host_to_device_async,
    hip_memset_async,
    hip_module_get_function,
    hip_module_launch_kernel,
    hip_module_load_data,
    hip_module_unload,
    hip_occupancy_max_active_blocks_per_multiprocessor,
    hip_set_device,
    hip_stream_create,
    hip_stream_destroy,
    hip_stream_synchronize,
)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def get_chip(device_id: int = 0) -> str:
    """Return the GCN arch name of a device, e.g. 'gfx942'.

    Feature flags such as ':sramecc+:xnack-' are stripped.
    """
    arch = hip_get_device_props(device_id)["gcn_arch_name"]
    return arch.split(":")[0]


# ---------------------------------------------------------------------------
# RAII GPU resource wrappers
# ---------------------------------------------------------------------------


class GpuBuffer:
    """RAII wrapper for a GPU memory buffer.

    Allocates device memory on construction and frees it on destruction.
    """

    def __init__(self, size_bytes: int) -> None:
        self._freed = True  # Guard against partial construction.
        self._size_bytes = size_bytes
        self._ptr = hip_malloc(size_bytes)
        if not self._ptr:
            raise RuntimeError(f"Failed to allocate GPU memory of size {size_bytes}")
        self._freed = False

    def __del__(self) -> None:
        self.free()

    def free(self) -> None:
        """Release the GPU buffer.

        Safe to call multiple times.
        """
        if not self._freed:
            hip_free(self._ptr)
            self._freed = True

    @property
    def ptr(self) -> "Any":
        """Ptr wrapping the raw device pointer (for hip_memcpy_* calls)."""
        return self._ptr

    @property
    def ptr_value(self) -> int:
        """Raw integer value of the device pointer (for kernel args)."""
        return int(self._ptr)

    @property
    def size_bytes(self) -> int:
        """Allocated size in bytes."""
        return self._size_bytes

    def copy_from_host(self, array, stream: "Optional[GpuStream]" = None) -> None:
        """Copy data from a numpy array on the host into this GPU buffer."""
        import numpy as np

        if not isinstance(array, np.ndarray):
            raise TypeError(f"expected numpy array, got {type(array)}")
        if array.nbytes > self._size_bytes:
            raise ValueError(
                f"array ({array.nbytes} bytes) exceeds buffer ({self._size_bytes} bytes)"
            )
        hip_memcpy_host_to_device_async(
            self._ptr,
            Ptr(array.ctypes.data),
            array.nbytes,
            stream._handle if stream is not None else None,
        )

    def copy_to_host(self, array, stream: "Optional[GpuStream]" = None) -> None:
        """Copy data from this GPU buffer into a numpy array on the host."""
        import numpy as np

        if not isinstance(array, np.ndarray):
            raise TypeError(f"expected numpy array, got {type(array)}")
        if array.nbytes > self._size_bytes:
            raise ValueError(
                f"array ({array.nbytes} bytes) exceeds buffer ({self._size_bytes} bytes)"
            )
        hip_memcpy_device_to_host_async(
            Ptr(array.ctypes.data),
            self._ptr,
            array.nbytes,
            stream._handle if stream is not None else None,
        )

    def memset(self, value: int = 0, stream: "Optional[GpuStream]" = None) -> None:
        """Fill this buffer with a byte value (low 8 bits of value)."""
        hip_memset_async(
            self._ptr,
            value,
            self._size_bytes,
            stream._handle if stream is not None else None,
        )


class GpuStream:
    """RAII wrapper for a HIP stream.

    Creates a new stream on construction and destroys it on destruction.
    """

    def __init__(self) -> None:
        self._destroyed = True  # Guard against partial construction.
        self._handle = hip_stream_create()
        self._destroyed = False

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        """Destroy the stream.

        Safe to call multiple times.
        """
        if not self._destroyed:
            hip_stream_destroy(self._handle)
            self._destroyed = True

    def synchronize(self) -> None:
        """Block until all work in this stream is complete."""
        hip_stream_synchronize(self._handle)

    @property
    def handle(self) -> "Any":
        """Raw Ptr handle for use in lower-level HIP calls."""
        return self._handle


class GpuEvent:
    """RAII wrapper for a HIP event.

    Creates a new event on construction and destroys it on destruction.
    """

    def __init__(self) -> None:
        self._destroyed = True  # Guard against partial construction.
        self._handle = hip_event_create()
        self._destroyed = False

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        """Destroy the event.

        Safe to call multiple times.
        """
        if not self._destroyed:
            hip_event_destroy(self._handle)
            self._destroyed = True

    def record(self) -> None:
        """Record this event in the default (null) stream."""
        hip_event_record(self._handle, Ptr(0))

    def synchronize(self) -> None:
        """Block until this event has been recorded."""
        hip_event_synchronize(self._handle)

    def elapsed_ms(self, start: "GpuEvent") -> float:
        """Return elapsed time in milliseconds between start and this event."""
        return hip_event_elapsed_time(start._handle, self._handle)

    def elapsed_ns(self, start: "GpuEvent") -> int:
        """Return elapsed time in nanoseconds between start and this event."""
        return int(self.elapsed_ms(start) * 1_000_000)

    @property
    def handle(self) -> "Any":
        """Raw Ptr handle for use in lower-level HIP calls."""
        return self._handle


class GpuFunction:
    """Thin wrapper around a HIP kernel function handle.

    The lifetime is tied to the owning GpuModule; do not call
    GpuModule.unload while a GpuFunction from that module is in use.
    """

    def __init__(self, handle, module: "GpuModule") -> None:
        self._handle = handle
        # Keep a reference so the module is not unloaded while we exist.
        self._module = module

    def launch(
        self,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args: list,
    ) -> None:
        """Launch the kernel with the given grid/block dimensions.

        HIP copies the argument values into its own buffer during the
        launch call, so the local ctypes structures can be freed as soon
        as this method returns.
        """
        # kernel_args and kernel_ptr_arr must remain alive across the call;
        # HIP copies the argument buffer internally during launch.
        kernelParams, kernel_args, kernel_ptr_arr = _create_kernel_args(*args)
        hip_module_launch_kernel(
            self._handle,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            kernelParams,
        )

    @property
    def handle(self) -> "Any":
        """Raw Ptr handle for use in lower-level HIP calls."""
        return self._handle


def occupancy_max_active_blocks_per_multiprocessor(
    function: GpuFunction,
    block_size: int,
    dyn_shared_mem_per_blk: int = 0,
) -> int:
    """Return the maximum number of blocks per multiprocessor for a kernel.

    Args:
        function: The GpuFunction to query.
        block_size: Number of threads per block.
        dyn_shared_mem_per_blk: Dynamic shared memory per block in bytes.

    Returns:
        Maximum number of active blocks per multiprocessor.
    """
    return hip_occupancy_max_active_blocks_per_multiprocessor(
        function.handle, block_size, dyn_shared_mem_per_blk
    )


class GpuModule:
    """RAII wrapper for a loaded HIP module.

    Loads an HSACO binary on construction and unloads it on destruction.
    Accepts either a file path (str) or raw binary bytes.
    """

    def __init__(self, hsaco) -> None:
        self._unloaded = True  # Guard against partial construction.
        if isinstance(hsaco, str):
            with open(hsaco, "rb") as f:
                hsaco_binary = f.read()
        elif isinstance(hsaco, (bytes, bytearray)):
            hsaco_binary = bytes(hsaco)
        else:
            raise TypeError(f"expected str path or bytes, got {type(hsaco)}")
        self._module = hip_module_load_data(hsaco_binary)
        self._unloaded = False

    def __del__(self) -> None:
        self.unload()

    def unload(self) -> None:
        """Unload the module.

        Safe to call multiple times.
        """
        if not self._unloaded:
            hip_module_unload(self._module)
            self._unloaded = True

    def get_function(self, name: str) -> GpuFunction:
        """Return a GpuFunction for the named kernel entry point."""
        handle = hip_module_get_function(self._module, name.encode())
        return GpuFunction(handle, self)


# ---------------------------------------------------------------------------
# Kernel argument array wrappers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class InputArray:
    """Numpy array passed as a read-only kernel input.

    Data is copied from host to GPU before the first launch. No copy-
    back is performed after the kernel runs.
    """

    array: Any


@dataclasses.dataclass
class OutputArray:
    """Numpy array used as a kernel output.

    No host-to-device copy is performed before the launch. After the
    first iteration the GPU buffer is copied back to the host array.
    """

    array: Any


@dataclasses.dataclass
class InOutArray:
    """Numpy array that is both read and written by the kernel.

    Data is copied host-to-device before the first launch and device-to-
    host after the first iteration.
    """

    array: Any


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Pairs numpy arrays with GpuBuffer objects for easy H<->D sync.

    Use register to create the GPU-side buffer for an array. The manager
    tracks the pair by the array's identity (id), so the same numpy object
    must be passed to subsequent calls.

    Example::

        mm = MemoryManager()
        buf_a = mm.register(a)
        buf_b = mm.register(b, upload=False)
        mm.sync_from_gpu(b)
        mm.release_all()
    """

    def __init__(self) -> None:
        # Maps id(array) -> (GpuBuffer, array) pairs. The array reference
        # prevents the object from being GC'd and its id reused.
        self._buffers: dict[int, tuple[GpuBuffer, Any]] = {}

    def register(self, array, upload: bool = True) -> GpuBuffer:
        """Allocate a GPU buffer and optionally upload array data.

        If the array is already registered, the existing buffer is returned
        without re-uploading.

        Args:
            array: Numpy array to pair with the new GPU buffer.
            upload: If True (default), copy the host data to GPU immediately.
                    If False, only allocate the buffer without copying.
        """
        import numpy as np

        if not isinstance(array, np.ndarray):
            raise TypeError(f"expected numpy array, got {type(array)}")
        key = id(array)
        if key in self._buffers:
            buf, _ = self._buffers[key]
            return buf
        buf = GpuBuffer(array.nbytes)
        if upload:
            buf.copy_from_host(array)
        self._buffers[key] = (buf, array)
        return buf

    def sync_to_gpu(self, array) -> None:
        """Copy current host data to the paired GPU buffer."""
        buf, _ = self._get_entry(array)
        buf.copy_from_host(array)

    def sync_from_gpu(self, array) -> None:
        """Copy GPU buffer data back to the paired host array."""
        buf, _ = self._get_entry(array)
        buf.copy_to_host(array)

    def get_buffer(self, array) -> GpuBuffer:
        """Return the GpuBuffer paired with array."""
        buf, _ = self._get_entry(array)
        return buf

    def release(self, array) -> None:
        """Free the GPU buffer for array and remove the tracking entry."""
        key = id(array)
        if key not in self._buffers:
            raise KeyError("array is not registered with this MemoryManager")
        buf, _ = self._buffers.pop(key)
        buf.free()

    def release_all(self) -> None:
        """Free all tracked GPU buffers."""
        for buf, _ in self._buffers.values():
            buf.free()
        self._buffers.clear()

    def _get_entry(self, array):
        key = id(array)
        if key not in self._buffers:
            raise KeyError("array is not registered with this MemoryManager")
        return self._buffers[key]


# ---------------------------------------------------------------------------
# Kernel argument construction
# ---------------------------------------------------------------------------

# Maps numpy integer dtypes to the ctypes type of the same width.
_NP_INT_TO_CTYPE: dict[Any, Any] = {}
_NP_FLOAT_TO_CTYPE: dict[Any, Any] = {}


def _init_dtype_maps() -> None:
    import numpy as np

    _NP_INT_TO_CTYPE.update(
        {
            np.int8: ctypes.c_int8,
            np.int16: ctypes.c_int16,
            np.int32: ctypes.c_int32,
            np.int64: ctypes.c_int64,
            np.uint8: ctypes.c_uint8,
            np.uint16: ctypes.c_uint16,
            np.uint32: ctypes.c_uint32,
            np.uint64: ctypes.c_uint64,
        }
    )
    _NP_FLOAT_TO_CTYPE.update(
        {
            np.float32: ctypes.c_float,
            np.float64: ctypes.c_double,
        }
    )


_init_dtype_maps()


_CTYPES_SCALAR_TYPES = (
    ctypes.c_int8,
    ctypes.c_int16,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_double,
    ctypes.c_void_p,
)


def create_kernel_args(*args: Any):
    """Build a kernel arguments Ptr from a mix of types.

    Supported argument types:

    * GpuBuffer — passes the device pointer as c_void_p.
    * numpy.ndarray — passes the host data pointer as c_void_p.
    * int — packed as c_int32.
    * float — packed as c_float.
    * numpy.integer — packed as the matching ctypes integer type (width-preserving).
    * numpy.floating — packed as the matching ctypes float type (width-preserving).
    * ctypes scalar instances (c_int8, c_int16, c_int32, c_int64, c_float,
      c_double, c_void_p) — passed as-is.

    Returns:
        Tuple of (kernelParams, kernel_args, kernel_ptr_arr):

        * kernelParams: Ptr to pass to hip_module_launch_kernel.
        * kernel_args: ctypes structure (must remain alive until after launch).
        * kernel_ptr_arr: Array of pointers into the structure fields (must
          remain alive until after launch).
    """
    return _create_kernel_args(*args)


def _create_kernel_args(*args: Any):
    """Internal implementation of create_kernel_args."""
    import numpy as np

    c_args = []
    c_struct_fields = []
    for i, arg in enumerate(args):
        if isinstance(arg, GpuBuffer):
            c_args.append(ctypes.c_void_p(arg.ptr_value))
            c_struct_fields.append((f"_field{i}", ctypes.c_void_p))
        elif isinstance(arg, np.ndarray):
            c_args.append(ctypes.c_void_p(arg.ctypes.data))
            c_struct_fields.append((f"_field{i}", ctypes.c_void_p))
        elif isinstance(arg, np.integer):
            ctype = _NP_INT_TO_CTYPE.get(type(arg), ctypes.c_int32)
            c_args.append(ctype(int(arg)))
            c_struct_fields.append((f"_field{i}", ctype))
        elif isinstance(arg, np.floating):
            ctype = _NP_FLOAT_TO_CTYPE.get(type(arg), ctypes.c_float)
            c_args.append(ctype(float(arg)))
            c_struct_fields.append((f"_field{i}", ctype))
        elif isinstance(arg, int):
            c_args.append(ctypes.c_int32(arg))
            c_struct_fields.append((f"_field{i}", ctypes.c_int32))
        elif isinstance(arg, float):
            c_args.append(ctypes.c_float(arg))
            c_struct_fields.append((f"_field{i}", ctypes.c_float))
        elif isinstance(arg, _CTYPES_SCALAR_TYPES):
            c_args.append(arg)
            c_struct_fields.append((f"_field{i}", type(arg)))
        else:
            raise TypeError(f"unsupported argument type: {type(arg)}")

    if not c_args:
        return None, None, None

    class _Args(ctypes.Structure):
        _fields_ = c_struct_fields

    kernel_args = _Args()
    for i, arg in enumerate(c_args):
        setattr(kernel_args, f"_field{i}", arg)

    ptr_arr_t = ctypes.c_void_p * len(c_args)
    kernel_args_addr = ctypes.addressof(kernel_args)
    kernel_ptr_arr = ptr_arr_t(
        *[
            kernel_args_addr + getattr(_Args, f"_field{i}").offset
            for i in range(len(c_args))
        ]
    )

    kernelParams = Ptr(ctypes.addressof(kernel_ptr_arr))
    return kernelParams, kernel_args, kernel_ptr_arr


# ---------------------------------------------------------------------------
# Internal kernel launch helpers
# ---------------------------------------------------------------------------


class _TimedLaunch:
    """Context manager that brackets a GPU kernel launch with HIP event timing.

    Records start/stop events around the with body, synchronizes on
    exit, and exposes the wall time as elapsed_ns.
    """

    def __init__(self, start_event: GpuEvent, stop_event: GpuEvent, flush_llc=None):
        self._start = start_event
        self._stop = stop_event
        self._flush_llc = flush_llc
        self.elapsed_ns: int = 0

    def __enter__(self):
        if self._flush_llc is not None:
            self._flush_llc.flush_llc()
        self._start.record()
        return self

    def __exit__(self, *_):
        self._stop.record()
        self._stop.synchronize()
        self.elapsed_ns = self._stop.elapsed_ns(self._start)


# ---------------------------------------------------------------------------
# Main execution entry point
# ---------------------------------------------------------------------------


def execute_hsaco(
    hsaco,
    kernel_name: str,
    arguments: list,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    num_iterations: int = 1,
    device_id: Optional[int] = None,
    flush_llc: Optional[Any] = None,
    verify_fn: Optional[Callable] = None,
    memory_manager: Optional[MemoryManager] = None,
) -> List[int]:
    """Execute a pre-compiled HSACO kernel on GPU.

    Args:
        hsaco: Path to the .hsaco file (str) or raw binary bytes.
        kernel_name: Name of the kernel entry point.
        arguments: List of kernel arguments. Each item may be:
            * InputArray — numpy array copied H->D before launch only.
            * OutputArray — numpy array allocated on GPU; copied D->H after
              the first iteration.
            * InOutArray — numpy array copied H->D before launch and D->H
              after the first iteration.
            * A raw numpy.ndarray — treated as InOutArray.
            * A scalar (int, float, numpy scalar, or ctypes scalar).
        grid_dim: (gridX, gridY, gridZ).
        block_dim: (blockX, blockY, blockZ).
        num_iterations: Number of kernel launches (for timing).
        device_id: GPU device to use. None keeps the current device.
        flush_llc: Optional object with initialize()/flush_llc()/cleanup()
            methods called around each iteration for LLC flushing.
        verify_fn: Optional callable invoked as verify_fn(arguments) after
            the first iteration to validate results.
        memory_manager: Optional MemoryManager to use. If None, a temporary
            one is created and released on return.

    Returns:
        List of execution times in nanoseconds, one per iteration.
    """
    import numpy as np

    hip_init()
    if device_id is not None:
        hip_set_device(device_id)

    # Clear any sticky HIP error from a previous failed call in this process.
    hip_clear_last_error()

    # Normalise raw ndarray → InOutArray.
    arguments = [InOutArray(a) if isinstance(a, np.ndarray) else a for a in arguments]

    owns_mm = memory_manager is None
    mm = MemoryManager() if owns_mm else memory_manager

    # Register arrays with the memory manager.
    for arg in arguments:
        if isinstance(arg, (InputArray, InOutArray)):
            mm.register(arg.array, upload=True)
        elif isinstance(arg, OutputArray):
            mm.register(arg.array, upload=False)

    # Build the flat argument list passed to GpuFunction.launch.
    launch_args = [
        (
            mm.get_buffer(arg.array)
            if isinstance(arg, (InputArray, OutputArray, InOutArray))
            else arg
        )
        for arg in arguments
    ]

    module = GpuModule(hsaco)
    function = module.get_function(kernel_name)

    start_event = GpuEvent()
    stop_event = GpuEvent()
    times_ns = []

    llc_initialized = False
    if flush_llc is not None:
        flush_llc.initialize()
        llc_initialized = True

    try:
        for it in range(num_iterations):
            with _TimedLaunch(start_event, stop_event, flush_llc) as t:
                function.launch(grid_dim, block_dim, launch_args)
            times_ns.append(t.elapsed_ns)

            if it == 0:
                for arg in arguments:
                    if isinstance(arg, (OutputArray, InOutArray)):
                        mm.sync_from_gpu(arg.array)
                if verify_fn is not None:
                    verify_fn(arguments)
    finally:
        if llc_initialized:
            flush_llc.cleanup()
        if owns_mm:
            mm.release_all()

    return times_ns
