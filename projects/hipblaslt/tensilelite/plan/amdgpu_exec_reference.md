# `amdgpu_exec` Reference

**Version:** 0.1.0
**Install location:** `~/.tensile/lib/python<version>/site-packages/amdgpu_exec/` (Python 3.12 in the current environment; the exact version depends on the active interpreter)
**Build backend:** scikit-build-core 0.12.2 (CMake + Ninja, two nanobind C++ extensions)
**Requires:** Python ≥ 3.10, numpy ≥ 2.1.0, ml_dtypes ≥ 0.5.0
**License:** Apache 2.0 with LLVM Exceptions; adapted from the [aster](https://github.com/iree-org/aster) project
**No staleness check** — distributed as a binary wheel; there is no `_build_info.py` and no source-tree scan.

---

## Public API surface (`__init__.py`)

```python
# Compilation (compile.py + _compile_asm C++ extension)
from amdgpu_exec import (
    compile_asm_to_hsaco,   # str → bytes
    compile_asm_from_file,  # Path → bytes
    compile_hsaco,          # LLVM IR str → bytes
    compile_asm,            # low-level: asm str → ELF bytes (C++ extension)
    link_binary,            # low-level: ELF bytes → HSACO bytes (C++ extension)
    llvmir_to_asm,          # LLVM IR str → asm str (C++ extension)
    AbiVersion, OptLevel, CompileOptions,
)

# Runtime (runtime.py + _runtime_module C++ extension)
from amdgpu_exec import (
    GpuModule, GpuFunction, GpuBuffer, GpuStream, GpuEvent,
    InputArray, OutputArray, InOutArray,
    MemoryManager,
    create_kernel_args,
    execute_hsaco,
    get_chip,
    occupancy_max_active_blocks_per_multiprocessor,
)
```

---

## Compilation functions

### `get_chip(device_id=0) -> str`

`runtime.py:64` — calls `hipGetDevicePropertiesR0600`, extracts `gcn_arch_name`, strips everything after the first `:`.

```python
chip = get_chip()        # e.g. "gfx950"
chip = get_chip(1)       # device 1
```

### `compile_asm_to_hsaco(asm: str, chip: str = None) -> bytes`

`compile.py:22` — if `chip` is `None`, calls `get_chip()`. Then calls the C++ `compile_asm` (LLVM MC assembler) followed by `link_binary` (LLD linker). Returns complete HSACO bytes ready to pass to `GpuModule` or `execute_hsaco`. No files written.

```python
hsaco = compile_asm_to_hsaco(asm_string, chip="gfx950")
```

### `compile_asm_from_file(path, chip=None) -> bytes`

Reads the `.s` file and delegates to `compile_asm_to_hsaco`.

---

## Runtime classes

### `GpuModule` — `runtime.py:305`

Loads an HSACO binary into the HIP runtime (`hipModuleLoadData`).

```python
module = GpuModule(hsaco)          # bytes or path string
fn     = module.get_function(name) # → GpuFunction
module.unload()                    # explicit release; also called by __del__
```

- `GpuModule` does **not** cache or deduplicate. Every call to `GpuModule(hsaco)` loads a new `hipModule_t`.
- For I-cache rotation: create N independent `GpuModule` instances from the same HSACO bytes and cycle them across launches.
- The M10 `tensilelite_runtime` module must be placed **outside** any directory that rocisa scans; amdgpu_exec itself has no such scan.

### `GpuFunction` — `runtime.py:241`

Obtained from `GpuModule.get_function(name)`. Do not construct directly.

```python
fn.launch(
    grid  = (gridX, gridY, gridZ),
    block = (blockX, blockY, blockZ),
    args  = [...]   # see argument types below
)
```

**Kernel argument types** accepted in `args` (`runtime.py:543–569`):

| Python type | Passed as |
|---|---|
| `GpuBuffer` | `c_void_p(buf.ptr_value)` — device pointer |
| `np.ndarray` | `c_void_p(array.ctypes.data)` — **host** pointer; use `GpuBuffer` for device |
| `np.integer` | width-preserving ctypes integer (uint8 … int64) |
| `np.floating` | `c_float` or `c_double` |
| `int` | `c_int32` |
| `float` | `c_float` |
| ctypes scalar (`c_int32`, `c_void_p`, …) | passed as-is |

### `GpuBuffer` — `runtime.py:78`

Device memory allocation (`hipMalloc`).

```python
buf = GpuBuffer(size_bytes)
buf.ptr_value          # int — raw device pointer address
buf.ptr                # Ptr object (uintptr_t wrapper)
buf.size_bytes         # int
buf.copy_from_host(array, stream=None)   # H→D async
buf.copy_to_host(array, stream=None)     # D→H async
buf.memset(value=0, stream=None)
buf.free()             # explicit release; also called by __del__
```

To use as a kernel argument: `ctypes.c_void_p(buf.ptr_value)`, or pass `buf` directly in `GpuFunction.launch` args list.

**Rotating buffers:** pre-allocate N `GpuBuffer` instances of the same size and cycle them round-robin across kernel launches to prevent write-combining effects.

### `GpuEvent` — `runtime.py:196`

HIP event wrapper for timing.

```python
start = GpuEvent(); stop = GpuEvent()
start.record()
fn.launch(...)
stop.record(); stop.synchronize()
elapsed_ms = stop.elapsed_ms(start)   # float
elapsed_ns = stop.elapsed_ns(start)   # int
```

### `GpuStream` — `runtime.py:163`

HIP stream wrapper.

```python
stream = GpuStream()
stream.synchronize()
stream.destroy()
```

### `InputArray`, `OutputArray`, `InOutArray` — `runtime.py:347–378`

Thin `@dataclass` wrappers around a numpy array used by `execute_hsaco`:

| Type | H→D before launch | D→H after iter 0 |
|---|---|---|
| `InputArray(array)` | ✓ | ✗ |
| `OutputArray(array)` | ✗ | ✓ |
| `InOutArray(array)` | ✓ | ✓ |
| bare `np.ndarray` | treated as `InOutArray` | ✓ |

### `MemoryManager` — `runtime.py:385`

Tracks `{id(array) → (GpuBuffer, array)}` pairs. Used internally by `execute_hsaco`; also usable directly.

```python
mm = MemoryManager()
buf = mm.register(array, upload=True)  # allocate + optional H→D
mm.sync_to_gpu(array)
mm.sync_from_gpu(array)
mm.get_buffer(array) -> GpuBuffer
mm.release(array)
mm.release_all()
```

---

## `execute_hsaco` — `runtime.py:629`

High-level convenience function: loads module, allocates buffers, runs N iterations, returns timings.

```python
times_ns: list[int] = execute_hsaco(
    hsaco,                          # bytes/bytearray or path str
    kernel_name: str,
    arguments: list,                # InputArray / OutputArray / InOutArray / scalars
    grid_dim   = (1, 1, 1),
    block_dim  = (64, 1, 1),
    num_iterations = 1,
    device_id  = None,              # calls hipSetDevice if set
    flush_llc  = None,              # object with initialize/flush_llc/cleanup
    verify_fn  = None,              # called after iter 0 with D→H results
    memory_manager = None,          # use existing MemoryManager; caller owns release
)
```

**Internal lifecycle:**
1. `hip_init()` — idempotent, `std::call_once` in C++ layer
2. Optional `hip_set_device(device_id)`
3. `hip_clear_last_error()`
4. Normalize bare `np.ndarray` → `InOutArray`
5. Create (or use provided) `MemoryManager`; register all arrays
6. Build `launch_args`: array types → `GpuBuffer`; scalars pass through
7. `GpuModule(hsaco)` + `module.get_function(kernel_name)` — **fresh load every call**
8. Loop N times: record start event → optional LLC flush → `function.launch(...)` → record stop → `elapsed_ns` appended
9. After iteration 0: D→H copy for `OutputArray`/`InOutArray`; call `verify_fn(arguments)`
10. `finally`: `flush_llc.cleanup()`; `mm.release_all()` if temporary

**Returns:** `list[int]` — nanoseconds per iteration.

**Key limitation:** `execute_hsaco` loads and unloads the module on every call. For benchmarking loops, use `GpuModule` / `GpuFunction` / `GpuBuffer` directly and manage lifetimes yourself.

---

## Low-level execution pattern (for benchmarking)

Bypass `execute_hsaco` when you need:
- Module reuse across iterations (no per-call `hipModuleLoadData` overhead)
- Rotating output buffers
- I-cache rotation (N independent `GpuModule` copies)

```python
import ctypes
from amdgpu_exec import GpuModule, GpuBuffer, GpuEvent, get_chip
from amdgpu_exec import compile_asm_to_hsaco

hsaco = compile_asm_to_hsaco(asm_str, chip=get_chip())

# I-cache rotation: N independent modules
N_MODULES = 4
modules = [GpuModule(hsaco) for _ in range(N_MODULES)]
fns = [m.get_function("my_kernel") for m in modules]

# Rotating output buffers
N_BUFS = 8
out_bufs = [GpuBuffer(output_size_bytes) for _ in range(N_BUFS)]

# Input buffer (shared across iterations)
a_buf = GpuBuffer(a.nbytes); a_buf.copy_from_host(a)
b_buf = GpuBuffer(b.nbytes); b_buf.copy_from_host(b)

times_ns = []
for i in range(warmup + iters):
    fn  = fns[i % N_MODULES]
    out = out_bufs[i % N_BUFS]
    args = [ctypes.c_void_p(a_buf.ptr_value),
            ctypes.c_void_p(b_buf.ptr_value),
            ctypes.c_void_p(out.ptr_value),
            ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K)]
    start = GpuEvent(); stop = GpuEvent()
    start.record()
    fn.launch(grid_dim, block_dim, args)
    stop.record(); stop.synchronize()
    if i >= warmup:
        times_ns.append(stop.elapsed_ns(start))

# Read back last output
result = np.empty(output_shape, dtype=np.float32)
out_bufs[(warmup + iters - 1) % N_BUFS].copy_to_host(result)
```

---

## HIP initialization timing

**`import amdgpu_exec` does not trigger any HIP API call.** The C++ extension loads `libamdhip64.so` lazily on first use. `hipInit(0)` and `hipSetDevice(0)` are called inside `hip_init()` which runs at the start of `execute_hsaco`, or on the first direct use of `GpuBuffer`/`GpuModule`. The `hip_init()` call is protected by `std::call_once` and is idempotent.

**Implication for `tensilelite_profiler`:** `import tensilelite_profiler` must happen before any `GpuBuffer`, `GpuModule`, or `execute_hsaco` call — but `import amdgpu_exec` alone is safe to do first.

---

## Build system (read-only — do not modify)

`amdgpu_exec` is installed as a **read-only binary wheel** (`~/.tensile/lib/python<version>/site-packages/amdgpu_exec/`). There is no buildable source tree available. The wheel was built with scikit-build-core 0.12.2 using MLIR CMake macros from the [aster](https://github.com/iree-org/aster) project; rebuilding requires a full LLVM/MLIR build environment that is not present.

**Do not attempt to modify `amdgpu_exec`.** New GPU primitives needed by the harness (e.g. `BoundedBuffer` for M8) must be implemented as standalone nanobind modules (e.g. `tensilelite_bounds`, `tensilelite_runtime`) following the rocisa CMake pattern, not as extensions to `amdgpu_exec`.

Internally, `amdgpu_exec` uses two nanobind C++ extensions:
- `_compile_asm` — LLVM MC + LLD for assembly compilation/linking
- `_runtime_module` — `dlopen`/`dlsym` to load `libamdhip64.so` at runtime (no compile-time HIP SDK dependency)

---

## `Ptr` type

All HIP handles (modules, functions, events, streams, device pointers) are represented as `Ptr` — a plain `uintptr_t` wrapper with `__init__(int)`, `__int__() -> int`, `__bool__()`. `GpuBuffer.ptr_value` returns `int(self._ptr)`. This differs from the upstream aster project which uses PyCapsule for handles; the `runtime.py` docstring at lines 10–12 documents this explicitly.

## `occupancy_max_active_blocks_per_multiprocessor`

```python
n: int = occupancy_max_active_blocks_per_multiprocessor(
    function: GpuFunction,
    block_size: int,
    dyn_shared_mem_per_blk: int = 0,
)
```

Wraps `hipOccupancyMaxActiveBlocksPerMultiprocessor`. Returns the maximum number of blocks that can simultaneously reside on one compute unit.
