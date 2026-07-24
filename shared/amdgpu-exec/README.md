# amdgpu-exec

A Python package for compiling AMDGPU assembly kernels and executing them on
AMD GPUs. Assembly is compiled to HSA code objects using LLVM MC and LLD.
The HIP runtime is loaded dynamically — no compile-time ROCm dependency.

## Overview

`amdgpu-exec` provides two capabilities via nanobind C++ extensions:

1. **Assembly compilation** (`_compile_asm`) — takes AMDGPU ISA source,
   assembles it with LLVM MC, and links it to an HSA code object with LLD.
2. **HIP runtime** (`_runtime_module`) — wraps device, module, kernel launch,
   memory, event, and stream APIs; loads `libamdhip64.so` at import time via
   `dlopen`.


## Requirements

- Clang compiler > 20+
- LLVM development libraries (LLVMAMDGPUAsmParser, lldELF, etc.)
- ROCm / HIP runtime (`libamdhip64.so`) — required at runtime only, not build time
- CMake 3.21+, Ninja
- Python 3.10+ with pip
- nanobind 2.9+

## Build

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
rocm-sdk init
pip install --no-build-isolation -e .
```

Or invoke CMake directly:

```bash
source .venv/bin/activate
cmake -S . -B build -GNinja \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_PREFIX_PATH=/opt/rocm/llvm/lib/cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja -C build
```

## Python API

### Compilation

```python
from amdgpu_exec import compile_asm, link_binary, llvmir_to_asm
from amdgpu_exec import compile_asm_to_hsaco, compile_asm_from_file, compile_hsaco

# Assemble ISA source → ELF object (bytes)
elf = compile_asm(asm_src=my_asm, chip="gfx942")

# Link ELF → HSA code object (bytes)
hsaco = link_binary(elf)

# Convenience helpers (assemble + link in one call)
hsaco = compile_asm_to_hsaco(asm_src, chip="gfx942")
hsaco = compile_asm_from_file(pathlib.Path("kernel.s"))

# Compile LLVM IR → HSA code object
hsaco = compile_hsaco(llvm_ir_str, chip="gfx942")
```

### High-level kernel execution

The simplest way to run a kernel: pass `InputArray`/`OutputArray`/`InOutArray`
wrappers and let `execute_hsaco` manage GPU memory, timing, and cleanup.

```python
import numpy as np
from amdgpu_exec import execute_hsaco, InputArray, OutputArray

A = np.ones(1024, dtype=np.float32)
B = np.zeros(1024, dtype=np.float32)

times_ns = execute_hsaco(
    hsaco=hsaco_bytes,
    kernel_name="my_kernel",
    arguments=[InputArray(A), OutputArray(B)],
    grid_dim=(16, 1, 1),
    block_dim=(64, 1, 1),
    num_iterations=10,
)
print(f"median: {sorted(times_ns)[5] / 1e6:.3f} ms")
```

### RAII GPU resource wrappers

For finer control, use the RAII classes directly:

```python
from amdgpu_exec import (
    GpuBuffer, GpuModule, GpuFunction, GpuStream, GpuEvent,
    MemoryManager, create_kernel_args, get_chip,
)

chip = get_chip()                         # e.g. "gfx942"

module = GpuModule(hsaco_bytes)           # accepts bytes or a file path
fn = module.get_function("my_kernel")

buf = GpuBuffer(A.nbytes)
buf.copy_from_host(A)

fn.launch(grid=(16, 1, 1), block=(64, 1, 1), args=[buf])

buf.copy_to_host(B)
buf.free()
module.unload()
```

`MemoryManager` pairs numpy arrays with `GpuBuffer` objects for bulk H↔D sync:

```python
mm = MemoryManager()
buf_a = mm.register(A)            # allocates and uploads
buf_b = mm.register(B, upload=False)
# … launch kernel …
mm.sync_from_gpu(B)
mm.release_all()
```

## Building wheels with cibuildwheel

[cibuildwheel](https://cibuildwheel.pypa.io) builds manylinux wheels for
CPython 3.10–3.13. The configuration lives in `pyproject.toml` under
`[tool.cibuildwheel]`.

Install cibuildwheel, then run it against the repository root:

```bash
pip install cibuildwheel
cibuildwheel --platform linux .
```

Wheels land in `wheelhouse/`. The build sequence for each interpreter is:

1. `dnf install -y libzstd-devel` — provides the zstd system library that
   ROCm's LLVM requires at configure time.
2. Install the ROCm SDK Python package and run `rocm-sdk init` to populate the
   toolchain.
3. Build the extension with scikit-build-core (CMake + Ninja).
4. `auditwheel repair` bundles any non-system shared libraries into the wheel.

The resulting wheels have no compile-time ROCm dependency; `libamdhip64.so` is
loaded at runtime via `dlopen`.

To target a specific Python version or override the manylinux image:

```bash
cibuildwheel --platform linux --only cp312-manylinux_x86_64 .
```

## Testing

```bash
source .venv/bin/activate
pytest tests/
```
