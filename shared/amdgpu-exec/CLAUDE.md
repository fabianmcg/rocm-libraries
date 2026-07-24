# amdgpu-exec

A Python package that compiles AMDGPU assembly kernels using LLVM MC and LLD,
then executes them on AMD GPUs via a dynamically loaded HIP runtime.
Written in C++20 with Python bindings provided by nanobind.

## Project Structure

- `python/lib/CompileAsm.cpp` — LLVM MC-based AMDGPU ASM compilation and LLD linking to produce HSA code objects
- `python/lib/RuntimeModule.cpp` — nanobind Python bindings for the HIP runtime (device management, memory, kernel launch, events, streams)
- `python/lib/hip.h` — HIP type declarations and dynamic-loader API table; loads `libamdhip64.so` at runtime via `dlopen`
- `python/amdgpu_exec/__init__.py` — Python package; re-exports all symbols from `_compile_asm` and `_runtime_module`

---

# Project Info

- Source root: `~/amdgpu-exec`
- Build directory: `~/amdgpu-exec/build`
- Virtual environment: `.venv` (at `~/amdgpu-exec/.venv`)

---

# Building

Install the ROCm SDK, build dependencies, and the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "rocm[devel]"
rocm-sdk init
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

Or build with CMake directly (rocm-sdk is auto-detected from PATH):

```bash
source .venv/bin/activate
cmake -S . -B build -GNinja \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

To override the ROCm LLVM cmake path explicitly:

```bash
cmake -S . -B build -GNinja -DROCM_LLVM_CMAKE_DIR=/custom/llvm/lib/cmake ...
```

When using rocm-sdk, set `LD_LIBRARY_PATH` before running tests so the dynamic
LLVM shared library is found:

```bash
export LD_LIBRARY_PATH=$(rocm-sdk path --root)/llvm/lib:$LD_LIBRARY_PATH
```

Before running `ninja` or `pytest`, activate the venv:

```bash
source .venv/bin/activate
```

Never call `cmake --install`; only build targets.

---

# Testing

```bash
source .venv/bin/activate
pytest tests/
```

Always verify tests pass before considering a change complete. Add both positive cases (expected success) and negative cases (expected failure) where meaningful. Avoid redundant or trivially useless tests.

---

# Coding Style

- Only use `auto` when the right-hand side is a cast, a constructor, or an iterator type; spell out the full type otherwise.
- Always pass cheap-to-copy types by value.
- Don't put braces around a single-line statement.
- Use `break`, `continue`, and inverted conditions to reduce nesting depth.
- Prefer early returns.
- If an `if` body ends with a `return`, omit the `else`:
  ```cpp
  // Prefer:
  if (cond) {
    return val;
  }
  // else body
  ```
- Prefer signed types over unsigned; cast unsigned values to signed when needed.
- Never use `\p` or `\c` in C++ comments.
- Use camelBack naming: only struct/class/enum names begin with a capital letter; all other identifiers start with lowercase.
- End comments with a full stop. Assertion and diagnostic messages start with a lowercase letter and do not end with a full stop.

---

# Code Quality

- Keep functions short and focused (~40 lines max); split when they grow larger.
- One responsibility per function or class; avoid mixing concerns.
- Code should be self-documenting through clear naming. Comments explain *why*, not *what*.
- Keep comments brief — one sentence is usually enough.
- Commit messages should be short and factual, describing what changed and why.
- Avoid redundant abstractions; don't design for hypothetical future use.

---

# General Guidelines

- If it is unclear which directory or worktree to use, ask the user.
