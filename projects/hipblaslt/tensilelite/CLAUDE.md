# AI Agent Guidance

This file provides guidance for AI coding agents when working with code in this repository.

## Overview

TensileLite is an auto-tuning framework for generating and selecting high-performance GPU kernels for tensor contractions (GEMM and related operations) on AMD GPUs. It is a component of hipBLASLt. The Python package (`Tensile/`) drives kernel generation and benchmarking; `rocisa/` provides a C++ (Nanobind-wrapped) assembly generation module; `include/` and `src/` form the C++ runtime library; and `client/` contains the benchmark executable.

## Working environment

```bash
# If you are outside the docker, and if you are asked to run using a docker. Ask the user for the container name.
docker exec <container> bash -ilc "command"

# If you are asked to run using a venv on Linux. Ask the user for the root of the venv
source <path-to-venv>/bin/activate && (the rest of the commands)
```

## Repository layout

```
tensilelite/
├── Tensile/                          # Python package — codegen, benchmarking, library logic
│   ├── bin/Tensile                   # Entry-point CLI
│   ├── Tensile.py                    # Top-level orchestrator: executeStepsInConfig()
│   ├── BenchmarkProblems.py          # Phase 1: kernel candidate generation + benchmarking
│   ├── LibraryLogic.py               # Phase 2: heuristic selection logic generation
│   ├── ClientWriter.py               # Phase 3: library + client wrapping
│   ├── KernelWriter.py               # Assembly codegen (main loop, scheduling, subtile)
│   ├── KernelWriterAssembly.py       # getSourceFileString() — drives .s → .o → .co
│   ├── KernelWriterActivationFunction.py  # Activation (ReLU, GELU, etc.) codegen
│   ├── SolutionStructs/
│   │   ├── Solution.py               # Parameter validation + assignDerivedParameters()
│   │   └── Problem.py                # ProblemType definition and validation
│   ├── Contractions.py               # Problem taxonomy (GEMM, batched, grouped, sparse, stream-k)
│   ├── LibraryIO.py                  # YAML/MsgPack serialization
│   ├── Common/                       # Global params (globalParameters), arch tables, utilities
│   ├── Components/                   # Modular kernel building blocks
│   │   └── Subtile/                  # Subtile kernel sub-package
│   ├── TensileCreateLibrary/         # Standalone library creation (no benchmarking)
│   │   └── Run.py                    # writeSolutionsAndKernels(), processKernelSource()
│   └── Tests/
│       ├── common/                   # YAML kernel integration tests (gemm, streamk, sparse, …)
│       └── unit/                     # Python unit tests (pytest)
├── rocisa/                           # C++ assembly IR module (Nanobind)
├── include/Tensile/                  # C++ runtime headers
├── src/                              # C++ runtime implementation
├── client/                           # C++ benchmark executable source
└── tests/                            # C++ host-library gtest
```

## Three-Phase Workflow

1. **BenchmarkProblems** (`Tensile/BenchmarkProblems.py`): Generates kernel candidates from a YAML problem spec, builds them with rocisa, benchmarks on hardware. Output: `1_BenchmarkProblems/`, `2_BenchmarkData/`.

2. **LibraryLogic** (`Tensile/LibraryLogic.py`): Analyzes benchmark data to pick the best kernel per problem size, generating heuristic selection logic as YAML/MsgPack. Output: `3_LibraryLogic/`.

3. **ClientWriter** (`Tensile/ClientWriter.py`): Wraps the selected kernels in a C++ library and generates the benchmark client. Output: `4_LibraryClient/`.

Entry point: `Tensile/bin/Tensile` → `Tensile/Tensile.py:Tensile()` → `executeStepsInConfig()`.

## Building

```bash
# One-time: install rocisa as editable package (after clone or rocisa C++ changes)
invoke rocisa

# Build C++ client to build_tmp/ (default location)
invoke build-client

# Detect local GPU architecture
invoke get-gpu-arch

# Override GPU target and ROCm path
invoke build-client --gpu-targets gfx942 --rocm-path /opt/rocm-6.4.0

# Export compile_commands.json (for IDE tooling / clangd)
invoke build-client --export-compile-commands
```

| What changed | Rebuild command |
|---|---|
| `rocisa/` C++ sources or `pyproject.toml`/`CMakeLists.txt` | `invoke rocisa` |
| `client/` C++ sources | `invoke build-client` |
| `Tensile/*.py` | No rebuild (editable install, imported directly) |

For custom CMake builds, cmake presets, linting, rebuilding assembly, CMake options, and supported targets — see `AGENTS_reference.md`. Read that file automatically whenever the task involves any of those topics.

## Testing

### Unit tests (fast — no client build needed)
```bash
tox -e unit -- Tensile/Tests/unit
tox -e unit -- Tensile/Tests/unit/test_KernelWriter.py
tox -e unit -- Tensile/Tests/unit/test_emitMfmaInstruction.py::test_mfma_f32_16x16x4
```

### Full test suite (builds client + runs all common tests)
```bash
tox -e py3 -- Tensile/Tests -m common
```

### Run by category marker
```bash
tox -e py3 -- Tensile/Tests -m gemm
tox -e py3 -- Tensile/Tests -m streamk
tox -e py3 -- Tensile/Tests -m gfx94x
```

### Run a single YAML integration test
```bash
# After invoke build-client:
Tensile/bin/Tensile Tensile/Tests/common/gemm/fp16_use_e.yaml tensile-out

# With a custom-built client:
Tensile/bin/Tensile Tensile/Tests/common/gemm/fp16_use_e.yaml tensile-out \
    --prebuilt-client=my-build/tensilelite-client/tensilelite-client
```

### Debug: single worker, custom client args
```bash
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx942 --clean" \
TENSILE_NUM_PYTEST_WORKERS=1 tox -e py3 -- Tensile/Tests -m common
```

### C++ gtest (host library)
```bash
cmake -DTENSILELITE_BUILD_TESTING=ON --preset tensilelite -S .. -B my-build
cmake --build my-build --parallel
ctest --test-dir my-build
```

### Lint / format
```bash
tox -e lint    # flake8 (pyflakes errors only; E/W codes ignored)
tox -e format  # black (line-length=100) on Common/, TensileCreateLibrary/, Utilities/
tox -e isort   # isort (black profile) on same directories
```

## Where to add tests

### Python unit tests
Location: `Tensile/Tests/unit/test_<module>.py`. Naming mirrors the module under test. Use standard pytest fixtures; mock GPU calls with `unittest.mock`. The `tox -e unit` env installs rocisa via `pip install`, so rocisa is importable without a prior `invoke build-client`.

### YAML integration tests
Location: `Tensile/Tests/common/<category>/<test>.yaml`. Categories: `gemm`, `groupedgemm`, `streamk`, `sparse`, `gradient`, `gsu`, `exception`, `flags`, `client`.

### C++ host tests
Location: `tests/`. Gated by `TENSILELITE_BUILD_TESTING=ON`.

## Key Python modules

| Module | Role |
|--------|------|
| `Tensile/KernelWriter.py` | Assembly codegen — `kernelBody()` / `kernelBodySubtile()` / `makeSchedule()` (~10 600 lines) |
| `Tensile/KernelWriterAssembly.py` | `getSourceFileString()` — drives `.s → .o → .co`; occupancy calculators |
| `Tensile/KernelWriterActivationFunction.py` | Activation (ReLU, GELU, etc.) codegen |
| `Tensile/SolutionStructs/Solution.py` | `assignDerivedParameters()` — fills all derived tiling fields, validates, sets `Valid` |
| `Tensile/SolutionStructs/Problem.py` | `ProblemType` definition and validation |
| `Tensile/BenchmarkProblems.py` | Phase 1 — fork permutation expansion, parallel codegen, client invocation |
| `Tensile/LibraryLogic.py` | Phase 2 — `LogicAnalyzer`: reads CSVs, selects winner per problem size, emits YAML/MsgPack |
| `Tensile/ClientWriter.py` | Phase 3 — `writeClientConfigIni()`, `runClient()`, subprocess invocation of `tensilelite-client` |
| `Tensile/Contractions.py` | Problem taxonomy mirrored by C++ runtime types |
| `Tensile/LibraryIO.py` | YAML/MsgPack serialization |
| `Tensile/Common/` | `globalParameters`, `defaultSolution`, `validParameters`, architecture tables |
| `Tensile/Components/` | Modular kernel building blocks — `Component.find(writer)` dispatches by ISA/dtype |
| `Tensile/TensileCreateLibrary/Run.py` | `writeSolutionsAndKernels()`, `processKernelSource()`, `generateLogicDataAndSolutions()` |

## rocisa

`rocisa/` is a C++ module compiled with `amdclang++` and bound into Python via Nanobind. `KernelWriter.py` imports instruction classes directly:

```python
from rocisa.instruction import (
    BufferLoadB128, DSStoreB64, MFMAInstruction, SWaitCnt, VAddU32, ...
)
from rocisa import rocIsa, countInstruction
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import Module, KernelBody, Label, TextBlock
```

`rocIsa.getInstance()` returns the process-wide singleton. The `rocIsaPass` optimization pass inserts `s_waitcnt`, validates register lifetimes, and applies architecture-specific fixups.

Normal install (once after cloning, or after C++ changes):
```bash
invoke rocisa    # editable pip install
```

Staleness check: `rocisa/rocisa/__init__.py` compares mtimes of all `.cpp/.hpp/.h/.def/.inc` against the `.so`. Import raises with a "rebuild" message if stale.

## C++ runtime library

`include/Tensile/` and `src/` implement the runtime selected at hipBLASLt call time.

| Header | Role |
|--------|------|
| `Tensile.hpp` | Top-level runtime API: `findBestSolution()`, `run()` |
| `ContractionProblem.hpp` | `ContractionProblemGemm` — problem descriptor (M, N, K, types, strides) |
| `ContractionSolution.hpp` | `ContractionSolution` — kernel descriptor (tiling, launch bounds) |
| `SolutionLibrary.hpp` | Abstract library interface: `findBestSolution(problem, hardware)` |
| `hip/HipSolutionAdapter.hpp` | Loads `.co` files, manages `hipModule_t`, launches kernels |

## Subtile

Subtile (`UseSubtileImpl`, `numSubTiles`) is a kernel variant that splits the output tile into multiple subtiles processed in sequence by the same wave, increasing LDS data reuse.

- `UseSubtileImpl=True` triggers `kernelBodySubtile()` (`KernelWriter.py:4873`).
- `numSubTiles` is auto-set to `2` (`Solution.py:2186`).
- `VectorWidthA/B` reduced until divisible by `numSubTiles * VectorWidth` (`Solution.py:2213–2247`).

`Components/Subtile/` sub-package:

| File | Role |
|------|------|
| `SubtileGeometry.py` | Tile geometry, MFMA layout selection, `emitMfmaCode()` |
| `Kernel.py` | Main subtile kernel logic: `preLoop()`, `mainLoop()` |
| `SubtileGREmit.py` | Global-read emit, TDM descriptor init |
| `SubtileLREmit.py` | Local-read emit, LDS buffer swap |
| `SubtileScaleEmit.py` | MX scale GR/LR emit |
| `InstructionEmitter.py` | rocisa instruction wrapping with subtile-specific waitcnt tracking |
| `InstructionScheduler.py` | Dependency-aware instruction scheduler for the subtile main loop |
| `LogicalScheduler.py` | Higher-level scheduling of GR/LW/LR/MFMA phases |

Activations (ReLU, GELU, etc.) are applied natively per-subtile inside `KernelWriterActivationFunction.py`. The `ActivationType` enum cases must be exhaustive — a missing case silently falls through. The bypass flag (`activation-no-guard`) disables the NaN guard.

## License headers

New source files MUST begin with the short SPDX license header.

C / C++ / HIP files:
```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
```

Python / shell / CMake / YAML files:
```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

The header goes at the very top of the file, immediately after a `#!` shebang line if one is present. Do NOT paste the legacy verbose MIT block into new files.

Existing files carrying the verbose MIT block MAY be migrated when already editing them, but only when it does not materially grow the PR.

## Pull requests

Always write PR descriptions using the rocm-libraries PR template. Fill in every section (use "N/A" where a section genuinely does not apply):

```markdown
## Motivation
<why this change is needed: the problem, bug, or feature being addressed>

## Technical Details
<what changed and how; key design decisions and trade-offs>

## Test Plan
<how the change was/should be validated: builds, unit/gtest, smoke, manual steps>

## Test Result
<outcome of the test plan: passing suites, benchmark numbers, before/after>

## Submission Checklist
- [ ] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.

## Risk level
<None/Low/Medium/High, with a short justification>

**Associated ticket**: <JIRA/issue id, or N/A>
```

Use the `users/<github-username>/<branch-name>` branch convention and base PRs on `develop`.

## Gotchas

- **Stale rocisa**: editing any `.cpp/.hpp` under `rocisa/` without running `invoke rocisa` causes `ImportError: rocisa C++ sources are newer than the built _rocisa.so`. The error lists the modified files.
- **`tox -e unit` vs `tox -e py3`**: `unit` is fast (no client build); `py3` builds the client inside its `commands` block. Override CMake/client args via `TENSILELITE_CLIENT_ARGS`; parallelism via `TENSILE_NUM_PYTEST_WORKERS` (default 4).
- **Direct `pytest` outside tox**: requires `invoke rocisa` first for the `rocisa` import to succeed.
- **Two test trees**: `Tensile/Tests/` (YAML kernel tests, run via `tox`/`pytest`) vs `tests/` (C++ host-library gtest, gated by `TENSILELITE_BUILD_TESTING=ON`).
- **Solution cache invalidation**: the 12-char SHA cache key covers `ConstantParams`, `ForkParams`, `ParamGroups`, `CustomKernels`, `InternalSupportParams`. It does NOT cover changes to `KernelWriter.py` or rocisa — delete `tensile-out/` manually after codegen changes.
- **`ARCH` auto-detection**: if `make co` picks the wrong arch (e.g. missing `xnack` suffix), manually pass `ARCH="gfx942:xnack-"`.
- **Subtile + activation bypass**: `ActivationType` enum cases must be exhaustive in all subtile epilogue paths. A missing case silently falls through. Always add a guard when adding new activation types.
- **`rocisa.egg-info/` and `rocisa/build/`**: normal artifacts from editable install and cmake build — do not commit.
- **`Tensile.sh` / `Tensile.bat`**: deprecated wrappers — use `Tensile/bin/Tensile` directly.
- **MX scale on gfx1250**: requires `TDMInst` unless `StreamK > 0`; enforced in `_deriveAndValidateMXScaleLayoutAndTransport()`.
- **`GlobalSplitUAlgorithm` + dot2**: `dot2` kernels do not support `MultipleBufferSingleKernel` — auto-demoted to `MultipleBuffer`.
