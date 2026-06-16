# TensileLite Primer

## Summary

TensileLite is the kernel generation and auto-tuning subsystem of hipBLASLt. It takes a YAML "solution" configuration describing a GEMM variant, generates AMD GPU assembly via the `rocisa` C++ module, assembles and links it into a code object (`.co`), benchmarks candidates on hardware, then produces heuristic selection logic (`3_LibraryLogic/`) consumed by the C++ runtime at hipBLASLt call time. The Python package (`Tensile/`) drives generation and benchmarking; `rocisa/` provides instruction-level assembly IR via Nanobind; `include/`+`src/` are the C++ runtime; and `client/` is the benchmark executable.

---

## Repository Layout

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
│   ├── KernelWriterBase.py           # ABC for all kernel writers
│   ├── KernelWriterModules.py        # Shared kernel fragments (preamble, epilogue, etc.)
│   ├── KernelWriterActivationFunction.py  # Activation (ReLU, GELU, etc.) codegen
│   ├── KernelWriterBetaOnly.py       # Beta-only (scale C) helper kernel writer
│   ├── KernelWriterConversion.py     # Type-conversion helper kernel writer
│   ├── KernelWriterReduction.py      # Reduction helper kernel writer
│   ├── SolutionStructs/
│   │   ├── Solution.py               # Parameter validation + assignDerivedParameters()
│   │   └── Problem.py                # ProblemType definition and validation
│   ├── Contractions.py               # Problem taxonomy (GEMM, batched, grouped, sparse, stream-k)
│   ├── LibraryIO.py                  # YAML/MsgPack serialization (parseSolutionsFile, etc.)
│   ├── Common/                       # Global params (globalParameters), arch tables, utilities
│   ├── Components/                   # Modular kernel building blocks
│   │   ├── MAC_*.py                  # MFMA/WMMA/VALU MAC variants per data type
│   │   ├── LocalRead.py              # ds_read emission
│   │   ├── LraTileAssignment.py      # Local-read address tile mapping
│   │   ├── GlobalWriteBatch.py       # Global-write batching
│   │   ├── ComputeStoreVgprs.py      # Store VGPR layout (VALU / MFMA / swap)
│   │   ├── StreamK.py                # Stream-K persistent loop
│   │   ├── PersistentLoop.py         # Persistent loop open/close
│   │   ├── GSU.py                    # Global split-U
│   │   ├── SIA.py                    # Scheduling inter-iteration algorithm
│   │   ├── PackData.py               # Data packing (TF32, FP16 → FP32, etc.)
│   │   ├── CustomSchedule.py         # User-provided schedule overrides
│   │   ├── Signature.py              # Kernel argument signature generation
│   │   └── Subtile/                  # Subtile kernel sub-package (see §Subtile Project)
│   ├── TensileCreateLibrary/         # Standalone library creation (no benchmarking)
│   │   ├── Run.py                    # writeSolutionsAndKernels(), processKernelSource()
│   │   └── ParseArguments.py         # CLI argument parsing for TensileCreateLibrary
│   ├── TensileLogic/                 # Logic file parsing and merging utilities
│   └── Tests/
│       ├── common/                   # YAML kernel integration tests
│       │   ├── gemm/                 # Standard GEMM configs
│       │   ├── groupedgemm/          # Grouped GEMM configs
│       │   ├── streamk/              # Stream-K configs
│       │   ├── sparse/               # Sparse GEMM configs
│       │   ├── gradient/             # Gradient configs
│       │   ├── gsu/                  # Global split-U configs
│       │   ├── exception/            # Expected-error configs
│       │   └── flags/                # Flag/option coverage configs
│       └── unit/                     # Python unit tests (pytest)
│           ├── test_KernelWriter*.py
│           ├── test_Configuration.py
│           ├── test_CustomSchedule*.py
│           └── ... (one file per module)
├── rocisa/                           # C++ assembly IR module (Nanobind)
├── include/Tensile/                  # C++ runtime headers
├── src/                              # C++ runtime implementation
├── client/                           # C++ benchmark executable source
└── tests/                            # C++ host-library gtest
```

---

## Three-Phase Workflow

```
YAML Config
    │
    ▼
[Phase 1] BenchmarkProblems.py
  BenchmarkProcess → fork permutations
  _generateForkedSolutions() → ParallelMap2(_generate_single_solution)
  → Solution.assignDerivedParameters()
  → writeBenchmarkFiles() → writeSolutionsAndKernels()
  → processKernelSource() → KernelWriterAssembly.getSourceFileString()
  → _getKernelSource() → kernelBody() | kernelBodySubtile()
  → rocisa emits IR → .s → amdclang++/llvm-mc → .o → lld → .co
  → runClient() → tensilelite-client benchmark run
  Output: 1_BenchmarkProblems/, 2_BenchmarkData/
    │
    ▼
[Phase 2] LibraryLogic.py
  generateLogic() → analyzeProblemType() → LogicAnalyzer.__init__()
  → reads .csv + .yaml benchmark data
  → picks best kernel per (M, N, K) using GFLOPS/s ranking
  → emits YAML/MsgPack heuristic tables
  Output: 3_LibraryLogic/
    │
    ▼
[Phase 3] ClientWriter.py
  writeClientConfig() + writeRunScript()
  → wraps selected kernels in a static .a
  → generates ClientParameters.ini for tensilelite-client
  Output: 4_LibraryClient/
```

---

## Config → ASM → Object: Detailed Call Chain

Starting from a YAML solution config passed to `Tensile/bin/Tensile`:

### 1. Entry: `Tensile.py:executeStepsInConfig()`
Reads `BenchmarkProblems`, `LibraryLogic`, `ClientParameters` sections from YAML and dispatches each phase.

### 2. `BenchmarkProblems.main()` → `_benchmarkProblemType()`
(`BenchmarkProblems.py:513`)

- Constructs a `BenchmarkProcess` from the config (holds fork parameter space and problem sizes).
- For each `benchmarkStep` (each fork permutation group):
  - Calls `_generateForkedSolutions()` → calls `ParallelMap2(_generate_single_solution, ...)` using `joblib.Parallel`.

### 3. `_generate_single_solution(perm, problemType, ...)` (`BenchmarkProblems.py:153`)
- Merges constant params + fork perm into a raw dict.
- Resolves `WavefrontSize` from ISA caps (`HasWave32`).
- Converts `MatrixInstruction` list → MI tiling parameters via `matrixInstructionToMIParameters()`.
- Instantiates `Solution(config, splitGSU, ...)` — `Solution.__init__` calls `assignDerivedParameters()`.
- Returns the `Solution` object if `solution["Valid"]`, else `None`.

### 4. `Solution.assignDerivedParameters(state, ...)` (`Solution.py:1456`)
Large static method that fills every derived tiling field:
- `WavefrontSize`, `MaxLDS` from ISA caps.
- `numSubTiles`, `VectorWidthA/B` (with subtile divisibility enforcement).
- `_GlobalAccumulation` (PartialsBuffer for StreamK, SingleBuffer/MultipleBuffer for GSU).
- MX scale layout and TDM transport selection via `_deriveAndValidateMXScaleLayoutAndTransport()`.
- Occupancy hints: `CUOccupancy`, `MathClocksUnrolledLoop`.
- Validates all MI parameters and sets `state["Valid"] = False` with `reject()` on any incompatibility.

### 5. `writeBenchmarkFiles()` → `writeSolutionsAndKernels()` (`BenchmarkProblems.py:370`, `TensileCreateLibrary/Run.py:418`)
- Deduplicates kernels by `getKeyNoInternalArgs()`.
- Instantiates `KernelWriterAssembly(assembler, debugConfig)`.
- Calls `ParallelMap2(processKernelSource, asmIter, ...)` — fans out kernel codegen across CPU threads.

### 6. `processKernelSource(kernelWriterAssembly, data, outOptions, splitGSU, kernel)` (`Run.py:216`)
- Calls `kernelWriter.setRocIsa(data, outOptions)` to configure the rocisa singleton.
- Calls `kernelWriter.getSourceFileString(kernel)` → returns `(errcode, asm_string)`.
- Returns `KernelCodeGenResult(err, src, header, asmFilename, objFilename, isa, wavefrontSize, cuocc, pgr, mathClocks)`.

### 7. `KernelWriterAssembly.getSourceFileString(kernel)` (`KernelWriterAssembly.py:176`)
- For custom kernels: reads pre-written `.s` source; computes occupancy from ELF via `compute_occupancy_from_asm_source()`.
- For generated kernels: calls `_getKernelSource(kernel)`.

### 8. `KernelWriter._getKernelSource(kernel)` (`KernelWriter.py:10404`)
- Calls `_initKernel(kernel, tPA, tPB)` to set up register pools, state structs.
- Dispatches to:
  - `kernelBody(kernel, tPA, tPB)` — standard MFMA tiled loop (`KernelWriter.py:5256`)
  - `kernelBodySubtile(kernel, tPA, tPB)` — subtile variant (`KernelWriter.py:4873`), when `kernel["UseSubtileImpl"]`

### 9. `kernelBody()` structure (`KernelWriter.py:5256`)
```
functionSignature()         # kernel args, .amdgpu_kernel metadata
defineAndResources()        # SGPR/VGPR allocation, SRD setup
Component.StreamK.preLoop() # stream-k init (if StreamK > 0)
Component.PersistentLoop.openPersistentLoop()
setupNewTile()              # global-read pointer setup, prefetch GR
[if PrefetchGlobalRead]:
  openShadowInit() → initC() → closeShadowInit()
  preLoopLocalWriteDo()     # write prefetched data to LDS
makeSchedule()              # instruction scheduling for main loop
[main summation loop]:
  globalReadDo()            # buffer_load_* for A and B
  localWriteDo()            # ds_store_* to LDS
  localReadDo()             # ds_load_* from LDS
  MAC instructions          # mfma_*/wmma_* via Components/MAC_*.py
  globalReadInc()           # pointer increment
[tail loop / NGLL]
globalWrite()               # store C/D, apply alpha/beta, activation, bias
functionEnd()
```

### 10. rocisa instruction emission
`KernelWriter.py` emits each instruction by importing from `rocisa.instruction` (e.g. `BufferLoadB128`, `DSStoreB64`, `MFMAInstruction`, `SWaitCnt`). The `rocisa` module (C++, Nanobind) handles:
- Instruction construction and validation.
- Assembly pass (`rocIsaPass`) for register scheduling and `waitcnt` insertion.
- Final text serialization to the `.s` string.

### 11. Assembly → `.o` → `.co`
Back in `writeSolutionsAndKernels()` (`Run.py:495`):
```python
writeAssembly(assemblyTmpPath, result)          # writes .s file
assemble(result):
    asmToolchain.assembler(gfx, wave, s_path, o_path)  # amdclang++ / llvm-mc
buildAssemblyCodeObjectFiles(linker, bundler, ...)     # lld → .co / .hsaco
```
The `.co` is placed in the device library directory (`<destRoot>/<arch>/`) for lazy loading by the C++ runtime at hipBLASLt call time.

---

## Build and Rebuild

```bash
cd rocm-libraries/projects/hipblaslt/tensilelite

# One-time: install rocisa as editable package (after clone or rocisa C++ changes)
invoke rocisa

# Build C++ client to build_tmp/ (default location)
invoke build-client

# Override GPU target and ROCm path
invoke build-client --gpu-targets gfx942 --rocm-path /opt/rocm-6.4.0

# Export compile_commands.json (for IDE tooling / clangd)
invoke build-client --export-compile-commands

# Enable rocprof SDK support
invoke build-client --enable-rocprof

# Custom CMake build in a non-default directory
cmake --preset tensilelite -S .. -B my-build
cmake --build my-build --parallel
```

| What changed | Rebuild command |
|---|---|
| `rocisa/` or `stinkytofu` C++ sources | `invoke rocisa` |
| `rocisa/pyproject.toml` or `CMakeLists.txt` | `invoke rocisa` |
| `client/` C++ sources | `invoke build-client` |
| `Tensile/*.py` | No rebuild (editable install, imported directly) |

Use `ccache` to accelerate: `sudo apt install ccache` — detected automatically by invoke.

### CMake options

| Option | Default | Purpose |
|---|---|---|
| `TENSILELITE_ENABLE_HOST` | ON | Build C++ runtime library |
| `TENSILELITE_ENABLE_CLIENT` | ON | Build benchmark client executable |
| `TENSILELITE_ENABLE_AUTOBUILD` | OFF | Generate `Tensile.sh` wrapper scripts (deprecated) |
| `TENSILELITE_BUILD_TESTING` | OFF | Build C++ host-library gtest |
| `GPU_TARGETS` | (detected) | Semicolon-separated list of gfx targets |

### Stale rocisa detection

If any `.cpp/.hpp/.h/.def/.inc` under `rocisa/` is newer than `_rocisa.so`, the import raises:
```
ImportError: rocisa C++ sources are newer than the built _rocisa.so — bindings are stale.
  Modified: .../stinkytofu/src/ir/asm/Function.cpp
  Rebuild:  cmake --build <build_dir> --target _rocisa
```
Run `invoke rocisa` to fix.

---

## Testing

### Full test suite (builds client + runs all common tests)
```bash
tox -e py3 -- Tensile/Tests -m common
```

### Unit tests only (fast — no client build needed)
```bash
tox -e unit -- Tensile/Tests/unit
```

### Run a single unit test file
```bash
tox -e unit -- Tensile/Tests/unit/test_KernelWriter.py
tox -e unit -- Tensile/Tests/unit/test_Configuration.py
```

### Run a single unit test function
```bash
tox -e unit -- Tensile/Tests/unit/test_emitMfmaInstruction.py::test_mfma_f32_16x16x4
```

### Run a single YAML integration test
```bash
# After invoke build-client:
Tensile/bin/Tensile Tensile/Tests/common/gemm/fp16_use_e.yaml tensile-out

# With a custom-built client:
Tensile/bin/Tensile Tensile/Tests/common/gemm/fp16_use_e.yaml tensile-out \
    --prebuilt-client=my-build/tensilelite-client/tensilelite-client
```

### Run tests by category marker
```bash
tox -e py3 -- Tensile/Tests -m gemm
tox -e py3 -- Tensile/Tests -m streamk
tox -e py3 -- Tensile/Tests -m gfx94x
tox -e py3 -- Tensile/Tests -m Float8
```

### Debug build + single worker (serialized for easier debugging)
```bash
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx942 --clean" \
TENSILE_NUM_PYTEST_WORKERS=1 tox -e py3 -- Tensile/Tests -m common
```

### Coverage
```bash
tox -e coverage        # full (unit + common tests)
tox -e coverage-unit   # unit tests only
# Outputs: HTML, XML, JSON reports + terminal summary
```

### Lint / format
```bash
tox -e lint    # flake8 (pyflakes errors only; E/W codes ignored)
tox -e format  # black (line-length=100) on Common/, TensileCreateLibrary/, Utilities/
tox -e isort   # isort (black profile) on same directories
```

### C++ gtest (host library)
```bash
cmake -DTENSILELITE_BUILD_TESTING=ON --preset tensilelite -S .. -B my-build
cmake --build my-build --parallel
ctest --test-dir my-build
```

### Test markers (from `pytest.ini`)
Architecture: `gfx11`, `gfx12`, `gfx94x`, `gfx950`, `gfx1250`, plus `xfail-gfxNNN`/`skip-gfxNNN`.
Data type: `Float`, `Double`, `Half`, `BFloat16`, `Int8`, `Float8`, `BFloat8`, `Float8BFloat8`, `Float4`, `Float6`, `BFloat6`.

### Environment variables
| Variable | Default | Effect |
|---|---|---|
| `TENSILE_NUM_PYTEST_WORKERS` | `4` | Parallel pytest worker count |
| `TENSILELITE_CLIENT_ARGS` | (empty) | Extra args forwarded to `invoke build-client` during tox |

---

## Where to Add Tests

### Python unit tests
Location: `Tensile/Tests/unit/test_<module>.py`.

Naming mirrors the module under test: `test_KernelWriter.py`, `test_Configuration.py`, `test_emitMfmaInstruction.py`. Use standard `pytest` fixtures. Mock GPU calls with `unittest.mock`. The `tox -e unit` environment installs rocisa via `pip install {toxinidir}/rocisa/` so rocisa is importable without a prior `invoke build-client`.

Example minimal unit test:
```python
# Tensile/Tests/unit/test_MyModule.py
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
from Tensile.MyModule import myFunction

def test_myFunction_basic():
    result = myFunction(input=42)
    assert result == expected
```

### YAML integration tests
Location: `Tensile/Tests/common/<category>/<test>.yaml`.

Each YAML must specify:
```yaml
GlobalParameters:
  PrintLevel: 1

BenchmarkProblems:
  - - ProblemType:
        OperationType: GEMM
        DataType: f16
        TransposeA: False
        TransposeB: True
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
        - EdgeType: ["ShiftPtr"]
      ForkParameters:
        - MatrixInstruction: [[16,16,16,1]]
        - WorkGroup: [[16,16,1]]
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [128, 128, 1, 128]

LibraryLogic: ~
ClientParameters: ~
```

Categories: `gemm`, `groupedgemm`, `streamk`, `sparse`, `gradient`, `gsu`, `exception`, `flags`, `client`.

### C++ host tests
Location: `tests/`. Gated by `TENSILELITE_BUILD_TESTING=ON`. Tests the C++ runtime (`include/Tensile/`, `src/`) without GPU hardware.

---

## Key Python Modules: Detailed Reference

### `Tensile.py` — Entry point
`executeStepsInConfig(config, outputPath, ...)` reads the YAML config top-level keys (`GlobalParameters`, `BenchmarkProblems`, `LibraryLogic`, `ClientParameters`) and dispatches each phase in sequence.

### `BenchmarkProblems.py` — Phase 1
Key functions:
- `_generate_single_solution(perm, problemType, constantParams, assembler, debugConfig, isaInfoMap)` (`line 153`) — instantiates one `Solution` from a fork permutation.
- `_generateForkedSolutions(...)` (`line 197`) — fans out over all permutations with `ParallelMap2`.
- `_benchmarkProblemType(...)` (`line 513`) — outer loop: iterates benchmark steps, checks cache, calls `writeBenchmarkFiles()` then `runClient()`.
- `writeBenchmarkFiles(stepBaseDir, solutions, problemSizes, ...)` (`line 370`) — deduplicates kernels, creates `KernelWriterAssembly`, calls `writeSolutionsAndKernels()`.

Caching: a 12-hex-char SHA of `ConstantParams + ForkParams + ...` keys (`_computeCacheKey()`) is used to skip codegen + build if nothing changed.

### `LibraryLogic.py` — Phase 2
Key functions and classes:
- `generateLogic(config, benchmarkDataPath, libraryLogicPath, ...)` (`line 1427`) — reads all `.csv`+`.yaml` pairs in `benchmarkDataPath`, dispatches to `analyzeProblemType()`.
- `analyzeProblemType(problemType, problemSizeGroups, ...)` (`line 48`) — merges data from multiple size groups, constructs `LogicAnalyzer`.
- `LogicAnalyzer` (`line 240`) — merges solutions across size groups (dedup by hash), reads performance data from `.csv`, runs the selection algorithm. Key internal steps:
  - Builds a `(problem_size → solution_idx)` winner table by finding the max GFLOPS/s per size.
  - Generates heuristic ranges/tiles logic (the `3_LibraryLogic/` YAML/MsgPack) that the C++ runtime uses.
- `handle_frequency_issue()` + `read_max_freq()` (`line 1528`) — GPU frequency sanity check; warns if max observed frequency is below hardware spec.

### `ClientWriter.py` — Phase 3 + client invocation
Key functions:
- `runClient(libraryLogicPath, forBenchmark, ...)` (`line 231`) — builds the client (if needed) and invokes `tensilelite-client` with the generated `ClientParameters.ini`.
- `writeClientConfigIni(...)` (`line 565`) — generates `ClientParameters.ini` from Python data: problem sizes, data types, bias args, activation args, code object paths, etc.
- `writeClientConfig(...)` (`line 729`) — generates the YAML client config (alternative to INI).
- `runNewClient(scriptPath, clientParametersPath, ...)` (`line 215`) — subprocess invocation of `tensilelite-client`.

### `KernelWriter.py` — Assembly codegen (largest module, ~10600 lines)
Key classes:
- `StateValues` (`line 137`) — dataclass tracking all mutable kernel state: register pool sizes (`bpeA`, `bpeCinternal`, etc.), loop/schedule state (`scheduleGlobalRead`, `numItersPLR`), subtile index (`SubTileIdx`), overflow flags.
- `KernelWriter` (`line 470`, abstract) — main codegen class. Key methods:
  - `__init__(assembler, debugConfig)` (`line 476`) — initializes `self.do[]` (enable/disable each code section), `self.db[]` (debug flags including `ConservativeWaitCnt`, `CheckValue1A/B`, `InitLds`).
  - `makeSchedule(kernel, tPA, tPB, ...)` (`line 638`) — builds the instruction interleaving schedule for a single summation iteration.
  - `_makeSubIterSchedule(kernel, tPA, tPB, ...)` (`line 858`) — builds the inner sub-iteration schedule (local-read FIFO tracking, MFMA dependency analysis).
  - `setupNewTile(kernel, tPA, tPB, ...)` (`line 2599`) — emits global-read pointer init, stagger, prefetch setup for a new output tile.
  - `kernelBody(kernel, tPA, tPB)` (`line 5256`) — emits the complete standard kernel body.
  - `kernelBodySubtile(kernel, tPA, tPB)` (`line 4873`) — emits the subtile kernel body.
  - `_getKernelSource(kernel)` (`line 10404`) — top-level codegen dispatcher.

### `KernelWriterAssembly.py` — ASM file production
`KernelWriterAssembly(KernelWriter)` (`line 141`). Key methods:
- `getSourceFileString(kernel)` (`line 176`) — for custom kernels reads `.s` source and optionally computes occupancy; for generated kernels calls `_getKernelSource()`.
- `getSgprOccupancy(sgprs)`, `getVgprOccupancy(...)`, `getOccupancy(...)` — occupancy calculators against ISA register caps.
- `setRocIsa(data, outOptions)` (`KernelWriter.py:10449`) — configures the rocisa singleton with ISA data and output options (e.g. `outputNoComment`).

### `SolutionStructs/Solution.py` — Parameter validation
`Solution(collections.abc.Mapping)` (`line 471`). Key methods:
- `__init__(config, splitGSU, ...)` (`line 476`) — fills `_state` from `config` + `defaultSolution` defaults, validates parameter types, calls `assignDerivedParameters()`.
- `assignDerivedParameters(state, splitGSU, ...)` (`line 1456`, static) — derives all tiling fields. Critical derivations: `numSubTiles` (1 or 2), `VectorWidthA/B` (divisibility by `numSubTiles`), MX scale layout/transport, occupancy, `_GlobalAccumulation` mode.
- `getKernels()` (`line 561`) — returns `[self]` (one kernel per solution for standard GEMM).
- `_deriveAndValidateMXScaleLayoutAndTransport(state, asmCaps, archCaps, ...)` (`line 57`, module-level) — resolves `MXLoadInst`/`MXScaleFormat`/`TDMInst` triple against ISA caps.

### `Contractions.py` — Problem taxonomy
Defines `ProblemType`, `FreeIndex`, `BatchIndex`, `BoundIndex`, `SizeMapping`, `Solution`, `InternalArgsSupport`. These mirror the C++ runtime types and are used by `LibraryLogic` to build the heuristic library.

### `LibraryIO.py` — Serialization
`parseSolutionsFile(solutionsFileName, ...)` — reads `.yaml` files containing solution lists. `writeLibraryLogicYaml(...)` / msgpack equivalents — write the heuristic logic. All I/O goes through this module.

### `Common/` — Global parameters and utilities
- `GlobalParameters.py`: `globalParameters` dict (CpuThreads, PrintLevel, etc.), `defaultSolution` dict (all solution defaults), `assignGlobalParameters()`.
- `ValidParameters.py`: `validParameters` dict — the allowed values for each solution key; used by `Solution` to reject invalid configs.
- `Architectures.py`: `isaToGfx()`, `gfxToVariants()`, ISA capability tables.

### `TensileCreateLibrary/Run.py` — Kernel build pipeline
- `writeSolutionsAndKernels(outputPath, asmToolchain, srcToolchain, solutions, kernels, ...)` (`line 418`) — parallel codegen + assemble + link.
- `processKernelSource(kernelWriterAssembly, data, outOptions, splitGSU, kernel)` (`line 216`) — single-kernel codegen; returns `KernelCodeGenResult`.
- `removeInvalidSolutionsAndKernels(results, kernels, solutions, ...)` (`line 266`) — filters out failed kernels; exits if `not errorTolerant`.
- `passPostKernelInfoToSolution(results, kernels, solutions, ...)` (`line 294`) — writes `CUOccupancy`, `MathClocksUnrolledLoop` back into each solution after codegen.
- `writeAssembly(asmPath, result)` (`line 375`) — writes `.s` file to disk.
- `generateLogicDataAndSolutions(logicFiles, args, assembler, isaInfoMap)` (`line 798`) — entry point for `TensileCreateLibrary` (used by hipBLASLt CMake, no benchmarking).

---

## Components/ — Modular Kernel Building Blocks

All components subclass `Component` (from `Tensile/Component.py`). `Component.find(writer)` selects the right variant by checking kernel caps (MFMA vs WMMA, data type, ISA). Key components:

| File | Component class(es) | Role |
|---|---|---|
| `MAC_F32.py` | `MAC_F32_Plain` | SGEMM MFMA/VALU MAC |
| `MAC_F16.py` | `MAC_F16_Plain`, `FMA_F16_Packed`, `FMA_F16_NonPacked` | HGEMM MAC variants |
| `MAC_F16_HPA.py` | `FMA_F16_HPA_DOT2`, `FMA_F16_HPA_MAD_MIX` | F16 with HPA |
| `MAC_BF16_HPA.py` | `FMA_BF16_HPA`, `FMA_BF16_HPA_DOT2` | BF16 with HPA |
| `MAC_F64.py` | `FMA_F64C_Plain` | DGEMM |
| `MAC_I8X4.py` | `MAC_I8X4_Plain` | INT8 SIMD MAC |
| `LocalRead.py` | — | `ds_load_*` emission; FIFO scheduling |
| `LraTileAssignment.py` | `LraTileAssignmentMFMA`, `LraTileAssignmentTransposedMFMA`, `*FP8`, `*F4`, `*F6` | Local-read register/address mapping per tile and data type |
| `GlobalWriteBatch.py` | — | Output store batching with alpha/beta application |
| `ComputeStoreVgprs.py` | `ComputeStoreVgprsMFMA`, `ComputeStoreVgprsMFMASwap`, `ComputeStoreVgprsVALU` | Store-VGPR layout selection |
| `StreamK.py` | — | Persistent/stream-K workgroup partition and accumulation |
| `PersistentLoop.py` | `PersistentLoopOn`, `PersistentLoopOff` | Loop open/close around tile iterations |
| `GSU.py` | — | Global split-U partial-sum accumulation |
| `SIA.py` | — | Schedule inter-iteration algorithm variants |
| `PackData.py` | — | TF32, FP4/FP6 packing before MFMA |
| `CustomSchedule.py` | — | User-supplied schedule override loading |
| `Signature.py` | — | `.amdgpu_kernel` metadata and argument list |
| `Subtile/` | See §Subtile Project | Subtile-specific geometry, emit, scheduling |

---

## rocisa — C++ Assembly IR Module

`rocisa/` is a C++ module compiled with `amdclang++` and bound into Python via **Nanobind**. `KernelWriter.py` imports instruction classes directly:

```python
from rocisa.instruction import (
    BufferLoadB128, BufferLoadB32, BufferLoadB64,
    DSLoadB128, DSLoadB32, DSStoreB64,
    MFMAInstruction, MXMFMAInstruction, SMFMAInstruction,
    SWaitCnt, SBarrier, SMovB32, VAddU32, ...
)
from rocisa import rocIsa, countInstruction, countGlobalRead, countLocalRead
from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import Module, KernelBody, Label, StructuredModule, TextBlock
from rocisa.container import vgpr, sgpr, accvgpr, mgpr, RegisterContainer
```

`rocIsa.getInstance()` returns the process-wide singleton. `setData(data)` / `setOutputOptions(outOptions)` configure it per-kernel.

The `rocIsaPass` optimization pass (`rocisa.asmpass`) runs after codegen to insert `s_waitcnt`, validate register lifetimes, and apply architecture-specific fixups.

Staleness check: `rocisa/rocisa/__init__.py` compares mtimes of all `.cpp/.hpp/.h/.def/.inc` against the `.so`. Pre-built wheels skip this check (no `_build_info.py`).

---

## Printing, Compiling, and Running a Single Config

### Generate ASM for a single YAML test
```bash
invoke build-client   # one-time
Tensile/bin/Tensile Tensile/Tests/common/gemm/sgemm.yaml tensile-out
```
Assembly files land in:
```
tensile-out/1_BenchmarkProblems/<problem>/00_Final/source/build_tmp/SOURCE/assembly/*.s
```

### Rebuild only the `.co` after editing ASM
```bash
# Edit the .s file, then:
make co TENSILE_OUT=tensile-out                           # auto-detect arch
make co TENSILE_OUT=tensile-out ARCH="gfx942" WAVE=64    # gfx9 explicit
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32   # gfx11 explicit
make co TENSILE_OUT=tensile-out ARCH="gfx942:xnack-" ASM_ARGS="-v" LINK_ARGS="-v"
```
The generated `Makefile` detects changed `.s` files (via mtimes) and rebuilds only affected `.o` files and the final `.co`.

### Run the client after rebuild
```bash
./build_tmp/tensilelite-client/tensilelite-client \
    --config-file tensile-out/1_BenchmarkProblems/.../ClientParameters.ini \
    --num-elements-to-validate 100 \
    --num-warmups 2 \
    --num-benchmarks 10
```

---

## tensilelite-client: Structure and Reference

The benchmark executable (`client/`) is the harness that runs compiled kernels against problem sizes, validates correctness, and reports performance. It is invoked by the Python benchmarking pipeline (via `runClient()` in `ClientWriter.py`) during Phase 1, or directly by developers.

### Source layout

```
client/
├── main.cpp                      # Entry point: option parsing, run loop orchestration
├── cpu_gemm_driver.cpp           # CPU reference GEMM (for ReferenceValidator)
├── src/
│   ├── ProgramOptions.cpp        # CLI/INI option definitions and parsing (no Boost)
│   ├── ClientProblemFactory.cpp  # Builds ContractionProblemGemm from CLI args
│   ├── DataInitialization.cpp    # Allocates + fills A/B/C/D/E/bias/scale tensors on GPU
│   │                             #   LRU cache for rotating buffers; pristine-copy management
│   ├── SolutionIterator.cpp      # Iterates over candidate solutions (all or best-only)
│   ├── BenchmarkTimer.cpp        # GPU/CPU timing (hipEvent or std::chrono)
│   ├── ReferenceValidator.cpp    # CPU vs GPU output comparison
│   ├── PerformanceReporter.cpp   # Computes GFLOPS/s, efficiency, writes results.csv
│   ├── ResultFileReporter.cpp    # CSV column definitions and output
│   ├── HardwareMonitor.cpp       # Polls GPU clocks, temperature, power (rocm-smi)
│   ├── HardwareMonitorListener.cpp  # Integrates HardwareMonitor into run loop
│   ├── MetaRunListener.cpp       # Sequences all RunListeners per problem/solution
│   ├── LibraryUpdateReporter.cpp # Writes winner indices for library-update workflows
│   ├── Rotating.cpp              # Rotating buffer management (cache-cold benchmarking)
│   ├── Profiler.cpp              # rocprof SDK integration (TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
│   ├── TimingEvents.cpp          # Detailed per-launch timing instrumentation
│   ├── ProgressListener.cpp      # Progress printing to stdout
│   ├── CSVStackFile.cpp          # Nested CSV output support
│   └── TypedId.cpp               # Typed ID utilities
└── include/                      # All matching headers
    ├── RunListener.hpp            # Abstract base for all listeners
    ├── ResultReporter.hpp         # Abstract base for result reporters
    ├── SolutionIterator.hpp       # TopSolutionIterator / AllSolutionIterator
    ├── DataInitialization.hpp     # InitMode enum, DataInitialization class
    ├── ReferenceValidator.hpp
    ├── BenchmarkTimer.hpp
    ├── PerformanceReporter.hpp
    └── ...
```

### Run loop

`main.cpp` wires together a chain of `RunListener` objects (`RunListener.hpp`) — `MetaRunListener`, `BenchmarkTimer`, `ReferenceValidator`, `PerformanceReporter`, `ResultFileReporter`, `HardwareMonitorListener` — and drives them through:

```
ClientProblemFactory → problems[]
  └── SolutionIterator → solutions[]
        └── for each (problem, solution):
              DataInitialization.initializeInputs()  // fill tensors, pristine copy
              [warmup runs × --num-warmups]
              [if --sync-after-warmups] hipStreamSynchronize()
              [timed benchmark runs × --num-benchmarks]:
                for each sync × --num-syncs-per-benchmark:
                  for each enqueue × --num-enqueues-per-sync:
                    hipLaunchKernelGGL(kernel)
                  [GPU or CPU timer record]
              ReferenceValidator.validate()          // CPU gemm + element compare
              PerformanceReporter.report()           // GFLOPS/s, efficiency → CSV
```

`SolutionIterator` has two modes (selected by `--best-solution`):
- **Default (`AllSolutionIterator`)**: sweeps every solution in the library for each problem (exhaustive — used for benchmarking to find winners).
- **`TopSolutionIterator`** (`--best-solution`): uses the heuristic to pick the predicted winner only (fast validation / smoke test).

### Flush kernel

`flush_icache()` (`main.cpp:84`) is a HIP kernel that issues `s_icache_inv` + 16 `s_nop 0` to invalidate the instruction cache. Launched by `estimate_flush_kernel_time()` when `--icache-flush-args true` is set.

### Key CLI options

All options can appear on the command line or in an INI file (`--config-file`). Options repeat in INI as `key = value`, one per line.

**Problem definition**

| Option | Default | Description |
|---|---|---|
| `--problem-identifier` | — | Einstein notation (e.g. `Cijk_Ailk_Bjlk`) |
| `--problem-size,-p` | — | Comma-separated sizes per dim; repeatable |
| `--type` | None | Sets all data types at once |
| `--a-type`, `--b-type`, `--c-type`, `--d-type`, `--e-type` | None | Per-matrix data types |
| `--alpha-type`, `--beta-type` | None | Scalar types |
| `--compute-input-type-A/B` | None | Compute-input type for mixed precision |
| `--f32-xdl-math-op` | None | xf32 compute for float matrices |
| `--strided-batched` | true | Strided-batched vs general batched |
| `--grouped-gemm` | false | Grouped GEMM |
| `--sparse` | 0 | Sparse mode (0=none, 1=A sparse, 2=B sparse) |
| `--high-precision-accumulate` | false | HPA accumulation |
| `--mx-a-block`, `--mx-b-block` | 0 | MX block size for A/B |
| `--mx-a-type`, `--mx-b-type` | E8 | MX scale element type |
| `--swizzle-tensor-a/b` | false | Enable input tensor swizzle |
| `--mx-scale-format` | 0 | MX scale format (0=none, 1=pre-swizzle) |
| `--deterministic-mode` | false | No U-splitting across workgroups |

**Epilogue / activations**

| Option | Default | Description |
|---|---|---|
| `--activation-type` | None | `relu`, `gelu`, `abs`, `exp`, `sigmoid`, `tanh`, `clippedrelu`, `leakyrelu`, `none` |
| `--activation-hpa` | false | Activation uses HPA dtype |
| `--activation-no-guard` | false | Disable NaN guard in activation |
| `--activation-additional-args` | — | Extra FP args (e.g. leaky-relu slope) |
| `--activation-enum-args` | [None] | Enum-typed activation args |
| `--use-bias` | 0 | Enable bias tensor |
| `--bias-source` | 3 | Bias source dimension index |
| `--bias-type-args` | [None] | Bias element type |
| `--use-scaleAB` | "" | ScaleA/B mode |
| `--use-scaleCD` | false | ScaleC/D mode |
| `--use-scaleAlphaVec` | 0 | Per-element alpha scaling |
| `--use-e` | false | Enable E tensor |
| `--use-gradient` | false | Gradient mode |
| `--output-amaxD` | false | Output absolute max of D |
| `--use-user-args` | false | User argument struct as kernel input |

**Data initialization**

| Option | Default | Description |
|---|---|---|
| `--init-a/b/c/d/e` | Random/Random/Random/Zero/Zero | Init mode per tensor |
| `--init-alpha/beta` | Two/Two | Scalar init |
| `--init-bias/scaleA/B/C/D/AlphaVec` | One/Two/... | Scale/bias init modes |
| `--init-seed` | 0 | `srand` seed |
| `--pristine-on-gpu` | true | Keep pristine GPU copy to avoid re-uploads each benchmark |
| `--c-equal-d` | false | Use same buffer for C and D |
| `--offset-a/b/c/d/e` | 0 | Buffer start offset |
| `--a/b/c/d/e-strides` | — | Override default strides |
| `--rotating-buffer-size` | 0 | MB of rotating buffers (cache-cold benchmarks) |
| `--rotating-buffer-mode` | 0 | Rotating mode |

**Benchmarking**

| Option | Default | Description |
|---|---|---|
| `--num-warmups` | 0 | Warmup iterations before timing |
| `--sync-after-warmups` | true | Sync GPU after warmups |
| `--num-benchmarks` | 1 | Benchmark iterations |
| `--num-enqueues-per-sync` | 1 | Kernel launches per sync |
| `--max-enqueues-per-sync` | -1 | Cap on auto-increased enqueues |
| `--num-syncs-per-benchmark` | 1 | Syncs per benchmark |
| `--min-flops-per-sync` | 0 | Auto-increase enqueues for small problems |
| `--use-gpu-timer` | true | Use `hipEvent` timer; false = `std::chrono` |
| `--skip-slow-solution-ratio` | 0.0 | Skip solutions below this fraction of best during warmup |
| `--granularity-threshold` | 0.0 | Skip solutions with wave granularity below threshold |
| `--prediction-threshold` | 2.0 | Skip solutions with low predicted performance |
| `--best-solution` | false | Run only library-selected winner (TopSolutionIterator) |
| `--selection-only` | false | Print kernel selections without running them |
| `--sleep-percent` | 0 | Sleep between launches |
| `--hardware-monitor` | true | Poll GPU clocks/power via rocm-smi |
| `--performance-metric` | DeviceEfficiency | Metric for results: `DeviceEfficiency`, `CUEfficiency` |

**Validation**

| Option | Default | Description |
|---|---|---|
| `--num-elements-to-validate` | 0 | Elements compared against CPU reference (0 = skip) |
| `--print-valids` | false | Print passing elements |
| `--print-max` | -1 | Max elements to print (-1 = all) |
| `--bounds-check` | Disable | 1=sentinel, 2=front guard page, 3=back guard page, 4=both |
| `--exit-on-error` | false | Stop immediately on validation failure |
| `--print-tensor-a/b/c/d/ref/bias` | false | Dump tensor values to stdout |
| `--print-tensor-scale-alpha-vec` | false | Print ScaleAlphaVec |
| `--print-tensor-amaxd` | false | Print AmaxD from CPU and GPU |
| `--dump-tensors` | false | Binary dump instead of text |

**I-cache benchmarking**

| Option | Default | Description |
|---|---|---|
| `--icache-flush-args` | [false] | Flush instruction cache between runs |
| `--icache-rotate-copies` | 0 | Extra module copies for cold-miss (0=off, -1=auto, N=N extra) |
| `--icache-rotate-size` | 64 | Cache budget (KB) for `-1` auto mode |

Auto mode: on Linux, parses `.co` ELF symbol table to find minimum kernel size, computes copies needed to overflow the I-cache (`N = icache_rotate_size * 2 * 1024 / min_kernel_size`). Non-Linux uses `icache_rotate_size` directly as the extras count.

**Output**

| Option | Default | Description |
|---|---|---|
| `--results-file` | `results.csv` | CSV output path |
| `--log-file` | — | Log file path |
| `--log-file-append` | false | Append to log instead of overwrite |
| `--log-level` | Debug | Verbosity: `Debug`, `Info`, `Warning`, `Error` |
| `--library-update-file` | "" | Write winner indices + speeds for library update |
| `--library-update-comment` | false | Include solution name as comment in update file |
| `--csv-export-extra-cols` | false | Export winner info extra columns |
| `--csv-merge-same-problems` | false | Merge CSV rows with same problem ID |
| `--PrintWinnersOnly` | false | Print only winning solutions |
| `--timing-instrumentation` | false | Emit detailed per-launch timing to stderr |

**Library / solution control**

| Option | Default | Description |
|---|---|---|
| `--library-file,-l` | — | YAML solution library (default: embedded) |
| `--code-object,-c` | — | Code object file(s) (default: embedded); repeatable |
| `--solution-start-idx` | -1 | First solution index to run |
| `--num-solutions` | -1 | Number of solutions to run (-1 = all) |
| `--problem-start-idx` | 0 | First problem index |
| `--num-problems` | -1 | Number of problems (-1 = all) |
| `--prob-sol-map` | — | `[probIdx, solIdx]` pairs to pin specific solution per problem |
| `--max-workspace-size` | 32 MB | Workspace for stream-K / GSU |
| `--device-idx` | 0 | GPU device index |
| `--use-default-stream` | false | Use default HIP stream |
| `--config-file` | — | INI config file(s); all above options can go here |

**Misc**

| Option | Default | Description |
|---|---|---|
| `--kernel-language` | Any | Filter by kernel language (`Assembly`, `Source`, `Any`) |
| `--prune-mode` | PruneRandom | Sparse matrix pruning mode |
| `--metadata-layout` | 0 | Sparse metadata layout |
| `--a/b/c/d-ops` | — | Tensor operations applied (transpose, conjugate, etc.) |
| `--rocprof-counter` | — | rocprof counter names (requires `--enable-rocprof` build) |

### Example: manual client invocation
```bash
./build_tmp/tensilelite-client/tensilelite-client \
    --config-file tensile-out/1_BenchmarkProblems/.../ClientParameters.ini \
    --num-warmups 2 \
    --num-benchmarks 10 \
    --num-elements-to-validate 1000 \
    --results-file my_results.csv \
    --log-level Info
```

### Performance reporting
`PerformanceReporter` computes:
- **GFLOPS/s** = `2 * M * N * K / time_ns` (adjusted for batch count)
- **DeviceEfficiency** = GFLOPS/s / peak_GFLOPS (from `HardwareMonitor` clock readings)

`results.csv` columns include: problem sizes, solution index, solution name, time (ns), GFLOPS/s, efficiency, validation pass/fail, GPU clock (MHz).

---

## C++ Runtime Library

`include/Tensile/` and `src/` implement the runtime that selects and dispatches kernels at hipBLASLt call time.

| Header | Role |
|---|---|
| `Tensile.hpp` | Top-level runtime API: `findBestSolution()`, `run()` |
| `ContractionProblem.hpp` | `ContractionProblemGemm` — problem descriptor (M, N, K, types, strides) |
| `ContractionSolution.hpp` | `ContractionSolution` — kernel descriptor (tiling, launch bounds) |
| `SolutionLibrary.hpp` | Abstract library interface: `findBestSolution(problem, hardware)` |
| `MasterSolutionLibrary.hpp` | Concrete library that holds multiple arch-specific sub-libraries |
| `EmbeddedLibrary.hpp` | Library backed by byte arrays embedded in the binary |
| `hip/HipSolutionAdapter.hpp` | Loads `.co` files, manages `hipModule_t`, launches kernels |
| `hip/HipHardware.hpp` | GPU hardware descriptor (arch, CU count, wavefront size) |
| `ContractionSolution.cpp` | `KernelArguments` construction and `hipLaunchKernelGGL` dispatch |

The runtime flow at hipBLASLt call time:
```
hipblasLtMatmul() → tensile_host.cpp:findSolution()
  → MasterSolutionLibrary::findBestSolution(problem, hardware)
  → HipSolutionAdapter::launchKernel(solution, inputs, outputs, stream)
  → hipLaunchKernelGGL (loaded from .co / embedded byte array)
```

---

## Subtile Project

**Subtile** (`UseSubtileImpl`, `numSubTiles`) is a kernel variant where the output tile is split into multiple subtiles processed in sequence by the same wave. The benefit is increased reuse of the input A/B data already loaded into LDS while writing different portions of the output.

### Key parameters (resolved in `Solution.assignDerivedParameters()`)
- `UseSubtileImpl` (bool) — enables subtile path; triggers `kernelBodySubtile()`.
- `numSubTiles` — auto-set to `2` when `UseSubtileImpl=True`, else `1` (`Solution.py:2186`).
- `VectorWidthA/B` — reduced until divisible by `numSubTiles * VectorWidth` (`Solution.py:2213–2247`).
- `subtileLdsSwizzle` (bool) — per-subtile LDS swizzle pattern to avoid bank conflicts.

### Dispatch in `KernelWriter._getKernelSource()` (`KernelWriter.py:10413`)
```python
if not kernel["UseSubtileImpl"]:
    (error, kb) = self.kernelBody(kernel, tensorParametersA, tensorParametersB)
else:
    (error, kb) = self.kernelBodySubtile(kernel, tensorParametersA, tensorParametersB)
```

### `kernelBodySubtile()` structure (`KernelWriter.py:4873`)
```
functionSignature()
defineAndResources()
[if MXBlock]: scale stride shift
StreamK.preLoop()
PersistentLoop.openPersistentLoop()
setupNewTile()           # global-read init for first subtile
defineTdmSgprs()         # TDM descriptor SGPRs (deferred post-setupNewTile)
globalReadDTLInitCommonSgpr()
[for each subtile]:
  SubTileIdx tracking via self.states.SubTileIdx
  main summation loop (local-read, MFMA, local-write, global-read-inc)
  SubTileIdx = (SubTileIdx + 1) % numSubTiles
epilogue (alpha/beta, activation, bias, global-write) per subtile
functionEnd()
```

`_subtileDtileBaseVgpr` (`KernelWriter.py:4880`) — first VGPR of the D-tile accumulator. On gfx1250 (MIArchVgpr), WMMA writes to regular VGPRs so the store path aliases ValuC to these VGPRs directly.

### `Components/Subtile/` — Subtile sub-package

| File | Role |
|---|---|
| `SubtileGeometry.py` | Tile geometry classes: `ABGRGeometry`, `ABLRGeometry`, `CDTileGeometry`, `MXScaleInputGeometry`, `TileInfo`, `RegisterTileInfo`; MFMA layout selection (`selectABGeometry`, `selectDGeometry`); `emitMfmaCode()` |
| `Kernel.py` | Main subtile kernel logic: `preLoop()`, `mainLoop()`; imports all emit helpers; instruction class imports from rocisa |
| `SubtileGREmit.py` | Global-read emit: `globalReadDoSubtile()`, `graTileAssignment()`, `graInitPointer()`, `emitSubtileBufferLoad()`, TDM descriptor init (`initTDMDescriptorSubtile`) |
| `SubtileLREmit.py` | Local-read emit: `localReadDoSubtile()`, `lraTileAssignment()`, `emitSubtileDsRead()`, LDS buffer swap |
| `SubtileScaleEmit.py` | MX scale GR/LR emit: `globalReadDoScaleSubtile()`, `localReadDoScaleSubtile()`, swizzled DTL init, pointer updates |
| `InstructionEmitter.py` | `InstructionEmitter` class — wraps rocisa instructions with subtile-specific waitcnt tracking (`SWaitCntEx`) |
| `InstructionScheduler.py` | `instructionSchedule()` — dependency-aware instruction scheduler for the subtile main loop; `_SlotPlacer`, `_SchedulingRules` |
| `LogicalScheduler.py` | `LogicalScheduler` — higher-level scheduling of GR/LW/LR/MFMA phases across iterations |

### Activations in subtile
ReLU and other activations are applied natively per-subtile inside `KernelWriterActivationFunction.py`. Critical pitfall: `ActivationType` enum cases must be handled in the correct order; a missing case silently falls through to the wrong activation. The activation bypass flag (`activation-no-guard`) disables the NaN guard. See `memory/project_subtile_epilogue.md` for known `ActivationType` case ordering issues and the bypass flag pattern.

---

## Known Issues / Gotchas

- **Stale rocisa**: editing any `.cpp/.hpp` under `rocisa/` without running `invoke rocisa` causes `ImportError: rocisa C++ sources are newer than the built _rocisa.so`. The error message lists the modified files.
- **`tox -e unit` vs `tox -e py3`**: `unit` is fast (no client build); `py3` builds the client inside its `commands` block — use `TENSILELITE_CLIENT_ARGS` to pass custom cmake flags (e.g. `--build-type Debug`).
- **Direct `pytest` outside tox**: requires `invoke rocisa` first for the `rocisa` import to succeed.
- **`ARCH` auto-detection**: if `make co` picks the wrong arch (e.g. missing `xnack` suffix), manually pass `ARCH="gfx942:xnack-"`.
- **`Tensile.sh` / `Tensile.bat`**: deprecated wrappers — use `Tensile/bin/Tensile` directly.
- **PyYAML/msgpack missing in raw cmake**: bypassing `invoke build` may fail mid-device-lib build. Point `Python_EXECUTABLE` at the venv (`build/venv/bin/python`).
- **`rocisa.egg-info/` and `rocisa/build/`**: normal artifacts from editable install and cmake build — do not commit.
- **`tox -e py3` worker count**: defaults to 4; use `TENSILE_NUM_PYTEST_WORKERS=1` to serialize runs for easier debugging.
- **Subtile + activation bypass**: the `ActivationType` enum cases must be exhaustive in all subtile epilogue paths. A missing case silently falls through. Always add a guard when adding new activation types to subtile paths.
- **Solution cache invalidation**: the 12-char cache key covers `ConstantParams`, `ForkParams`, `ParamGroups`, `CustomKernels`, `InternalSupportParams`. Changing any of these forces a rebuild. The cache does NOT cover changes to `KernelWriter.py` or rocisa — you must `--clean` or delete `tensile-out/` manually after codegen changes.
- **`GlobalSplitUAlgorithm` + dot2**: `dot2` kernels do not support `MultipleBufferSingleKernel` (`Solution.py:1507`) — auto-demoted to `MultipleBuffer`.
- **MX scale on gfx1250**: requires `TDMInst` unless `StreamK > 0`; enforced in `_deriveAndValidateMXScaleLayoutAndTransport()`.
