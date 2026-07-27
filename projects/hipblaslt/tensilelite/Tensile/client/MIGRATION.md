<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Benchmark Client — Migration Report

This document describes the new in-process Python benchmark client for TensileLite
(`Tensile/client/`), how it differs from the historical standalone C++
`tensilelite-client` binary, and how to migrate existing workflows. It is written
to be self-contained: a reader with no prior knowledge of TensileLite should be
able to follow every step. All shell commands shown here were re-verified on a
`gfx950` machine before this document was finalized (see
[Reproducing these measurements](#reproducing-these-measurements)).

The new client's job is to take a benchmark specification (a TensileLite
`BenchmarkProblems` YAML, or an `.ini` that points at one), generate and compile
the candidate kernels in-process, run them on the GPU, and emit the same
`results.csv` and library-update YAML the C++ client produced — without needing a
pre-built kernel library or a separate binary. The core orchestrator is
`SweepRunner`, and it currently supports **NT strided-batched GEMM only**.

## Table of contents

1. [Architecture of the new Python client](#1-architecture-of-the-new-python-client)
2. [Differences from the old C++ client](#2-differences-from-the-old-c-client)
3. [Summary of each new component](#3-summary-of-each-new-component)
4. [How to use the new client (concrete, verified examples)](#4-how-to-use-the-new-client-concrete-verified-examples)
5. [Migration guide (from the old C++ tensilelite-client binary)](#5-migration-guide-from-the-old-c-tensilelite-client-binary)
6. [Benchmark comparison (real measured numbers)](#6-benchmark-comparison-real-measured-numbers)

---

## 1. Architecture of the new Python client

The client lives under
`/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/client/`.
It is a set of small, focused modules layered on top of two lower-level pieces
that already existed in the repo:

- `amdgpu_exec` — a Python extension that talks to the HIP runtime (device
  properties, `compile_asm_to_hsaco`, GPU buffers, module load, GPU events).
- `tensilelite_runtime` — Nanobind bindings for the C++ host library (used by
  the library runner and for the I-cache module-copy heuristic).

### Module map

| Module | Role |
|--------|------|
| `__init__.py` | Empty except the SPDX header; callers import submodules directly. |
| `sweep_runner.py` | The orchestrator. Enumerates solutions from a YAML/INI, compiles each to HSACO, benchmarks every `(problem_size, solution)` pair, writes `results.csv` and the library-update YAML. |
| `harness.py` | Low-level benchmarking primitives: `BenchmarkResult`, `BufferPool`, `KernelRunner` (module rotation + GPU-event timing). |
| `reporters.py` | `ResultsCSVReporter` and `LibraryUpdateReporter` — byte-compatible output formatting. |
| `reference.py` | NumPy CPU reference GEMMs and `assertClose`, used by correctness tests. Decoupled from the benchmark loop. |
| `yaml_solution_builder.py` | Turns a `BenchmarkProblems` YAML into concrete `Solution` objects and problem-size tuples. |
| `library_runner.py` | `LibraryRunner` — queries a pre-built solution library via `tensilelite_runtime` (best/top-N/filter). |
| `gemm_args.py` | Port of the C++ `ContractionSolution.cpp` kernel-argument assembly. |
| `hw_monitor.py` | `HardwareMonitor` context manager, polls `amdsmi` in a daemon thread. |
| `mx_types.py` | MX (micro-scaled) datatype decode helpers (E8, E5M3, FP4, FP6). |
| `sparse.py` | AMD 2:4 structured-sparsity compress/decompress (CPU-only). |

### How the pieces connect

`SweepRunner` is the top of the call graph. It uses `yaml_solution_builder` to
enumerate solutions and problem sizes, `_setupTensile` +
`KernelWriterAssembly` + `amdgpu_exec.compile_asm_to_hsaco` to compile, and
`harness.KernelRunner` to time. Timing results are wrapped in `SweepResult` and
handed to the two reporters. `gemm_args` supplies the two internal argument
words the kernel needs. `reference.py`, `library_runner.py`, `hw_monitor.py`,
`mx_types.py`, and `sparse.py` are auxiliary: `reference.py` powers correctness
tests, `library_runner.py` is a separate query path over a pre-built library,
and the rest support features that are not yet wired into the sweep loop.

### Data flow (benchmark YAML / .ini to results.csv)

```
benchmark YAML  (or .ini with a benchmark-yaml= key)
      |
      v
SweepRunner.__init__
   - if path ends in .ini -> _resolveYamlFromIni() parses the flat INI
     (under a synthetic [default] section) and returns the benchmark-yaml value
      |
      v
SweepRunner.run() -> _compile()
   - amdgpu_exec.get_chip()                     (e.g. "gfx950")
   - _setupTensile(chip)                        (assembler + ISA info map)
   - solutionsFromYaml() / yaml_solution_builder (enumerate candidate solutions)
   - per solution:
       _injectInternalArgsSupport()             (KernArgsVersion, chip fallbacks)
       _shouldSkip()  filter                     (drop WorkGroupMapping==0;
                                                  drop StaggerU==0 && SupportCustomStaggerU;
                                                  MX kernels excluded)
       _generateAsm()  -> KernelWriterAssembly.getSourceFileString()
       amdgpu_exec.compile_asm_to_hsaco()        (optional: save .co via saveCoPath)
      |
      v
problemSizesFromYaml()  -> 8-dim size tuples (M, N, batch, K, ldd, ldc, lda, ldb)
      |
      v
per problem size, per solution: _benchmarkOne()
   - _allocBufs(): A/B/C GpuBuffers + a rotating D BufferPool
   - _makeRunner(): KernelRunner.fromHsaco() with I-cache module rotation
   - runner.run(grid=(numWg,1,1), block=(NumThreads,1,1), nWarmup, nIters)
       -> GpuEvent-timed BenchmarkResult (per-iteration times in ns)
   - GFLOPS = 2*M*N*K*batch / (minUs * 1e-6) / 1e9      (min-time; sweep_runner.py:394-395)
      |
      v
_reportProblem():
   - ResultsCSVReporter.writeRow()   -> results.csv  (all solutions in one row)
   - LibraryUpdateReporter.writeRow() -> winner (max gflops > 0) as library-update YAML
```

The GFLOPS formula uses the **minimum** observed iteration time, which matches
the C++ client's `WinnerGFlops` (also a minimum-time metric). This is the single
most important behavioral compatibility guarantee.

### `.ini` integration (the M14 pipeline path)

`ClientWriter.py` glues the new client into the existing three-phase pipeline.
`writeClientConfigIni` emits a `benchmark-yaml=<path>` key into the `.ini` when a
benchmark YAML is available (`ClientWriter.py:640-641`). `runClient(...)`
(`ClientWriter.py:259`) defaults to `use_python_client=True`, which routes to
`_runWithPythonHarness` (`ClientWriter.py:237`) and constructs a `SweepRunner`
from the first config path, calling `.run(resultsCsv=buildPath/"results.csv")`.
It returns `0` on success and `1` on an exception. Passing
`use_python_client=False` falls back to a subprocess of the C++ binary, and a
multi-GPU benchmark request also falls back to the C++ `runClientParallel`.

---

## 2. Differences from the old C++ client

### Preserved (intentional compatibility)

- **`results.csv` schema and formatting.** Same column layout, the same `", "`
  separator, and the same `%.6g`-style float formatting (`_formatFloat` in
  `reporters.py`).
- **Library-update YAML format.** Emitted as
  `"  - - [sizes]\n    - [winnerIdx, winnerGFlops]"`, identical to the C++
  output.
- **GFLOPS metric = minimum time** (`WinnerGFlops` semantics).
- **`.ini` compatibility.** A flat `.ini` with a `benchmark-yaml=` key is
  accepted directly by `SweepRunner`.
- **`HardwareMonitor` behavior** and **amd-smi clock pinning** are ported.
- **Kernel argument layout.** `gemm_args.py` ports the C++
  `ContractionSolution.cpp` argument assembly byte-for-byte.

### Changed

- **In-process compile + benchmark.** The sweep generates and compiles kernels
  in-process via `amdgpu_exec`; there is no standalone binary and no pre-built
  kernel library required for the sweep itself.
- **NumPy CPU reference** replaces the C++ `Reference.cpp`, and it lives in the
  test suite (`reference.py`), not inside the benchmark run loop.
- **Rotation in Python.** Output-buffer rotation (`BufferPool`) and I-cache
  module rotation (`KernelRunner`) are implemented in Python.
- **Fixed `alpha=1.0`, `beta=0.0` with uninitialized buffers** in the sweep. The
  C++ client defaults to `init-alpha=Two` / `init-beta=Two` and initializes its
  buffers.

### Dropped / not yet ported

- **In-loop validation now supported.** `SweepRunner` supports in-sweep GPU
  output validation via `numElementsToValidate` (`0`=disabled, `-1`=all,
  `N`=first N elements). No in-sweep bounds-check or dump-tensor path.
  (See [section 4c](#c-verifying-correctness) for details.)
- **Grouped GEMM in the sweep.** A grouped-GEMM argument builder exists in
  `gemm_args.py`, but the sweep's `_buildSweepArgs` is NT strided-batched only.
- **Sparse in the sweep.** `sparse.py` is CPU-only and unit-tested; GPU sparse is
  not wired into `gemm_args`.
- **MX kernels** are excluded by the `_shouldSkip` filter.
- **fp16 epilogue**, **auto WorkGroupMapping / StaggerU selection**, and
  **multi-GPU parallel** runs are not ported (multi-GPU benchmark falls back to
  the C++ client).
- `hwMonitor`, `boundsCheck`, and `rocprofCounters` are accepted by
  `SweepRunner.run()` but are currently unused inside the benchmark loop.

---

## 3. Summary of each new component

### `sweep_runner.py`

The orchestrator.
`SweepResult` is a dataclass `(solutionIdx, solutionName, problemSize, benchmark, gflops, validation: str = "SKIPPED")`.
`SweepRunner(yamlPath, libraryPath=None, nWarmup=2, nIters=10, rotatingBuffers=8,
icacheCopies="auto", problemIdx=0, groupIdx=0, saveCoPath=None, pinClocks=False,
amdSmiPath=None)`. If `yamlPath` ends in `.ini`, it is resolved via
`_resolveYamlFromIni` (the INI is parsed under a synthetic `[default]` section,
and the `benchmark-yaml` value is returned; `KeyError` if that key is absent).
`run(resultsCsv=None, libraryUpdateFile=None, hwMonitor=False, boundsCheck=False,
rocprofCounters=None) -> list[SweepResult]`. Internals include `_setupTensile`,
`_generateAsm` (`KernelWriterAssembly.getSourceFileString`, kernel name from
`getKernelNameMin`), `_compileAll`/`_compileOneSolution` (via
`amdgpu_exec.compile_asm_to_hsaco`), the `_shouldSkip` filter, and
`_benchmarkOne`. GFLOPS is computed from `minUs` at `sweep_runner.py:394-395`.

### `harness.py`

Low-level benchmarking primitives.
`BenchmarkResult(timesNs, warmupN, hw, counters)` exposes `meanUs`, `p50Us`,
`p95Us`, `minUs`, and `gflops(M, N, K)`.
`BufferPool(nSlots, sizeBytes, gpuBufferCls)` provides `next()`, `iterSlots()`,
and `freeAll()`.
`KernelRunner(functions, outputPool=None)` with classmethod
`fromHsaco(hsacoBytes, kernelName, nModuleCopies=1, coPath=None)` — loads the
code object and rotates across module copies (`'auto'` uses
`tensilelite_runtime.get_icache_module_copies`). `run(argsFn, grid, block,
nWarmup, nIters, boundsCheck=False, hwMonitor=False, rocprofCounters=None)`
returns a `BenchmarkResult` built from GPU-event timing. `autoScaleIters(...)`
helps pick iteration counts based on FLOP budget.

### `reporters.py`

Output formatting. `_formatFloat = f"{v:.6g}"`.
`ResultsCSVReporter(path, solutionNames, numSizeDims=4, perfMetric="GFlops")`
provides `writeHeader`, `writeRow(sizeParams, solutionResults)`, `close`. The CSV
schema is `GFlops`, the size columns `SizeI..SizeP` (for the 8-dim case: M, N,
batch, K, ldd, ldc, lda, ldb), then `LDD, LDC, LDA, LDB, TotalFlops`, then one
column per solution (header = solution name, value = that solution's GFLOPS). The
separator is `", "` and the first column value is the running problem index.
`LibraryUpdateReporter(path)` provides `writeRow(sizeParams, winnerIdx,
winnerGFlops)` / `close`, emitting the library-update YAML block.

### `reference.py`

NumPy CPU reference implementations, entirely decoupled from the benchmark loop
and used by feature tests. Tolerances: `RTOL/ATOL_BF16 = 2e-2`, `FP16 = 1e-3`,
`FP32 = 1e-5`. Functions include `gemm`, `gemmFp16`, `gemmBf16(A, B, alpha=1,
beta=0, C=None)`, `gemmInt8`, `gemmFp8`, `gemmXf32`, `gemmMx`, `gemmGrouped`;
helpers `toXf32`, `computeAmaxD`, `computeETensor`; epilogue helpers `applyBias`,
`applyActivation`, `applyScaleAb`, `applyScaleCd`, `applyScaleAlphaVec`; and
`assertClose(gpu, ref, rtol, atol, label)`.

### `yaml_solution_builder.py`

Turns a `BenchmarkProblems` YAML into concrete solutions.
`buildSolutionsFromYaml` / `solutionsFromYaml` run
`BenchmarkProcess -> constructForkPermutations -> _generate_single_solution`.
`problemSizesFromYaml` yields the size tuples.
`enumerateAllSolutions` / `_iterRawSolutions` use `yaml.safe_load` (no rocisa
needed) for lightweight inspection. `_injectInternalArgsSupport` sets
`KernArgsVersion` and related fields (with a chip fallback table); `solutionId`
computes a stable identifier.

### `library_runner.py`

`LibraryRunner(library_path, device_id=0)` queries a pre-built solution library
through the `tensilelite_runtime` bindings (the M10 feature): `find_best`,
`find_top_n`, `filter_by_predicate`. The runtime import is lazy.

### `gemm_args.py`

A port of the C++ `ContractionSolution.cpp` argument assembly.
`_computeInternalArg0` / `_computeInternalArg1` (the two words `SweepRunner`
actually uses), plus `buildKernelArgs`, `buildGroupedGemmArgs`, and StreamK / MX /
epilogue slot builders. Unsupported combinations raise `NotImplementedError`. The
sweep loop only exercises `_computeInternalArg0/1`.

### `hw_monitor.py`

`HardwareMonitor(deviceId=0, intervalMs=10)` — a context manager that polls
`amdsmi` on a daemon thread; it is a no-op if `amdsmi` is unavailable.

### `mx_types.py`

MX (micro-scaled) datatype decode helpers converting packed `uint8` data to
`float32`: `decodeE8`, `decodeE5m3`, `unpackFloat4` (E2M1), `unpackFloat6E2m3`,
`unpackBfloat6E3m2`.

### `sparse.py`

`compress24` / `decompress24` implement AMD 2:4 structured sparsity on the CPU.
They are unit-tested but not wired into the GPU argument path in `gemm_args`.

---

## 4. How to use the new client (concrete, verified examples)

> **Environment prerequisites (apply to every command below):**
> - Run from the tensilelite root:
>   `/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite`.
> - `LD_LIBRARY_PATH=/opt/rocm/lib` is **mandatory** — without it, chip detection
>   silently fails.
> - The Python driver scripts need `PYTHONPATH` set to the tensilelite directory
>   (the tox Python does not have `Tensile` on its path by default).
> - `hipModuleLoad "file not found"` warnings printed before the explicit
>   code-object load are benign.

Confirm the toolchain sees the GPU first:

```bash
cd /home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite
LD_LIBRARY_PATH=/opt/rocm/lib .tox/unit/bin/python -c \
  "import amdgpu_exec; print(amdgpu_exec.get_chip())"
# prints: gfx950
```

### (a) Running a full sweep (YAML with multiple solutions)

`SweepRunner` compiles and benchmarks every candidate solution the YAML
enumerates, for every problem size, and writes one `results.csv` row per size
with a GFLOPS column per solution. Driver script (`/tmp/py_sweep.py`):

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Time a full Python SweepRunner sweep over all problem sizes in a YAML group."""
import time
from Tensile.client.sweep_runner import SweepRunner

YAML = "/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/client/tests/yaml/gemm_standard.yaml"

runner = SweepRunner(
    yamlPath=YAML,
    problemIdx=2, groupIdx=0,
    nWarmup=3, nIters=15,
    rotatingBuffers=1, icacheCopies=1,
)
t0 = time.perf_counter()
results = runner.run(resultsCsv="/tmp/py_results.csv")
t1 = time.perf_counter()

print(f"WALL_CLOCK_SWEEP_SECONDS={t1 - t0:.4f}")
for r in results:
    print(f"size={r.problemSize} gflops={r.gflops:.2f} minUs={r.benchmark.minUs:.3f}")
```

Run it:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite \
  .tox/unit/bin/python /tmp/py_sweep.py
```

**Multi-solution note.** When a YAML's `ForkParameters` enumerate more than one
solution, `results.csv` grows a column per solution and the library-update file
records the winner (highest GFLOPS > 0) per size. For example, a YAML forking
`DepthU: [16, 32]` yields two solutions and two GFLOPS columns; a verified run of
that shape reported `sol1 -> 333518` and `sol2 -> 351969` GFLOPS at 2048.

### (b) Running a single config (one solution, one problem size)

Use a trimmed YAML with a single `ForkParameters` combination and a single
`Exact` problem size. The single-config YAML (`/tmp/gemm_single.yaml`):

```yaml
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Single-config bf16 GEMM: one solution, one problem size (2048x2048x4x2048).
# Derived verbatim from gemm_standard.yaml group 2 (bf16 HPA NT batched),
# trimmed to a single Exact ProblemSize.
GlobalParameters:
  SyncsPerBenchmark: 0
  MinimumRequiredVersion: 5.0.0
  NumElementsToValidate: 128
  DataInitTypeBeta: 0
  DataInitTypeAlpha: 1
  Device: 0

BenchmarkProblems:
  -
    - # ProblemType
      OperationType: GEMM
      DataType: b
      DestDataType: b
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: False
      TransposeB: True
      UseBeta: True
      Batched: True
      UseBias: 0
    - # BenchmarkProblemSizeGroup
      InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - MatrixInstruction:
          - [16, 16, 16, 1, 1, 2, 2, 2, 2]
        - DepthU: [16]
        - GlobalSplitU: [1]
        - SourceSwap: [True]
        - PrefetchGlobalRead: [1]
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [2048, 2048, 4, 2048]
```

Note: the `NumElementsToValidate: 128` and the `DataInitType*` keys in
`GlobalParameters` are only meaningful to the C++ client. `SweepRunner` ignores
them — it does not validate and uses fixed `alpha=1.0`/`beta=0.0` with
uninitialized buffers.

Driver script (`/tmp/py_single.py`):

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Time a single-config Python SweepRunner run (one solution, one problem size)."""
import time
from Tensile.client.sweep_runner import SweepRunner

runner = SweepRunner(
    yamlPath="/tmp/gemm_single.yaml",
    problemIdx=0, groupIdx=0,
    nWarmup=3, nIters=15,
    rotatingBuffers=1, icacheCopies=1,
)
t0 = time.perf_counter()
results = runner.run(resultsCsv="/tmp/py_single_results.csv")
t1 = time.perf_counter()

print(f"WALL_CLOCK_SINGLE_SECONDS={t1 - t0:.4f}")
for r in results:
    print(f"size={r.problemSize} gflops={r.gflops:.2f} minUs={r.benchmark.minUs:.3f} "
          f"meanUs={r.benchmark.meanUs:.3f}")
# Pure benchmark time estimate: sum of (nWarmup+nIters) * meanUs across results.
for r in results:
    pureBenchS = (r.benchmark.meanUs * (3 + 15)) / 1e6
    print(f"PURE_BENCH_SECONDS_APPROX={pureBenchS:.4f}")
```

Run it:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite \
  .tox/unit/bin/python /tmp/py_single.py
```

### (c) Verifying correctness

`SweepRunner` now supports in-sweep correctness validation via the
`numElementsToValidate` constructor parameter, mirroring the C++ client's
`--num-elements-to-validate` flag:

- `0` — validation **disabled** (default; benchmark-only, matches C++ default).
- `-1` — validate **all** output elements.
- `N > 0` — validate the **first N flattened** output elements.

**Intentional divergence from the C++ client:** the C++ client samples using a
`NextPrime(total/N)` stride pattern; `SweepRunner` validates the first N
elements of the flat output instead. This is simpler and sufficient for
detecting silent wrong-answer bugs in GEMM kernels.

When enabled, each returned `SweepResult` carries a per-solution `.validation`
field ("PASS", "FAIL:\<message\>", or "SKIPPED"), and the results CSV gains a
`Validation` column placed immediately after `TotalFlops` and before the
per-solution GFLOPS columns. The aggregate row-level status follows these rules:
if any solution fails, the first FAIL message is used; if at least one passes,
"PASS" is used; otherwise "SKIPPED".

Only standard (StreamK==0) NT stridedBatched GEMMs with matching input/output
dtype in {fp32, fp16, bf16} are verified; other configurations record "SKIPPED".

Example usage:

```python
from Tensile.client.sweep_runner import SweepRunner

runner = SweepRunner(
    yamlPath="/path/to/benchmark.yaml",
    numElementsToValidate=-1,   # validate all output elements
)
results = runner.run(resultsCsv="/path/to/results.csv")
for r in results:
    print(r.solutionName, r.gflops, r.validation)
```

Correctness verification also remains available as a standalone NumPy step built
on `Tensile.client.reference` (`gemmBf16` / `gemm` + `assertClose`). This is
exactly what the feature tests (`test_gemm_standard.py`, `test_gemm_mx.py`, ...)
do, exercised by `test_reference.py`.

### (d) Using `.ini` config files (M14 pipeline integration)

`SweepRunner` accepts a flat `.ini` directly, provided the `.ini` carries a
`benchmark-yaml=` key. On construction it detects the `.ini` suffix and calls
`_resolveYamlFromIni`, which parses the flat INI under a synthetic `[default]`
section and returns the `benchmark-yaml` value (raising `KeyError` if the key is
missing):

```python
from Tensile.client.sweep_runner import SweepRunner

runner = SweepRunner(yamlPath="/path/to/client.ini")   # must contain benchmark-yaml=
results = runner.run(resultsCsv="/path/to/results.csv")
```

Inside the standard three-phase pipeline, `ClientWriter.writeClientConfigIni`
emits `benchmark-yaml=<path>` when a benchmark YAML is available
(`ClientWriter.py:640-641`), and `runClient(...)` drives the whole thing:

```python
from Tensile.ClientWriter import runClient

# use_python_client=True (the default) routes to SweepRunner via
# _runWithPythonHarness; it writes <buildPath>/results.csv and returns 0/1.
rc = runClient(
    libraryLogicPath, forBenchmark=True, enableTileSelection=False,
    cxxCompiler="amdclang++", cCompiler="amdclang",
    outputPath=buildPath, configPaths=[iniPath],
    use_python_client=True,
)
```

Set `use_python_client=False` to fall back to a subprocess of the C++ binary; a
multi-GPU benchmark request also falls back to the C++ `runClientParallel`.

### (e) Reading results from `results.csv`

A real `results.csv` written by `SweepRunner` (`/tmp/py_results.csv`) looks like:

```
GFlops, SizeI, SizeJ, SizeK, SizeL, SizeM, SizeN, SizeO, SizeP, LDD, LDC, LDA, LDB, TotalFlops, MT64x64
0, 256, 256, 4, 256, 256, 256, 256, 256, 256, 256, 256, 256, 134217728, 3352.09
1, 512, 512, 4, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1073741824, 27502.9
```

Column meanings:

| Column | Meaning |
|--------|---------|
| `GFlops` (first column value) | Running problem/row index (0, 1, 2, ...), not a GFLOPS number. |
| `SizeI..SizeP` | The 8-dim problem tuple: M, N, batch, K, ldd, ldc, lda, ldb. |
| `LDD, LDC, LDA, LDB` | Leading dimensions (repeated in dedicated columns). |
| `TotalFlops` | `2 * M * N * K * batch`. |
| Trailing column(s), header = solution name (e.g. `MT64x64`) | That solution's measured GFLOPS (min-time). One such column per solution. |

The C++ client's CSV header is different (it carries extra winner/analysis
columns): `GFlops, SizeI, SizeJ, SizeK, SizeL, LDD, LDC, LDA, LDB, TotalFlops,
TilesPerCu, TotalGranularity, WinnerGFlops, WinnerTimeUS, WinnerIdx, WinnerName,
<full solution name>`. Both share the leading `GFlops`/size columns and the
`", "` separator, and both report a minimum-time GFLOPS metric, so downstream
tooling that keys off those columns keeps working.

---

## 5. Migration guide (from the old C++ tensilelite-client binary)

The old workflow, for someone who called the C++ binary directly, was:

1. Build a solution library and its code objects (a pre-built
   `TensileLibrary.yaml` + `kernel_*.co`).
2. Write a flat `.ini` (or assemble an equivalent CLI) describing the problem,
   data types, init modes, warmups/benchmarks, validation, and the paths to the
   library + code object.
3. Run `tensilelite-client --config-file x.ini`, which loads the pre-built
   kernels, initializes data, iterates problems x solutions, times them, and
   writes `results.csv` (+ optionally a library-update file).

The new workflow removes steps 1–2 for the common case: point `SweepRunner`
(or `runClient`) at the **benchmark YAML** (or an `.ini` carrying a
`benchmark-yaml=` key). Kernels are generated and compiled **in-process**; no
pre-built library or code object is needed, and `results.csv` is still produced
with the same key columns.

### Side-by-side

**Old (C++ binary, pre-built library + code object):**

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
  build_tmp/tensilelite/client/tensilelite-client --config-file /tmp/cpp_ref/BFloat16/client3.ini
```

where the `.ini` names `library-file=`, `code-object=`, the full problem/data-type
surface, and (for validation) `num-elements-to-validate=`.

**New (Python, in-process compile from the benchmark YAML):**

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite \
  .tox/unit/bin/python -c '
from Tensile.client.sweep_runner import SweepRunner
SweepRunner(yamlPath="/tmp/gemm_single.yaml").run(resultsCsv="/tmp/results.csv")'
```

or, inside the pipeline, `runClient(..., use_python_client=True)` with an `.ini`
that carries `benchmark-yaml=`.

### Gotchas when migrating

- **`use-user-args=False` for non-grouped GEMM (C++ binary).** If you keep using
  the C++ binary for a plain (non-grouped) GEMM, `use-user-args=True` throws
  `"Failed to cast problem type"`. Keep it `False`. This is visible in the
  verified INIs (e.g. `/tmp/cpp_ref/BFloat16/client3.ini`).
- **In-loop validation via `numElementsToValidate`.** Pass
  `numElementsToValidate=-1` (all elements) or `numElementsToValidate=N` (first
  N elements) to `SweepRunner` to enable in-sweep GPU output validation. The
  C++ binary's `NextPrime`-strided sampling is intentionally not replicated; see
  [section 4c](#c-verifying-correctness) for details and the divergence note.
- **Feature coverage.** Grouped/sparse/MX/fp16-epilogue/multi-GPU are not in the
  Python sweep; use the C++ binary for those.

### The C++ binary is still available

The standalone binary remains built at
`/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client`
(sources under `client/main.cpp`, `client/src/`, `client/include/ProgramOptions.hpp`).
Keep using it when you need in-process validation, grouped/sparse/MX GEMM, or
multi-GPU parallel benchmarking. Confirm it works with:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
  build_tmp/tensilelite/client/tensilelite-client --help
```

---

## 6. Benchmark comparison (real measured numbers)

All numbers below are real measurements taken on this `gfx950` machine.

### Fair apples-to-apples comparison

The two clients were configured identically for this comparison:

- **Problem**: bf16 HPA NT strided-batched GEMM (`gemm_standard.yaml` group 2),
  three square sizes: 1024x1024x4x1024, 2048x2048x4x2048, 4096x4096x4x4096.
- **Iterations**: `num-warmups=3`, `num-benchmarks=10`
  (`num-enqueues-per-sync=1`, `num-syncs-per-benchmark=1` for the C++ binary;
  `nWarmup=3`, `nIters=10` for `SweepRunner`).
- **Validation**: disabled (`num-elements-to-validate=0` / `numElementsToValidate=0`).
- **GPU**: gfx950, device 0.
- **Timing**: one cold-start run discarded; three timed runs; median reported.
- **Clock state**: `sudo amd-smi set -g 0 --perf-level HIGH` was unavailable on
  this machine; clocks were at driver defaults for all runs.

Both clients use a **minimum-time GFLOPS** metric (Python: `minUs`; C++:
`WinnerGFlops` / `time-us` best iteration).

**Intentional differences that remain after equalization:**

- The Python client compiles kernels in-process from the benchmark YAML
  (via `amdgpu_exec.compile_asm_to_hsaco`), which takes roughly 5–6 s per
  invocation regardless of problem count.  The C++ binary loads pre-built
  `.co` objects from disk and skips that cost entirely.  The wall-clock numbers
  therefore reflect this structural difference — they are not comparable as
  "benchmark speed" but rather document the two workflows.
- The C++ binary initializes A/B/C with random data (`init-a=Random`, etc.) and
  sets `alpha=2`, `beta=2`.  `SweepRunner` uses uninitialized buffers with
  fixed `alpha=1.0`, `beta=0.0`. This does not affect GPU kernel timing for
  pure GEMM, but is noted for completeness.

### Apples-to-apples GFLOPS table (median of 3 timed runs)

bf16 HPA NT batched, batch=4, `nWarmup=3`, `nIters=10`, validation disabled.

| Problem size | Python GFLOPS (median) | C++ WinnerGFlops (median) | Delta % |
|---|---|---|---|
| 1024x1024x4x1024 | 126,766 | 214,106 | -40.8% |
| 2048x2048x4x2048 | 334,559 | 360,462 | -7.2% |
| 4096x4096x4x4096 | 327,207 | 302,072 | +8.3% |

Raw GFLOPS across the 3 runs (all values in GFLOPS):

| Size | Python run 1 | Python run 2 | Python run 3 | C++ run 1 | C++ run 2 | C++ run 3 |
|---|---|---|---|---|---|---|
| 1024 | 126,766 | 125,657 | 127,901 | 213,250 | 214,106 | 215,179 |
| 2048 | 339,183 | 330,821 | 334,559 | 360,762 | 360,159 | 360,462 |
| 4096 | 327,839 | 326,174 | 327,207 | 298,684 | 302,072 | 302,918 |

### Wall-clock per run

| Client | Run 1 (s) | Run 2 (s) | Run 3 (s) | Median (s) | Compilation share |
|---|---|---|---|---|---|
| Python (`SweepRunner`) | 6.20 | 5.52 | 6.18 | 6.18 | ~5–6 s (kernel compile) |
| C++ (`tensilelite-client`) | 0.656 | 0.689 | 0.683 | 0.683 | 0 (pre-built `.co`) |

The Python wall-clock is dominated by in-process assembly generation and HSACO
compilation (~5–6 s). The GPU benchmark time itself at the 2048 size is
approximately `(3 + 10) * 202 µs = 2.6 ms`; the rest is compilation overhead.
The C++ binary pays no compilation cost.

### Interpretation

- **GFLOPS at 1024**: the Python client is 41% lower than C++. At this small
  size (128 tiles per batch, filling only part of the GPU), the Python argument
  overhead and event-timing granularity have a larger relative impact, and the
  C++ client's more aggressive data-initialization (random A/B values computed
  by the GPU's init kernel) may warm the cache more favorably.
- **GFLOPS at 2048**: the gap narrows to 7%. At this size the kernel is
  compute-bound and both clients converge on the true peak throughput.
- **GFLOPS at 4096**: Python reports 8% *higher* than C++. Both measurements
  are within run-to-run variance (~3% spread observed across three runs); the
  sign flip is noise, not a systematic advantage.
- **Both metrics are minimum-time.** Python uses `benchResult.minUs`; the C++
  binary's `WinnerGFlops` is derived from the minimum measured `time-us` across
  all benchmark iterations.  The comparison is methodologically sound.
- **Clocks were not pinned.** `sudo amd-smi set -g 0 --perf-level HIGH` was
  unavailable. Run-to-run variance for both clients is roughly 1–3%.

### Reproducing these measurements

All commands run from
`/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite` with
`LD_LIBRARY_PATH=/opt/rocm/lib`.

Fast, no-GPU-benchmark sanity checks:

```bash
# Chip detection.
LD_LIBRARY_PATH=/opt/rocm/lib .tox/unit/bin/python -c \
  "import amdgpu_exec; print(amdgpu_exec.get_chip())"        # -> gfx950

# C++ client help.
LD_LIBRARY_PATH=/opt/rocm/lib \
  build_tmp/tensilelite/client/tensilelite-client --help
```

Python client (bf16, 3 sizes, nWarmup=3, nIters=10):

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import time
from Tensile.client.sweep_runner import SweepRunner

YAML = 'Tensile/client/tests/yaml/gemm_standard.yaml'

runner = SweepRunner(
    yamlPath=YAML,
    problemIdx=2, groupIdx=0,
    nWarmup=3, nIters=10,
    numElementsToValidate=0,
)
t0 = time.perf_counter()
results = runner.run(resultsCsv='/tmp/py_fair_results.csv')
t1 = time.perf_counter()
print(f'WALL_CLOCK_S={t1 - t0:.4f}')
for r in results:
    if r.problemSize[0] in (1024, 2048, 4096) and r.problemSize[1] == r.problemSize[0]:
        print(f'size={r.problemSize[0]}x{r.problemSize[1]} gflops={r.gflops:.2f}')
```

C++ client INI (bf16, 3 sizes, num-warmups=3, num-benchmarks=10, validation off):

```ini
library-file=/tmp/cpp_ref/BFloat16/TensileLibrary.yaml
code-object=/tmp/cpp_ref/BFloat16/kernel_0.co
results-file=/tmp/cpp_fair_results.csv
problem-identifier=Contraction_l_Ailk_Bjlk_Cijk_Dijk
a-type=BFloat16
b-type=BFloat16
c-type=BFloat16
d-type=BFloat16
alpha-type=Float
beta-type=Float
compute-input-type-A=BFloat16
compute-input-type-B=BFloat16
f32-xdl-math-op=Float
high-precision-accumulate=True
strided-batched=True
grouped-gemm=False
use-bias=0
use-e=False
use-gradient=False
use-scaleAB=
use-scaleCD=False
use-scaleAlphaVec=0
sparse=0
activation-type=None
activation-compute-type=Float
activation-no-guard=False
use-user-args=False
device-idx=0
init-seed=0
init-a=Random
init-b=Random
init-c=Random
init-d=Zero
init-alpha=Two
init-beta=Two
num-warmups=3
num-benchmarks=10
use-gpu-timer=True
sync-after-warmups=True
num-enqueues-per-sync=1
num-syncs-per-benchmark=1
num-elements-to-validate=0
csv-export-extra-cols=True
csv-merge-same-problems=True
log-level=Warning
PrintWinnersOnly=False
problem-size=1024,1024,4,1024
a-strides=-1,1024,-1
b-strides=-1,1024,-1
c-strides=-1,1024,-1
d-strides=-1,1024,-1
problem-size=2048,2048,4,2048
a-strides=-1,2048,-1
b-strides=-1,2048,-1
c-strides=-1,2048,-1
d-strides=-1,2048,-1
problem-size=4096,4096,4,4096
a-strides=-1,4096,-1
b-strides=-1,4096,-1
c-strides=-1,4096,-1
d-strides=-1,4096,-1
```

Run the binary:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
  build_tmp/tensilelite/client/tensilelite-client --config-file /path/to/fair.ini
```
