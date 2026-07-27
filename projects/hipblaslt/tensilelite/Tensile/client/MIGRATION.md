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

- **In-loop validation.** `SweepRunner` performs no correctness check: it
  allocates uninitialized buffers and never reads back `D`. There is no
  `--num-elements-to-validate`, no in-sweep bounds-check, and no
  print/dump-tensor path. (Correctness is a separate NumPy step — see
  [section 4c](#c-verifying-correctness-numpy-only-separate-from-the-sweep).)
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
`SweepResult` is a dataclass `(solutionIdx, solutionName, problemSize, benchmark, gflops)`.
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

### (c) Verifying correctness (NumPy only, separate from the sweep)

This is the single most important behavioral difference to understand.
**`SweepRunner` is benchmark-only. It never checks results.** It allocates
uninitialized device buffers and never reads back the output `D`. There is no
`--num-elements-to-validate`, `--verify-all`, or bounds-check flag in
`SweepRunner`.

Correctness verification is a **separate NumPy step** built on
`Tensile.client.reference`: compute the reference GEMM on the CPU
(`gemmBf16` / `gemm`) and compare with `assertClose`. This is exactly what the
feature tests (`test_gemm_standard.py`, `test_gemm_mx.py`, ...) do, exercised by
`test_reference.py`.

The `--num-elements-to-validate` concept belongs to the **C++ client**:

- `-1` = validate **all** output elements (stride stays 1).
- `128` = validate roughly 128 sampled elements (stride = `NextPrime(total/128)`).
- `0` = validation **disabled** (the C++ default).

The following NumPy driver (`/tmp/py_verify.py`) demonstrates both the
"verify all" and "verify 128" modes on the host. It computes the bf16 reference
and compares it against a stand-in output array, so it isolates the reference +
comparison cost (it does not launch the GPU kernel):

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Time the Python-side numpy correctness verification cost: all vs 128 elements."""
import time
import numpy as np
import ml_dtypes
from Tensile.client import reference

M = N = K = 2048
rng = np.random.default_rng(0)
A = rng.standard_normal((M, K)).astype(ml_dtypes.bfloat16)
B = rng.standard_normal((K, N)).astype(ml_dtypes.bfloat16)  # (K, N)
C = np.zeros((M, N), dtype=ml_dtypes.bfloat16)
alpha, beta = 1.0, 0.0

# Stand-in GPU result identical to the reference, so assertClose passes.
Dref_bootstrap = reference.gemmBf16(A, B, alpha, beta, C)
gpuOut = np.array(Dref_bootstrap)

# ---- Mode 1: verify ALL elements ----
t0 = time.perf_counter()
Dref = reference.gemmBf16(A, B, alpha, beta, C)
reference.assertClose(gpuOut, Dref, reference.RTOL_BF16, reference.ATOL_BF16, "D")
t1 = time.perf_counter()
print(f"VERIFY_ALL_SECONDS={t1 - t0:.4f}  (elements={M*N})")

# ---- Mode 2: verify only 128 sampled output elements ----
nSample = 128
flatIdx = np.linspace(0, M * N - 1, nSample, dtype=np.int64)
rows = flatIdx // N
cols = flatIdx % N
t0 = time.perf_counter()
Af = A.astype(np.float32)
Bf = B.astype(np.float32)
refSample = np.empty(nSample, dtype=np.float32)
for i in range(nSample):
    refSample[i] = np.dot(Af[rows[i], :], Bf[:, cols[i]])
refSample = (alpha * refSample).astype(ml_dtypes.bfloat16)
gpuSample = gpuOut[rows, cols]
reference.assertClose(gpuSample, refSample, reference.RTOL_BF16, reference.ATOL_BF16, "D128")
t1 = time.perf_counter()
print(f"VERIFY_128_SECONDS={t1 - t0:.4f}  (elements={nSample})")
```

Run it:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite \
  .tox/unit/bin/python /tmp/py_verify.py
```

If you need in-process, in-loop GPU output validation against a reference, that
capability currently lives only in the C++ client
(`--num-elements-to-validate`); it has not been ported into `SweepRunner`.

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
- **No in-loop validation in the Python sweep.** If your old flow relied on
  `num-elements-to-validate`, keep validating with the C++ binary, or verify
  separately with NumPy (`Tensile.client.reference`); see
  [section 4c](#c-verifying-correctness-numpy-only-separate-from-the-sweep).
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

All numbers below are real measurements taken on this `gfx950` machine. Two
verification modes are referenced throughout:

- **"verify all"** = the C++ client validates **every** output element
  (`num-elements-to-validate=-1`). For the NumPy step, it means computing and
  comparing the full output tensor.
- **"verify 128"** = the C++ client validates roughly **128 sampled** elements
  (`num-elements-to-validate=128`, stride = `NextPrime(totalAllocated/128)`).
  For the NumPy step, 128 output elements are sampled along a linear index.

### Timing table

| Scenario | Client | Verify mode | Wall-clock (s) | GFLOPS |
|----------|--------|-------------|----------------|--------|
| Single config 2048x2048x4x2048 | Python (SweepRunner) | none (bench-only) | 6.08 (compile-dominated); pure bench ~0.004 | 329808 (min-time) |
| Single config 2048x2048x4x2048 | C++ | validate ALL (-1) | 3.52 / 3.62 (two runs) | WinnerGFlops ~361066 / 360689 |
| Single config 2048x2048x4x2048 | C++ | validate 128 | 0.57 / 0.73 (two runs) | WinnerGFlops ~361295 / 360992 |
| NumPy verification 2048x2048 (4.19M elems) | Python (reference) | verify ALL | 0.0879 | n/a |
| NumPy verification 2048x2048 (128 sampled) | Python (reference) | verify 128 | 0.0038 | n/a |
| Sweep 7 sizes (256..4096 + 2 non-square) bf16 | Python (SweepRunner) | none | 6.09 (compile-dominated) | per-size below |
| Sweep 3 sizes (1024/2048/4096) | C++ | validate ALL (-1) | 23.46 | 1024->216039, 2048->360916, 4096->281921 |
| Bonus multi-solution sweep (DepthU [16,32], 2 sols, 2048) | Python | none | 6.21 | sol1->333518, sol2->351969 |

Python 7-size sweep per-size GFLOPS (min-time): 256->3352, 512->27503,
1024->131505, 2048->338250, 4096->328199, 256x512->7498, 512x256->7566.

### Interpretation

- **Python wall-clock is compilation-dominated.** Both the single-config
  (~6.08 s) and 7-size sweep (~6.09 s) wall-clocks are dominated by in-process
  kernel compilation (~6 s), not by the benchmark itself: the pure GPU benchmark
  for the single config is ~0.004 s. The C++ timings **exclude** compilation
  because they load a pre-built `.co`.
- **C++ full validation is single-threaded CPU** and grows with problem size.
  This is why the 3-size C++ sweep takes 23.46 s — the 4096 case dominates. For
  the single config, "verify all" vs "verify 128" is ~3.5 s vs ~0.6 s (roughly
  6x).
- **NumPy verification is much faster** because it uses BLAS-threaded matmul:
  ~0.088 s (all) vs ~0.0038 s (128), roughly 23x.
- **The GFLOPS numbers differ between clients** (Python min-time ~330k vs C++
  `WinnerGFlops` ~361k at 2048). Both are real; the gap comes from measurement
  and argument-setup differences, not correctness. Both use a minimum-time
  metric.
- **Caveats.** Thermal variance is roughly +/-10% and clocks were not pinned
  (hence the two-run spreads shown for the C++ single-config rows). `/usr/bin/time`
  is not installed on this machine, so a `perf_counter` wrapper was used for
  wall-clock timing.

### Reproducing these measurements

All commands run from
`/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite` with
`LD_LIBRARY_PATH=/opt/rocm/lib`.

Fast, no-GPU-benchmark sanity checks (re-verified for this document):

```bash
# Chip detection.
LD_LIBRARY_PATH=/opt/rocm/lib .tox/unit/bin/python -c \
  "import amdgpu_exec; print(amdgpu_exec.get_chip())"        # -> gfx950

# Reporter unit tests (no GPU).
LD_LIBRARY_PATH=/opt/rocm/lib .tox/unit/bin/python -m pytest \
  Tensile/client/tests/test_sweep_runner.py -k "Reporter" -q # -> 10 passed, 6 deselected

# C++ client help.
LD_LIBRARY_PATH=/opt/rocm/lib \
  build_tmp/tensilelite/client/tensilelite-client --help
```

Python client runs (GPU):

```bash
# Single config.
LD_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite \
  .tox/unit/bin/python /tmp/py_single.py

# Full sweep.
LD_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite \
  .tox/unit/bin/python /tmp/py_sweep.py

# NumPy verification (all vs 128), no GPU kernel launched.
LD_LIBRARY_PATH=/opt/rocm/lib \
PYTHONPATH=/home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite \
  .tox/unit/bin/python /tmp/py_verify.py
```

C++ client timed runs. The wrapper `/tmp/run_cpp_timed.py` runs the binary via
`subprocess`, times it with `perf_counter`, and prints the returncode,
wall-clock, any validation/PASS/FAIL lines, and the CSV rows. Its essential body:

```python
# /tmp/run_cpp_timed.py (essential contents)
CLIENT = ".../build_tmp/tensilelite/client/tensilelite-client"
env = dict(os.environ); env["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
t0 = time.perf_counter()
res = subprocess.run([CLIENT, "--config-file", iniPath],
                     capture_output=True, text=True, env=env, timeout=1200)
t1 = time.perf_counter()
# prints RETURNCODE, WALL_CLOCK_SECONDS=(t1-t0), validation lines, and
# per-row "CSV size=(...) GFlops=... WinnerGFlops=..." parsed from resultsCsv.
```

Invoke it (or run the binary directly):

```bash
# Single config, validate ALL (-1).
LD_LIBRARY_PATH=/opt/rocm/lib .tox/unit/bin/python /tmp/run_cpp_timed.py \
  /tmp/cpp_single_validate_all.ini /tmp/cpp_single_all_results.csv single-all

# Single config, validate 128.
LD_LIBRARY_PATH=/opt/rocm/lib .tox/unit/bin/python /tmp/run_cpp_timed.py \
  /tmp/cpp_single_validate_128.ini /tmp/cpp_single_128_results.csv single-128

# 3-size sweep, validate ALL (-1).
LD_LIBRARY_PATH=/opt/rocm/lib .tox/unit/bin/python /tmp/run_cpp_timed.py \
  /tmp/cpp_sweep_validate_all.ini /tmp/cpp_sweep_all_results.csv sweep-all

# Equivalent direct invocation.
LD_LIBRARY_PATH=/opt/rocm/lib \
  build_tmp/tensilelite/client/tensilelite-client --config-file /tmp/cpp_single_validate_all.ini
```

The two single-config C++ INIs are identical except for one line —
`/tmp/cpp_single_validate_all.ini` sets `num-elements-to-validate=-1` (validate
all) while `/tmp/cpp_single_validate_128.ini` sets `num-elements-to-validate=128`
(validate ~128 sampled). Both point at the pre-built
`/tmp/cpp_ref/BFloat16/TensileLibrary.yaml` and
`/tmp/cpp_ref/BFloat16/kernel_0.co`. For reference, the full "validate all" INI:

```ini
library-file=/tmp/cpp_ref/BFloat16/TensileLibrary.yaml
code-object=/tmp/cpp_ref/BFloat16/kernel_0.co
results-file=/tmp/cpp_single_all_results.csv
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
num-benchmarks=15
use-gpu-timer=True
sync-after-warmups=True
num-enqueues-per-sync=1
num-syncs-per-benchmark=1
num-elements-to-validate=-1
csv-export-extra-cols=True
csv-merge-same-problems=True
log-level=Warning
PrintWinnersOnly=False
problem-size=2048,2048,4,2048
a-strides=-1,2048,-1
b-strides=-1,2048,-1
c-strides=-1,2048,-1
d-strides=-1,2048,-1
```

The 3-size sweep "validate ALL" row in the timing table was measured with
`/tmp/cpp_sweep_validate_all.ini`. It is the same shape as the single-config INI
shown above (same `library-file`, `code-object`, data types, and
`use-user-args=False`), sets `num-elements-to-validate=-1`, and lists three
`problem-size=` blocks with their strides:

```ini
num-elements-to-validate=-1
...
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

Note: the pre-existing `/tmp/cpp_ref/BFloat16/client3.ini` has the same three
problem sizes but **no** `num-elements-to-validate` line, so the C++ default of
`0` (validation disabled) applies — it is not equivalent to the "validate ALL"
row. Add `num-elements-to-validate=-1` to enable full validation.
