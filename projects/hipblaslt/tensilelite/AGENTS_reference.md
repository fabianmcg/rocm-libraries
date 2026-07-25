# TensileLite Reference

Supplementary reference for `AGENTS.md` — load this when you need test commands, custom builds, linting, CMake options, or supported targets.

## Python Harness (primary testing path for M1–M13)

The Python client harness (`Tensile/client/`) provides assembly-level kernel compilation,
benchmarking, and CSV reporting that match the C++ tensilelite-client in output format.
Use it for development, CI, and parity validation — it does not require a pre-built
solution library (compiles kernels from YAML via the Tensile assembler on demand).

```bash
# All harness tests (unit + GPU)
tox -e unit

# Only profiler counter tests (requires tensilelite_profiler C extension)
tox -e unit -k requires_rocprof

# Generate parity report after running parity tests
tox -e unit -- Tensile/client/tests/test_parity.py --generate-parity-report

# Parity report is written to Tensile/client/parity_report.md
```

### SweepRunner — benchmark all solutions in a YAML

```python
from Tensile.client.sweep_runner import SweepRunner

runner = SweepRunner(
    yamlPath="Tensile/client/tests/yaml/gemm_standard.yaml",
    nWarmup=3,
    nIters=15,
    rotatingBuffers=8,
    icacheCopies="auto",
    problemIdx=2,   # bf16 HPA group
    groupIdx=0,
)
results = runner.run(
    resultsCsv="results.csv",
    libraryUpdateFile="library_update.yaml",
)
# results is a list of SweepResult(solutionIdx, solutionName, problemSize, benchmark, gflops)
```

### LibraryRunner — dispatch via a pre-built production library

```python
from Tensile.client.library_runner import LibraryRunner

runner = LibraryRunner(
    libraryPath="path/to/TensileLibrary.yaml",
    coPath="path/to/TensileLibrary_gfx950.co",
)
# Find best solution for a problem and benchmark it
result = runner.run(problemSize=(1024, 1024, 4, 1024))
```

## Running Tests

```bash
# Full test suite (builds client + runs all common tests)
tox -e py3 -- Tensile/Tests -m common

# Python unit tests only (skips the long client build; requires a prior build)
tox -e unit -- Tensile/Tests/unit

# Run a specific test category
tox -e py3 -- Tensile/Tests -m gemm

# Run a single test directly (after a prior `invoke build-client`)
Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out
```

## Custom CMake Build

```bash
cmake --preset tensilelite -S .. -B my-custom-build
cmake --build my-custom-build --parallel

# Run test with custom client path
./my-custom-build/Tensile.sh Tensile/Tests/common/<test>.yaml tensile-out \
    --prebuilt-client=my-custom-build/tensilelite-client/tensilelite-client

# Build with custom args (e.g., Debug + specific GPU)
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox -e py3 -- Tensile/Tests -m common
```

Iterate on rocisa C++ without re-pip-installing:

```bash
cd rocisa && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ ..
make -j8
```

`invoke build-client` accepts `--clean`, `--build-dir`, `--build-type`, `--gpu-targets`, `--rocm-path`, `--export-compile-commands`, `--bundle-python-deps`, `--enable-rocprof`, `--cxx-flags-release`. See `tasks.py`.

## Linting and Formatting

```bash
tox -e lint          # flake8 (pyflakes errors only, E/W ignored)
tox -e format        # black (line-length=100) on Common/, TensileCreateLibrary/, Utilities/Decorators/
tox -e isort         # isort (black profile) on same directories
```

## Rebuilding Assembly Without Full Rerun

After a Tensile run creates `tensile-out/`, you can edit assembly and rebuild only object code:

```bash
make co TENSILE_OUT=tensile-out                          # auto-detect arch
make co TENSILE_OUT=tensile-out ARCH="gfx942" WAVE=64   # gfx9 explicit
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32  # gfx11 explicit
```

## CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `TENSILELITE_ENABLE_HOST` | ON | Build C++ runtime library |
| `TENSILELITE_ENABLE_CLIENT` | ON | Build benchmark client |
| `TENSILELITE_ENABLE_AUTOBUILD` | OFF | Auto-rebuild rocisa wrapper scripts |
| `TENSILELITE_BUILD_TESTING` | OFF | Build C++ host library tests |
| `GPU_TARGETS` | (detected) | Semicolon-separated list of gfx targets |

## Supported Targets

GPU architectures (see `Tensile/Common/Architectures.py`): gfx900, gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1010/1011/1012, gfx1030, gfx1100/1101/1102, gfx1200/1201, gfx1250 (each with optional `:xnack+/-`).

Test markers for architectures (see `pytest.ini`): `gfx11`, `gfx12`, `gfx94x`, `gfx950`, `gfx1250`, plus per-arch `xfail-gfxNNN` / `skip-gfxNNN`. Data type markers: `Float`, `Double`, `Half`, `BFloat16`, `Int8`, `Float8`/`BFloat8` (OCP and `_fnuz` NANOO variants), mixed `Float8BFloat8`, `Float4`, `Float6`, `BFloat6`.
