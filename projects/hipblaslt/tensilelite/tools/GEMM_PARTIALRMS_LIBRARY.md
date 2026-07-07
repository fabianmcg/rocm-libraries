# Building a TensileLite device library for the GEMM+PartialRMS kernel

This document explains how to compile the GEMM+PartialRMS (K1) kernel into a
TensileLite device library (`.hsaco` / `.co` / `.dat`) that hipBLASLt can load
at runtime. It covers the two-step process: generating a LibraryLogic YAML from
the known solution parameters, then invoking `TensileCreateLibrary` to compile it.

---

## Background

At build time, hipBLASLt compiles kernels via `TensileCreateLibrary`, which takes
**LibraryLogic YAML** files as input (already-selected solutions, not benchmark
configs) and produces:

| Artifact | Purpose |
|---|---|
| `Kernels.so-000-<arch>.hsaco` | Compiled AMDGPU ELF containing all kernels |
| `TensileLibrary_..._<arch>.co` | Clang-offload-bundled code object |
| `TensileLibrary_..._<arch>.dat.zlib` | msgpack solution library for runtime dispatch |
| `TensileLibrary_lazy_<arch>.dat.zlib` | Lazy-load dispatch index |
| `TensileLiteLibrary_lazy_<arch>_Mapping.dat.zlib` | Lazy filename mapping |

The LibraryLogic YAML is a 12-element YAML sequence encoding the kernel
parameters, the problem type, and a size→solution selection table.

---

## Prerequisites

- `~/.tensile` venv with `rocisa` and `Tensile` importable (run `invoke rocisa`
  once if not already installed).
- `amdclang++` on PATH (provided by ROCm).
- A `gfx950` GPU is not required for compilation, only for running the kernel.

---

## Step 1: Generate the LibraryLogic YAML

`tools/gen_partialrms_logic.py` builds the K1 solution from
`tools/gemm_partial_rms_k1.yaml`, converts its state to YAML-safe primitives,
and writes a LibraryLogic YAML that `TensileCreateLibrary` can process.

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite

python tools/gen_partialrms_logic.py \
    --chip gfx950 \
    --out-dir /tmp/partialrms_logic
```

This writes:

```
/tmp/partialrms_logic/
  gfx950/
    Equality/
      PartialRMS_BF16_TN.yaml
```

The YAML contains one solution entry for the 256×256×64 MFMA bf16 TN GEMM with
PartialRMS epilogue (`MT256x256x64`, `StreamK=3`, `PartialRMS=True`).

### What the script does

1. Calls `build_k1_solution('gfx950', ...)` using the YAML config.
2. Extracts the fully-expanded solution state dict via `sol.getKernels()[0]`.
3. Converts Tensile-typed objects (`DataType`, `ActivationType`,
   `SemanticVersion`, `InternalSupportParams`) to plain Python primitives.
4. Applies three type coercions required by `TensileCreateLibrary`:
   - `BufferStore` → `bool`
   - `GlobalReadPerMfma` → `float`
   - `StaggerUStride` → `int`
5. Writes the 12-element LibraryLogic list and verifies it with a YAML
   round-trip check.

---

## Step 2: Compile with TensileCreateLibrary

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite

mkdir -p /tmp/partialrms_output

python -m Tensile.TensileCreateLibrary \
    --architecture=gfx950 \
    --cxx-compiler=$(which amdclang++) \
    --jobs=$(nproc) \
    /tmp/partialrms_logic/ \
    /tmp/partialrms_output/ \
    HIP
```

Typical output:

```
TensileCreateLibrary
...
no type mismatches found
1 unique solution
1 kernel processed
```

Compilation takes approximately 6 seconds for one kernel.

### Output layout

```
/tmp/partialrms_output/
  library/
    gfx950/
      Kernels.so-000-gfx950.hsaco          (~174 KB)
      TensileLibrary_..._gfx950.co         (~17 KB)
      TensileLibrary_..._gfx950.dat.zlib   (~2.2 KB)
      TensileLibrary_lazy_gfx950.dat.zlib
      TensileLiteLibrary_lazy_gfx950_Mapping.dat.zlib
```

---

## Step 3: Verify (optional, requires gfx950 GPU)

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite

python tools/bench_gemm_rms.py
# Expected: "verification PASSED", ~870+ TFLOPS K1, ~530 GB/s epilogue
```

The existing unit tests also exercise the kernel end-to-end:

```bash
pytest Tensile/Tests/unit/test_gemm_partial_rms.py -v          # 490 tests
pytest Tensile/Tests/unit/test_gemm_partial_rms_epilogue.py -v  # 7 tests
```

---

## Kernel parameters

The K1 kernel is defined in `tools/gemm_partial_rms_k1.yaml`. Key parameters:

| Parameter | Value |
|---|---|
| Data type | bf16 (TN GEMM, `DataType=b`) |
| MacroTile | 256×256 |
| DepthU | 64 |
| MatrixInstruction | `[16, 16, 32, 1, 1, 4, 8, 4, 2]` (MI16x16, wg0=4, wg1=2) |
| StreamK | 3 (persistent kernel with StreamK atomic reduction) |
| StreamKForceDPOnly | 1 |
| PartialRMS | True (subtile PartialRMS epilogue enabled) |
| UseSubtileImpl | True |

To build with a different `wg_n` (controls MT1 and the number of N-tiles):

```bash
python tools/bench_gemm_rms.py --wg-n 1   # MT1=64
python tools/bench_gemm_rms.py --wg-n 2   # MT1=128 (default)
```

---

## How hipBLASLt invokes TensileCreateLibrary at build time

hipBLASLt's CMake build (`cmake/HipBLASLtCodegen.cmake`) chains two custom
commands:

```bash
# Step 1: validate logic files
python -m Tensile.TensileLogic <LOGIC_PATH> --known-bugs known_bugs.yaml --check-all

# Step 2: compile kernels
python -m Tensile.TensileCreateLibrary \
    --architecture=gfx942;gfx90a \
    --cxx-compiler=$(which amdclang++) \
    --jobs=$(nproc) \
    [--logic-filter="gfx942/Equality/*"] \
    projects/hipblaslt/library/ \
    <build>/Tensile/ \
    HIP
```

The logic YAMLs under `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/`
were produced from offline benchmarking runs and committed to the repository.
`TensileCreateLibrary` only compiles them — it does not benchmark or select solutions.

The recommended build path is `invoke build -a gfx942 -f 'gfx942/Equality/*'`
from `projects/hipblaslt`, which handles the venv, rocisa install, and
`--logic-filter` automatically.
