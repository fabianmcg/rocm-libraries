# TensileLite + amdgpu-exec Integration Guide

This document explains how to integrate **TensileLite** (AMD GPU kernel generator) with
**amdgpu-exec** (self-contained GPU kernel executor) to compile and run a GEMM kernel directly
from Python, without any external ROCm toolchain paths.

## Overview

| Component | Role |
|-----------|------|
| **TensileLite** | Generates a GCN assembly string (`.s` text) from a kernel configuration |
| **amdgpu-exec** | Compiles the assembly to an HSA code object and executes it via HIP |

amdgpu-exec ships with an embedded assembler and linker — it requires no external `amdclang++`,
no `ROCM_PATH`, and no ROCm installation path. You give it an assembly string and a chip name;
it gives back nanosecond timings.

## Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │  TensileLite (assembly generation)                       │
  │                                                          │
  │  chip string                                             │
  │      │                                                   │
  │      ▼                                                   │
  │  make_isa_info(chip)                                     │
  │    probe caps via amdgpu_exec.compile_asm()  ──────────┐ │
  │    inject into rocisa singleton via setData()           │ │
  │      │                                        amdgpu-exec│
  │      ▼                                        embedded   │
  │  Solution(config, isaInfoMap)                 assembler  │
  │    validate tile parameters                   (one-way)  │
  │    derive MacroTile, NumThreads, etc.                   │ │
  │      │                                        ◄──────────┘ │
  │      ▼                                                   │
  │  KernelWriterAssembly.getSourceFileString()              │
  │    emit GCN instructions via rocisa                      │
  │      │                                                   │
  │      └─────── asm_str ──────────────────────────────────┤
  │                                                          │
  └──────────────────────────────────────────────────────────┘
                        │
                        ▼  (raw .s text)
  ┌─────────────────────────────────────────────────────────┐
  │  amdgpu-exec (compile + execute)                         │
  │                                                          │
  │  compile_asm_to_hsaco(asm_str, chip)                     │
  │    → assemble: asm_str → ELF object (embedded clang)     │
  │    → link: ELF → HSA code object (.hsaco bytes)          │
  │                                                          │
  │  execute_hsaco(hsaco, kernel_name, args, grid, block)    │
  │    → hipModuleLoadData(hsaco_bytes)                      │
  │    → hipModuleGetFunction(kernel_name)                   │
  │    → hipModuleLaunchKernel(...)                          │
  │    → hipDeviceSynchronize()                              │
  │    → D→H copy + verify_fn callback                      │
  │    → return List[int]  (nanosecond timings)              │
  └─────────────────────────────────────────────────────────┘
```

---

## ISA Capability Probing via amdgpu-exec

TensileLite's `rocisa` module needs to know what instructions the target GPU supports before it
can generate assembly. Normally it calls `amdclang++` to probe this. Here we replace that with
`amdgpu_exec.compile_asm()`, which uses the same mechanism via the embedded assembler.

### How rocisa probes capabilities (C++)

```cpp
// hardware_caps.hpp:tryAssembler — simplified
bool tryAssembler(IsaVersion isa, string assemblerPath, string asmSnippet) {
    // runs: assemblerPath -x assembler -target amdgcn-amdhsa -mcpu=gfxNNN -
    // with asmSnippet piped to stdin
    // returns: true if exit code 0 (instruction supported)
}
```

### Equivalent Python with amdgpu-exec

```python
def _try_asm(chip: str, body: str) -> bool:
    src = f"""
.amdgcn_target "amdgcn-amd-amdhsa--{chip}"
.text
.globl _probe
.p2align 8
.type _probe,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel _probe
  .amdhsa_next_free_vgpr 256
  .amdhsa_next_free_sgpr 0
  ...
.end_amdhsa_kernel
.text
_probe:
  {body}       ← the instruction being tested
  s_endpgm
"""
    try:
        amdgpu_exec.compile_asm(src, chip)  # uses embedded assembler
        return True
    except Exception:
        return False
```

The probing phase runs ~60 small test compilations (one per ISA capability flag) and typically
completes in a few seconds.

### Injecting capabilities into rocisa

Once the capability maps are built, they are injected directly into the `rocisa` singleton,
bypassing `rocIsa.init()` entirely:

```python
isa_info = rocisa.IsaInfo()
isa_info.__setstate__((asm_caps, arch_caps, reg_caps, asm_bugs))
rocisa.rocIsa.getInstance().setData({isa: isa_info})
```

After this call, `KernelWriterAssembly` can read capabilities via `ti.getAsmCaps()` /
`ti.getArchCaps()` / `ti.getRegCaps()` as if `init()` had been called normally.

---

## TensileLite Internals

### Solution config dict

A `Solution` wraps all kernel parameters and validates them. Minimum fields for a float32 GEMM:

```python
config = {
    "ProblemType": {
        "OperationType":   "GEMM",
        "DataType":        0,       # 0 = float32 ('S')
        "DestDataType":    0,
        "ComputeDataType": 0,
        "TransposeA":      False,   # A: M×K (column-major in memory)
        "TransposeB":      True,    # B: N×K (column-major, B^T is K×N logically)
        "UseBeta":         True,
        "Batched":         True,    # required for StridedBatched
        "StridedBatched":  True,
        # all other ProblemType fields → safe defaults
    },
    "InternalSupportParams": defaultInternalSupportParams,  # KernArgsVersion=2
    "ISA":             [9, 5, 0],   # gfx950
    "CodeObjectVersion": "6",       # code object version 6 for ROCm 7+
    # MFMA tile: use matrixInstructionToMIParameters to convert 9-item MI spec
    # v_mfma_f32_16x16x4f32, 1×1 wavegroup, 4×4 wavetile → MacroTile 256×64
    "GlobalSplitU":    1,
    "KernelLanguage":  "Assembly",
}
```

After construction, `Solution` fills derived parameters:
- `MacroTile0 = WorkGroup[0] * ThreadTile[0]` = 64
- `MacroTile1 = WorkGroup[1] * ThreadTile[1]` = 64
- `NumThreads = WorkGroup[0] * WorkGroup[1] * WorkGroup[2]` = 256
- `WavefrontSize = 64` (for gfx9xx)
- `WorkGroupMapping = 8` (default WGM)

### Assembly generation

```python
kwa = KernelWriterAssembly(assembler, debugConfig)

# Wire rocisa state into the writer (reads from the singleton populated by make_isa_info)
ti = rocisa.rocIsa.getInstance()
kwa.setRocIsa(ti.getData(), ti.getOutputOptions())

kernel = solution.getKernels()[0]
err, asm_str = kwa.getSourceFileString(kernel)  # → the .s text

kernel_name = getKernelNameMin(kernel, splitGSU=False)
```

`getSourceFileString` calls `kernelBody()` which uses rocisa instruction objects to emit:
- AMDHSA kernel metadata (register counts, LDS size, argument descriptors)
- Global read / LDS write / LDS read instructions
- MFMA (matrix fused multiply-accumulate) instructions (gfx9xx) or FMA (if no MatrixInstruction)
- Waitcnt and synchronization instructions
- Global write epilogue (alpha/beta scaling)

---

## amdgpu-exec Internals

### Compilation

```python
hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
```

Internally this does:
1. `compile_asm(asm_str, chip)` → ELF object bytes (embedded clang assembler)
2. `link_binary(elf_bytes)` → HSA code object bytes

No files are written to disk; everything is in memory.

### Execution

```python
times_ns = amdgpu_exec.execute_hsaco(
    hsaco=hsaco,
    kernel_name=kernel_name,
    arguments=[...],
    grid_dim=(numWG, 1, 1),
    block_dim=(256, 1, 1),
    num_iterations=10,
    verify_fn=my_verify_fn,
)
```

- Before the first iteration: uploads `InputArray` and `InOutArray` buffers to device
- Launches the kernel `num_iterations` times, measuring with HIP events
- After the first iteration: downloads `InOutArray` and `OutputArray`, calls `verify_fn`
- Returns a `List[int]` of per-iteration execution times in nanoseconds

---

## Kernel Argument Layout

TensileLite kernels with `KernArgsVersion=2`, `UseUniversalArgs=True`, and `Batched=True` expect
the following argument layout. The kernel uses **column-major** (Fortran-order) storage internally:
stride-1 is along the free dimension (M for A/D/C, N for B), and `strideX0` is the stride along
the reduction/K dimension.

| Slot | Name | Type | Description |
|------|------|------|-------------|
| 0 | GemmInfo | uint32 | `1` — gemmCount=1, argType=0 |
| 1 | kernel_info0 | uint32 | `(StaggerU << 16) \| GSU` |
| 2 | kernel_info1 | int32 | `(wgmxcc << 16) \| WGM` |
| 3 | numWG | uint32 | `ceil(M/MT0) * ceil(N/MT1)` |
| 4 | SizesFree0 | uint32 | M |
| 5 | SizesFree1 | uint32 | N |
| 6 | SizesFree2 | uint32 | batch count (1 for non-batched) |
| 7 | SizesSum0 | uint32 | K |
| 8 | D | ptr | output (M×N col-major) |
| 9 | C | ptr | bias/C matrix (M×N col-major) |
| 10 | A | ptr | A matrix (M×K col-major) |
| 11 | B | ptr | B matrix (N×K col-major) |
| 12 | strideD0 | uint32 | stride along N for D = M |
| 13 | strideD1 | uint32 | batch stride D = 0 |
| 14 | strideC0 | uint32 | stride along N for C = M |
| 15 | strideC1 | uint32 | batch stride C = 0 |
| 16 | strideA0 | uint32 | stride along K for A = M |
| 17 | strideA1 | uint32 | batch stride A = 0 |
| 18 | strideB0 | uint32 | stride along K for B = N |
| 19 | strideB1 | uint32 | batch stride B = 0 |
| 20 | alpha | float32 | scaling factor for A*B |
| 21 | beta | float32 | scaling factor for C |

### Column-major storage

The TensileLite kernel hardcodes stride-1 along the free dimension (i for A/C/D, j for B).
All matrices must be allocated in Fortran/column-major order:

```python
a = np.asfortranarray(np.random.randn(M, K).astype(np.float32))  # M×K col-major
b = np.asfortranarray(np.random.randn(N, K).astype(np.float32))  # N×K col-major
c = np.asfortranarray(np.zeros((M, N), dtype=np.float32))
d = np.asfortranarray(np.zeros((M, N), dtype=np.float32))
# Reference: same formula, numpy handles correctly
d_ref = alpha * (np.asarray(a) @ np.asarray(b).T) + beta * np.asarray(c)
```

Strides for col-major arrays (stride = number of elements to advance one step in that dimension):
- `strideA0 = M` — advancing one K step in A costs M elements (col-major M×K)
- `strideB0 = N` — advancing one K step in B costs N elements (col-major N×K)
- `strideD0 = strideC0 = M` — advancing one N step costs M elements (col-major M×N)

### Encoding `kernel_info0`

```python
# StaggerU: stagger pattern for global reads (default 32)
# GSU: GlobalSplitU (1 = no split)
kernel_info0 = (solution["StaggerU"] << 16) | solution["GlobalSplitU"]
```

### Encoding `kernel_info1`

```python
# wgmxcc: XCC work-group mapping (1 = single XCC, safe default)
# WGM: WorkGroupMapping (tile-mapping order, default 8)
kernel_info1 = (1 << 16) | solution["WorkGroupMapping"]
```

---

## Grid and Block Dimensions

```python
MT0 = solution["MacroTile0"]          # columns per tile = WorkGroup[0] * ThreadTile[0]
MT1 = solution["MacroTile1"]          # rows per tile    = WorkGroup[1] * ThreadTile[1]

wg_x = math.ceil(M / MT0)            # work-groups along M
wg_y = math.ceil(N / MT1)            # work-groups along N
numWG = wg_x * wg_y                  # total work-groups (flattened for KernArgsVersion=2)

grid_dim  = (numWG, 1, 1)            # all WGs in x dimension
block_dim = (solution["NumThreads"], 1, 1)   # threads per block
```

For the default config (`WorkGroup=[16,16,1]`, `ThreadTile=[4,4]`):
- `MT0 = MT1 = 64`
- `NumThreads = 256`
- For a 256×256 GEMM: `numWG = 4 * 4 = 16`, `grid_dim = (16, 1, 1)`

---

## Performance Measurement

```python
times_ns = amdgpu_exec.execute_hsaco(...)       # list of nanosecond timings
best_ms  = min(times_ns) / 1e6
tflops   = 2 * M * N * K / (min(times_ns) * 1e-9) / 1e12
```

The factor of 2 accounts for both a multiply and an add per element of K.

---

## Running the Example

```bash
cd /path/to/tensilelite
source ~/.tensile/bin/activate

# Basic 256×256×256 float32 GEMM
python tensile_gemm_example.py

# Larger problem with more timing iterations
python tensile_gemm_example.py --M 2048 --N 2048 --K 1024 --iterations 20

# Explicit chip (defaults to auto-detect)
python tensile_gemm_example.py --chip gfx942 --M 512 --N 512 --K 512
```

Expected output:
```
device     : gfx942
problem    : M=256, N=256, K=256
alpha/beta : 1.0 / 0.0

Probing ISA capabilities for gfx942 via amdgpu-exec embedded assembler...
MacroTile  : 64×64
NumThreads : 256

Generating assembly...
Kernel     : Cijk_Ailk_Bljk_SB_MT064x064x008_...
Assembly   : 38,421 chars

Compiling assembly to HSACO...
HSACO size : 12,288 bytes

verification: PASSED

=== timing ===
iterations : 10
best       : 0.042 ms
avg        : 0.044 ms
TFLOPS     : 0.802  (2 * 256 * 256 * 256 / best)
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `tensile_gemm_example.py` | This example |
| `rocisa/rocisa/include/hardware_caps.hpp` | `initAsmCaps` — source of truth for cap table |
| `rocisa/rocisa/include/base.hpp` | `IsaInfo` struct, `setData` API |
| `Tensile/KernelWriterAssembly.py` | `KernelWriterAssembly.getSourceFileString()` |
| `Tensile/SolutionStructs/Solution.py` | `Solution` constructor and validation |
| `Tensile/Common/GlobalParameters.py` | `defaultInternalSupportParams`, `assignGlobalParameters` |
