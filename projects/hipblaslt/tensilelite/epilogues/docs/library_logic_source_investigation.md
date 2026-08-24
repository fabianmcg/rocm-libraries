# Library Logic Source Investigation

## 1. What are the library logic files and where do they live?

The committed library logic files live at:

```
library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/
  <arch-name>/           # e.g. aquavanjaram, gfx950, gfx1200, ...
    <cu-variant>/        # e.g. gfx942, gfx942_80cu, gfx942_152cu, ...
      Equality/          # exact-match lookup (per problem size → best kernel index)
      GridBased/         # range-based grid heuristic
      FreeSize/          # free-dimension heuristic
      StreamK/           # stream-k variant
      Experimental/      # gated behind --experimental flag
```

2,574 YAML files total. Each encodes one (architecture, problem type, layout) combination. The YAML structure is: header → architecture name + ISA → device IDs → full ProblemType dict → indexed solution pool → LibraryLogic section (exact-size winners + range decision tree).

**No comments or metadata inside the YAML files point back to a source benchmark YAML or run ID.** Provenance is tracked exclusively through git history.

---

## 2. What are the source tuning YAMLs?

**The source benchmark YAMLs are not stored in this repository.** They are external artifacts produced per-tuning-run.

The files in `tensilelite/Tensile/Tests/common/` are **correctness tests**, not production tuning sweeps — they use small ForkParameter lists, `SyncsPerBenchmark: 0`, and numerical validation. They are not the source of the checked-in library logic.

The **`utilities/geko/`** tool (GEMM Kernel Optimizer) is the mechanism for production tuning:

1. Capture a real workload: `HIPBLASLT_LOG_MASK=64` produces a YAML list of GEMM shapes.
2. `geko --configure` reads the workload log and generates per-GEMM Tensile benchmark YAMLs in `optimizations/`.
3. `geko --optimize` runs the full three-phase Tensile pipeline, producing `3_LibraryLogic/*.yaml`.
4. The resulting files are reviewed, diffed against the existing library logic, and committed.

For epilogue-specific kernels, the benchmark YAMLs live in **`tensilelite/epilogues/yaml/`** (e.g. `tune_subtile_bf16_prmsq_8192.yaml`, `gemm_partial_rms_k1_rowmajor.yaml`). These are the source for epilogue-related library logic commits.

---

## 3. The full pipeline from YAML to committed library logic

### Step 1 — Write a benchmark YAML

Contains:
- `GlobalParameters:` — ISA, timing settings, validation depth.
- `BenchmarkProblems:` — `[ProblemType, BenchmarkProblemSizeGroup]` pairs. `ForkParameters` is a cartesian-product sweep over kernel parameters. `BenchmarkFinalParameters.ProblemSizes` lists the exact M/N/K/batch sizes.
- `LibraryLogic:` — triggers Phase 2 analysis.

### Step 2 — Run `Tensile/bin/Tensile config.yaml output_dir`

Entry point: `Tensile/Tensile.py:executeStepsInConfig()`.

**Phase 1 (BenchmarkProblems):** Expands ForkParameters, constructs Solution objects (invalid combos silently dropped), compiles all survivors to `.co` code objects via rocisa + amdclang++ (SHA-256 cached), then runs the `tensilelite-client` C++ binary measuring GFlops per (problem size, solution) pair. Output: `2_BenchmarkData/*.csv`.

**Phase 2 (LibraryLogic):** Reads the CSV. Prunes solutions that never win on any size. Builds a range-logic decision tree mapping M/N/K regions to the best solution index. Records exact-match winners. Writes one `LibraryLogic.yaml` per `(ScheduleName, ProblemType)`. **Output: `3_LibraryLogic/*.yaml` — these are the files that get committed to the library.**

**Phase 3 (TensileCreateLibrary):** Takes the `3_LibraryLogic/*.yaml` files and compiles the winning kernels into a deployable device library (`.co`/`.dat`). This is what `invoke build` runs at build time using the already-committed logic files.

### Step 3 — Review and integrate

Copy `3_LibraryLogic/*.yaml` into the appropriate subdirectory of `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/<cu-variant>/<Equality|GridBased|...>/` and commit.

### Step 4 — Build-time validation

CMake runs `TensileLogic --check-all` against the logic path before `TensileCreateLibrary`. Can also be run standalone via `python scripts/run_tensile_logic_check.py`.

---

## 4. Is there a single tuning YAML covering the full problem space?

**No.** There is no single master YAML in the repository. The library logic was accumulated incrementally through many separate tuning runs, each targeting a specific (dtype, layout, GPU, problem-size range). The git log for `library/src/amd_detail/rocblaslt/src/Tensile/Logic/` reflects this clearly — PRs add tuning for one dtype/arch combination at a time.

`utilities/geko/` is the recommended mechanism for workload-driven tuning. For epilogue-specific kernels, `tensilelite/epilogues/yaml/` contains the best-documented in-tree tuning inputs.

---

## 5. How to regenerate the library logic files

**For existing problem types (retuning a specific arch/dtype):**

```bash
# 1. Run the benchmark sweep
python Tensile/bin/Tensile sweep.yaml /tmp/out --gpu-targets gfx942

# 2. Copy Phase 2 output to library
cp /tmp/out/3_LibraryLogic/*.yaml \
  library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942/Equality/

# 3. Validate
python scripts/run_tensile_logic_check.py

# 4. Test compilation
invoke build -ca gfx942 -f 'aquavanjaram/gfx942/Equality/*'
```

**For GEKO-driven workload tuning:**

```bash
cd utilities/geko
./bin/geko --search --workload-log hipblaslt-log-mask64.yaml --devices=0,1,2,3
# Produces merged LibraryLogic YAMLs in libs/ ready to commit
```

---

## 6. Key files

| Path | Role |
|------|------|
| `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/` | 2,574 committed library logic YAML files (Phase 2 output) |
| `device-library/CMakeLists.txt` | CMake entry point; sets `HIPBLASLT_LIBLOGIC_PATH` |
| `cmake/HipBLASLtCodegen.cmake` | `hipblaslt_create_device_library()` — runs TensileLogic validation + TensileCreateLibrary |
| `tensilelite/Tensile/Tensile.py` | Three-phase entry point (`executeStepsInConfig`) |
| `tensilelite/Tensile/BenchmarkProblems.py` | Phase 1: kernel compilation + GPU benchmarking |
| `tensilelite/Tensile/LibraryLogic.py` | Phase 2: benchmark CSV → LibraryLogic YAML |
| `tensilelite/Tensile/TensileCreateLibrary/` | Phase 3: LibraryLogic YAML → `.co`/`.dat` device library |
| `utilities/geko/` | GEKO optimizer: workload-log → Tensile YAML → LibraryLogic → merged libs |
| `tensilelite/epilogues/yaml/` | Epilogue-specific benchmark/tuning YAMLs (PartialRMS, MX-FP8 quant) |
| `tensilelite/epilogues/docs/TUNING_PIPELINE.md` | Most detailed existing write-up of the pipeline |
| `tensilelite/epilogues/docs/LIBRARY_CREATION_GUIDE.md` | Step-by-step YAML → device library guide |
| `scripts/run_tensile_logic_check.py` | Validates committed logic files |
| `tensilelite/Tensile/TensileLogic/known_bugs.yaml` | Known-bad solutions excluded from validation |

---

## 7. Gaps and unknowns

- **Source tuning YAMLs are not in the repo.** The sweep config for each individual tuning run is not committed alongside the resulting library logic. Git commit messages reference JIRA IDs and describe problem sizes, but the original `BenchmarkProblems` YAML is not preserved.

- **No single authoritative "full sweep" config.** The library accumulated through many discrete PRs from different authors across different dtypes and arches.

- **GEKO is the intended automation layer** but its generated benchmark YAMLs are also runtime artifacts, not stored.

- **Epilogue-specific tuning YAMLs** in `tensilelite/epilogues/yaml/` are the best-documented in-tree examples of production tuning inputs, specific to PartialRMS and MX-FP8 dynamic-quantization epilogues for gfx950.
