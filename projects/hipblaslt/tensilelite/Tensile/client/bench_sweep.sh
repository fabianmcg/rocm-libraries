#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# bench_sweep.sh: run the full Tensile pipeline then compare Python vs C++ sweep.
#
# Usage:
#   ./Tensile/client/bench_sweep.sh YAML [NUM_ELEMENTS_TO_VALIDATE] [OUTPUT_DIR]
#
# Steps:
#   1. BenchmarkProblems -> LibraryLogic -> ClientWriter  (Tensile pipeline)
#   2. Python SweepRunner in library mode against the built library
#   3. C++ tensilelite-client against the same library
#
# Environment variables:
#   TENSILE_ARCH    GPU architecture (default: auto-detected via amdgpu_exec)
#   TENSILE_PYTHON  override Python interpreter path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${TENSILE_PYTHON:-${TENSILE_ROOT}/.tox/unit/bin/python}"
CPP_CLIENT="${TENSILE_ROOT}/build_tmp/tensilelite/client/tensilelite-client"
TENSILE_BIN="${TENSILE_ROOT}/Tensile/bin/Tensile"

YAML="${1:?Usage: bench_sweep.sh YAML [validate] [output_dir]}"
VALIDATE="${2:--1}"
OUTPUT_DIR="${3:-/tmp/bench_sweep_$(basename "${YAML%.yaml}")_$$}"
ARCH="${TENSILE_ARCH:-}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}"

# Auto-detect arch if not set.
if [[ -z "${ARCH}" ]]; then
    ARCH="$("${PYTHON}" -c "import sys; sys.path.insert(0,'${TENSILE_ROOT}'); import amdgpu_exec; print(amdgpu_exec.get_chip().split(':')[0])")"
fi

echo "======================================================================"
echo "bench_sweep.sh"
echo "  YAML       : ${YAML}"
echo "  arch       : ${ARCH}"
echo "  validate   : ${VALIDATE} (-1 = all elements)"
echo "  output-dir : ${OUTPUT_DIR}"
echo "  python     : ${PYTHON}"
echo "======================================================================"

mkdir -p "${OUTPUT_DIR}"
PIPELINE_DIR="${OUTPUT_DIR}/pipeline"
PIPELINE_LOG="${OUTPUT_DIR}/pipeline.log"

# --- Phase 1: Run the Tensile pipeline ---
echo ""
echo "--- Phase 1: BenchmarkProblems -> LibraryLogic -> ClientWriter ---"
time "${PYTHON}" "${TENSILE_BIN}" "${YAML}" "${PIPELINE_DIR}" --gpu-targets "${ARCH}" \
    > "${PIPELINE_LOG}" 2>&1 && echo "Pipeline: OK" || {
    echo "Pipeline FAILED — see ${PIPELINE_LOG}"
    tail -20 "${PIPELINE_LOG}"
    exit 1
}

# Locate library and INI produced by the pipeline.
LIBRARY="$(find "${PIPELINE_DIR}/4_LibraryClient" -name "TensileLibrary_${ARCH}.yaml" 2>/dev/null | head -1)"
CPP_INI="$(find "${PIPELINE_DIR}/4_LibraryClient/source" -name "ClientParameters_*.ini" ! -name "*Granularity*" 2>/dev/null | head -1)"

if [[ -z "${LIBRARY}" ]]; then
    echo "ERROR: TensileLibrary_${ARCH}.yaml not found under ${PIPELINE_DIR}/4_LibraryClient"
    exit 1
fi
echo "  library : ${LIBRARY}"
echo "  ini     : ${CPP_INI:-<not found>}"

# --- Phase 2: Python SweepRunner (library mode) ---
echo ""
echo "--- Phase 2: Python SweepRunner (library mode) ---"
time "${PYTHON}" - <<PYEOF
import sys
sys.path.insert(0, '${TENSILE_ROOT}')
from Tensile.client.sweep_runner import SweepRunner
runner = SweepRunner('${YAML}', libraryPath='${LIBRARY}',
                     numElementsToValidate=${VALIDATE})
results = runner.run(resultsCsv='${OUTPUT_DIR}/python_results.csv')
total = len(results)
passed = sum(1 for r in results if getattr(r, 'validation', 'SKIPPED') == 'PASS')
failed = sum(1 for r in results if str(getattr(r, 'validation', '')).startswith('FAIL'))
print(f"results: {total}, pass: {passed}, fail: {failed}")
PYEOF

# --- Phase 3: C++ tensilelite-client ---
if [[ -z "${CPP_INI}" ]]; then
    echo ""
    echo "--- Phase 3: C++ client skipped (no ClientParameters.ini found) ---"
elif [[ ! -f "${CPP_CLIENT}" ]]; then
    echo ""
    echo "--- Phase 3: C++ client skipped (binary not found at ${CPP_CLIENT}) ---"
else
    echo ""
    echo "--- Phase 3: C++ tensilelite-client ---"
    TMP_INI="$(mktemp /tmp/bench_sweep_XXXXXX.ini)"
    cp "${CPP_INI}" "${TMP_INI}"
    sed -i "s|results-file=.*|results-file=${OUTPUT_DIR}/cpp_results.csv|" "${TMP_INI}"
    for KEY in "num-elements-to-validate=${VALIDATE}" \
               "num-benchmarks=10" \
               "num-syncs-per-benchmark=1" \
               "num-warmups=3"; do
        K="${KEY%%=*}"; V="${KEY#*=}"
        sed -i "s/^${K}=.*/${K}=${V}/" "${TMP_INI}"
        grep -q "^${K}=" "${TMP_INI}" || echo "${K}=${V}" >> "${TMP_INI}"
    done
    time "${CPP_CLIENT}" --config-file "${TMP_INI}"
    rm -f "${TMP_INI}"
fi

echo ""
echo "Done. Results in: ${OUTPUT_DIR}"
