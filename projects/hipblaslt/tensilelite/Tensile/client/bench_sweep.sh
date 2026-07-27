#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# bench_sweep.sh: compare wall-clock time for a full sweep between the Python
# SweepRunner and the C++ tensilelite-client.
#
# Usage:
#   ./Tensile/client/bench_sweep.sh [YAML] [NUM_ELEMENTS_TO_VALIDATE]
#
# YAML defaults to Tensile/client/tests/yaml/gemm_standard.yaml
# NUM_ELEMENTS_TO_VALIDATE defaults to -1 (all elements)
#
# The Python client runs SweepRunner in compile mode directly against the YAML.
# The C++ client requires a pre-built library; pass TENSILE_CPP_INI to point
# at a ClientParameters.ini from a prior pipeline run. If absent, only the
# Python time is reported.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${TENSILE_CPP_PYTHON:-${TENSILE_ROOT}/.tox/unit/bin/python}"
CPP_CLIENT="${TENSILE_ROOT}/build_tmp/tensilelite/client/tensilelite-client"
YAML="${1:-${SCRIPT_DIR}/tests/yaml/gemm_standard.yaml}"
VALIDATE="${2:--1}"
CPP_INI="${TENSILE_CPP_INI:-}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}"

echo "======================================================================"
echo "bench_sweep.sh"
echo "  YAML     : ${YAML}"
echo "  validate : ${VALIDATE} (${VALIDATE} == -1 means all elements)"
echo "  python   : ${PYTHON}"
echo "======================================================================"

# --- Python sweep ---
echo ""
echo "--- Python SweepRunner ---"
time "${PYTHON}" - <<PYEOF
import sys
sys.path.insert(0, '${TENSILE_ROOT}')
from Tensile.client.sweep_runner import SweepRunner
runner = SweepRunner('${YAML}', numElementsToValidate=${VALIDATE})
results = runner.run()
total = len(results)
passed = sum(1 for r in results if getattr(r, 'validation', 'SKIPPED') == 'PASS')
failed = sum(1 for r in results if str(getattr(r, 'validation', '')).startswith('FAIL'))
print(f"results: {total}, pass: {passed}, fail: {failed}")
PYEOF

# --- C++ client (optional) ---
if [[ -z "${CPP_INI}" ]]; then
    echo ""
    echo "--- C++ client: skipped (set TENSILE_CPP_INI=<path/to/ClientParameters.ini>) ---"
elif [[ ! -f "${CPP_CLIENT}" ]]; then
    echo ""
    echo "--- C++ client: skipped (binary not found at ${CPP_CLIENT}) ---"
else
    echo ""
    echo "--- C++ tensilelite-client (INI: ${CPP_INI}) ---"
    TMP_INI="$(mktemp /tmp/bench_sweep_XXXXXX.ini)"
    cp "${CPP_INI}" "${TMP_INI}"
    sed -i "s/^num-elements-to-validate=.*/num-elements-to-validate=${VALIDATE}/" "${TMP_INI}"
    grep -q "num-elements-to-validate" "${TMP_INI}" || echo "num-elements-to-validate=${VALIDATE}" >> "${TMP_INI}"
    time "${CPP_CLIENT}" --config-file "${TMP_INI}"
    rm -f "${TMP_INI}"
fi

echo ""
echo "Done."
