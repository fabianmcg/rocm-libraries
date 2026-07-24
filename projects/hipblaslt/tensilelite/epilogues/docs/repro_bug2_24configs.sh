#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Reproducer for Bug 2: runs exactly the 24 failing PartialRMS configs
# (LDSB0 + DirectToLds=0 + MT1=256) with a guaranteed clean cache.
#
# The 24 configs are split across three BenchmarkProblems groups:
#   MT64x256  DepthU=64   (8 configs)
#   MT64x256  DepthU=128  (8 configs)
#   MT256x256 DepthU=64   (8 configs)
#
# Due to order-dependence, not all 24 fail in every run. In isolation the
# first group typically fails (8/24); all 24 fail when the full sweep
# (tune_prms_8192.yaml) precedes this YAML. See repro_bug2_dtl0_wgn2.sh
# for the end-to-end reproducer that reliably triggers all 24.
#
# This script is useful for:
#   - Quickly checking whether the bug is present (any FAILED = bug present).
#   - Post-fix verification: a correct fix produces 0 failures here AND in
#     the full sweep.
#
# Usage (from the tensilelite root):
#   source ~/.tensile/bin/activate
#   bash epilogues/docs/repro_bug2_24configs.sh
#
# Override output directory:
#   OUT_DIR=/tmp/my_repro bash epilogues/docs/repro_bug2_24configs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILELITE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML="${SCRIPT_DIR}/repro_bug2_24configs.yaml"
OUT_DIR="${OUT_DIR:-/tmp/repro_bug2_24configs}"

# Resolve Python: prefer ~/.tensile venv, fall back to python3.
_VENV_PYTHON="${HOME}/.tensile/bin/python"
if [[ -x "${_VENV_PYTHON}" ]]; then
    PYTHON="${PYTHON:-${_VENV_PYTHON}}"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "==> Bug 2 reproducer: 24 configs (LDSB0 + DirectToLds=0 + MT1=256)"
echo "    Tensile root : ${TENSILELITE_ROOT}"
echo "    YAML         : ${YAML}"
echo "    Output dir   : ${OUT_DIR}"
echo "    Python       : ${PYTHON}"
echo ""
echo "    Configs: MT64x256xDU64 (8), MT64x256xDU128 (8), MT256x256xDU64 (8)"
echo "    Expected: >=8 FAILED (all 24 fail only after the full sweep warm-up)."
echo ""

# Always start from a clean output directory.
rm -rf "${OUT_DIR}"

TENSILE_LOG=$(mktemp /tmp/tensile_24configs_XXXXXX.log)
trap 'rm -f "${TENSILE_LOG}"' EXIT
echo "    TENSILE_LOG       : ${TENSILE_LOG}"

"${PYTHON}" "${TENSILELITE_ROOT}/Tensile/bin/Tensile" \
    "${YAML}" "${OUT_DIR}" --debug -v \
    >"${TENSILE_LOG}" 2>&1 || true

PASS=$(grep -c ',PASSED,' "${TENSILE_LOG}" 2>/dev/null || true)
FAIL=$(grep -c ',FAILED,' "${TENSILE_LOG}" 2>/dev/null || true)

echo "==> Results:"
grep ',FAILED,' "${TENSILE_LOG}" 2>/dev/null \
    | grep -oP 'MT\d+x\d+x\d+[^,]*(?=,FAILED,)' \
    | while read -r k; do echo "  FAIL: ${k}"; done || true
grep ',PASSED,' "${TENSILE_LOG}" 2>/dev/null \
    | grep -oP 'MT\d+x\d+x\d+[^,]*(?=,PASSED,)' \
    | while read -r k; do echo "  PASS: ${k}"; done || true

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed (out of 24 configs)"

if [[ ${FAIL} -gt 0 ]]; then
    echo "==> Bug 2 present: ${FAIL}/24 configs failed."
    exit 0
else
    echo "==> No failures detected in isolation."
    echo "    Run the full sweep to trigger all 24: bash epilogues/docs/repro_bug2_dtl0_wgn2.sh"
    exit 1
fi
