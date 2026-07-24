#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Reproducer for Bug 2: PartialRMS LDSB0 + DirectToLds=0 + MT1=256 (wg_n=2).
#
# Runs the full tune_prms_8192.yaml sweep on gfx950. The failing kernels only
# appear when all four groups (A–D) are benchmarked together — the bug is
# order- or load-dependent and does NOT reproduce from the isolated per-DepthU
# YAMLs alone.
#
# Expected outcome (on a quiet gfx950 system):
#   104 PASSED, 24 FAILED
#
# Failing tiles (all LDSB0 + DTLA0 + MT1=256):
#   MT64x256x64   (8 variants: PGR×SU×PKA combinations)
#   MT64x256x128  (8 variants)
#   MT256x256x64  (8 variants)
#
# The same geometries with DirectToLds=1 (Groups C) pass. The bug is
# in the GPU kernel; root cause has not yet been isolated.
#
# Usage (from the tensilelite root):
#   source ~/.tensile/bin/activate
#   bash epilogues/docs/repro_bug2_dtl0_wgn2.sh
#
# Override output directory:
#   OUT_DIR=/tmp/my_repro bash epilogues/docs/repro_bug2_dtl0_wgn2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILELITE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML="${TENSILELITE_ROOT}/epilogues/yaml/tune_prms_8192.yaml"
OUT_DIR="${OUT_DIR:-/tmp/repro_bug2_dtl0_wgn2}"

# Resolve Python: prefer ~/.tensile venv, fall back to python3.
_VENV_PYTHON="${HOME}/.tensile/bin/python"
if [[ -x "${_VENV_PYTHON}" ]]; then
    PYTHON="${PYTHON:-${_VENV_PYTHON}}"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "==> Bug 2 reproducer: PartialRMS LDSB0 + DirectToLds=0 + MT1=256"
echo "    Tensile root : ${TENSILELITE_ROOT}"
echo "    YAML         : ${YAML}"
echo "    Output dir   : ${OUT_DIR}"
echo "    Python       : ${PYTHON}"
echo ""
echo "    Expected outcome: 104 passed, 24 failed."
echo "    Failing tiles: MT64x256x64, MT64x256x128, MT256x256x64 (all DTLA0)."
echo "    Passing control: same tiles with DirectToLds=1 (Groups A/C) pass."
echo ""
echo "    NOTE: failures only appear when all four YAML groups are run together."
echo "    The isolated per-DepthU YAMLs may show all-pass."
echo ""

# Always start from a clean output directory to avoid --use-cache reuse.
rm -rf "${OUT_DIR}"

TENSILE_LOG=$(mktemp /tmp/tensile_bug2_XXXXXX.log)
trap 'rm -f "${TENSILE_LOG}"' EXIT

# Tensile exits non-zero (1) when any kernel fails validation.
"${PYTHON}" "${TENSILELITE_ROOT}/Tensile/bin/Tensile" \
    "${YAML}" "${OUT_DIR}" \
    >"${TENSILE_LOG}" 2>&1 || true

PASS=$(grep -c ',PASSED,' "${TENSILE_LOG}" 2>/dev/null || true)
FAIL=$(grep -c ',FAILED,' "${TENSILE_LOG}" 2>/dev/null || true)

echo "==> Benchmark results:"
grep ',PASSED,' "${TENSILE_LOG}" 2>/dev/null \
    | grep -oP 'MT\d+x\d+x\d+[^,]*(?=,PASSED,)' \
    | while read -r k; do echo "  PASS: ${k}"; done || true
grep ',FAILED,' "${TENSILE_LOG}" 2>/dev/null \
    | grep -oP 'MT\d+x\d+x\d+[^,]*(?=,FAILED,)' \
    | while read -r k; do echo "  FAIL: ${k}"; done || true

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"

if [[ ${FAIL} -eq 24 ]]; then
    echo "==> Bug 2 reproduced: exactly 24 DTLA0+MT1=256 failures."
    exit 0
elif [[ ${FAIL} -gt 0 ]]; then
    echo "==> Partial reproduction: ${FAIL} failures (expected 24)."
    echo "    System load may have shifted the failing set — see bug doc."
    exit 1
else
    echo "==> Bug NOT reproduced: all kernels passed (unexpected on an unpatched tree)."
    exit 2
fi
