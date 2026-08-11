#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Integration test for the tensilelite-client Deepseek-scale mainloop (PGR=0) paths.
#
# Tests all three modes:
#   A-only: --use-deepseek-scale-a
#   B-only: --use-deepseek-scale-b
#   A+B:    --use-deepseek-scale-a --use-deepseek-scale-b
#
# Steps:
#   1. Run the Tensile pipeline to generate LibraryLogic YAMLs for all modes.
#   2. Compile a device library (YAML format) with TensileCreateLibrary.
#   3. Run the client for each mode and shape, verify PASSED.
#
# Usage:
#   epilogues/scripts/test_client_deepseek_scale.sh [--chip CHIP] [--client PATH] [--out-dir DIR]
#
# Arguments:
#   --chip    GPU architecture string (default: gfx950).
#   --client  Path to the tensilelite-client binary (required if not on PATH).
#   --out-dir Scratch directory for generated files (default: /tmp/ds_test_<chip>).
#
# Prerequisites:
#   * Python environment with TensileLite installed (or run from tensilelite/ root).
#   * An AMD GPU matching the chip argument must be present and accessible.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CHIP="gfx950"
CLIENT_BIN=""
OUT_DIR=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --chip)    CHIP="$2";       shift 2 ;;
        --client)  CLIENT_BIN="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2";    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="/tmp/ds_test_${CHIP}"
fi

LIB_DIR="${OUT_DIR}/library_yaml"

# ── Resolve client binary ─────────────────────────────────────────────────────
if [[ -z "$CLIENT_BIN" ]]; then
    if command -v tensilelite-client &>/dev/null; then
        CLIENT_BIN="$(command -v tensilelite-client)"
    else
        echo "error: tensilelite-client not found; pass --client <path>" >&2
        exit 1
    fi
fi

if [[ ! -x "$CLIENT_BIN" ]]; then
    echo "error: client binary not executable: $CLIENT_BIN" >&2
    exit 1
fi

# ── Helper ────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0

check() {
    local label="$1"
    local output="$2"
    # Look for PASSED in the data rows; INVALID or absence means failure.
    if echo "$output" | grep -q "^0,.*,PASSED,"; then
        echo "PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $label"
        echo "$output" | grep "^0," | head -1 | cut -c1-200 >&2 || true
        FAIL=$((FAIL + 1))
    fi
}

# ── Step 1: Run Tensile pipeline ──────────────────────────────────────────────
echo "==> Running Tensile pipeline for $CHIP ..."
TENSILELITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The benchmark phase validates with its own reference; a FAILED result there
# does not indicate a kernel bug.  Correctness is checked by the client below
# with --num-elements-to-validate -1.
set +e
python3 "${TENSILELITE_ROOT}/Tensile/bin/Tensile" \
    "${TENSILELITE_ROOT}/epilogues/yaml/gemm_deepseek_scale_client.yaml" \
    "${OUT_DIR}/tensile_out"
TENSILE_EXIT=$?
set -e
if [[ ! -d "${OUT_DIR}/tensile_out/3_LibraryLogic" ]]; then
    echo "error: Tensile pipeline did not produce 3_LibraryLogic (exit ${TENSILE_EXIT})" >&2
    exit 1
fi

# ── Step 2: Build device library ─────────────────────────────────────────────
echo "==> Building device library (YAML format) ..."
python3 -m Tensile.TensileCreateLibrary \
    --library-format=yaml \
    --architecture "$CHIP" \
    "${OUT_DIR}/tensile_out/3_LibraryLogic" \
    "$LIB_DIR" \
    HIP

# Locate the non-lazy YAML and its code object.
LIB_YAML="$(find "${LIB_DIR}/library/${CHIP}" -maxdepth 1 \
    -name "TensileLibrary_BB_*.yaml" ! -name "*lazy*" | head -1)"
if [[ -z "$LIB_YAML" ]]; then
    echo "error: could not find non-lazy library YAML under ${LIB_DIR}/library/${CHIP}" >&2
    exit 1
fi
LIB_CO="${LIB_YAML%.yaml}.co"
if [[ ! -f "$LIB_CO" ]]; then
    echo "error: code object not found: $LIB_CO" >&2
    exit 1
fi
echo "  Library YAML: $LIB_YAML"
echo "  Code object : $LIB_CO"

# ── Common client arguments ───────────────────────────────────────────────────
# Inputs: fp8 e4m3 (F8); output: f32 (S); compute: f32.
# Transpose: A=TN (TransposeA=True, TransposeB=False).
BASE_ARGS=(
    --library-file      "$LIB_YAML"
    --code-object       "$LIB_CO"
    --problem-identifier "Contraction_l_Alik_Bljk_Cijk_Dijk"
    --a-type Float8 --b-type Float8 --c-type Float --d-type Float
    --compute-input-type-A Float8 --compute-input-type-B Float8
    --high-precision-accumulate
    --f32-xdl-math-op Float
    --alpha-type Float
    --num-benchmarks 1
    --num-elements-to-validate -1
    --device-idx 0
)

# ── Step 3: A-only mode ───────────────────────────────────────────────────────
echo "==> Testing DeepseekScaleA-only ..."

for MN in "128,128" "256,256"; do
    M="${MN%,*}"
    N="${MN#*,}"
    K=128
    label="A-only M=${M} N=${N} K=${K}"
    echo "  ==> $label ..."
    OUT="$("$CLIENT_BIN" "${BASE_ARGS[@]}" \
        --use-deepseek-scale-a \
        --problem-size "${M},${N},1,${K}" 2>&1)" || true
    check "$label" "$OUT"
done

# ── Step 4: B-only mode ───────────────────────────────────────────────────────
echo "==> Testing DeepseekScaleB-only ..."

for MN in "128,128" "256,256"; do
    M="${MN%,*}"
    N="${MN#*,}"
    K=128
    label="B-only M=${M} N=${N} K=${K}"
    echo "  ==> $label ..."
    OUT="$("$CLIENT_BIN" "${BASE_ARGS[@]}" \
        --use-deepseek-scale-b \
        --problem-size "${M},${N},1,${K}" 2>&1)" || true
    check "$label" "$OUT"
done

# ── Step 5: A+B combined mode ─────────────────────────────────────────────────
echo "==> Testing DeepseekScaleA+B ..."

for MN in "128,128" "256,256"; do
    M="${MN%,*}"
    N="${MN#*,}"
    K=128
    label="A+B M=${M} N=${N} K=${K}"
    echo "  ==> $label ..."
    OUT="$("$CLIENT_BIN" "${BASE_ARGS[@]}" \
        --use-deepseek-scale-a \
        --use-deepseek-scale-b \
        --problem-size "${M},${N},1,${K}" 2>&1)" || true
    check "$label" "$OUT"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
