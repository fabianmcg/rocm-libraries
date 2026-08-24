# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Run the single best PartialRMS config (MT256x256, MIWT8x8, WG2x2) and
# summarise GFlops/time_us for the 8192^3 point from the CSV results.

set -euo pipefail

OUTDIR="${1:-/tmp/tensile_prms_best}"
LOGFILE="${OUTDIR}.log"
YAML="Tensile/Tests/common/gemm/gfx950/gemm_partial_rms_best.yaml"

# Activate venv and set required library path.
source ~/.tensile/bin/activate
export LD_LIBRARY_PATH=/opt/rocm/lib

echo "Output dir : $OUTDIR"
echo "Log file   : $LOGFILE"
echo "YAML       : $YAML"
echo "---"

mkdir -p "$OUTDIR"

python Tensile/bin/Tensile "$YAML" "$OUTDIR" 2>&1 | tee "$LOGFILE"

echo ""
echo "=== Summary for 8192 x 8192 x 1 x 8192 ==="

# Parse performance directly from the log: the Tensile client emits one CSV
# line per measurement. The log is more reliable than the benchmark CSV,
# which can merge multiple data rows onto a single line.
# Log column layout (after csv-parsing the quoted size field):
#   [0]=run  [1]=prob-progress  [2]=sol-progress  [3]=op
#   [4]=size-tuple  [5]=bias  [6]=factor  [7]=activation
#   [8]=solution  [9]=validation  [10]=time-us  [11]=gflops(MFlops)

python3 - "$LOGFILE" <<'EOF'
import csv, sys

logfile = sys.argv[1]
target_size = "(8192,8192,1,8192)"

print(f"{'Solution':<60} {'GFlops':>12} {'TimeUs':>12}")
print("-" * 86)

found = False
with open(logfile) as fh:
    for raw_line in fh:
        line = raw_line.strip()
        # Quick pre-filter before parsing.
        if "8192,8192,1,8192" not in line:
            continue
        # Skip header and non-data lines.
        if not line[0].isdigit():
            continue
        try:
            parts = next(csv.reader([line]))
        except Exception:
            continue
        if len(parts) < 12:
            continue
        if parts[4].strip() != target_size:
            continue
        sol     = parts[8].strip()
        time_us = parts[10].strip()
        mflops  = parts[11].strip()
        if not time_us or not mflops:
            continue
        try:
            t = float(time_us)
            g = float(mflops) / 1e3  # convert MFlops -> GFlops
        except ValueError:
            continue
        short = sol[:57] + "..." if len(sol) > 60 else sol
        print(f"{short:<60} {g:>12.3f} {t:>12.1f}")
        found = True

if not found:
    print("No 8192^3 measurement found in log — check the log for errors.")
EOF
