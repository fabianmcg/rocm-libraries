# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Parse four benchmark CSVs from the baselines sweep and compute:

  Step 1: winner-tile (MT384x256) comparison table across no-epi / chain /
          MegaFused / beta*C, with delta and ratio columns.
  Step 2: Pearson correlation of MegaFused delta_fused with tile area and wgCount
          across all tiles present in both the MegaFused and no-epi CSVs.

CSV format (same as parse_beta_sweep.py):
  row 0 = header, col 9 = TotalFlops, cols 11+ = per-solution GFlops.
  time_us = TotalFlops / GFlops / 1000.
"""

import csv
import math
import re
import statistics
from pathlib import Path

BENCHMARK_DIR = Path("/tmp/baselines-sweep/2_BenchmarkData")
TOTAL_FLOPS_COL = 9
META_COLS = 11

# Flag substrings used to identify each CSV's role.
FLAG_MFE1 = "_MFE1_"
FLAG_MFE0_CHAIN = "_MFE0_"
FLAG_PRMSRA1 = "_PRMSRA1_"
FLAG_BBS_BH = "BBS_BH_"
FLAG_BBS_H = "BBS_H_"
FLAG_PRMS0 = "_PRMS0_"


def loadCsv(path):
    """Return (role, dict[mt_key -> {best_us, median_us, M, N, wtM, wtN, area, wgCount}])."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    # Strip blank rows.
    rows = [r for r in rows if any(c.strip() for c in r)]
    header = [c.strip() for c in rows[0]]
    dataRows = rows[1:]
    if not dataRows:
        return None, {}
    totalFlops = float(dataRows[0][TOTAL_FLOPS_COL])
    role = identifyRole(header)
    records = {}
    for colIdx in range(META_COLS, len(header)):
        name = header[colIdx]
        points = []
        for r in dataRows:
            if colIdx < len(r):
                cell = r[colIdx].strip()
                if cell and cell.lower() not in ("nan", "inf", "") and float(cell) > 0:
                    points.append(float(cell))
        if not points:
            continue
        mt = parseMt(name)
        if mt is None:
            continue
        key, M, N, wtM, wtN = mt
        area = M * N
        wgCount = math.ceil(8192 / M) * math.ceil(8192 / N)
        best_us = totalFlops / max(points) / 1000.0
        median_us = totalFlops / statistics.median(points) / 1000.0
        # Keep the faster (lower time) entry if the same MT appears in multiple columns.
        if key in records:
            if median_us < records[key]["median_us"]:
                records[key] = dict(best_us=best_us, median_us=median_us,
                                    M=M, N=N, wtM=wtM, wtN=wtN, area=area, wgCount=wgCount)
        else:
            records[key] = dict(best_us=best_us, median_us=median_us,
                                M=M, N=N, wtM=wtM, wtN=wtN, area=area, wgCount=wgCount)
    return role, records


def identifyRole(header):
    """Return one of 'mfe', 'noepi', 'chain', 'beta' by scanning solution column names."""
    for name in header[META_COLS:]:
        if FLAG_MFE1 in name:
            return "mfe"
        if FLAG_MFE0_CHAIN in name and FLAG_PRMSRA1 in name:
            return "chain"
        if FLAG_BBS_BH in name:
            return "beta"
        if FLAG_BBS_H in name and FLAG_PRMS0 in name:
            return "noepi"
    # Fallback: inspect all names together.
    allNames = " ".join(header[META_COLS:])
    if FLAG_MFE1 in allNames:
        return "mfe"
    if FLAG_MFE0_CHAIN in allNames and FLAG_PRMSRA1 in allNames:
        return "chain"
    if FLAG_BBS_BH in allNames:
        return "beta"
    return "noepi"


def parseMt(name):
    """Return (key, M, N, wtM, wtN) or None."""
    mMatch = re.search(r"MT(\d+)x(\d+)x\d+", name)
    wMatch = re.search(r"MIWT(\d+)_(\d+)", name)
    if not mMatch or not wMatch:
        return None
    M, N = int(mMatch.group(1)), int(mMatch.group(2))
    wtM, wtN = int(wMatch.group(1)), int(wMatch.group(2))
    return f"MT{M}x{N}", M, N, wtM, wtN


def pearson(xs, ys):
    """Return Pearson correlation coefficient."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def loadAll():
    """Return dict keyed by role."""
    csvPaths = sorted(BENCHMARK_DIR.glob("*.csv"))
    print(f"CSV files found in {BENCHMARK_DIR}:")
    for p in csvPaths:
        print(f"  {p.name}")
    print()
    roleData = {}
    for p in csvPaths:
        role, data = loadCsv(p)
        print(f"  {p.name} -> role={role!r}, {len(data)} MT keys")
        if role and role not in roleData:
            roleData[role] = data
        elif role in roleData:
            # Merge (keep faster times).
            for k, v in data.items():
                if k not in roleData[role] or v["median_us"] < roleData[role][k]["median_us"]:
                    roleData[role][k] = v
    print()
    return roleData


def step1(roleData):
    """Print winner-tile comparison table."""
    winner = "MT384x256"
    print("=" * 70)
    print("STEP 1  Winner-tile comparison  (MT384x256)")
    print("=" * 70)
    roles = [("noepi", "A  no-epi"),
             ("chain", "B  chain"),
             ("mfe",   "C  MegaFused"),
             ("beta",  "D  beta*C")]
    times = {}
    for roleKey, label in roles:
        data = roleData.get(roleKey, {})
        rec = data.get(winner)
        if rec is None:
            print(f"  {label}: NOT FOUND in CSV")
            continue
        times[roleKey] = rec
        print(f"  {label}: best={rec['best_us']:.1f} us  median={rec['median_us']:.1f} us")

    print()
    # Deltas using median.
    baseKey = "noepi"
    if baseKey not in times:
        print("  ERROR: no-epi baseline not found; cannot compute deltas")
        return
    base = times[baseKey]
    for roleKey, label in roles[1:]:
        if roleKey not in times:
            continue
        rec = times[roleKey]
        dMed = rec["median_us"] - base["median_us"]
        dBest = rec["best_us"] - base["best_us"]
        print(f"  delta_{roleKey} (median): {dMed:+.1f} us   (best: {dBest:+.1f} us)")

    print()
    if "mfe" in times and "beta" in times:
        dFused = times["mfe"]["median_us"] - base["median_us"]
        dBeta = times["beta"]["median_us"] - base["median_us"]
        if dBeta != 0:
            ratio = dFused / dBeta
            print(f"  ratio_fused_to_beta (median deltas): {ratio:.3f}")
        dFusedBest = times["mfe"]["best_us"] - base["best_us"]
        dBetaBest = times["beta"]["best_us"] - base["best_us"]
        if dBetaBest != 0:
            ratioBest = dFusedBest / dBetaBest
            print(f"  ratio_fused_to_beta (best deltas):   {ratioBest:.3f}")


def step2(roleData):
    """Print per-tile MegaFused delta correlation table."""
    print()
    print("=" * 70)
    print("STEP 2  Per-tile delta_fused correlation")
    print("=" * 70)
    mfeData = roleData.get("mfe", {})
    noepiData = roleData.get("noepi", {})
    commonKeys = sorted(set(mfeData) & set(noepiData))
    print(f"Paired tiles (present in both mfe and noepi): {len(commonKeys)}")
    print()

    recs = []
    for key in commonKeys:
        mfe = mfeData[key]
        noepi = noepiData[key]
        deltaFused = mfe["median_us"] - noepi["median_us"]
        recs.append(dict(
            mt=key,
            area=mfe["area"],
            wgCount=mfe["wgCount"],
            M=mfe["M"],
            N=mfe["N"],
            noepi_us=noepi["median_us"],
            mfe_us=mfe["median_us"],
            deltaFused=deltaFused,
        ))

    areas = [r["area"] for r in recs]
    wgCounts = [r["wgCount"] for r in recs]
    deltas = [r["deltaFused"] for r in recs]
    mfeUs = [r["mfe_us"] for r in recs]

    print(f"  Pearson corr(area,        delta_fused) = {pearson(areas, deltas):+.3f}")
    print(f"  Pearson corr(area,        mfe_us)      = {pearson(areas, mfeUs):+.3f}")
    print(f"  Pearson corr(wgCount,     delta_fused) = {pearson(wgCounts, deltas):+.3f}")
    print()

    # Per-tile table sorted by area.
    recs.sort(key=lambda r: r["area"])
    header = f"{'MT':<14} {'area':>8} {'wgCount':>8} {'noepi_us':>10} {'mfe_us':>10} {'delta_fused':>12}"
    print(header)
    print("-" * len(header))
    for r in recs:
        print(f"{r['mt']:<14} {r['area']:>8} {r['wgCount']:>8} "
              f"{r['noepi_us']:>10.1f} {r['mfe_us']:>10.1f} {r['deltaFused']:>+12.1f}")


def main():
    roleData = loadAll()
    step1(roleData)
    step2(roleData)


if __name__ == "__main__":
    main()
