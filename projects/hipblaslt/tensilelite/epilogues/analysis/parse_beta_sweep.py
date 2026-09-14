# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Correlate MatrixInstruction shape with beta*C overhead on gfx950.

benchmark_beta.yaml has two BenchmarkProblems groups with identical fork space,
differing only by UseBeta (group 1 False -> BBS_H, group 2 True -> BBS_BH). We
measure, per solution, the extra time the beta*C path costs:

    delta_us = time(with_beta) - time(without_beta)

and ask which MatrixInstruction shapes minimise it. Times come only from the
benchmark CSV (col 9 = TotalFlops, cols 11+ = per-solution GFlops); the
LibraryLogic "winner" number is a model prediction and is never used here.

    time_us = TotalFlops / GFlops / 1000

The YAML runs NumBenchmarks=1 (one data row), so single-shot variance is large;
the whole benchmark was repeated three times with the unmodified YAML and we
take the median delta per config across the three runs. No warmup / enqueue
knobs were added (they perturb the measured kernel time), so the per-run
methodology is exactly the shipped one.

MatrixInstruction geometry is recovered from the kernel name:
    MT<M>x<N>x<DepthU> , MIWT<wtM>_<wtN>  (per-wave tile, units of the 16x16 MI)
    waveGroupM = M / (16*wtM) , waveGroupN = N / (16*wtN)   (waves in M / N)
"""

import csv
import re
import statistics
from pathlib import Path

RUN_DIRS = [
    Path("/tmp/beta-sweep2/2_BenchmarkData"),
    Path("/tmp/beta-sweep3/2_BenchmarkData"),
    Path("/tmp/beta-sweep4/2_BenchmarkData"),
]
NOBETA = ("Cijk_Alik_Bljk_BBS_H_UserArgs_00.csv", "Cijk_Alik_Bljk_BBS_H_UserArgs_")
BETA = ("Cijk_Alik_Bljk_BBS_BH_UserArgs_00.csv", "Cijk_Alik_Bljk_BBS_BH_UserArgs_")
META_COLS = 11
TOTALFLOPS_COL = 9


def loadCsv(path, prefix):
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and r[0].strip() != ""]
    header = [c.strip() for c in rows[0]]
    dataRows = rows[1:]
    totalFlops = float(dataRows[0][TOTALFLOPS_COL])
    out = {}
    for col in range(META_COLS, len(header)):
        name = header[col]
        if not name.startswith(prefix):
            continue
        points = []
        for r in dataRows:
            if col < len(r):
                cell = r[col].strip()
                if cell and cell.lower() != "nan" and float(cell) > 0:
                    points.append(float(cell))
        if points:
            out[name[len(prefix):]] = totalFlops / max(points) / 1000.0
    return out


def parseMi(key):
    m, n, du = map(int, re.match(r"MT(\d+)x(\d+)x(\d+)", key).groups())
    wtM, wtN = map(int, re.search(r"MIWT(\d+)_(\d+)", key).groups())
    return dict(
        M=m, N=n, DU=du, wtM=wtM, wtN=wtN,
        wgM=m // (16 * wtM), wgN=n // (16 * wtN),
        pgr=re.search(r"_PGR(\d)_", key).group(1),
        sk=re.search(r"_SK(\d)_", key).group(1),
    )


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def main():
    perRun = []
    for d in RUN_DIRS:
        perRun.append((loadCsv(d / NOBETA[0], NOBETA[1]), loadCsv(d / BETA[0], BETA[1])))

    keys = set(perRun[0][0])
    for nb, be in perRun:
        keys &= set(nb) & set(be)
    keys = sorted(keys)

    recs = []
    for k in keys:
        nbs = [nb[k] for nb, _ in perRun]
        bes = [be[k] for _, be in perRun]
        deltas = [b - n for n, b in zip(nbs, bes)]
        mi = parseMi(k)
        recs.append(dict(
            key=k, nb=statistics.median(nbs), delta=statistics.median(deltas),
            deltaSpread=max(deltas) - min(deltas), **mi,
        ))

    print(f"# runs aggregated : {len(perRun)}  (median delta per config)")
    print(f"# paired configs  : {len(recs)}")
    print(f"# fastest no-beta  : {min(r['nb'] for r in recs):.1f} us")
    print(f"# median deltaSpread across runs: "
          f"{statistics.median([r['deltaSpread'] for r in recs]):.1f} us (single-shot noise band)")

    deltas = [r["delta"] for r in recs]
    print("\nPearson corr(median delta, MI quantity):")
    for name, f in [
        ("no-beta time", lambda r: r["nb"]),
        ("tile area M*N", lambda r: r["M"] * r["N"]),
        ("aspect M/N", lambda r: r["M"] / r["N"]),
        ("tile N", lambda r: r["N"]),
        ("tile M", lambda r: r["M"]),
        ("waveGroupN", lambda r: r["wgN"]),
        ("waveGroupM", lambda r: r["wgM"]),
        ("waveTileN", lambda r: r["wtN"]),
        ("waveTileM", lambda r: r["wtM"]),
    ]:
        print(f"  {name:14}: {pearson([f(r) for r in recs], deltas):+.3f}")

    groupBy("median delta by waveGroup [wgM,wgN] (the MI wave layout)",
            recs, lambda r: (r["wgM"], r["wgN"]))
    groupBy("median delta by aspect class", recs,
            lambda r: "M-heavy(M>N)" if r["M"] > r["N"] else ("N-heavy(N>M)" if r["N"] > r["M"] else "square"))
    groupBy("median delta by PGR", recs, lambda r: "PGR" + r["pgr"])

    print("\n--- lowest-delta 15 configs (beta-resistant) ---")
    printRows(sorted(recs, key=lambda r: r["delta"])[:15])
    print("\n--- highest-delta 15 configs (beta-costly) ---")
    printRows(sorted(recs, key=lambda r: -r["delta"])[:15])


def groupBy(label, recs, key):
    buckets = {}
    for r in recs:
        buckets.setdefault(key(r), []).append(r["delta"])
    print(f"\n{label}:")
    for g in sorted(buckets, key=lambda x: statistics.mean(buckets[x])):
        v = buckets[g]
        print(f"  {str(g):>14}: n={len(v):3d}  mean={statistics.mean(v):6.1f}  "
              f"median={statistics.median(v):6.1f}  min={min(v):6.1f}  max={max(v):6.1f}")


def printRows(rows):
    print(f"  {'MT':<14}{'wTile':>8}{'wGrp':>7}{'PGR':>4}{'SK':>3}"
          f"{'nb_us':>8}{'delta':>8}{'spread':>8}")
    for r in rows:
        mt = f"MT{r['M']}x{r['N']}"
        wtile = f"{r['wtM']}x{r['wtN']}"
        wgrp = f"{r['wgM']}x{r['wgN']}"
        print(f"  {mt:<14}{wtile:>8}{wgrp:>7}{r['pgr']:>4}{r['sk']:>3}"
              f"{r['nb']:>8.0f}{r['delta']:>8.1f}{r['deltaSpread']:>8.1f}")


if __name__ == "__main__":
    main()
