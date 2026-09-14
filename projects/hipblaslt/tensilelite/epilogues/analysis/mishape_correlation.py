# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Combine MegaFused epilogue instruction counts (parse_mishape.py) with the
per-tile timing deltas (parse_baselines.out) to test the MI-shape hypotheses:

  - corr(wave-tile area, fused-pass instruction count)  -> does codegen scale?
  - linear fit fusedInstr = a*wtArea + b -> fixed-cost fraction (H1).
  - aggregate work = fusedInstr * numWG -> is total epilogue work tile-invariant?
  - corr(wave-tile area, delta_us) and outlier leverage (H5).
"""

import math
import statistics

# From parse_mishape.out: (MT, wtM, wtN, next_free_vgpr, lds_bytes, fusedInstr, execInstr).
INSTR = {
    "MT576x64":  (9,  4,  344, 163840, 3870, 5494),
    "MT448x128": (14, 4,  472, 147456, 5770, 8422),
    "MT128x384": (2,  24, 440, 131072, 5378, 7538),
    "MT512x128": (16, 4,  472, 163840, 6567, 9603),
    "MT384x192": (12, 6,  504, 147456, 7370, 10776),
    "MT256x384": (8,  12, 512, 163840, 9780, 14328),
    "MT384x256": (12, 8,  512, 163840, 9691, 14235),
}

# From parse_baselines.out: (M, N, wgCount, noepi_us, mfe_us, delta_us).
TIMING = {
    "MT576x64":  (576, 64,  1920, 1320.4, 1397.0, 76.6),
    "MT128x384": (128, 384, 1408, 895.9,  1026.2, 130.3),
    "MT448x128": (448, 128, 1216, 1010.3, 1239.0, 228.7),
    "MT128x512": (128, 512, 1024, 774.8,  914.8,  140.0),
    "MT512x128": (512, 128, 1024, 780.2,  875.0,  94.7),
    "MT384x192": (384, 192, 946,  802.9,  879.5,  76.6),
    "MT192x448": (192, 448, 817,  855.2,  1000.3, 145.1),
    "MT448x192": (448, 192, 817,  846.0,  987.9,  142.0),
    "MT256x384": (256, 384, 704,  763.0,  892.8,  129.8),
    "MT384x256": (384, 256, 704,  751.2,  874.4,  123.3),
}


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def linfit(xs, ys):
    """Least-squares slope, intercept."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    return a, b


def main():
    keys = list(INSTR.keys())
    wtArea = [INSTR[k][0] * INSTR[k][1] for k in keys]
    fused = [INSTR[k][4] for k in keys]
    execE = [INSTR[k][5] for k in keys]
    wg = [TIMING[k][2] for k in keys]
    delta = [TIMING[k][5] for k in keys]

    print("=" * 78)
    print("H1/H3: does fused-pass instruction count scale with wave-tile area?")
    print("=" * 78)
    print(f"  corr(wtArea, fusedInstr)  = {pearson(wtArea, fused):+.4f}")
    print(f"  corr(wtArea, execInstr)   = {pearson(wtArea, execE):+.4f}")
    a, b = linfit(wtArea, fused)
    print(f"  fusedInstr ~= {a:.1f}*wtArea + {b:.1f}")
    winner = INSTR["MT384x256"]
    fixedFrac = b / winner[4]
    print(f"  fixed intercept fraction at winner (MT384x256) = {b:.0f}/{winner[4]} "
          f"= {fixedFrac*100:.1f}%")
    print(f"  => instruction count is ~LINEAR in wtArea; fixed cost is small.")
    print()
    print(f"  {'MT':<11}{'wtArea':>7}{'fused':>7}{'fused/wtArea':>14}")
    for k in keys:
        wa = INSTR[k][0] * INSTR[k][1]
        print(f"  {k:<11}{wa:>7}{INSTR[k][4]:>7}{INSTR[k][4]/wa:>14.1f}")

    print()
    print("=" * 78)
    print("Aggregate epilogue work = fusedInstr * numWorkgroups  (occupancy=1)")
    print("=" * 78)
    agg = [f * w for f, w in zip(fused, wg)]
    print(f"  {'MT':<11}{'wtArea':>7}{'fused':>7}{'numWG':>8}{'fused*WG (M)':>14}")
    for i, k in enumerate(keys):
        print(f"  {k:<11}{wtArea[i]:>7}{fused[i]:>7}{wg[i]:>8}{agg[i]/1e6:>14.2f}")
    print(f"  mean={statistics.mean(agg)/1e6:.2f}M  stdev={statistics.pstdev(agg)/1e6:.2f}M "
          f"cv={statistics.pstdev(agg)/statistics.mean(agg)*100:.1f}%")
    print(f"  corr(wtArea, fused*WG) = {pearson(wtArea, agg):+.4f}")
    print(f"  => aggregate epilogue instruction work is ~tile-area-INVARIANT.")

    print()
    print("=" * 78)
    print("H5: corr(wtArea, delta_us) on this 7-tile subset, with outlier removed")
    print("=" * 78)
    print(f"  corr(wtArea, delta_us)          = {pearson(wtArea, delta):+.4f}")
    print(f"  corr(fusedInstr, delta_us)      = {pearson(fused, delta):+.4f}")
    print(f"  corr(numWG, delta_us)           = {pearson(wg, delta):+.4f}")
    # Remove the MT448x128 leverage outlier.
    idx = [i for i, k in enumerate(keys) if k != "MT448x128"]
    wa2 = [wtArea[i] for i in idx]
    d2 = [delta[i] for i in idx]
    print(f"  corr(wtArea, delta_us) w/o MT448x128 = {pearson(wa2, d2):+.4f}  (n=6)")

    # Full 10-tile from timing only (area vs delta) for reference.
    allWa = [TIMING[k][0] * TIMING[k][1] for k in TIMING]
    allD = [TIMING[k][5] for k in TIMING]
    print(f"  corr(MTarea, delta_us) full 10 tiles = {pearson(allWa, allD):+.4f}  (n=10)")
    allW = [TIMING[k][2] for k in TIMING]
    print(f"  corr(numWG, delta_us) full 10 tiles  = {pearson(allW, allD):+.4f}  (n=10)")


if __name__ == "__main__":
    main()
