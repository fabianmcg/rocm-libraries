# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Extract MegaFused epilogue instruction counts per tile from built .s files.

For each MegaFused (MFE1) kernel assembly, this locates the epilogue section
boundaries by banner comments, then counts real instructions per category in:

  - fused pass       [Global Write Elements, first Global Write Batch #0)
                     = residual load + residual-add + rmsSum + gamma + ResidualOut
                       store + cross-wave RMS reduction (MegaFused-specific work).
  - nonedge store    [first Global Write Batch #0, first Global Write Edge Batch #0)
                     = executed bf16 D store (aligned problem never takes Edge path).
  - executed epilogue = fused pass + nonedge store (Edge batches are dead code).

Instructions are AMDGPU opcodes only (lines whose first token starts with a
known opcode prefix); comments, labels, and directives are excluded.
"""

import re
from pathlib import Path

ASM_DIR = Path(
    "/tmp/baselines-sweep/1_BenchmarkProblems/"
    "Cijk_Alik_Bljk_BBS_H_PRMS_RA_MFE_UserArgs_00/00_Final/caches/"
    "2d8888d81401/source/build_tmp/SOURCE/assembly"
)

# MT -> (assembly hash fragment, wave-tile M, wave-tile N) for representative tiles.
# Picked to span the surviving tile-area range 36864 .. 98304.
TILES = [
    ("MT576x64",  "yMkZD96FHHbTVzSJ", 9,  4),
    ("MT448x128", "fWwsnFOcRhzhhuSZ", 14, 4),
    ("MT128x384", "MuRpmy8iUWWeY7wW", 2,  24),
    ("MT512x128", "LEjYLuADH8fx8Kit", 16, 4),
    ("MT384x192", "PKuBwwn9DC3x0SxL", 12, 6),
    ("MT256x384", "JnlfQ0m8UEh2GmOs", 8,  12),
    ("MT384x256", "YUnVh8K705iHVAc2", 12, 8),
]

OPCODE_RE = re.compile(r"^(s_|v_|ds_|buffer_|global_|flat_|scratch_|image_)")

CATEGORIES = {
    "buffer_load":     re.compile(r"^buffer_load"),
    "buffer_store":    re.compile(r"^buffer_store"),
    "buf_store_short": re.compile(r"^buffer_store_short"),
    "buf_store_dwx2": re.compile(r"^buffer_store_dwordx2"),
    "v_pk_add_f32":    re.compile(r"^v_pk_add_f32"),
    "v_add_f32":       re.compile(r"^v_add_f32"),
    "v_fma/v_fmac":    re.compile(r"^v_fma|^v_fmac"),
    "v_pk_mul_f32":    re.compile(r"^v_pk_mul_f32"),
    "v_mul_f32":       re.compile(r"^v_mul_f32"),
    "v_cmp":           re.compile(r"^v_cmp"),
    "v_cndmask":       re.compile(r"^v_cndmask"),
    "s_waitcnt":       re.compile(r"^s_waitcnt"),
    "s_and_saveexec":  re.compile(r"^s_and_saveexec|^s_andn2_saveexec"),
    "s_mov_b64_exec":  re.compile(r"^s_mov_b64 exec|^s_mov_b64 s\[\d+:\d+\], exec"),
    "ds_bpermute":     re.compile(r"^ds_bpermute"),
    "s_barrier":       re.compile(r"^s_barrier"),
    "v_cvt_pk*bf16":   re.compile(r"^v_cvt_pk.*bf16|^v_cvt_pk_bf16"),
}


def stripLine(raw):
    """Return the opcode-relevant token stream, or '' for non-instruction lines."""
    # Drop trailing // or /* comments for matching, but keep 'exec' operands.
    s = raw.strip()
    if not s or s.startswith("/*") or s.startswith("."):
        return ""
    if s.endswith(":"):
        return ""
    # Strip trailing line comment.
    s = re.split(r"//", s)[0].strip()
    if not OPCODE_RE.match(s):
        return ""
    return s


def findBoundaries(lines):
    """Return (epiStart, storeStart, edgeStart, lastMfma, endLine) 0-based indices."""
    epiStart = storeStart = edgeStart = lastMfma = endLine = None
    for i, raw in enumerate(lines):
        if epiStart is None and "/* Global Write Elements" in raw:
            epiStart = i
        if storeStart is None and "/* Global Write Batch #0" in raw:
            storeStart = i
        if edgeStart is None and "/* Global Write Edge Batch #0" in raw:
            edgeStart = i
        if "v_mfma" in raw:
            lastMfma = i
        if "s_endpgm" in raw:
            endLine = i
    return epiStart, storeStart, edgeStart, lastMfma, endLine


def countRange(lines, lo, hi):
    """Count total instructions and per-category counts in [lo, hi)."""
    total = 0
    cats = {k: 0 for k in CATEGORIES}
    for raw in lines[lo:hi]:
        s = stripLine(raw)
        if not s:
            continue
        total += 1
        for name, rx in CATEGORIES.items():
            if rx.match(s):
                cats[name] += 1
    return total, cats


def readVgpr(lines):
    """Return (numVgpr, numAccVgpr, nextFreeVgpr) from header comments/directives."""
    numVgpr = numAcc = nextFree = None
    for raw in lines[:60]:
        m = re.search(r"Num VGPR\s*=\s*(\d+)", raw)
        if m:
            numVgpr = int(m.group(1))
        m = re.search(r"Num AccVGPR\s*=\s*(\d+)", raw)
        if m:
            numAcc = int(m.group(1))
        m = re.search(r"amdhsa_next_free_vgpr\s+(\d+)", raw)
        if m:
            nextFree = int(m.group(1))
    return numVgpr, numAcc, nextFree


def findAsm(fragment):
    for p in ASM_DIR.glob("*.s"):
        if fragment in p.name:
            return p
    return None


def main():
    print(f"{'MT':<11} {'wtArea':>6} {'vgpr':>5} {'acc':>4} {'nfV':>4} "
          f"{'occ':>4} {'fused':>6} {'nonEdge':>7} {'exec':>6}")
    print("-" * 70)
    rows = []
    for mt, frag, wtM, wtN in TILES:
        p = findAsm(frag)
        if p is None:
            print(f"{mt:<11} ASM NOT FOUND ({frag})")
            continue
        lines = p.read_text().splitlines()
        epi, store, edge, lastMfma, end = findBoundaries(lines)
        numVgpr, numAcc, nextFree = readVgpr(lines)
        # gfx950 unified VGPR file = 512 per lane; occupancy waves/SIMD.
        occ = 512 // nextFree if nextFree else 0
        wtArea = wtM * wtN
        fusedTot, fusedCats = countRange(lines, epi, store)
        nonEdgeTot, nonEdgeCats = countRange(lines, store, edge)
        execTot, execCats = countRange(lines, epi, edge)
        mfmaOk = lastMfma < epi
        rows.append((mt, wtArea, wtM, wtN, numVgpr, numAcc, nextFree, occ,
                     fusedTot, nonEdgeTot, execTot, fusedCats, nonEdgeCats,
                     execCats, mfmaOk, epi, store, edge))
        print(f"{mt:<11} {wtArea:>6} {numVgpr:>5} {numAcc:>4} {nextFree:>4} "
              f"{occ:>4} {fusedTot:>6} {nonEdgeTot:>7} {execTot:>6}")

    print("\nlast v_mfma precedes epilogue (post-MFMA tail) for all tiles: "
          + str(all(r[14] for r in rows)))

    print("\n=== Per-category counts in FUSED PASS (MegaFused-specific) ===")
    cats = list(CATEGORIES.keys())
    hdr = f"{'category':<18}" + "".join(f"{r[0][:9]:>10}" for r in rows)
    print(hdr)
    print("-" * len(hdr))
    for c in cats:
        line = f"{c:<18}" + "".join(f"{r[11][c]:>10}" for r in rows)
        print(line)

    print("\n=== Per-category counts in EXECUTED EPILOGUE (fused + nonEdge store) ===")
    print(hdr)
    print("-" * len(hdr))
    for c in cats:
        line = f"{c:<18}" + "".join(f"{r[13][c]:>10}" for r in rows)
        print(line)

    print("\n=== Boundary line numbers (1-based) ===")
    print(f"{'MT':<11} {'epiStart':>9} {'storeStart':>11} {'edgeStart':>10}")
    for r in rows:
        print(f"{r[0]:<11} {r[15]+1:>9} {r[16]+1:>11} {r[17]+1:>10}")


if __name__ == "__main__":
    main()
