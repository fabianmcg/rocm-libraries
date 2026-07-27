# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""bench_comparison.py: full-pipeline bf16 GEMM benchmark comparison.

Runs BenchmarkProblems -> LibraryLogic -> ClientWriter, then benchmarks
the resulting library with both the Python SweepRunner and the C++ client.
Produces a Markdown report comparing cold/warm Python GFLOPS vs C++ GFLOPS.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

thisDir = Path(__file__).resolve().parent
# parents[2] of __file__ itself = tensilelite root (Tensile/client/file -> [0]=client [1]=Tensile [2]=tensilelite).
tensileliteRoot = Path(__file__).resolve().parents[2]

# When run as a script, Python replaces sys.path[0] with the script directory,
# removing the '' entry that normally resolves to the CWD (tensilelite root).
# Insert the root explicitly so 'Tensile' package is always importable.
_tensileRootStr = str(tensileliteRoot)
if _tensileRootStr not in sys.path:
    sys.path.insert(0, _tensileRootStr)
tensileBin = tensileliteRoot / "Tensile" / "bin" / "Tensile"
clientExe = tensileliteRoot / "build_tmp" / "tensilelite" / "client" / "tensilelite-client"
defaultCompanionYaml = thisDir / "bench_comparison.yaml"


# ---------------------------------------------------------------------------
# Phase 0 — setup helpers.
# ---------------------------------------------------------------------------


def detectArch(argArch):
    """Return arch string: use argArch if truthy, else detect from GPU."""
    if argArch:
        return argArch
    import amdgpu_exec
    return amdgpu_exec.get_chip().split(":")[0]


def dropPageCache():
    """Drop OS page cache; best-effort, failure never aborts the script."""
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=False,
        )
    except Exception:
        pass


def printBanner(args, arch, outputDir):
    """Print a clear summary of what is about to run."""
    print("=" * 72)
    print("bench_comparison: full-pipeline bf16 GEMM benchmark")
    print(f"  YAML          : {args.yaml}")
    print(f"  arch          : {arch}")
    print(f"  output-dir    : {outputDir}")
    print(f"  validate      : {args.num_elements_to_validate}")
    print(f"  num-benchmarks: {args.num_benchmarks}")
    print(f"  num-warmups   : {args.num_warmups}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Phase 1-3 — pipeline.
# ---------------------------------------------------------------------------


def runPipeline(yamlPath, pipelineDir, arch, logPath):
    """Run the full Tensile pipeline (BenchmarkProblems -> ClientWriter)."""
    cmd = [sys.executable, str(tensileBin), yamlPath, pipelineDir,
           "--gpu-targets", arch]
    print(f"Running pipeline; log -> {logPath}")
    with open(logPath, "w") as f:
        result = subprocess.run(
            cmd, cwd=str(tensileliteRoot), stdout=f, stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Tensile pipeline failed (rc={result.returncode}); see {logPath}"
        )


def _findLibraryYaml(pipelineDir, arch):
    """Return path to TensileLibrary*.yaml under 4_LibraryClient/{arch}/."""
    direct = f"{pipelineDir}/4_LibraryClient/library/{arch}/TensileLibrary*.yaml"
    matches = glob.glob(direct)
    if not matches:
        matches = glob.glob(
            f"{pipelineDir}/4_LibraryClient/**/TensileLibrary*.yaml", recursive=True
        )
    if not matches:
        raise FileNotFoundError(
            f"TensileLibrary*.yaml not found; tried:\n  {direct}\n"
            f"  {pipelineDir}/4_LibraryClient/**/TensileLibrary*.yaml"
        )
    return matches[0]


def _findIniPath(pipelineDir):
    """Return the first non-Granularity ClientParameters_*.ini path."""
    matches = glob.glob(
        f"{pipelineDir}/4_LibraryClient/source/ClientParameters_*.ini"
    )
    matches = [p for p in matches if "_Granularity" not in p]
    if not matches:
        raise FileNotFoundError(
            f"No ClientParameters_*.ini found under "
            f"{pipelineDir}/4_LibraryClient/source/"
        )
    return matches[0]


def locateArtifacts(pipelineDir, arch):
    """Return (libraryYaml, iniPath) from 4_LibraryClient; raise if missing."""
    libraryYaml = _findLibraryYaml(pipelineDir, arch)
    libDir = os.path.dirname(libraryYaml)
    coFiles = glob.glob(os.path.join(libDir, "*.co"))
    if not coFiles:
        raise FileNotFoundError(f"No .co files found next to library: {libDir}")
    iniPath = _findIniPath(pipelineDir)
    return libraryYaml, iniPath


# ---------------------------------------------------------------------------
# Phase 2 — Python benchmark (library mode).
# ---------------------------------------------------------------------------


def parsePythonResults(results):
    """Convert SweepResult list to {(M,N,batch,K): (gflops, validation)}."""
    bySize = {}
    for r in results:
        size = tuple(r.problemSize[:4])
        existing = bySize.get(size)
        if existing is None or r.gflops > existing[0]:
            bySize[size] = (r.gflops, r.validation)
    return bySize


def _runOnePythonBench(companionYaml, libraryYaml, validateN, nWarmup, nIters,
                       outDir, runIndex):
    """Run SweepRunner once; return (wall, bySize)."""
    from Tensile.client.sweep_runner import SweepRunner
    csvPath = os.path.join(outDir, f"python_run{runIndex}.csv")
    t0 = time.perf_counter()
    runner = SweepRunner(
        yamlPath=companionYaml,
        libraryPath=libraryYaml,
        numElementsToValidate=validateN,
        nWarmup=nWarmup,
        nIters=nIters,
    )
    results = runner.run(resultsCsv=csvPath)
    wall = time.perf_counter() - t0
    return wall, parsePythonResults(results)


def runPythonBench(companionYaml, libraryYaml, validateN, nWarmup, nIters, outDir):
    """Run SweepRunner 3 times; run 0 is cold (cleared cache), 1-2 are warm."""
    from Tensile.client.sweep_runner import _cachedIsaInfoMap
    # Clear cache so run 0 is genuinely cold (forces ISA detection).
    _cachedIsaInfoMap.cache_clear()
    dropPageCache()
    runs = []
    for i in range(3):
        print(f"  Python bench run {i} ...")
        wall, bySize = _runOnePythonBench(
            companionYaml, libraryYaml, validateN, nWarmup, nIters, outDir, i
        )
        runs.append({"wall": wall, "bySize": bySize})
        print(f"    wall={wall:.1f}s, sizes={len(bySize)}")
    return runs


# ---------------------------------------------------------------------------
# Phase 3 — C++ benchmark.
# ---------------------------------------------------------------------------


def parseCppCsv(csvPath):
    """Parse the C++ results CSV; return {(M,N,batch,K): gflops}.

    The CSV produced by the client has the format:
      GFlops, SizeI, SizeJ, SizeK, SizeL, ..., <KernelName>
    The 'GFlops' column holds a per-benchmark run index, not the actual
    performance. The last column (the kernel name) contains the real GFLOPS.
    """
    bySize = {}
    with open(csvPath, newline="") as f:
        reader = csv.DictReader(f)
        # Strip whitespace from all header names.
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        # Assumes a single-solution library so the last column holds winner
        # GFLOPS per row; a multi-solution library needs max-over-kernel-columns.
        gflopsField = reader.fieldnames[-1] if reader.fieldnames else None
        if not gflopsField:
            return bySize
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                gflops = float(row.get(gflopsField, "nan"))
            except ValueError:
                continue
            if not math.isfinite(gflops) or gflops <= 0:
                continue
            try:
                sizeI = int(row["SizeI"])
                sizeJ = int(row["SizeJ"])
                sizeK = int(row["SizeK"])
                sizeL = int(row["SizeL"])
            except (KeyError, ValueError):
                continue
            size = (sizeI, sizeJ, sizeK, sizeL)
            bySize[size] = max(bySize.get(size, 0.0), gflops)
    return bySize


def _patchIni(srcIniPath, patchedIniPath, overrides):
    """Write a copy of srcIniPath with overrides applied.

    overrides is a dict of key->value strings. Keys already present in the
    source are replaced; keys not present are appended. The custom po::store
    always lets the last loaded file win, so we write a per-run INI rather
    than relying on command-line flags to override INI values.
    """
    with open(srcIniPath) as f:
        lines = f.readlines()
    replaced = set()
    patched = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in overrides:
            patched.append(f"{key}={overrides[key]}\n")
            replaced.add(key)
        else:
            patched.append(line)
    for key, val in overrides.items():
        if key not in replaced:
            patched.append(f"{key}={val}\n")
    with open(patchedIniPath, "w") as f:
        f.writelines(patched)


def _runOneCppBench(iniPath, validateN, nBench, nWarmup, outDir, runIndex):
    """Run the C++ client once; return (wall, bySize)."""
    csvPath = os.path.join(outDir, f"cpp_run{runIndex}.csv")
    logPath = os.path.join(outDir, f"cpp_run{runIndex}.log")
    # Write a patched INI so our overrides aren't silently lost to the
    # custom po::store that lets the last-loaded file always win.
    patchedIni = os.path.join(outDir, f"cpp_run{runIndex}.ini")
    _patchIni(iniPath, patchedIni, {
        "results-file": csvPath,
        "num-syncs-per-benchmark": "10",
        "num-benchmarks": str(nBench),
        "num-warmups": str(nWarmup),
        "num-elements-to-validate": str(validateN),
    })
    cmd = [str(clientExe), "--config-file", patchedIni]
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, cwd=str(tensileliteRoot), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    wall = time.perf_counter() - t0
    with open(logPath, "w") as f:
        f.write(result.stdout)
    if "DID_NOT_SATISFY_ASSERTS" in result.stdout:
        raise RuntimeError(
            f"C++ client validation failed (DID_NOT_SATISFY_ASSERTS) on run "
            f"{runIndex}; see {logPath}"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ client failed (rc={result.returncode}) on run {runIndex}; "
            f"see {logPath}"
        )
    bySize = parseCppCsv(csvPath)
    return wall, bySize


def runCppBench(iniPath, validateN, nBench, nWarmup, outDir):
    """Run the C++ client 3 times; raise on DID_NOT_SATISFY_ASSERTS."""
    if not clientExe.exists():
        raise FileNotFoundError(
            f"C++ client not found: {clientExe}\n"
            "Run `invoke build-client` to build it."
        )
    dropPageCache()
    runs = []
    for i in range(3):
        print(f"  C++ bench run {i} ...")
        wall, bySize = _runOneCppBench(iniPath, validateN, nBench, nWarmup, outDir, i)
        runs.append({"wall": wall, "bySize": bySize})
        print(f"    wall={wall:.1f}s, sizes={len(bySize)}")
    return runs


# ---------------------------------------------------------------------------
# Phase 4 — report helpers.
# ---------------------------------------------------------------------------


def median(values):
    """Return statistics.median for non-empty list, nan otherwise."""
    if not values:
        return float("nan")
    return statistics.median(values)


def pythonColdBySize(runs):
    """Return the cold-run (run 0) bySize dict."""
    return runs[0]["bySize"]


def pythonWarmBySize(runs):
    """Return median GFLOPS per size from warm runs (1 and 2), and median wall."""
    allSizes = set(runs[1]["bySize"]) | set(runs[2]["bySize"])
    bySize = {}
    for size in allSizes:
        vals = []
        for run in runs[1:]:
            entry = run["bySize"].get(size)
            if entry is not None:
                vals.append(entry[0])
        if vals:
            bySize[size] = median(vals)
    warmWall = median([runs[1]["wall"], runs[2]["wall"]])
    return bySize, warmWall


def cppMedianBySize(runs):
    """Return median GFLOPS per size across all 3 C++ runs, and median wall."""
    allSizes = set()
    for run in runs:
        allSizes.update(run["bySize"].keys())
    bySize = {}
    for size in allSizes:
        vals = [run["bySize"][size] for run in runs if size in run["bySize"]]
        if vals:
            bySize[size] = median(vals)
    cppWall = median([r["wall"] for r in runs])
    return bySize, cppWall


def _fmtSize(size):
    """Format (M,N,batch,K) tuple as MxNxbatchxK string."""
    return "x".join(str(x) for x in size)


def _fmtDelta(pyGflops, cppGflops):
    """Format percentage delta (pyGflops - cpp) / cpp * 100 with sign."""
    if cppGflops == 0 or not math.isfinite(cppGflops) or not math.isfinite(pyGflops):
        return "N/A"
    delta = (pyGflops - cppGflops) / cppGflops * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _coldTable(coldBySize):
    """Return Markdown rows for the cold Python run."""
    lines = ["| Size | GFLOPS | Validation |", "| --- | --- | --- |"]
    for size in sorted(coldBySize):
        gflops, validation = coldBySize[size]
        lines.append(f"| {_fmtSize(size)} | {gflops:.1f} | {validation} |")
    return "\n".join(lines)


def _warmTable(warmBySize):
    """Return Markdown rows for the warm Python run."""
    lines = ["| Size | GFLOPS |", "| --- | --- |"]
    for size in sorted(warmBySize):
        lines.append(f"| {_fmtSize(size)} | {warmBySize[size]:.1f} |")
    return "\n".join(lines)


def _cppTable(cppBySize):
    """Return Markdown rows for C++ median results."""
    lines = ["| Size | GFLOPS | Validation |", "| --- | --- | --- |"]
    for size in sorted(cppBySize):
        lines.append(f"| {_fmtSize(size)} | {cppBySize[size]:.1f} | PASS (no DID_NOT_SATISFY_ASSERTS) |")
    return "\n".join(lines)


def _deltaTable(warmBySize, cppBySize):
    """Return Markdown delta table comparing warm Python vs C++ median."""
    common = sorted(set(warmBySize) & set(cppBySize))
    if not common:
        return "No common sizes to compare."
    lines = ["| Size | Python warm (GFLOPS) | C++ median (GFLOPS) | Delta |",
             "| --- | --- | --- | --- |"]
    for size in common:
        pyG = warmBySize[size]
        cppG = cppBySize[size]
        lines.append(
            f"| {_fmtSize(size)} | {pyG:.1f} | {cppG:.1f} | {_fmtDelta(pyG, cppG)} |"
        )
    return "\n".join(lines)


def _buildReportSections(reportPath, args, arch, pyRuns, coldBySize, warmBySize,
                         warmWall, cppBySize, cppWall, cppRuns):
    """Return the list of Markdown section strings for the report."""
    pyRun0Wall = pyRuns[0]["wall"]
    pyRun1Wall = pyRuns[1]["wall"]
    pyRun2Wall = pyRuns[2]["wall"]
    nConfigs = sum(len(r["bySize"]) for r in pyRuns[:1])
    nSizes = len(coldBySize)
    validationNote = (
        f"num-elements-to-validate={args.num_elements_to_validate} "
        f"({'all elements' if args.num_elements_to_validate == -1 else str(args.num_elements_to_validate) + ' elements'})"
    )
    return [
        "# bench_comparison: total sweep wall-clock report",
        "",
        "## Configuration",
        f"- arch: {arch}",
        f"- yaml: {args.yaml}",
        f"- output-dir: {os.path.dirname(reportPath)}",
        f"- validation: {validationNote}",
        f"- num-benchmarks: {args.num_benchmarks}",
        f"- num-warmups: {args.num_warmups}",
        f"- configs benchmarked: {nConfigs}",
        f"- problem sizes: {nSizes}",
        "",
        "## Total sweep wall-clock (all configs × all sizes)",
        "",
        "| Run | Client | Wall-clock (s) | Note |",
        "| --- | --- | --- | --- |",
        f"| 0 | Python | {pyRun0Wall:.2f} | cold (includes ISA detection) |",
        f"| 1 | Python | {pyRun1Wall:.2f} | warm |",
        f"| 2 | Python | {pyRun2Wall:.2f} | warm |",
        f"| median warm | Python | {warmWall:.2f} | median of runs 1-2 |",
        f"| 0 | C++ | {cppRuns[0]['wall']:.2f} | cold page cache |",
        f"| 1 | C++ | {cppRuns[1]['wall']:.2f} | |",
        f"| 2 | C++ | {cppRuns[2]['wall']:.2f} | |",
        f"| median | C++ | {cppWall:.2f} | median of 3 runs |",
        "",
        f"**Python warm speedup vs C++: {cppWall / warmWall:.1f}×**",
        f"**Python cold vs C++: {cppWall / pyRun0Wall:.1f}×**",
    ]


def writeReport(reportPath, args, arch, pyRuns, cppRuns):
    """Compose and write the Markdown report; also print to stdout."""
    coldBySize = pythonColdBySize(pyRuns)
    warmBySize, warmWall = pythonWarmBySize(pyRuns)
    cppBySize, cppWall = cppMedianBySize(cppRuns)
    sections = _buildReportSections(
        reportPath, args, arch, pyRuns, coldBySize, warmBySize, warmWall,
        cppBySize, cppWall, cppRuns,
    )
    report = "\n".join(sections) + "\n"
    print(report)
    with open(reportPath, "w") as f:
        f.write(report)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def buildParser():
    """Return the argument parser for bench_comparison."""
    parser = argparse.ArgumentParser(
        description="Full-pipeline bf16 GEMM benchmark comparison."
    )
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: /tmp/tensilelite_bench_<ts>).")
    parser.add_argument("--yaml", default=str(defaultCompanionYaml),
                        help="Full-pipeline benchmark YAML.")
    parser.add_argument("--arch", default=None,
                        help="GPU architecture (e.g. gfx950); auto-detected if omitted.")
    parser.add_argument("--num-elements-to-validate", type=int, default=-1,
                        help="Elements to validate (-1 = all, 0 = skip).")
    parser.add_argument("--num-benchmarks", type=int, default=10)
    parser.add_argument("--num-warmups", type=int, default=3)
    return parser


def main():
    """Entry point: parse args, run pipeline, benchmark, write report."""
    parser = buildParser()
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = args.output_dir or f"/tmp/tensilelite_bench_{timestamp}"
    os.makedirs(outputDir, exist_ok=True)
    pipelineDir = os.path.join(outputDir, "pipeline")
    os.makedirs(pipelineDir, exist_ok=True)

    arch = detectArch(args.arch)
    printBanner(args, arch, outputDir)

    pipelineLog = os.path.join(outputDir, "pipeline.log")
    runPipeline(args.yaml, pipelineDir, arch, pipelineLog)

    libraryYaml, iniPath = locateArtifacts(pipelineDir, arch)
    print(f"Library YAML : {libraryYaml}")
    print(f"INI path     : {iniPath}")

    print("\nRunning Python benchmark (library mode) ...")
    pyRuns = runPythonBench(
        args.yaml, libraryYaml,
        args.num_elements_to_validate, args.num_warmups, args.num_benchmarks,
        outputDir,
    )

    print("\nRunning C++ benchmark ...")
    cppRuns = runCppBench(
        iniPath, args.num_elements_to_validate, args.num_benchmarks,
        args.num_warmups, outputDir,
    )

    reportPath = os.path.join(outputDir, "bench_report.md")
    writeReport(reportPath, args, arch, pyRuns, cppRuns)
    print(f"\nDONE. Report written to: {reportPath}")


if __name__ == "__main__":
    main()
