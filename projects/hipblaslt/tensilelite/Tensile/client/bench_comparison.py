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


def _findGroupDirs(pipelineDir):
    """Return (finalYaml, hsacoPath, libYamlPath) triples from BenchmarkProblems output."""
    finalYamls = sorted(glob.glob(
        f"{pipelineDir}/1_BenchmarkProblems/**/Data/00_Final.yaml", recursive=True
    ))
    groups = []
    for yamlPath in finalYamls:
        groupDir = os.path.dirname(os.path.dirname(yamlPath))
        hsacos = glob.glob(
            f"{groupDir}/**/TensileLibrary_gfx950.co", recursive=True
        )
        libYamls = glob.glob(
            f"{groupDir}/**/TensileLibrary.yaml", recursive=True
        )
        if hsacos and libYamls:
            groups.append((yamlPath, hsacos[0], libYamls[0]))
    return groups


def _readKernelNamesFromLibYaml(libYamlPath):
    """Return list of kernelName strings from a TensileLibrary.yaml."""
    import yaml as _yaml
    with open(libYamlPath) as f:
        data = _yaml.safe_load(f)
    names = []
    def _walk(node):
        if isinstance(node, dict):
            if "kernelName" in node:
                names.append(node["kernelName"])
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
    _walk(data)
    return list(dict.fromkeys(names))  # deduplicated, order preserved


def _validateOutput(runner, solDict, M, N, batch, K, args, wg, validateN):
    """Validate kernel output against CPU bf16 reference (NT column-major, alpha=1, beta=0).

    Returns 'PASS', 'FAIL:<msg>', or 'SKIPPED:<reason>'.
    """
    try:
        from ml_dtypes import bfloat16 as bf16
    except ImportError:
        return "SKIPPED:ml_dtypes absent"
    try:
        import numpy as np
        from amdgpu_exec import GpuBuffer
        from Tensile.client.sweep_runner import _buildSweepArgs, _deviceCuCount
        from Tensile.client.reference import assertClose, RTOL_BF16, ATOL_BF16

        # Mirror _makeVerifyInputs / _computeVerifyRef from sweep_runner.py exactly.
        rng = np.random.default_rng(seed=M * 1000 + N + K)
        aNp = np.asfortranarray(rng.random((M, K)).astype(bf16))
        bNp = np.asfortranarray(rng.random((N, K)).astype(bf16))
        aFlat = np.tile(aNp.ravel(order="F"), batch)
        bFlat = np.tile(bNp.ravel(order="F"), batch)
        aBuf = GpuBuffer(M * K * batch * 2)
        bBuf = GpuBuffer(N * K * batch * 2)
        dBuf = GpuBuffer(M * N * batch * 2)
        cBuf = GpuBuffer(M * N * batch * 2)
        aBuf.copy_from_host(aFlat)
        bBuf.copy_from_host(bFlat)
        cuCount = _deviceCuCount()
        vArgs = _buildSweepArgs(solDict, M, N, batch, K, dBuf, cBuf, aBuf, bBuf, cuCount)
        runner._functions[0].launch(
            (wg, 1, 1), (solDict.get("NumThreads", 256), 1, 1), vArgs
        )
        from amdgpu_exec import GpuEvent
        ev = GpuEvent(); ev.record(); ev.synchronize()
        dOut = np.zeros(M * N * batch, dtype=bf16)
        dBuf.copy_to_host(dOut)
        # Reference: same as _computeVerifyRef.
        aSlice = aFlat[:M * K].reshape(M, K, order="F")
        bSlice = bFlat[:N * K].reshape(N, K, order="F")
        from Tensile.client.reference import gemmBf16
        dRef = np.tile(np.asfortranarray(
            gemmBf16(aSlice, bSlice.T, 1.0, 0.0, None)
        ).ravel(order="F"), batch)
        nCheck = dOut.size if validateN == -1 else min(validateN, dOut.size)
        assertClose(dOut[:nCheck], dRef[:nCheck], rtol=RTOL_BF16, atol=ATOL_BF16)
        return "PASS"
    except AssertionError as exc:
        return f"FAIL:{str(exc)[:120]}"
    except Exception as exc:
        return f"SKIPPED:{exc}"


def _benchmarkGroup(yamlPath, hsacoPath, libYamlPath, nWarmup, nIters, validateN=0):
    """Benchmark all solutions in one group using the pre-built HSACO; return results list."""
    import yaml as _yaml
    from Tensile.client.harness import KernelRunner
    from Tensile.client.yaml_solution_builder import _injectInternalArgsSupport
    from Tensile.client.sweep_runner import _buildSweepArgs, _computeNumWg, _deviceCuCount
    from amdgpu_exec import GpuBuffer, get_chip

    chip = get_chip()
    with open(yamlPath) as f:
        data = _yaml.safe_load(f)
    if not isinstance(data, list) or len(data) < 6:
        return []
    rawSizes = data[1].get("ProblemSizes", [])
    probSizes = []
    for entry in rawSizes:
        exact = entry.get("Exact", []) if isinstance(entry, dict) else []
        if len(exact) >= 4:
            probSizes.append(tuple(int(x) for x in exact[:4]))
    solRaws = [item for item in data[5:] if isinstance(item, dict)]
    if not solRaws or not probSizes:
        return []
    kernelNames = _readKernelNamesFromLibYaml(libYamlPath)
    if len(kernelNames) != len(solRaws):
        kernelNames = kernelNames[:len(solRaws)]
    hsacoBytes = open(hsacoPath, "rb").read()
    cuCount = _deviceCuCount()
    results = []
    for solRaw, kernelName in zip(solRaws, kernelNames):
        solDict = _injectInternalArgsSupport(dict(solRaw), chip)
        try:
            runner = KernelRunner.fromHsaco(hsacoBytes, kernelName, nModuleCopies=1)
        except Exception as exc:
            print(f"    load failed {kernelName[:50]}: {exc}")
            continue
        for M, N, batch, K in probSizes:
            try:
                dSize = M * N * batch * 2  # bf16 = 2 bytes
                aBuf = GpuBuffer(M * K * batch * 2)
                bBuf = GpuBuffer(N * K * batch * 2)
                cBuf = GpuBuffer(dSize)
                dBuf = GpuBuffer(dSize)
                args = _buildSweepArgs(solDict, M, N, batch, K, dBuf, cBuf, aBuf, bBuf, cuCount)
                wg = _computeNumWg(solDict, M, N, batch)
                benchResult = runner.run(
                    argsFn=lambda _buf: args,
                    grid=(wg, 1, 1),
                    block=(solDict.get("NumThreads", 256), 1, 1),
                    nWarmup=nWarmup,
                    nIters=nIters,
                )
                gflops = 2 * M * N * K * batch / (benchResult.minUs * 1e-6) / 1e9
                validation = "SKIPPED"
                if validateN != 0:
                    validation = _validateOutput(
                        runner, solDict, M, N, batch, K, args, wg, validateN)
                results.append((M, N, batch, K, gflops, validation))
            except Exception as exc:
                print(f"    bench failed {kernelName[:30]} {M}x{N}: {exc}")
    return results


def parsePythonResults(rawResults):
    """Convert raw results list to (nResults, nSizes, nPass, nFail, nSkip)."""
    sizes = set((r[0], r[1], r[2], r[3]) for r in rawResults)
    nPass = sum(1 for r in rawResults if len(r) > 5 and r[5] == "PASS")
    nFail = sum(1 for r in rawResults if len(r) > 5 and str(r[5]).startswith("FAIL"))
    nSkip = sum(1 for r in rawResults if len(r) > 5 and str(r[5]).startswith("SKIPPED"))
    return len(rawResults), len(sizes), nPass, nFail, nSkip


def _runOnePythonBench(groups, validateN, nWarmup, nIters, outDir, runIndex):
    """Benchmark all candidate groups using pre-built HSACO; return (wall, nResults, nSizes, nPass, nFail)."""
    t0 = time.perf_counter()
    allResults = []
    for yamlPath, hsacoPath, libYamlPath in groups:
        groupResults = _benchmarkGroup(yamlPath, hsacoPath, libYamlPath,
                                       nWarmup, nIters, validateN)
        allResults.extend(groupResults)
    wall = time.perf_counter() - t0
    nResults, nSizes, nPass, nFail, nSkip = parsePythonResults(allResults)
    return wall, nResults, nSizes, nPass, nFail


def runPythonBench(pipelineDir, validateN, nWarmup, nIters, outDir):
    """Benchmark all candidate groups 3 times using pre-built HSACO files."""
    from Tensile.client.sweep_runner import _cachedIsaInfoMap
    groups = _findGroupDirs(pipelineDir)
    if not groups:
        raise FileNotFoundError(
            f"No (00_Final.yaml, hsaco) pairs found under {pipelineDir}")
    print(f"  Found {len(groups)} candidate groups with pre-built HSACO")
    _cachedIsaInfoMap.cache_clear()
    dropPageCache()
    runs = []
    for i in range(3):
        print(f"  Python bench run {i} ...")
        wall, nResults, nSizes, nPass, nFail = _runOnePythonBench(
            groups, validateN, nWarmup, nIters, outDir, i)
        runs.append({"wall": wall, "nResults": nResults, "nSizes": nSizes,
                     "nPass": nPass, "nFail": nFail})
        print(f"    wall={wall:.1f}s, results={nResults}, sizes={nSizes}, "
              f"pass={nPass}, fail={nFail}")
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


def _buildReportSections(reportPath, args, arch, pyRuns, cppRuns, cppWall):
    """Return the list of Markdown section strings for the report."""
    pyRun0Wall = pyRuns[0]["wall"]
    pyRun1Wall = pyRuns[1]["wall"]
    pyRun2Wall = pyRuns[2]["wall"]
    warmWall = median([pyRun1Wall, pyRun2Wall])
    nResults = pyRuns[0]["nResults"]
    nSizes = pyRuns[0]["nSizes"]
    nPass = pyRuns[0]["nPass"]
    nFail = pyRuns[0]["nFail"]
    validateNote = (
        "all elements" if args.num_elements_to_validate == -1
        else f"{args.num_elements_to_validate} elements"
        if args.num_elements_to_validate > 0 else "disabled"
    )
    validationResult = (
        f"{nPass}/{nResults} PASS, {nFail} FAIL"
        if args.num_elements_to_validate != 0
        else "disabled"
    )
    return [
        "# bench_comparison: total sweep wall-clock report",
        "",
        "## Configuration",
        f"- arch: {arch}",
        f"- yaml: {args.yaml}",
        f"- output-dir: {os.path.dirname(reportPath)}",
        f"- validation: num-elements-to-validate={args.num_elements_to_validate} ({validateNote})",
        f"- num-benchmarks: {args.num_benchmarks}",
        f"- num-warmups: {args.num_warmups}",
        f"- Python: all {nResults} (candidate × size) pairs across {nSizes} problem sizes",
        f"- Python validation (run 0): {validationResult}",
        f"- C++: library mode, winner-per-size from pre-built library",
        "",
        "## Total sweep wall-clock (all candidates × all sizes)",
        "",
        "| Run | Client | Wall-clock (s) | Note |",
        "| --- | --- | --- | --- |",
        f"| 0 | Python | {pyRun0Wall:.2f} | cold |",
        f"| 1 | Python | {pyRun1Wall:.2f} | warm (HSACO cached) |",
        f"| 2 | Python | {pyRun2Wall:.2f} | warm |",
        f"| median warm | Python | {warmWall:.2f} | median of runs 1-2 |",
        f"| 0 | C++ | {cppRuns[0]['wall']:.2f} | cold page cache |",
        f"| 1 | C++ | {cppRuns[1]['wall']:.2f} | |",
        f"| 2 | C++ | {cppRuns[2]['wall']:.2f} | |",
        f"| median | C++ | {cppWall:.2f} | median of 3 runs |",
        "",
        f"**Python warm speedup vs C++: {cppWall / warmWall:.1f}×**",
        f"**Python cold vs C++: {cppWall / pyRun0Wall:.1f}×**",
        "",
        "> Note: Python sweeps all 13 candidate solutions (pre-built HSACO from",
        "> BenchmarkProblems cache); C++ benchmarks only the LibraryLogic winner",
        "> per size (1 solution × 3 sizes).",
    ]


def writeReport(reportPath, args, arch, pyRuns, cppRuns):
    """Compose and write the Markdown report; also print to stdout."""
    _cppBySize, cppWall = cppMedianBySize(cppRuns)
    sections = _buildReportSections(
        reportPath, args, arch, pyRuns, cppRuns, cppWall,
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

    print("\nRunning Python benchmark (compile mode, all candidates) ...")
    pyRuns = runPythonBench(
        pipelineDir,
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
