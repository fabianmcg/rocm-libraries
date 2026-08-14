# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build TensileLite Solution objects from benchmark YAML files.

Uses BenchmarkProcess -> constructForkPermutations -> _generate_single_solution
to drive the same pipeline that BenchmarkProblems uses, but serially and without
GPU benchmarking. This avoids the need to hand-construct Solution dicts.
"""

def _parseBenchmarkGroup(yamlPath, problemIdx=0, groupIdx=0):
    """Parse one BenchmarkProblems group into (process, step) via BenchmarkProcess."""
    from Tensile import LibraryIO
    from Tensile.BenchmarkStructs import BenchmarkProcess

    data = LibraryIO.readYAML(yamlPath)
    problems = data["BenchmarkProblems"][problemIdx]
    process = BenchmarkProcess(problems[0], problems[1 + groupIdx], False)
    return process, process[0]


def _buildSolutionsForProblemIdx(yamlPath, assembler, isaInfoMap, debugConfig,
                                  problemIdx, groupIdx, seen):
    """Build solutions for one BenchmarkProblems entry, deduplicating against seen."""
    from Tensile.BenchmarkStructs import constructForkPermutations
    from Tensile.BenchmarkProblems import _generate_single_solution

    process, step = _parseBenchmarkGroup(yamlPath, problemIdx, groupIdx)
    solutions = []
    for perm in constructForkPermutations(step.forkParams, step.paramGroups):
        solution = _generate_single_solution(
            perm, process.problemType, step.constantParams,
            assembler, debugConfig, isaInfoMap,
        )
        if solution is None or solution in seen:
            continue
        seen.add(solution)
        solutions.append(solution)
    return solutions


def buildSolutionsFromYaml(yamlPath, assembler, isaInfoMap, debugConfig,
                           problemIdx=None, groupIdx=0):
    """Build every forked Solution for a benchmark YAML group via the normal pipeline.

    When problemIdx is None (the default), all BenchmarkProblems entries are read.
    Returns a list of valid Solution objects (None results and duplicates removed).
    """
    from Tensile import LibraryIO

    if problemIdx is not None:
        return _buildSolutionsForProblemIdx(
            yamlPath, assembler, isaInfoMap, debugConfig, problemIdx, groupIdx, set()
        )

    data = LibraryIO.readYAML(yamlPath)
    nProblems = len(data["BenchmarkProblems"])
    seen = set()
    solutions = []
    for idx in range(nProblems):
        solutions.extend(
            _buildSolutionsForProblemIdx(
                yamlPath, assembler, isaInfoMap, debugConfig, idx, groupIdx, seen
            )
        )
    return solutions


def problemSizesFromYaml(yamlPath, problemIdx=0, groupIdx=0):
    """Return the ProblemSizes from a benchmark YAML as a list of raw size tuples.

    Each entry is the Exact size tuple (free0, free1, batch, bound, ...).
    The caller interprets the leading four elements according to the problem type.
    """
    _process, step = _parseBenchmarkGroup(yamlPath, problemIdx, groupIdx)
    return [tuple(int(x) for x in p.sizes) for p in step.problemSizes.problems]


def readTestAxes(yamlPath, section, mt1=None):
    """Read test axis lists from the TestAxes section of a benchmark YAML.

    section: key under TestAxes (e.g. "K1", "Pipeline").

    Returns a dict with keys present in that section:
      "M":       [(m, label), ...] — expanded from MMultipliers/MFractions/MOffsets/MFixed
                 against mt1 (required when any M* key is present).
      "NHidden": [...]
      "K":       [...]
    """
    from Tensile import LibraryIO
    data = LibraryIO.readYAML(yamlPath)
    cfg = data.get("TestAxes", {}).get(section, {})
    if not cfg:
        raise KeyError(f"TestAxes.{section} not found in {yamlPath}")

    result = {}

    if "NHidden" in cfg:
        result["NHidden"] = list(cfg["NHidden"])
    if "K" in cfg:
        result["K"] = list(cfg["K"])
    if "KNHidden" in cfg:
        result["KNHidden"] = [tuple(pair) for pair in cfg["KNHidden"]]

    mKeys = {"MMultipliers", "MFractions", "MOffsets", "MFixed"}
    if mKeys & set(cfg):
        if mt1 is None:
            raise ValueError(f"mt1 required to expand M shapes in TestAxes.{section}")
        seen = set()
        mShapes = []

        def addM(m, label):
            m = max(1, m)
            if m not in seen:
                seen.add(m)
                mShapes.append((m, label))

        for mult in cfg.get("MMultipliers", []):
            addM(mt1 * mult, f"{mult}xMT1")
        for num, den in cfg.get("MFractions", []):
            addM(mt1 * num // den, f"MT1_{num}d{den}")
        for baseMult, delta in cfg.get("MOffsets", []):
            m = mt1 * abs(baseMult) + delta
            sign = "p" if delta >= 0 else "m"
            addM(m, f"{abs(baseMult)}MT1{sign}{abs(delta)}")
        for m in cfg.get("MFixed", []):
            addM(m, f"M{m}")

        result["M"] = mShapes

    return result


def solutionId(solution):
    """Return a short stable string identifying a solution by its tile dimensions and flags."""
    mt0 = solution["MacroTile0"]
    mt1 = solution["MacroTile1"]
    parts = [f"MT{mt0}x{mt1}"]
    if solution.get("PartialRMSResidualAdd"):
        parts.append("res")
    elif solution.get("PartialRMS"):
        parts.append("nores")
    gt = str(solution.get("PartialRMSGammaType", "b") or "b").lower()
    if gt != "b":
        parts.append(f"g{gt}")
    if solution.get("PartialRMSResidualAdd"):
        rt = str(solution.get("PartialRMSResidualType", "b") or "b").lower()
        if rt != "b":
            parts.append(f"r{rt}")
    return "_".join(parts)


def solutionsFromYaml(yamlPath, assembler, isaInfoMap, debugConfig,
                      problemIdx=None, groupIdx=0):
    """Return all (solution, id) pairs produced by a benchmark YAML file.

    When problemIdx is None (the default), every BenchmarkProblems entry is read
    and their solutions are combined. Pass an explicit index to read one entry.
    The id string is derived from tile dimensions and epilogue flags.
    """
    solutions = buildSolutionsFromYaml(
        yamlPath, assembler, isaInfoMap, debugConfig, problemIdx, groupIdx,
    )
    return [(s, solutionId(s)) for s in solutions]
