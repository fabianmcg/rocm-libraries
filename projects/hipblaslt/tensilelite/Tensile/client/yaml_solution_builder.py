# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build TensileLite Solution objects from benchmark YAML files.

Uses BenchmarkProcess -> constructForkPermutations -> _generate_single_solution
to drive the same pipeline that BenchmarkProblems uses, but serially and without
GPU benchmarking. This avoids the need to hand-construct Solution dicts.

KernArgsVersion chip table (used as fallback when InternalSupportParams is absent):

    _chipToKernArgsVersion = {
        "gfx908": 0, "gfx90a": 0,
        "gfx940": 1, "gfx941": 1,
        "gfx942": 2, "gfx950": 2,
        "gfx1100": 1, "gfx1101": 1,
    }

Update this table when new architectures are supported.
"""

def _parseBenchmarkGroup(yamlPath, problemIdx=0, groupIdx=0):
    """Parse one BenchmarkProblems group into (process, step) via BenchmarkProcess."""
    from Tensile import LibraryIO
    from Tensile.BenchmarkStructs import BenchmarkProcess

    data = LibraryIO.readYAML(yamlPath)
    problems = data["BenchmarkProblems"][problemIdx]
    process = BenchmarkProcess(problems[0], problems[1 + groupIdx], False)
    return process, process[0]


def buildSolutionsFromYaml(yamlPath, assembler, isaInfoMap, debugConfig,
                           problemIdx=0, groupIdx=0):
    """Build every forked Solution for a benchmark YAML group via the normal pipeline.

    Returns a list of valid Solution objects (None results and duplicates removed).
    """
    from Tensile.BenchmarkStructs import constructForkPermutations
    from Tensile.BenchmarkProblems import _generate_single_solution

    process, step = _parseBenchmarkGroup(yamlPath, problemIdx, groupIdx)

    solutions = []
    seen = set()
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


def problemSizesFromYaml(yamlPath, problemIdx=0, groupIdx=0):
    """Return the ProblemSizes from a benchmark YAML as a list of raw size tuples.

    Each entry is the Exact size tuple (free0, free1, batch, bound, ...).
    The caller interprets the leading four elements according to the problem type.
    """
    _process, step = _parseBenchmarkGroup(yamlPath, problemIdx, groupIdx)
    return [tuple(int(x) for x in p.sizes) for p in step.problemSizes.problems]


def solutionId(solution):
    """Return a short stable string identifying a solution by its tile dimensions and flags."""
    mt0 = solution["MacroTile0"]
    mt1 = solution["MacroTile1"]
    parts = [f"MT{mt0}x{mt1}"]
    if solution.get("PartialRMSResidualAdd"):
        parts.append("res")
    elif solution.get("PartialRMS"):
        parts.append("nores")
    if solution.get("RstdScale"):
        parts.append("rstd")
    return "_".join(parts)


def solutionsFromYaml(yamlPath, assembler, isaInfoMap, debugConfig,
                      problemIdx=0, groupIdx=0):
    """Return all (solution, id) pairs produced by a benchmark YAML group.

    Solutions are enumerated exactly as the YAML specifies — no overrides.
    The id string is derived from tile dimensions and epilogue flags.
    """
    solutions = buildSolutionsFromYaml(
        yamlPath, assembler, isaInfoMap, debugConfig, problemIdx, groupIdx,
    )
    return [(s, solutionId(s)) for s in solutions]


# ---------------------------------------------------------------------------
# KernArgsVersion fallback table (no ArchitectureSet field in tuning YAMLs).
# ---------------------------------------------------------------------------

_chipToKernArgsVersion: dict[str, int] = {
    "gfx908": 0,
    "gfx90a": 0,
    "gfx940": 1,
    "gfx941": 1,
    "gfx942": 2,
    "gfx950": 2,
    "gfx1100": 1,
    "gfx1101": 1,
}


def _kernArgsVersionForChip(chip: str) -> int:
    """Return the KernArgsVersion for a given chip identifier.

    Raises NotImplementedError for unknown chips rather than falling back to
    version=0 silently, since a wrong version produces incorrect argument
    layouts.
    """
    if chip not in _chipToKernArgsVersion:
        raise NotImplementedError(
            f"unsupported chip for KernArgsVersion lookup: {chip}"
        )
    return _chipToKernArgsVersion[chip]


def _injectInternalArgsSupport(solution: dict, chip: str | None) -> dict:
    """Return a copy of solution augmented with InternalArgsSupport fields.

    Reads fields from the solution dict's InternalSupportParams sub-key when
    present; falls back to the chip->version table for YAMLs that predate the
    InternalSupportParams block.

    The injected keys use the verified YAML names:
      KernArgsVersion, SupportCustomWGM, SupportCustomStaggerU,
      SupportUserGSU, UseSFC, UseUniversalArgs

    Also injects the version >= 2 bit-field keys from the solution level:
      GlobalSplitUCoalesced, GlobalSplitUWorkGroupMappingRoundRobin
    """
    result = dict(solution)
    isp = solution.get("InternalSupportParams", {}) or {}

    if "KernArgsVersion" not in isp:
        if chip is None:
            from amdgpu_exec import get_chip
            chip = get_chip()
        version = _kernArgsVersionForChip(chip)
    else:
        version = int(isp["KernArgsVersion"])

    result["KernArgsVersion"] = version
    result["SupportCustomWGM"] = bool(isp.get("SupportCustomWGM", False))
    result["SupportCustomStaggerU"] = bool(isp.get("SupportCustomStaggerU", False))
    result["SupportUserGSU"] = bool(isp.get("SupportUserGSU", False))
    result["UseSFC"] = bool(isp.get("UseSFC", False))
    result["UseUniversalArgs"] = bool(isp.get("UseUniversalArgs", True))

    # Version >= 2 bit-field keys live at solution level, not inside InternalSupportParams.
    result["GlobalSplitUCoalesced"] = bool(
        solution.get("GlobalSplitUCoalesced", False)
    )
    result["GlobalSplitUWorkGroupMappingRoundRobin"] = bool(
        solution.get("GlobalSplitUWorkGroupMappingRoundRobin", False)
    )
    return result


def solutionsFromFinalYaml(yamlPath: str, assembler, isaInfoMap, debugConfig,
                           chip: str | None = None):
    """Parse a 00_Final.yaml (BenchmarkProblems output) into (Solution objects, problem_sizes).

    00_Final.yaml is a flat list:
      [0]: {MinimumRequiredVersion: ...}
      [1]: {ProblemSizes: [...]}
      [2..4]: BiasTypeArgs, ActivationArgs, GateTypeArgs
      [5+]: fully-resolved solution parameter dicts with ProblemType and ISA

    Returns (solutions, problemSizes) where solutions is a list of (Solution, id) pairs
    (same format as solutionsFromYaml) and problemSizes is a list of (M, N, batch, K) tuples.
    """
    import yaml
    from Tensile.BenchmarkProblems import _build_and_validate_solution

    with open(yamlPath) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list) or len(data) < 6:
        return [], []
    rawSizes = data[1].get("ProblemSizes", [])
    problemSizes = []
    for entry in rawSizes:
        exact = entry.get("Exact", []) if isinstance(entry, dict) else []
        if len(exact) >= 4:
            problemSizes.append(tuple(int(x) for x in exact[:4]))
    solutions = []
    for item in data[5:]:
        if not isinstance(item, dict):
            continue
        sol = _build_and_validate_solution(
            dict(item), assembler, debugConfig, isaInfoMap, silent=True
        )
        if sol is not None:
            solutions.append((sol, solutionId(sol)))
    return solutions, problemSizes


def _iterRawSolutions(yamlPath: str) -> list[tuple[int, int, dict]]:
    """Iterate raw solution dicts from a tuning YAML without building Solution objects.

    Tuning YAMLs store solutions under:
      BenchmarkProblems[*][1+].BenchmarkFinalParameters[group_idx]
        .SolutionSummationExpansion[sol_idx]

    Uses yaml.safe_load instead of LibraryIO so that rocisa is not required
    and unit tests can run without a compiled rocisa module.

    Returns a list of (group_idx, solution_idx, solution_dict) triples where
    group_idx is a flat index across all BenchmarkFinalParameters groups that
    contain at least one solution.  Returns an empty list when no
    SolutionSummationExpansion entries are found (e.g. on input-spec YAMLs
    that only carry ProblemSizes).
    """
    import yaml

    with open(yamlPath) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return []

    results = []
    group_idx = 0
    for problem_group in data.get("BenchmarkProblems", []):
        if not isinstance(problem_group, list):
            continue
        for step in problem_group[1:]:
            if not isinstance(step, dict):
                continue
            for bfp_entry in step.get("BenchmarkFinalParameters", []):
                if not isinstance(bfp_entry, dict):
                    continue
                expansion = bfp_entry.get("SolutionSummationExpansion", [])
                for sol_idx, sol in enumerate(expansion):
                    results.append((group_idx, sol_idx, sol))
                if expansion:
                    group_idx += 1
    return results


def enumerateAllSolutions(
    yamlPath: str,
    chip: str | None = None,
) -> list[tuple[int, int, dict]]:
    """Enumerate every solution dict from a tuning YAML with InternalArgsSupport fields.

    Generalizes the while True / except (IndexError, KeyError): break pattern
    used in bench_gemm_rmsnorm.py. Reads directly from BenchmarkFinalParameters
    rather than driving the full BenchmarkProblems pipeline — no assembler or
    isaInfoMap needed.

    Returns a list of (group_idx, solution_idx, solution_dict) triples where
    each solution_dict is augmented with the following keys:
      KernArgsVersion, SupportCustomWGM, SupportCustomStaggerU,
      SupportUserGSU, UseSFC, UseUniversalArgs,
      GlobalSplitUCoalesced, GlobalSplitUWorkGroupMappingRoundRobin

    If chip is None, amdgpu_exec.get_chip() is called lazily only when needed
    (i.e. when InternalSupportParams is absent from the solution dict).
    """
    raw = _iterRawSolutions(yamlPath)
    return [
        (g, s, _injectInternalArgsSupport(sol, chip))
        for g, s, sol in raw
    ]
