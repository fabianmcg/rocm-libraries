# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate a LibraryLogic YAML for the PartialRMS K1 (GEMM+PartialRMS) kernel.

Builds the solution from tools/gemm_partial_rms_k1.yaml, converts its state to
YAML-safe primitives, and writes a 12-element LibraryLogic list file that
TensileCreateLibrary can compile into a device library (.hsaco/.co/.dat).

Usage:
    python tools/gen_partialrms_logic.py [--chip gfx950] [--out-dir /path/to/logic]
"""

import argparse
import os
import sys
import yaml

_TENSILELITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR   = os.path.join(_TENSILELITE, "tools")
for _d in (_TOOLS_DIR, _TENSILELITE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from gemm_partialrms_colv2_helpers import setup_tensile, build_k1_solution
from Tensile.SolutionStructs.Naming import getKernelNameMin

# gfx950 device IDs recognised by TensileCreateLibrary.
_GFX950_DEVICE_IDS = ["Device 74a1", "Device 75a8", "Device 75a3", "Device 75a2"]


def _to_primitive(v):
    """Recursively convert Tensile-typed objects to YAML-safe Python primitives."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, list):
        return [_to_primitive(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_primitive(val) for k, val in v.items()}
    # DataType and similar objects expose .value as an int.
    if hasattr(v, "value") and isinstance(v.value, int):
        return v.value
    # ActivationType and other Tensile enums: stringify (yields e.g. "None").
    if hasattr(v, "__module__") and "Tensile" in str(getattr(v, "__module__", "")):
        return str(v)
    # SemanticVersion / ISA namedtuple: convert to [major, minor, patch].
    if hasattr(v, "major"):
        return [v.major, v.minor, v.patch]
    return str(v)


def _problem_type_state(pt):
    return {k: _to_primitive(pt[k]) for k in sorted(pt.keys())}


def _solution_state(sol, kernelName):
    d = {k: _to_primitive(v) for k, v in dict(sol).items()}

    # ISA namedtuple → [major, minor, patch].
    if hasattr(d.get("ISA"), "major"):
        d["ISA"] = [d["ISA"].major, d["ISA"].minor, d["ISA"].patch]

    # ProblemType lives in the outer YAML element; drop it from the solution dict.
    d.pop("ProblemType", None)

    # InternalSupportParams namedtuple → plain dict.
    isp = d.get("InternalSupportParams")
    if isp is not None and not isinstance(isp, dict):
        try:
            d["InternalSupportParams"] = dict(isp._asdict())
        except AttributeError:
            d["InternalSupportParams"] = dict(vars(isp))

    # Mandatory bookkeeping fields.
    d.update({"SolutionIndex": 0, "Valid": True, "KernelNameMin": kernelName,
              "BaseName": kernelName, "CustomKernelName": ""})

    # Type coercions required to pass TensileCreateLibrary validation.
    if "BufferStore" in d:
        d["BufferStore"] = bool(d["BufferStore"])
    if "GlobalReadPerMfma" in d:
        d["GlobalReadPerMfma"] = float(d["GlobalReadPerMfma"])
    if "StaggerUStride" in d:
        d["StaggerUStride"] = int(d["StaggerUStride"])
    return d


def generate(chip: str, outDir: str) -> str:
    """Build the K1 solution and write a LibraryLogic YAML to outDir.

    Returns the path to the written file.
    """
    logicDir = os.path.join(outDir, chip, "Equality")
    os.makedirs(logicDir, exist_ok=True)

    print(f"Setting up TensileLite for {chip} …")
    assembler, isaInfoMap, _ = setup_tensile(chip)

    print("Building K1 (GEMM+PartialRMS) solution …")
    sol = build_k1_solution(chip, assembler, isaInfoMap)
    assert sol["Valid"], "solution validation failed"

    kernel = sol.getKernels()[0]
    kernel.duplicate = False
    kernelName = getKernelNameMin(kernel, splitGSU=False)
    print(f"  KernelNameMin : {kernelName}")
    print(f"  MacroTile     : {sol['MacroTile0']}x{sol['MacroTile1']}")
    print(f"  NumThreads    : {sol['NumThreads']}")

    ptState  = _problem_type_state(sol["ProblemType"])
    solState = _solution_state(sol, kernelName)

    # 12-element LibraryLogic list expected by TensileCreateLibrary.
    logic = [
        {"MinimumRequiredVersion": "5.0.0"},
        chip,                   # ScheduleName
        chip,                   # ArchitectureName
        _GFX950_DEVICE_IDS,
        ptState,
        [solState],
        [0],                    # solution index ordering
        [[[4096, 4096, 1, 4096], [0, 0]]],  # ExactLogic (placeholder size)
        None,                   # RangeLogic
        None,
        "DeviceEfficiency",
        "Equality",             # LibraryType
    ]

    outPath = os.path.join(logicDir, "PartialRMS_BF16_TN.yaml")
    with open(outPath, "w") as fh:
        yaml.dump(logic, fh, default_flow_style=None, sort_keys=True, allow_unicode=True)

    # Round-trip sanity check.
    with open(outPath) as fh:
        check = yaml.safe_load(fh)
    assert isinstance(check, list) and len(check) == 12, "YAML round-trip failed"
    assert check[5][0]["KernelNameMin"] == kernelName, "KernelNameMin mismatch after round-trip"

    print(f"\nWrote LibraryLogic YAML → {outPath}")
    return outPath


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chip", default="gfx950",
                        help="GPU architecture (default: gfx950)")
    parser.add_argument("--out-dir", default="/tmp/partialrms_logic",
                        help="Root output directory for logic YAMLs (default: /tmp/partialrms_logic)")
    args = parser.parse_args()
    generate(args.chip, args.out_dir)


if __name__ == "__main__":
    main()
