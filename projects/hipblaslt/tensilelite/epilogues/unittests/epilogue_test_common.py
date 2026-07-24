# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared test-framework utilities for the epilogue GPU unit tests."""
import os
import sys

import numpy as np
import pytest

# tensilelite root; conftest.py already inserts this, kept here so the
# module is importable standalone (e.g. import-time checks).
TENSILE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if TENSILE_ROOT not in sys.path:
    sys.path.insert(0, TENSILE_ROOT)

YAML_DIR = os.path.join(TENSILE_ROOT, "epilogues", "yaml")

try:
    import amdgpu_exec
    import ml_dtypes
    HAVE_DEPS = True
except ImportError:
    amdgpu_exec = None
    ml_dtypes = None
    HAVE_DEPS = False


def yamlPath(name):
    """Return the absolute path to a named YAML file in epilogues/yaml/."""
    return os.path.join(YAML_DIR, name)


def _isGfx950():
    if not HAVE_DEPS:
        return False
    try:
        return amdgpu_exec.get_chip().startswith("gfx950")
    except Exception:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _isGfx950(),
    reason="requires amdgpu_exec + ml_dtypes and a gfx950 GPU",
)


def enumerateSolutions(yamlName, predicate=None):
    """Return (solution, id) pairs for all solutions in a benchmark YAML group.

    Calls setup_tensile + solutionsFromYaml at collection time. Returns [] when
    deps or GPU are unavailable. predicate, if given, filters the list.
    """
    if not HAVE_DEPS:
        return []
    try:
        from epilogues.tensilelite.partialrms_helpers import setup_tensile
        from epilogues.tensilelite.yaml_solution_builder import solutionsFromYaml
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = setup_tensile(chip)
        sols = solutionsFromYaml(yamlPath(yamlName), assembler, isaInfoMap, debugConfig)
    except Exception:
        return []
    if predicate is not None:
        sols = [(s, sid) for s, sid in sols if predicate(s)]
    return sols


def assertClose(gpu, ref, label, rtol=2e-2, atol=2e-2, kind="output"):
    """Assert gpu ≈ ref within tolerance; raise a detailed AssertionError on failure."""
    bad = np.where(~np.isfinite(gpu) | (np.abs(gpu - ref) > atol + rtol * np.abs(ref)))
    nBad = len(bad[0])
    if nBad == 0:
        return
    r, c = bad[0][0], bad[1][0]
    raise AssertionError(
        f"{kind} mismatch ({label}): {nBad} elements out of tolerance. "
        f"max_abs={np.nanmax(np.abs(gpu - ref)):.3e}, "
        f"first bad row={r}, col={c}: gpu={gpu[r, c]:.6f} ref={ref[r, c]:.6f}"
    )
