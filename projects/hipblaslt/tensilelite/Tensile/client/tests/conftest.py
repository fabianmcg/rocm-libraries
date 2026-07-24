# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest configuration for Tensile/client harness tests.

Defines HAVE_DEPS, requires_deps, and requires_gfx950. The chip is detected
lazily at test-setup time, not at collection time, so that HIP initialization
does not happen before session fixtures (important for M11 ROCprofiler-SDK).
"""

import functools

import pytest

try:
    import amdgpu_exec  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

requires_deps = pytest.mark.skipif(not HAVE_DEPS, reason="amdgpu_exec not installed")

# IMPORTANT: do NOT call get_chip() here. get_chip() triggers HIP initialization
# on first use. A module-level call runs during test COLLECTION, which would
# initialize HIP before the session-scoped profiler-import fixture in M11 runs
# and break rocprofiler_force_configure. Keep this as a plain marker and resolve
# the chip lazily in pytest_runtest_setup instead.
requires_gfx950 = pytest.mark.requires_gfx950


@functools.lru_cache(maxsize=1)
def _currentChip() -> str:
    from amdgpu_exec import get_chip
    return get_chip()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_gfx950: run only on a gfx950 GPU")


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_gfx950") is None:
        return
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    if _currentChip() != "gfx950":
        pytest.skip("requires gfx950 GPU")
