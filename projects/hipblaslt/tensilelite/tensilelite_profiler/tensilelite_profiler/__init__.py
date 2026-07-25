# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite ROCprofiler-SDK Python bindings.

Exposes hardware counter collection via rocprofiler_force_configure.
This module must be imported before any HIP call initialises HSA.
"""

from ._tensilelite_profiler import *  # noqa: F401, F403
