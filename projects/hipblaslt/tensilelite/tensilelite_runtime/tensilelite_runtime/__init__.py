# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite runtime Python bindings.

Provides ELF-based I-cache analysis helpers and (in M10) ContractionSolution
bindings. Import is safe before any HIP initialization.
"""

from ._tensilelite_runtime import *  # noqa: F401, F403
