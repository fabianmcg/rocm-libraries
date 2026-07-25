# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite sentinel-based bounds-check Python bindings.

Provides BoundedBuffer: a device allocation with a trailing sentinel region
that can be inspected after kernel execution to detect out-of-bounds writes.
Import is safe before any HIP initialization.
"""

from ._tensilelite_bounds import BoundedBuffer  # noqa: F401
