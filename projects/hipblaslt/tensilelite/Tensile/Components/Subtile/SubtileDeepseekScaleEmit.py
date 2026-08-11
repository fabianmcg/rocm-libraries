# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Helpers for DeepseekScale kernel-arg offset discovery.

The flat_load bolt-on (setupDeepseekMainloopScale, emitDeepseekScaleGR, etc.)
has been removed. DeepseekScale now uses the MX SA/SB scheduler path with
buffer_load DTL via SrdMXSA/B. This module is kept as a thin helper so that
SubtileScaleEmit.initDeepseekScaleSrd can call _scaleBufKernArgOffsets without
importing from KernelWriter directly.
"""


def _scaleBufKernArgOffsets(writer):
    """Return (offset_a_or_None, offset_b_or_None) byte offsets of ScaleABuf/ScaleBBuf.

    Byte offsets are relative to the per-GEMM kernel arg base (KernArgAddress after
    the common-args shift), computed by walking numStoreSgprNames from the argLoader
    current position.
    """
    base = writer.argLoader.getOffset()
    names = writer.states.numStoreSgprNames
    sizes = writer.states.numStoreSgprNameSizes
    off_a = off_b = None
    cur = base
    for name, size in zip(names, sizes):
        if name == "ScaleABuf":
            off_a = cur
        elif name == "ScaleBBuf":
            off_b = cur
        cur += size * 4
    return off_a, off_b
