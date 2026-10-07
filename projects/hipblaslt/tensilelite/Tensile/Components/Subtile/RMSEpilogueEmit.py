# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RMSEpilogueEmitter: fused ResidualAdd + PartialRMS (+ MXFP8Quant) epilogue.

Replacement emitter for the Subtile/CMS MegaFused epilogue. Milestone 1 is a
thin delegating shim over the existing SubtileMegaFusedEmitter; subsequent
milestones move the emission logic here, restructured to mirror the reference
kernel mega_fused_epilogue.mlir (load_raw / pair_shuffle / store_data /
combine_rowgroups), and integrate the AGPR-held accumulator (H) path that the
reference kernel does not model.
"""

from .SubtileMegaFusedEmit import SubtileMegaFusedEmitter


class RMSEpilogueEmitter:
    """Emit the fused RMS/Residual epilogue for the Subtile/CMS gfx950 kernel."""

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        # Milestone 1: delegate verbatim. Later milestones replace this with a
        # native implementation and drop the delegate.
        self._delegate = SubtileMegaFusedEmitter(writer, kernel)

    def emit(self, vgprTiles):
        return self._delegate.emit(vgprTiles)
