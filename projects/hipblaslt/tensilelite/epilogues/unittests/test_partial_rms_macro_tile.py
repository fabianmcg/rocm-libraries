# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validator tests for the PartialRMS MacroTile multiple-of-64 constraint."""

import pytest

from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Solution import _validatePartialRMS


def _makeState(mt0, mt1):
    return {
        "PartialRMS": True,
        "UseSubtileImpl": True,
        "ISA": (9, 5, 0),
        "ProblemType": {"DataType": DataType("B"), "OutputAmaxD": False, "GroupedGemm": False},
        "StreamK": 0,
        "StreamKForceDPOnly": True,
        "MIArchVgpr": False,
        "PrefetchAcrossPersistent": 0,
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "MIWaveGroup": [1, 1],
        "MatrixInstN": 16,
        "MatrixInstM": 16,
        "MaxLDS": 65536,
        "WavefrontSize": 64,
        "Valid": True,
    }


def _isAccepted(mt0, mt1):
    state = _makeState(mt0, mt1)
    _validatePartialRMS(state, False)
    return state.get("Valid")


@pytest.mark.parametrize("mt0,mt1", [
    (320, 320),  # MT320x320 from best_bf16_8192 — must now be accepted
    (64, 64),
    (128, 256),
])
def test_multipleOf64Accepted(mt0, mt1):
    assert _isAccepted(mt0, mt1) is True


@pytest.mark.parametrize("mt0,mt1", [
    (100, 320),  # MT0 not a multiple of 64
    (320, 100),  # MT1 not a multiple of 64
    (32, 64),    # power of two but not a multiple of 64
])
def test_nonMultipleOf64Rejected(mt0, mt1):
    assert _isAccepted(mt0, mt1) is False
