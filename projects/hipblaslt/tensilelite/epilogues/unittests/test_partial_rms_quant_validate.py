# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validator tests for the PartialRMSQuant implication constraint."""

import pytest

from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Solution import _validatePartialRMS


def _state(quant, prms):
    return {
        "PartialRMS": prms, "PartialRMSQuant": quant, "RstdScale": False,
        "UseSubtileImpl": True, "ISA": (9, 5, 0),
        "ProblemType": {"DataType": DataType("B"), "OutputAmaxD": False, "GroupedGemm": False},
        "StreamK": 0, "StreamKForceDPOnly": True, "MIArchVgpr": False, "PrefetchAcrossPersistent": 0,
        "MacroTile0": 128, "MacroTile1": 128, "MIWaveGroup": [1, 1],
        "MatrixInstN": 16, "MatrixInstM": 16, "MaxLDS": 65536, "WavefrontSize": 64,
        "Valid": True,
    }


def test_quant_requires_partialrms():
    st = _state(quant=True, prms=False)
    _validatePartialRMS(st, False)
    assert st.get("Valid") is False


def test_quant_with_partialrms_accepted():
    st = _state(quant=True, prms=True)
    _validatePartialRMS(st, False)
    assert st.get("Valid") is True
