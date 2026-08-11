# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validator tests for the DeepseekScale mainloop scale constraint logic (no GPU required)."""

import pytest

from Tensile.SolutionStructs.Solution import _validateDeepseekScale


def _state():
    """Return a minimal DeepseekScale solution state dict that passes validation."""
    return {
        "UseDeepseekScaleA": True,
        "UseDeepseekScaleB": True,
        "UseSubtileImpl": True,
        "ISA": (9, 5, 0),
        "MIArchVgpr": False,
        "StreamK": 0,
        "StreamKForceDPOnly": True,
        "_ABTilePairA": "AB_B8",
        "DepthU": 128,
        "DeepseekScaleBlockK": 128,
        "AssertSummationElementMultiple": 128,
        "MacroTile1": 128,
        "TileQuant": False,
        "RstdScale": False,
        "PartialRMS": False,
        "PrefetchGlobalRead": 0,
        "ProblemType": {
            "HighPrecisionAccumulate": True,
            "GroupedGemm": False,
            "Gradient": False,
            "OutputAmaxD": False,
            "UseBias": 0,
            "BiasSrc": "D",
            "UseScaleAB": "",
            "UseScaleAlphaVec": 0,
            "UseScaleCD": False,
            "UseScaleD": False,
            "UseE": False,
            "ActivationType": "none",
        },
        "Valid": True,
    }


def test_deepseek_scale_base_valid():
    state = _state()
    _validateDeepseekScale(state, True)
    assert state["Valid"]


def test_deepseek_scale_reject_gradient():
    state = _state()
    state["ProblemType"]["Gradient"] = True
    _validateDeepseekScale(state, False)
    assert not state["Valid"]


def test_deepseek_scale_reject_output_amax_d():
    state = _state()
    state["ProblemType"]["OutputAmaxD"] = True
    _validateDeepseekScale(state, False)
    assert not state["Valid"]


def test_deepseek_scale_reject_bias_src_a():
    state = _state()
    state["ProblemType"]["UseBias"] = 1
    state["ProblemType"]["BiasSrc"] = "A"
    _validateDeepseekScale(state, False)
    assert not state["Valid"]


def test_deepseek_scale_reject_bias_src_b():
    state = _state()
    state["ProblemType"]["UseBias"] = 1
    state["ProblemType"]["BiasSrc"] = "B"
    _validateDeepseekScale(state, False)
    assert not state["Valid"]


# Each entry is a dict of ProblemType overrides that should still leave Valid True.
_PERMITTED_MODIFIERS = [
    {"UseScaleAB": "Scalar"},
    {"UseScaleAB": "Vector"},
    {"UseScaleAlphaVec": 1},
    {"UseScaleCD": True},
    {"UseScaleD": True},
    {"UseBias": 1, "BiasSrc": "D"},
    {"UseE": True},
    {"ActivationType": "relu"},
]


@pytest.mark.parametrize("overrides", _PERMITTED_MODIFIERS)
def test_deepseek_scale_permits_downstream_modifiers(overrides):
    # The per-K-block scale lives entirely in the mainloop MFMA operand; the
    # standard epilogue receives an already-scaled accumulator, so purely
    # downstream modifiers must not be rejected.
    state = _state()
    state["ProblemType"].update(overrides)
    _validateDeepseekScale(state, True)
    assert state["Valid"]
