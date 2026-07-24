# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

# Insert the tensilelite root so that `Tensile`, `rocisa`, and `epilogues`
# are all importable when pytest is invoked from any working directory.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Guard against the local epilogues/epilogue_harness/ (or the old
# epilogues/tensilelite/) shadowing the installed tensilelite package.
# After the rename to epilogue_harness/ the shadowing risk is gone, but
# this check catches any future regression immediately at collection time.
try:
    import tensilelite as _tensilelite  # noqa: E402
    _epilogues_path = str(_root / "epilogues")
    assert not str(_tensilelite.__file__).startswith(_epilogues_path), (
        f"tensilelite resolved to {_tensilelite.__file__!r}, which is inside "
        f"the epilogues tree — the installed package is being shadowed"
    )
except ImportError:
    pass
