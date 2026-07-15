# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

# Insert the tensilelite root so that `Tensile`, `rocisa`, and `epilogues`
# are all importable when pytest is invoked from any working directory.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
