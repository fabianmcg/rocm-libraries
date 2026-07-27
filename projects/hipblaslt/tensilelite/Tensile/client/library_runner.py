# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LibraryRunner: high-level interface for selecting and evaluating solutions.

Wraps the tensilelite_runtime C++ bindings (M10) to provide a Python-friendly
API for loading a TensileLibrary file, enumerating candidate solutions, and
querying Formocast performance predictions without running kernels on the GPU.
"""

from __future__ import annotations

from typing import List, Optional

# Import lazily to avoid HIP init at collection time.
_rt = None


def _runtime():
    global _rt
    if _rt is None:
        import tensilelite_runtime as rt
        _rt = rt
    return _rt


class LibraryRunner:
    """Load a TensileLibrary file and select solutions for a given problem.

    Parameters
    ----------
    library_path:
        Path to a TensileLibrary YAML or MsgPack file.
    device_id:
        HIP device index used for hardware queries.  Defaults to 0.
    """

    def __init__(self, library_path: str, device_id: int = 0) -> None:
        rt = _runtime()
        self._lib = rt.load_library(library_path)
        self._hw = rt.get_hardware(device_id)

    @property
    def hardware(self):
        """Hardware descriptor for the active device."""
        return self._hw

    @property
    def library(self):
        """Loaded TensileLibrary object."""
        return self._lib

    def find_best(self, prob) -> Optional[object]:
        """Return the single best solution for prob, or None if none matches."""
        return self._lib.find_best_solution(self._hw, prob)

    def find_top_n(self, n: int, prob) -> List[object]:
        """Return up to n solutions sorted by descending Formocast GFLOPS.

        Uses find_top_solutions which calls findAllSolutions internally (since
        SingleSolutionLibrary does not override findTopSolutions) and then
        sorts the result by Formocast prediction before truncating to n.
        """
        return self._lib.find_top_solutions(self._hw, prob, n)

    def filter_by_predicate(self, solutions: List[object], prob) -> List[object]:
        """Return the subset of solutions whose hardware and task predicates pass.

        Useful for narrowing a pre-fetched list without re-querying the library.
        """
        rt = _runtime()
        out = []
        for sol in solutions:
            if sol.eval_hardware_predicate(self._hw) and sol.eval_task_predicate(self._hw, prob):
                out.append(sol)
        return out
