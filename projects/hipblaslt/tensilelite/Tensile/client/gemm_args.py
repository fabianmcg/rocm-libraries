# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build raw kernel argument bytes for TensileLite GEMM kernels.

This module ports the argument-layout logic from
ContractionSolution.cpp:singleCallArgs (lines 548-1130) to Python.

Supported configuration subset (M1-M5; extended per milestone):
  - stridedBatched in {True, False}
  - streamK in {0, 3}  (M6+ adds 4, 5)
  - streamKAtomic = 0 only (atomic=1 skips workspace+flags block)
  - groupedGemm = False
  - GSU = 1, globalAccumulation in {0, 1, 2}
  - useInitialStrides = False
  - InternalArgsSupport.version <= 2, useSFC = False
  - expertSchedulingMode = 0, debugKernel = False

Unsupported combinations raise NotImplementedError, including:
  - GSU > 1 && streamK == 0
  - globalAccumulation = 3 (MBSK)
  - useSFC = True
  - version > 2
  - expertSchedulingMode != 0
  - debugKernel = True
  - streamKAtomic = 1
"""

from __future__ import annotations


def buildKernelArgs(
    solution_params: dict,
    problem_params: dict,
    tensors: dict,
) -> bytes:
    """Build the raw argument buffer for a TensileLite GEMM kernel.

    solution_params: solution dict from enumerateAllSolutions (includes
                     InternalArgsSupport fields injected by task 0.8).
    problem_params:  problem dimensions: M, N, K, lda, ldb, ldc, ldd,
                     alpha, beta, batch_count, strides, etc.
    tensors:         device pointers (int) keyed by 'A', 'B', 'C', 'D',
                     and optionally 'workspace'.

    Returns raw bytes suitable for passing as a ctypes c_char_p argument to
    the kernel launch via amdgpu_exec.GpuFunction.launch.

    Milestones 1-6 fill in the branches. This stub raises NotImplementedError
    for every configuration until the appropriate milestone implements it.
    """
    raise NotImplementedError(
        "buildKernelArgs is not yet implemented; will be filled in by M1-M6"
    )
