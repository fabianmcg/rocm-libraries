# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Helpers for the K1 (PartialRMS) + row_div pipeline.

Provides setup, solution building, and execution utilities for:
  K1: fused GEMM + PartialRMS epilogue (gfx950, bf16, row-major output)
    Called with free0=N_hidden, free1=M_tokens (swapped M↔N vs col-major).
    D[token, i]          = h1[token, i] * gamma[i]   (bf16, row-major M×N_hidden)
    partialBuf[token, t] = Σ_{i in tile t} h1[token, i]²  (f32, M_padded×n_d)
    where h1 = A^T @ W0 and n_d = ceil(N_hidden / MT0).

  row_div: second-stage kernel that divides D in-place using partialBuf.

StreamKForceDPOnly=1 ensures every WG computes a complete tile so the
accumulator is final at the PartialRMS epilogue hook.
"""

import math
import os
import sys

import numpy as np
import amdgpu_exec

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_TENSILE_DIR = os.path.dirname(os.path.dirname(_PKG_DIR))
if _TENSILE_DIR not in sys.path:
    sys.path.insert(0, _TENSILE_DIR)


# ---------------------------------------------------------------------------
# Magic-number fast-division helpers (mirrors ContractionSolution.cpp alg 2)
# ---------------------------------------------------------------------------


def _magic_number_alg2(d: int):
    """Return (magic, shift) for 32-bit unsigned division by d using algorithm 2."""
    if d == 0:
        return 0, 0
    d = d & 0xFFFFFFFF
    a = 0
    nc = (-1 - (-d) % d) & 0xFFFFFFFF
    p = 31
    q1 = 0x80000000 // nc
    r1 = 0x80000000 - q1 * nc
    q2 = 0x7FFFFFFF // d
    r2 = 0x7FFFFFFF - q2 * d
    while p < 64:
        p += 1
        if r1 >= nc - r1:
            q1 = 2 * q1 + 1
            r1 = 2 * r1 - nc
        else:
            q1 = 2 * q1
            r1 = 2 * r1
        if r2 + 1 >= d - r2:
            if q2 >= 0x7FFFFFFF:
                a = 1
            q2 = 2 * q2 + 1
            r2 = 2 * r2 + 1 - d
        else:
            if q2 >= 0x80000000:
                a = 1
            q2 = 2 * q2
            r2 = 2 * r2 + 1
        delta = d - 1 - r2
        if not (p < 64 and (q1 < delta or (q1 == delta and r1 == 0))):
            break
    magic = (q2 + 1) & 0xFFFFFFFF
    shift = p - 32
    if a:
        shift |= 0x80000000
    return magic, shift & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# TensileLite: setup and StreamK argument helpers
# ---------------------------------------------------------------------------


def setup_tensile(chip: str):
    from pathlib import Path
    from Tensile.Toolchain.Validators import validateToolchain
    from Tensile.Toolchain.Component import Assembler
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.GlobalParameters import assignGlobalParameters
    from Tensile.Common.Types import DebugConfig

    gfx = chip.split(":")[0]
    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa(gfx)
    isaInfoMap = makeIsaInfoMap([isa], cxx)
    assignGlobalParameters({}, isaInfoMap)
    assembler = Assembler(Path(cxx), co_version="6")
    debugConfig = DebugConfig()
    return assembler, isaInfoMap, debugConfig


def compute_sk3_dp_args(M: int, N: int, K: int, solution) -> dict:
    """Compute StreamK=3 kernel arguments for the ForceDPOnly mode.

    With ForceDPOnly=1, skTiles=0 so every WG runs in data-parallel mode.
    The grid equals the number of output tiles.
    """
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    depth_u = solution["DepthU"]

    tiles = math.ceil(M / MT0) * math.ceil(N / MT1)
    iters_per_tile = max(1, math.ceil(K / depth_u))

    magic_ipt, shift_ipt = _magic_number_alg2(iters_per_tile)

    sk_tiles = 0
    sk_iters_per_wg = 0
    sk_grid = tiles

    return {
        "iters_per_tile": np.uint32(iters_per_tile),
        "magic_iters_per_tile": np.uint32(magic_ipt),
        "shift_iters_per_tile": np.uint32(shift_ipt),
        "sk_iters_per_wg": np.uint32(sk_iters_per_wg),
        "sk_grid": np.uint32(sk_grid),
        "sk_tiles": np.uint32(sk_tiles),
    }


def _pack_kernel_info(solution) -> tuple:
    """Pack StaggerU and WorkGroupMapping fields into kernel_info0 / kernel_info1."""
    su = solution.get("StaggerU", 0)
    su_map = solution.get("StaggerUMapping", 0)
    ss = solution.get("_staggerStrideShift", 0)
    su_word = (su_map << 13) | ((ss << 8) & 0x1F00) | (su & 0xFF)
    ki0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))
    ki1 = np.uint32(
        (solution.get("WorkGroupMappingXCC", 1) << 16)
        | (solution["WorkGroupMapping"] & 0xFFFF)
    )
    return ki0, ki1


# ---------------------------------------------------------------------------
# Generate assembly
# ---------------------------------------------------------------------------


def generate_asm(solution, assembler, debugConfig):
    """Generate assembly string and kernel name from a solution."""
    import rocisa
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelNameMin

    kwa = KernelWriterAssembly(assembler, debugConfig)
    ti = rocisa.rocIsa.getInstance()
    kwa.setRocIsa(ti.getData(), ti.getOutputOptions())

    kernel = solution.getKernels()[0]
    kernel.duplicate = False
    err, asm_str = kwa.getSourceFileString(kernel)
    if err:
        raise RuntimeError(f"Assembly generation failed: {err}")

    kernel_name = getKernelNameMin(kernel, splitGSU=False)
    return asm_str, kernel_name


def compileSolution(solution):
    """Compile one solution: setup_tensile → generate_asm → compile_asm_to_hsaco.

    Returns (kernelName, hsaco, chip).
    """
    chip = amdgpu_exec.get_chip()
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    asmStr, kernelName = generate_asm(solution, assembler, debugConfig)
    hsaco = amdgpu_exec.compile_asm_to_hsaco(asmStr, chip)
    return kernelName, hsaco, chip


def buildSubtileArgs(free0, free1, bound, numWG, dOut, cIn, aOperand, bOperand,
                     skArgs, ki0, ki1, epilogueArgs, alpha=np.float32(1.0),
                     hasBeta=True):
    """Build the StreamK=3/ForceDPOnly subtile kernel argument list.

    All subtile kernels use StreamKForceDPOnly=1, which drops AddressWS and
    AddressFlags from the kernarg layout (see Tensile/Components/Signature.py).
    Four zero u64 batchOffset{D,C,A,B} args are appended at the tail for
    non-grouped-GEMM kernels (batch=1, so offsets are all zero).

    Pass hasBeta=False for kernels with UseBeta=False (e.g. TileQuant), which
    do not load a beta SGPR; epilogue args then start one slot earlier.

    Slot layout (0-indexed) with hasBeta=True:
      0        : kernel_info_flags (uint32, always 1)
      1, 2     : ki0, ki1
      3        : numWG
      4–7      : free0, free1, batch=1, bound
      8        : D (output)
      9        : C (input)
      10       : A operand
      11       : B operand
      12–19    : strides (free0,0, free0,0, bound,0, bound,0)
      20–21    : alpha=1.0, beta=0.0
      22–27    : SK decomposition args
      28+      : epilogue-specific args
      tail     : batchOffsetD, batchOffsetC, batchOffsetA, batchOffsetB (u64, all 0)

    With hasBeta=False slot 21 (beta) is absent; epilogue args start at slot 27.
    """
    args = [
        np.uint32(1), ki0, ki1, np.uint32(numWG),
        np.uint32(free0), np.uint32(free1), np.uint32(1), np.uint32(bound),
        dOut, cIn, aOperand, bOperand,
        np.uint32(free0), np.uint32(0),
        np.uint32(free0), np.uint32(0),
        np.uint32(bound), np.uint32(0),
        np.uint32(bound), np.uint32(0),
        np.float32(alpha),
    ]
    if hasBeta:
        args.append(np.float32(0.0))
    args.extend([
        skArgs["iters_per_tile"], skArgs["magic_iters_per_tile"],
        skArgs["shift_iters_per_tile"], skArgs["sk_iters_per_wg"],
        skArgs["sk_grid"], skArgs["sk_tiles"],
    ])
    args.extend(epilogueArgs)
    # batchOffset{D,C,A,B}: zero for batch=1 (no pointer-array offset needed).
    args.extend([np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0)])
    return args


def buildRowDivArgs(dRow, partialBuf, nHidden, nC, nD, invD, eps):
    """Build the 8-element argument list for the pipeline row_div kernel.

    Offset layout: 0 D ptr, 8 partialBuf ptr, 16 pad i32, 20 n, 24 n_c,
    28 n_d, 32 inv_d, 36 eps. dRow and partialBuf must be already-wrapped
    InOutArray/InputArray objects.
    """
    return [
        dRow,
        partialBuf,
        np.int32(0), np.int32(nHidden), np.int32(nC),
        np.int32(nD), np.float32(invD), np.float32(eps),
    ]
