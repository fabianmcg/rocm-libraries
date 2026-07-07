# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Helpers for the K1 (PartialRMS) + Colv2 (RMSNorm) pipeline.

Provides setup, solution building, and execution utilities for:
  K1: fused GEMM + PartialRMS epilogue (gfx950, bf16)
    D[m, n]          = h1[m, n] * gamma[n]         (bf16, col-major M×N_hidden)
    partialBuf[m, t] = Σ_{n in tile t} h1[m, n]²  (f32, row-major M_padded×N_tiles_N)
    where h1 = A^T @ W0.

  Colv2: RMSNorm kernel that divides D in-place using partialBuf.

StreamKForceDPOnly=1 ensures every WG computes a complete tile so the
accumulator is final at the PartialRMS epilogue hook.
"""

import math
import os
import sys
import time

import numpy as np
import amdgpu_exec

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_TENSILE_DIR = os.path.dirname(_TOOLS_DIR)
for _d in (_TOOLS_DIR, _TENSILE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

_DEFAULT_K1_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_partial_rms_k1.yaml")


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
# Build the K1 (PartialRMS) solution
# ---------------------------------------------------------------------------


def load_k1_config(yamlPath):
    """Load K1 problem type and flat solution parameters from a benchmark YAML.

    Returns (problemType, flatParams) where flatParams is a dict with all
    BenchmarkCommonParameters and ForkParameters merged into a single level.
    Each single-element list value is unwrapped to its scalar; MatrixInstruction
    keeps its 9-element list form.
    """
    from Tensile import LibraryIO

    data = LibraryIO.readYAML(yamlPath)
    problemType = data["BenchmarkProblems"][0][0]
    grp = data["BenchmarkProblems"][0][1]

    flatParams = {}
    for section in ("BenchmarkCommonParameters", "ForkParameters"):
        for entry in grp.get(section, []):
            for key, value in entry.items():
                if key == "MatrixInstruction":
                    # Keep the inner list as-is (9-element MI spec).
                    flatParams[key] = value[0]
                elif isinstance(value, list) and len(value) == 1:
                    flatParams[key] = value[0]
                else:
                    flatParams[key] = value

    return problemType, flatParams


def build_k1_solution(chip: str, assembler, isaInfoMap, wgN: int = 2,
                      yamlPath=_DEFAULT_K1_YAML, miOverride=None, residualAdd: bool = False):
    """Build a bf16 GEMM + PartialRMS kernel for gfx950.

    Loads problem type and solution parameters from yamlPath. wgN sets
    MIWaveGroup[1] (last element of the 9-element MI spec). miOverride
    replaces the full 9-element MI list when provided.
    """
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.GlobalParameters import defaultInternalSupportParams
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
        validateMIParameters,
    )

    gfx = chip.split(":")[0]
    isa = gfxToIsa(gfx)

    problemType, params = load_k1_config(yamlPath)
    params["PartialRMSResidualAdd"] = residualAdd

    if miOverride is not None:
        params["MatrixInstruction"] = miOverride
    else:
        # Apply wgN sugar: last element is MIWaveGroup[1].
        params["MatrixInstruction"][-1] = wgN

    mi9 = params.pop("MatrixInstruction")
    wavefrontSize = 64
    miParams = matrixInstructionToMIParameters(
        mi9, isa, wavefrontSize, problemType, workGroup=None, isaInfoMap=isaInfoMap
    )

    config = {
        "ProblemType": problemType,
        "InternalSupportParams": defaultInternalSupportParams,
        "ISA": [isa.major, isa.minor, isa.patch],
        "CodeObjectVersion": "6",
    }
    config.update(params)
    config.update(miParams)

    if not validateMIParameters(config, isaInfoMap):
        raise RuntimeError("MI parameter validation failed")

    solution = Solution(
        config,
        splitGSU=False,
        printSolutionRejectionReason=True,
        printIndexAssignmentInfo=False,
        assembler=assembler,
        isaInfoMap=isaInfoMap,
    )
    if not solution["Valid"]:
        raise RuntimeError("K1 solution was rejected — see reason above")
    return solution


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


# ---------------------------------------------------------------------------
# Run K1: GEMM + PartialRMS
# ---------------------------------------------------------------------------


def run_k1(hsaco: bytes, kernel_name: str, solution, M: int, N: int, K: int):
    """Execute the K1 fused GEMM+PartialRMS kernel and verify outputs.

    Checks:
      D (slot 8):       bf16 output, tol = 2e-2. D = h1 * gamma.
      partialBuf (2D):  fp32, tol = 1e-4. partialBuf[m, t] = Σ_{n in tile t} h1[m,n]^2.

    Numpy reference (uses bf16-rounded inputs to match kernel precision):
      a_ref     = bf16(A).T          (M x K, fp32)
      w0_ref    = bf16(W0)           (K x N, fp32)
      h1        = a_ref @ w0_ref     (M x N, fp32)
      D_ref     = bf16(h1 * gamma)   (M x N, bf16)
      sumsq_ref = 2D (M, N_tiles_N) per-tile column-block Σx²
    """
    import ml_dtypes

    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]

    N_tiles_N = math.ceil(N / MT1)
    M_padded = math.ceil(M / MT0) * MT0
    numWG = math.ceil(M / MT0) * N_tiles_N

    rng = np.random.default_rng(42)
    a_f32 = np.asfortranarray(rng.random((K, M), dtype=np.float32) * 0.1)
    w0_f32 = np.asfortranarray(rng.random((K, N), dtype=np.float32) * 0.1)

    a_bf16 = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(w0_f32.astype(ml_dtypes.bfloat16))

    c_bf16 = np.zeros((M, N), dtype=ml_dtypes.bfloat16, order="F")
    d_bf16 = np.zeros((M, N), dtype=ml_dtypes.bfloat16, order="F")

    gamma_f32 = rng.random(N, dtype=np.float32) + 0.5
    gamma_bf16 = gamma_f32.astype(ml_dtypes.bfloat16)

    # 2D partialBuf: shape (M_padded, N_tiles_N), row-major (C-order).
    # Byte offset for (m, t) = (m * N_tiles_N + t) * 4.
    partial_buf = np.zeros((M_padded, N_tiles_N), dtype=np.float32, order="C")

    # Numpy reference.
    a_ref = np.asarray(a_bf16).astype(np.float32)
    w0_ref = np.asarray(w0_bf16).astype(np.float32)
    h1 = a_ref.T @ w0_ref  # M x N, fp32
    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)
    d_ref = np.asarray((h1 * gamma_ref[np.newaxis, :]).astype(ml_dtypes.bfloat16))

    # 2D sumsq_ref: per-tile Σx² over each MT1-wide column block.
    sumsq_ref = np.zeros((M, N_tiles_N), dtype=np.float32)
    for t in range(N_tiles_N):
        col_lo = t * MT1
        col_hi = min((t + 1) * MT1, N)
        sumsq_ref[:, t] = np.sum(h1[:, col_lo:col_hi] ** 2, axis=1)

    sk_args = compute_sk3_dp_args(M, N, K, solution)
    kernel_info0, kernel_info1 = _pack_kernel_info(solution)

    ws_dummy = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    # Argument layout:
    #   slots 0-29: GEMM + StreamK args (same as RMSNorm)
    #   slot 30: RMSNormGamma (ptr, bf16)
    #   slot 31: PartialBuf   (ptr, fp32, 2D [M_padded, N_tiles_N]) [InOutArray]
    args = [
        np.uint32(1),  # 0: Gemm info
        kernel_info0,  # 1: kernel_info0
        kernel_info1,  # 2: kernel_info1
        np.uint32(numWG),  # 3: numWG
        np.uint32(M),  # 4: SizesFree0=M
        np.uint32(N),  # 5: SizesFree1=N
        np.uint32(1),  # 6: SizesFree2=batch
        np.uint32(K),  # 7: SizesSum0=K
        amdgpu_exec.InOutArray(d_bf16),  # 8: D (bf16)
        amdgpu_exec.InputArray(c_bf16),  # 9: C (bf16, beta=0)
        amdgpu_exec.InputArray(a_bf16),  # 10: A (K×M col-major)
        amdgpu_exec.InputArray(w0_bf16),  # 11: B (K×N col-major)
        amdgpu_exec.InputArray(ws_dummy),  # 12: AddressWS
        amdgpu_exec.InputArray(flags_dummy),  # 13: AddressFlags
        np.uint32(M),
        np.uint32(0),  # 14,15: strideD0=M, strideD1=0
        np.uint32(M),
        np.uint32(0),  # 16,17: strideC0=M, strideC1=0
        np.uint32(K),
        np.uint32(0),  # 18,19: strideA0=K, strideA1=0
        np.uint32(K),
        np.uint32(0),  # 20,21: strideB0=K, strideB1=0
        np.float32(1.0),  # 22: alpha
        np.float32(0.0),  # 23: beta
        sk_args["iters_per_tile"],  # 24: ItersPerTile
        sk_args["magic_iters_per_tile"],  # 25: MagicNumberItersPerTile
        sk_args["shift_iters_per_tile"],  # 26: MagicShiftItersPerTile
        sk_args["sk_iters_per_wg"],  # 27: SKItersPerWG
        sk_args["sk_grid"],  # 28: skGrid
        sk_args["sk_tiles"],  # 29: skTiles
        amdgpu_exec.InputArray(gamma_bf16),  # 30: RMSNormGamma (bf16)
        amdgpu_exec.InOutArray(partial_buf),  # 31: PartialBuf (fp32, 2D C-order)
    ]

    ok = False

    def verify(arguments):
        nonlocal ok
        d_gpu_bf16 = np.asarray(arguments[8].array)
        d_gpu_f32 = d_gpu_bf16.astype(np.float32)
        d_ref_f32 = d_ref.astype(np.float32)
        pb_flat = np.asarray(arguments[31].array)
        pb_gpu = pb_flat.reshape(M_padded, N_tiles_N)

        # Check D (bf16).
        rtol, atol = 2e-2, 2e-2
        diff_d = np.abs(d_gpu_f32[:M] - d_ref_f32[:M])
        tol_d = atol + rtol * np.abs(d_ref_f32[:M])
        bad_d = np.where(~np.isfinite(d_gpu_f32[:M]) | (diff_d > tol_d))
        d_ok = len(bad_d[0]) == 0

        # Check 2D partialBuf.
        rtol_p, atol_p = 1e-4, 1e-4
        diff_p = np.abs(pb_gpu[:M, :] - sumsq_ref)
        tol_p = atol_p + rtol_p * np.abs(sumsq_ref)
        bad_p = np.where(~np.isfinite(pb_gpu[:M, :]) | (diff_p > tol_p))
        p_ok = len(bad_p[0]) == 0

        ok = d_ok and p_ok
        max_abs_d = (
            float(np.nanmax(np.abs(d_gpu_f32[:M] - d_ref_f32[:M]))) if M > 0 else 0.0
        )
        max_abs_p = (
            float(np.nanmax(np.abs(pb_gpu[:M, :] - sumsq_ref))) if M > 0 else 0.0
        )

        if ok:
            print(
                f"verification: PASSED  "
                f"D max_abs={max_abs_d:.3e}  partialBuf max_abs={max_abs_p:.3e}"
            )
        else:
            print(f"verification: FAILED")
            if not d_ok:
                print(
                    f"  D: {len(bad_d[0])} elements out of tolerance, max_abs={max_abs_d:.3e}"
                )
                r, c = bad_d[0][0], bad_d[1][0]
                print(
                    f"    first bad D[{r},{c}]: gpu={d_gpu_f32[r,c]:.6f} "
                    f"ref={d_ref_f32[r,c]:.6f}"
                )
            if not p_ok:
                print(
                    f"  partialBuf: {len(bad_p[0])} entries out of tolerance, "
                    f"max_abs={max_abs_p:.3e}"
                )
                r, c = bad_p[0][0], bad_p[1][0]
                print(
                    f"    first bad partialBuf[{r},{c}]: gpu={pb_gpu[r,c]:.6f} "
                    f"ref={sumsq_ref[r,c]:.6f}"
                )

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(numWG, 1, 1),
        block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1,
        verify_fn=verify,
    )
    return ok
