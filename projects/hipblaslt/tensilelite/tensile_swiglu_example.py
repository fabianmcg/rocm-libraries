# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite + amdgpu-exec fused GEMM + SwiGLU example.

Demonstrates the SwiGLU fused epilogue on gfx950 (MI350):
  D = up * silu(gate)  where each wave independently splits its N slice into
  [gate | up] halves and computes y = up * silu(gate).

The GEMM accumulator has doubled N width: N_gemm = n_tiles * MacroTile1.
With MIWaveGroup=[1, wg_n], each MacroTile1-wide N tile is partitioned into
wg_n per-wave slices of width waveN = MacroTile1 // wg_n.  Each wave splits
its own slice at the midpoint: gate = slice[:, :waveN//2], up = slice[:, waveN//2:].
The results are concatenated to form the (M, N_out) output stored to D, where
N_out = N_gemm // 2.

Constraints:
  - N_gemm must be a positive multiple of MacroTile1 (multi-tile N supported)
  - mma_n = MacroTile1 // 16 // wg_n must be even and >= 2
  - alpha=1, beta=0 must be passed by the host
  - StreamKForceDPOnly=1 ensures complete tiles at the epilogue hook
  - gfx950 / bf16 only (ISA-specific assembly)

Usage:
    python tensile_swiglu_example.py [--M 64] [--K 128] [--wg-n 1] [--n-tiles 1]
                                     [--chip gfx950] [--iterations 10]
"""
import argparse
import math
import os
import struct
import sys
import time

import numpy as np
import amdgpu_exec

_TENSILE_DIR = os.path.dirname(os.path.abspath(__file__))
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
# TensileLite: setup and solution construction
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


def build_swiglu_solution(chip: str, assembler, isaInfoMap, wg_n: int = 1):
    """Build a bf16 GEMM + SwiGLU kernel config for gfx950.

    Uses:
      - MatrixInstruction 16x16x32 bf16, WaveTile(4,4)
        -> MacroTile0=64, MacroTile1=64*wg_n (== N_gemm for tile containment)
      - MIWaveGroup=[1, wg_n]: arbitrary wg_n is supported as long as
        mma_n = MacroTile1 // 16 // wg_n is even and >= 2.
      - UseSubtileImpl=True, StreamK=3, StreamKForceDPOnly=1
      - SwiGLU=True

    N_out = MacroTile1 // 2.  The caller passes N_gemm = MacroTile1 as SizesFree1.
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

    problem_type = {
        "OperationType": "GEMM",
        "DataType":      "b",   # bf16
        "DestDataType":  "b",   # bf16
        "ComputeDataType": "s", # fp32 accumulation
        "HighPrecisionAccumulate": True,
        "TransposeA": True,     # A: K×M col-major (TN layout — DTL requires TN)
        "TransposeB": False,    # B: K×N_gemm col-major
        "UseBeta": False,
        "Batched": True,
        "StridedBatched": True,
        "GroupedGemm": False,
        "UseBias": 0,
        "UseScaleAB": "",
        "UseScaleCD": False,
        "UseScaleAlphaVec": 0,
        "Sparse": 0,
    }

    # MatrixInstruction 9-item format (see matrixInstructionToMIParameters):
    #   [instM, instN, instK, instB, ?, wt1, wt0, wg0_waves, wg1_waves]
    mi9 = [16, 16, 32, 1,   # instM, instN, instK, instB
            1,               # mi[4]
            4, 4,            # wt1=4, wt0=4  (MIWaveTile=[4,4])
            1,               # mi[7] -> wg0 waves = 1
            wg_n]            # mi[8] -> wg1 waves = wg_n

    wavefrontSize = 64
    mi_params = matrixInstructionToMIParameters(
        mi9, isa, wavefrontSize, problem_type, workGroup=None, isaInfoMap=isaInfoMap
    )

    config = {
        "ProblemType":          problem_type,
        "InternalSupportParams": defaultInternalSupportParams,
        "ISA":                  [isa.major, isa.minor, isa.patch],
        "CodeObjectVersion":    "6",
        "GlobalSplitU":         1,
        "KernelLanguage":       "Assembly",
        "StreamK":              3,
        "StreamKForceDPOnly":   1,
        "StreamKAtomic":        0,
        "ScheduleIterAlg":      3,
        "PrefetchGlobalRead":   1,
        "DirectToLdsA":         1,
        "DirectToLdsB":         1,
        "UseSubtileImpl":       True,
        "SwiGLU":               True,
        "StaggerU":             0,
        "DepthU":               64,  # 2 * MI_K = 2 * 32
        "LdsPadA":              -1,
        "LdsPadB":              -1,
        "StoreVectorWidth":     -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "PreloadKernArgs":      False,
        "_1LDSBuffer":          0,
        "PrefetchAcrossPersistent": 0,
    }
    config.update(mi_params)

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
        raise RuntimeError("Solution was rejected — see reason above")
    return solution


def generate_asm(solution, assembler, debugConfig):
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
# StreamK arg computation for StreamK=3, StreamKForceDPOnly=1
# ---------------------------------------------------------------------------

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

    # ForceDPOnly: skTiles=0, skItersPerWG=0
    sk_tiles = 0
    sk_iters_per_wg = 0
    sk_grid = tiles  # grid == number of DP tiles

    return {
        "iters_per_tile":          np.uint32(iters_per_tile),
        "magic_iters_per_tile":    np.uint32(magic_ipt),
        "shift_iters_per_tile":    np.uint32(shift_ipt),
        "sk_iters_per_wg":         np.uint32(sk_iters_per_wg),
        "sk_grid":                 np.uint32(sk_grid),
        "sk_tiles":                np.uint32(sk_tiles),
    }


# ---------------------------------------------------------------------------
# Run the SwiGLU kernel
# ---------------------------------------------------------------------------

def run_swiglu(
    hsaco: bytes,
    kernel_name: str,
    solution,
    M: int,
    N_gemm: int,
    K: int,
    num_iterations: int,
):
    """Execute the fused GEMM+SwiGLU kernel and verify against a numpy reference.

    N_gemm must be a positive multiple of MacroTile1; N_out = N_gemm // 2 is
    the output width.  The kernel dispatches ceil(N_gemm/MT1) workgroups in the
    N direction; the last workgroup's store is clamped by SubtileNGuard, so the
    constraint is on the dispatch granularity (B-read alignment), not on the
    kernel's ability to handle partial tiles.

    Alpha=1.0, Beta=0.0 (no C contribution).
    """
    import ml_dtypes
    import os, sys as _sys
    _tests_unit = os.path.join(os.path.dirname(__file__), "Tensile", "Tests", "unit")
    if _tests_unit not in _sys.path:
        _sys.path.insert(0, _tests_unit)
    from swiglu_reference import swiglu_reference

    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    N_out = N_gemm // 2

    numWG = math.ceil(M / MT0) * math.ceil(N_gemm / MT1)

    rng = np.random.default_rng(42)
    # TN layout: TransposeA=True, TransposeB=False
    #   A is K×M col-major: shape (K,M)
    #   B is K×N_gemm col-major: shape (K,N_gemm)
    a_f32 = np.asfortranarray(rng.random((K, M), dtype=np.float32) * 0.1)
    b_f32 = np.asfortranarray(rng.random((K, N_gemm), dtype=np.float32) * 0.1)

    a_bf16 = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    b_bf16 = np.asfortranarray(b_f32.astype(ml_dtypes.bfloat16))

    c_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')
    d_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')

    # Geometry-free reference: global-split oracle independent of tile structure.
    a_ref     = np.asarray(a_bf16).astype(np.float32)
    b_ref     = np.asarray(b_bf16).astype(np.float32)
    d_ref_f32 = swiglu_reference(a_ref, b_ref)            # shape (M, N_out)
    d_ref_bf16 = d_ref_f32.astype(ml_dtypes.bfloat16)

    alpha = np.float32(1.0)
    beta  = np.float32(0.0)

    sk_args = compute_sk3_dp_args(M, N_gemm, K, solution)

    stagger_u         = solution.get("StaggerU", 0)
    stagger_u_mapping = solution.get("StaggerUMapping", 0)
    stagger_stride_shift = solution.get("_staggerStrideShift", 0)
    su_word = (stagger_u_mapping << 13) | ((stagger_stride_shift << 8) & 0x1F00) | (stagger_u & 0xFF)
    kernel_info0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))

    wgmxcc = solution.get("WorkGroupMappingXCC", 1)
    kernel_info1 = np.uint32((wgmxcc << 16) | (solution["WorkGroupMapping"] & 0xFFFF))

    block_dim = (solution["NumThreads"], 1, 1)
    grid_dim  = (numWG, 1, 1)

    # Argument layout (SwiGLU has no extra kernargs — no gamma/eps):
    #   index  0: Gemm info (u32)
    #   index  1: kernel_info0 (u32)
    #   index  2: kernel_info1 (u32)
    #   index  3: numWG (u32)
    #   index  4: SizesFree0=M (u32)
    #   index  5: SizesFree1=N_gemm (u32)   <- full GEMM N
    #   index  6: SizesFree2=batch (u32)
    #   index  7: SizesSum0=K (u32)
    #   index  8: D (InOutArray, M×N_out col-major)
    #   index  9: C (ptr)
    #   index 10: A (ptr)
    #   index 11: B (ptr)
    #   index 12: AddressWS (dummy)
    #   index 13: AddressFlags (dummy)
    #   then strides, alpha/beta, SK args
    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)
    args = [
        np.uint32(1),                           # Gemm info
        kernel_info0,                           # kernel_info0
        kernel_info1,                           # kernel_info1
        np.uint32(numWG),                       # numWG
        np.uint32(M),                           # SizesFree0
        np.uint32(N_gemm),                      # SizesFree1 (full GEMM N)
        np.uint32(1),                           # SizesFree2 (batch=1)
        np.uint32(K),                           # SizesSum0
        amdgpu_exec.InOutArray(d_bf16),         # D (bf16, M×N_out col-major)
        amdgpu_exec.InputArray(c_bf16),         # C (bf16, beta=0)
        amdgpu_exec.InputArray(a_bf16),         # A (K×M col-major, TransposeA=True)
        amdgpu_exec.InputArray(b_bf16),         # B (K×N_gemm col-major)
        amdgpu_exec.InputArray(ws_dummy),       # AddressWS (valid but unused)
        amdgpu_exec.InputArray(flags_dummy),    # AddressFlags (valid but unused)
        np.uint32(M), np.uint32(0),             # strideD0=M, strideD1=0
        np.uint32(M), np.uint32(0),             # strideC0=M, strideC1=0
        np.uint32(K), np.uint32(0),             # strideA0=K, strideA1=0
        np.uint32(K), np.uint32(0),             # strideB0=K, strideB1=0
        alpha,                                  # alpha
        sk_args["iters_per_tile"],              # ItersPerTile
        sk_args["magic_iters_per_tile"],        # MagicNumberItersPerTile
        sk_args["shift_iters_per_tile"],        # MagicShiftItersPerTile
        sk_args["sk_iters_per_wg"],             # SKItersPerWG
        sk_args["sk_grid"],                     # skGrid
        sk_args["sk_tiles"],                    # skTiles
    ]

    def verify(arguments):
        import ml_dtypes
        d_gpu_bf16 = np.asarray(arguments[8].array)
        d_gpu_f32  = np.asarray(d_gpu_bf16).astype(np.float32)
        d_ref_f32_local = np.asarray(d_ref_bf16).astype(np.float32)

        rtol, atol = 2e-2, 2e-2
        diff = np.abs(d_gpu_f32 - d_ref_f32_local)
        tol  = atol + rtol * np.abs(d_ref_f32_local)
        bad  = np.where(~np.isfinite(d_gpu_f32) | (diff > tol))
        ok   = len(bad[0]) == 0

        max_abs = float(np.nanmax(np.abs(d_gpu_f32 - d_ref_f32_local))) if d_gpu_f32.size else 0.0
        max_rel = float(np.nanmax(np.abs((d_gpu_f32 - d_ref_f32_local) /
                                          (np.abs(d_ref_f32_local) + 1e-8))))
        if ok:
            print(f"verification: PASSED  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
        else:
            print(f"verification: FAILED  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
                  f"mismatches={len(bad[0])}")
            for i in range(min(5, len(bad[0]))):
                r, c = bad[0][i], bad[1][i]
                print(f"    [{r},{c}] gpu={d_gpu_f32[r, c]:.6f}  ref={d_ref_f32_local[r, c]:.6f}  "
                      f"diff={d_gpu_f32[r, c]-d_ref_f32_local[r, c]:.3e}")

    return amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=grid_dim,
        block_dim=block_dim,
        num_iterations=num_iterations,
        verify_fn=verify,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TensileLite fused GEMM+SwiGLU example")
    p.add_argument("--M", type=int, default=64,   help="Output rows (arbitrary)")
    p.add_argument("--K", type=int, default=128,  help="Reduction dimension (arbitrary)")
    p.add_argument("--wg-n", type=int, default=1,
                   help="MIWaveGroup[1] (number of N wave-group waves)")
    p.add_argument("--n-tiles", type=int, default=1,
                   help="number of MacroTile1-wide N tiles; N_gemm = n_tiles * MacroTile1")
    p.add_argument("--chip", default=None, help="Target GPU (default: auto-detect)")
    p.add_argument("--iterations", "-i", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    chip = args.chip or amdgpu_exec.get_chip()
    print(f"device     : {chip}")

    if not chip.startswith("gfx950"):
        print(f"WARNING: SwiGLU is only implemented for gfx950; current chip={chip}")
        print("The kernel will likely be rejected by Solution validation.")

    print("Setting up TensileLite...")
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)

    print("Building SwiGLU solution...")
    solution = build_swiglu_solution(chip, assembler, isaInfoMap, wg_n=args.wg_n)
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    N_gemm = MT1 * args.n_tiles
    N_out  = N_gemm // 2
    print(f"problem    : M={args.M}, N_gemm={N_gemm}, N_out={N_out}, K={args.K}, n_tiles={args.n_tiles}")
    print(f"MacroTile  : {MT0}x{MT1}")
    print(f"MIWaveGroup: {solution['MIWaveGroup']}")
    print(f"NumThreads : {solution['NumThreads']}")
    print(f"DepthU     : {solution['DepthU']}\n")

    print("Generating assembly...")
    t0 = time.perf_counter()
    asm_str, kernel_name = generate_asm(solution, assembler, debugConfig)
    print(f"Gen time   : {time.perf_counter()-t0:.3f} s")
    print(f"Kernel     : {kernel_name}")
    print(f"Assembly   : {len(asm_str):,} chars\n")

    print("Compiling to HSACO...")
    t0 = time.perf_counter()
    hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
    print(f"Compile    : {time.perf_counter()-t0:.3f} s")
    print(f"HSACO size : {len(hsaco):,} bytes\n")

    print("Running kernel...")
    times_ns = run_swiglu(
        hsaco, kernel_name, solution,
        args.M, N_gemm, args.K, args.iterations
    )

    if times_ns:
        best_ms = min(times_ns) / 1e6
        avg_ms  = sum(times_ns) / len(times_ns) / 1e6
        print(f"\nbest : {best_ms:.3f} ms  avg : {avg_ms:.3f} ms  ({len(times_ns)} iters)")


if __name__ == "__main__":
    main()
