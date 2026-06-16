# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite + amdgpu-exec GEMM integration example.

TensileLite generates a GCN assembly string for a float32 GEMM kernel.
amdgpu-exec (self-contained, no external ROCm paths for compilation) then
compiles that assembly and runs it on the GPU.

Usage:
    python tensile_gemm_example.py [--M 256] [--N 256] [--K 256]
                                   [--alpha 1.0] [--beta 0.0]
                                   [--chip gfx950] [--iterations 10]
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import amdgpu_exec

# Ensure the TensileLite Python package is importable from the script's directory.
_TENSILE_DIR = os.path.dirname(os.path.abspath(__file__))
if _TENSILE_DIR not in sys.path:
    sys.path.insert(0, _TENSILE_DIR)


# ---------------------------------------------------------------------------
# TensileLite setup: ISA capabilities + solution + assembly generation
# ---------------------------------------------------------------------------


def setup_tensile(chip: str) -> tuple:
    """Initialize the TensileLite environment for the given chip.

    Uses amdclang++ (found automatically from standard ROCm locations) only
    for the rocisa ISA capability probing step — not for kernel compilation.
    Kernel compilation is handled entirely by amdgpu-exec's embedded toolchain.

    Returns (assembler, isaInfoMap, debugConfig).
    """
    from pathlib import Path
    from Tensile.Toolchain.Validators import validateToolchain
    from Tensile.Toolchain.Component import Assembler
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.GlobalParameters import assignGlobalParameters
    from Tensile.Common.Types import DebugConfig

    gfx = chip.split(":")[0]
    cxx = validateToolchain("amdclang++")  # found from ROCM_PATH/bin or /opt/rocm/bin
    isa = gfxToIsa(gfx)
    isaInfoMap = makeIsaInfoMap([isa], cxx)  # probes ISA caps via amdclang++
    assignGlobalParameters({}, isaInfoMap)
    assembler = Assembler(Path(cxx), co_version="6")
    debugConfig = DebugConfig()
    return assembler, isaInfoMap, debugConfig


def build_gemm_solution(chip: str, assembler, isaInfoMap):
    """Build and validate a float32 GEMM kernel config using TensileLite.

    Computes D = alpha * A * B^T + beta * C where:
      A : M×K  row-major  (TransposeA=False)
      B : N×K  row-major  (TransposeB=True, so logical B^T is K×N)
      C, D : M×N row-major

    Uses MFMA (v_mfma_f32_16x16x4f32) with a 4×4 wave-tile → MacroTile 64×64.

    Returns the validated Solution object.
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
        "DataType": 0,  # float32 ('S')
        "DestDataType": 0,
        "ComputeDataType": 0,
        "TransposeA": False,
        "TransposeB": True,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
        "GroupedGemm": False,
        "HighPrecisionAccumulate": False,
        "UseBias": 0,
        "UseScaleAB": "",
        "UseScaleCD": False,
        "UseScaleAlphaVec": 0,
        "Sparse": 0,
    }

    # 9-item MatrixInstruction: [M, N, K, B, wg0, wg1, wt0, wt1, depthU_ratio]
    # v_mfma_f32_16x16x4f32, WorkGroup=[16,16,1], WaveTile=[4,4], depthU_ratio=1
    mi9 = [
        16,
        16,
        4,
        1,  # M, N, K, B
        4,
        4,  # wg0=SubGroup0//(M//wfsize), wg1
        4,
        4,  # waveTile0, waveTile1
        1,
    ]  # depthU multiplier

    wavefrontSize = 64 if not isaInfoMap[isa].archCaps["HasWave32"] else 32

    mi_params = matrixInstructionToMIParameters(
        mi9, isa, wavefrontSize, problem_type, workGroup=None, isaInfoMap=isaInfoMap
    )

    config = {
        "ProblemType": problem_type,
        "InternalSupportParams": defaultInternalSupportParams,
        "ISA": [isa.major, isa.minor, isa.patch],
        "CodeObjectVersion": "6",
        "GlobalSplitU": 1,
        "KernelLanguage": "Assembly",
    }
    config.update(
        mi_params
    )  # merge MIBlock, MIWaveGroup, MIWaveTile, MFMA_BF16_1K, etc.

    if not validateMIParameters(config, isaInfoMap):
        raise RuntimeError(
            "MI parameter validation failed — adjust MatrixInstruction config"
        )

    solution = Solution(
        config,
        splitGSU=False,
        printSolutionRejectionReason=True,
        printIndexAssignmentInfo=False,
        assembler=assembler,
        isaInfoMap=isaInfoMap,
    )
    if not solution["Valid"]:
        raise RuntimeError(
            "Solution was rejected by TensileLite — see rejection reason above"
        )
    return solution


def generate_gemm_asm(solution, assembler, debugConfig) -> tuple:
    """Use KernelWriterAssembly to generate the GCN assembly string.

    This is TensileLite's sole output in this integration:
    a raw .s assembly text ready to hand off to amdgpu-exec.

    Returns (asm_str, kernel_name).
    """
    import rocisa
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelNameMin

    kwa = KernelWriterAssembly(assembler, debugConfig)
    ti = rocisa.rocIsa.getInstance()
    kwa.setRocIsa(ti.getData(), ti.getOutputOptions())

    kernel = solution.getKernels()[0]
    kernel.duplicate = False  # required by getSourceFileString
    err, asm_str = kwa.getSourceFileString(kernel)
    if err:
        raise RuntimeError(f"Assembly generation failed with error code {err}")

    kernel_name = getKernelNameMin(kernel, splitGSU=False)
    return asm_str, kernel_name


# ---------------------------------------------------------------------------
# amdgpu-exec: run the GEMM
# ---------------------------------------------------------------------------


def run_gemm(
    hsaco: bytes,
    kernel_name: str,
    solution,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
    num_iterations: int,
) -> list:
    """Execute the GEMM kernel via amdgpu-exec and verify correctness.

    The TensileLite kernel uses column-major (Fortran order) storage internally.
    Strides in the argument list are strides along the reduction dimension, not
    the leading dimension in the conventional BLAS sense.

    Kernel argument layout (KernArgsVersion=2, UseUniversalArgs=True, Batched=True):
      GemmInfo, kernel_info0, kernel_info1, numWG,
      SizesFree0(M), SizesFree1(N), SizesFree2(batch=1), SizesSum0(K),
      D, C, A, B,
      strideD0(=M), strideD1(=0), strideC0(=M), strideC1(=0),
      strideA0(=M), strideA1(=0), strideB0(=N), strideB1(=0),
      alpha, beta

    Strides for column-major arrays:
      A (M×K col-major): stride along K = M   → strideA0=M
      B (N×K col-major): stride along K = N   → strideB0=N
      C/D (M×N col-major): stride along N = M → strideD0=strideC0=M
    """
    rng = np.random.default_rng(42)
    # Allocate column-major (Fortran order) arrays — kernel expects col-major
    a = np.asfortranarray(rng.random((M, K), dtype=np.float32))  # M×K col-major
    b = np.asfortranarray(rng.random((N, K), dtype=np.float32))  # N×K col-major
    c = np.asfortranarray(rng.random((M, N), dtype=np.float32))  # M×N col-major
    d = np.asfortranarray(np.zeros((M, N), dtype=np.float32))  # M×N col-major (output)
    # CPU reference: D = alpha * A @ B.T + beta * C  (all in numpy row-major for reference)
    d_ref = np.float32(alpha) * (np.asarray(a) @ np.asarray(b).T) + np.float32(
        beta
    ) * np.asarray(c)

    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    numWG = math.ceil(M / MT0) * math.ceil(N / MT1)
    block_dim = (solution["NumThreads"], 1, 1)
    grid_dim = (numWG, 1, 1)

    # Encode internalArg0 matching ContractionSolution.cpp (KernArgsVersion=2):
    #   bits 31-16: packed stagger word = (staggerUMapping<<13 | staggerUStrideShift<<8 | staggerU&0xFF)
    #   bit  15:    gsuc (0 for default config)
    #   bit  14:    gsuwgmrr (0 for default config)
    #   bits 13-0:  GSU (masked to 14 bits for version < 3)
    stagger_u            = solution["StaggerU"]
    stagger_u_mapping    = solution["StaggerUMapping"]
    stagger_stride_shift = solution.get("_staggerStrideShift", 0)
    su_word = (stagger_u_mapping << 13) | ((stagger_stride_shift << 8) & 0x1F00) | (stagger_u & 0xFF)
    kernel_info0 = (su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF)

    # Encode internalArg1 matching ContractionSolution.cpp (version>=2, useSFC=False):
    #   bits 31-22: wgmxccg (0 = let runtime fill from hardware CU count)
    #   bits 21-16: wgmxcc (WorkGroupMappingXCC, default 1)
    #   bits 15-0:  WorkGroupMapping
    wgmxcc = solution.get("WorkGroupMappingXCC", 1)
    kernel_info1 = (wgmxcc << 16) | (solution["WorkGroupMapping"] & 0xFFFF)

    args = [
        np.uint32(1),  # GemmInfo
        np.uint32(kernel_info0),  # kernel_info0
        np.uint32(kernel_info1),  # kernel_info1
        np.uint32(numWG),  # numWG
        np.uint32(M),
        np.uint32(N),
        np.uint32(1),  # SizesFree0, SizesFree1, SizesFree2 (batch=1)
        np.uint32(K),  # SizesSum0
        amdgpu_exec.InOutArray(d),  # D (M×N col-major output)
        amdgpu_exec.InputArray(c),  # C (M×N col-major)
        amdgpu_exec.InputArray(a),  # A (M×K col-major)
        amdgpu_exec.InputArray(b),  # B (N×K col-major, TransposeB=True)
        np.uint32(M),
        np.uint32(0),  # strideD0=M, strideD1=0 (no batch)
        np.uint32(M),
        np.uint32(0),  # strideC0=M, strideC1=0
        np.uint32(M),
        np.uint32(0),  # strideA0=M, strideA1=0
        np.uint32(N),
        np.uint32(0),  # strideB0=N, strideB1=0
        np.float32(alpha),
        np.float32(beta),
    ]

    def verify(arguments):
        result = np.asarray(arguments[8].array)
        if np.allclose(result, d_ref, rtol=1e-4, atol=1e-4):
            print("verification: PASSED")
        else:
            max_err = np.max(np.abs(result - d_ref))
            print(f"verification: FAILED  max_err={max_err:.3e}")

    return amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=grid_dim,
        block_dim=block_dim,
        num_iterations=num_iterations,
        verify_fn=verify,
    )


def report_timing(times_ns: list, M: int, N: int, K: int):
    print(f"\n=== timing ===")
    if not times_ns:
        print("no timing samples (0 iterations)")
        return
    times_ms = [t / 1e6 for t in times_ns]
    best_ms = min(times_ms)
    avg_ms = sum(times_ms) / len(times_ms)
    tflops = 2 * M * N * K / (best_ms * 1e-3) / 1e12
    print(f"iterations : {len(times_ms)}")
    print(f"best       : {best_ms:.3f} ms")
    print(f"avg        : {avg_ms:.3f} ms")
    print(f"TFLOPS     : {tflops:.3f}  (2 * {M} * {N} * {K} / best)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="TensileLite + amdgpu-exec GEMM example")
    p.add_argument("--M", type=int, default=256)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument(
        "--chip",
        default=None,
        help="target GPU (default: auto-detect via amdgpu_exec.get_chip())",
    )
    p.add_argument("--iterations", "-i", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    chip = args.chip or amdgpu_exec.get_chip()
    print(f"device     : {chip}")
    print(f"problem    : M={args.M}, N={args.N}, K={args.K}")
    print(f"alpha/beta : {args.alpha} / {args.beta}\n")

    # --- TensileLite: initialize ISA environment ---
    print("Setting up TensileLite environment...")
    start_time = time.perf_counter()
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    end_time = time.perf_counter()
    print(f"Setup time : {end_time - start_time:.3f} s\n")

    # --- TensileLite: build and validate the GEMM kernel config ---
    start_time = time.perf_counter()
    solution = build_gemm_solution(chip, assembler, isaInfoMap)
    end_time = time.perf_counter()
    print(f"Build time : {end_time - start_time:.3f} s")
    print(f"MacroTile  : {solution['MacroTile0']}×{solution['MacroTile1']}")
    print(f"NumThreads : {solution['NumThreads']}\n")

    # --- TensileLite: generate the GCN assembly string ---
    print("Generating assembly...")
    start_time = time.perf_counter()
    asm_str, kernel_name = generate_gemm_asm(solution, assembler, debugConfig)
    end_time = time.perf_counter()
    print(f"Gen time   : {end_time - start_time:.3f} s")
    print(f"Kernel     : {kernel_name}")
    print(f"Assembly   : {len(asm_str):,} chars\n")

    # --- amdgpu-exec: compile (self-contained embedded toolchain) ---
    print("Compiling assembly to HSACO...")
    start_time = time.perf_counter()
    hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
    end_time = time.perf_counter()
    print(f"Compile time : {end_time - start_time:.3f} s")
    print(f"HSACO size : {len(hsaco):,} bytes\n")

    # --- amdgpu-exec: launch, verify, and time ---
    start_time = time.perf_counter()
    times_ns = run_gemm(
        hsaco,
        kernel_name,
        solution,
        args.M,
        args.N,
        args.K,
        args.alpha,
        args.beta,
        args.iterations,
    )
    end_time = time.perf_counter()
    print(f"\nTotal time : {end_time - start_time:.3f} s")
    report_timing(times_ns, args.M, args.N, args.K)


if __name__ == "__main__":
    main()
