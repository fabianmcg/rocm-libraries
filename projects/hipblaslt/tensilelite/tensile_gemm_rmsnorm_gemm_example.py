# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite fused GEMM + PartialRMS (K1 kernel) example.

Demonstrates the PartialRMS fused epilogue on gfx950:
  D = h1 * gamma           (bf16, M x N_hidden)
  partialBuf = Σx²         (fp32, M)   where h1 = A^T * W0

This is Phase 1 (K1) of a two-kernel RMSNorm pipeline. K1 computes the raw
per-row sum of squares (not divided by N) and writes to partialBuf. K2 would
read partialBuf, compute rstd = rsqrt(Σx²/N + eps), and apply it.

Row-containment constraint: N_hidden must equal MacroTile1 so each WG owns
exactly one complete output row. This is validated at launch time.

StreamKForceDPOnly=1 ensures every WG computes a complete tile (no K-split
partial fixup) so the accumulator is final at the PartialRMS epilogue hook.

Usage:
    python tensile_gemm_rmsnorm_gemm_example.py --phase k1
    python tensile_gemm_rmsnorm_gemm_example.py --phase k1 --wg-n 2
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import amdgpu_exec

_TENSILE_DIR = os.path.dirname(os.path.abspath(__file__))
if _TENSILE_DIR not in sys.path:
    sys.path.insert(0, _TENSILE_DIR)


# ---------------------------------------------------------------------------
# Re-use helpers from tensile_rmsnorm_example.py
# ---------------------------------------------------------------------------

def setup_tensile(chip: str):
    from tensile_rmsnorm_example import setup_tensile as _setup
    return _setup(chip)


def compute_sk3_dp_args(M: int, N: int, K: int, solution) -> dict:
    from tensile_rmsnorm_example import compute_sk3_dp_args as _compute
    return _compute(M, N, K, solution)


# ---------------------------------------------------------------------------
# Build the K1 (PartialRMS) solution
# ---------------------------------------------------------------------------

def build_k1_solution(chip: str, assembler, isaInfoMap, wg_n: int = 1):
    """Build a bf16 GEMM + PartialRMS kernel for gfx950.

    Mirror of build_rmsnorm_solution from tensile_rmsnorm_example.py, with
    PartialRMS=True instead of RMSNorm=True.
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
        "OperationType":    "GEMM",
        "DataType":         "b",    # bf16
        "DestDataType":     "b",    # bf16
        "ComputeDataType":  "s",    # fp32 accumulation
        "HighPrecisionAccumulate": True,
        "TransposeA":       True,   # A: K×M col-major (TN layout)
        "TransposeB":       False,  # B: K×N col-major
        "UseBeta":          True,
        "Batched":          True,
        "StridedBatched":   True,
        "GroupedGemm":      False,
        "UseBias":          0,
        "UseScaleAB":       "",
        "UseScaleCD":       False,
        "UseScaleAlphaVec": 0,
        "Sparse":           0,
    }

    # [instM, instN, instK, instB, mi4, wt1, wt0, wg0_waves, wg1_waves]
    mi9 = [16, 16, 32, 1, 1, 4, 4, 1, wg_n]

    wavefrontSize = 64
    mi_params = matrixInstructionToMIParameters(
        mi9, isa, wavefrontSize, problem_type, workGroup=None, isaInfoMap=isaInfoMap
    )

    config = {
        "ProblemType":           problem_type,
        "InternalSupportParams": defaultInternalSupportParams,
        "ISA":                   [isa.major, isa.minor, isa.patch],
        "CodeObjectVersion":     "6",
        "GlobalSplitU":          1,
        "KernelLanguage":        "Assembly",
        "StreamK":               3,
        "StreamKForceDPOnly":    1,
        "StreamKAtomic":         0,
        "ScheduleIterAlg":       3,
        "PrefetchGlobalRead":    1,
        "DirectToLdsA":          1,
        "DirectToLdsB":          1,
        "UseSubtileImpl":        True,
        "PartialRMS":            True,
        "StaggerU":              0,
        "DepthU":                64,
        "LdsPadA":               -1,
        "LdsPadB":               -1,
        "StoreVectorWidth":      -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "PreloadKernArgs":       False,
        "_1LDSBuffer":           0,
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
      D (slot 8):     bf16 output, tol = 2e-2. D = h1 * gamma.
      partialBuf (slot N+1): fp32, tol = 1e-4. partialBuf[m] = Σ_n h1[m,n]^2.

    Numpy reference (uses bf16-rounded inputs to match kernel precision):
      a_ref  = bf16(A).T          (M x K, fp32)
      w0_ref = bf16(W0)           (K x N, fp32)
      h0     = a_ref @ w0_ref     (M x N, fp32)   [shape: M x N_hidden]
      h1     = h0                 (phase-1a: no residual)
      D_ref  = bf16(h1 * gamma)   (M x N, bf16)
      sumsq  = Σ_n h1[:,n]^2      (M, fp32)        [raw sum, NOT mean]
    """
    import ml_dtypes

    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]

    if N != MT1:
        raise ValueError(
            f"Row-containment violated: N={N} must equal MacroTile1={MT1}."
        )

    M_padded = math.ceil(M / MT0) * MT0  # padded row count
    numWG    = math.ceil(M / MT0) * math.ceil(N / MT1)

    rng = np.random.default_rng(42)
    a_f32  = np.asfortranarray(rng.random((K, M), dtype=np.float32) * 0.1)
    w0_f32 = np.asfortranarray(rng.random((K, N), dtype=np.float32) * 0.1)

    a_bf16  = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(w0_f32.astype(ml_dtypes.bfloat16))

    c_bf16 = np.zeros((M, N), dtype=ml_dtypes.bfloat16, order='F')
    d_bf16 = np.zeros((M, N), dtype=ml_dtypes.bfloat16, order='F')

    gamma_f32  = rng.random(N, dtype=np.float32) + 0.5
    gamma_bf16 = gamma_f32.astype(ml_dtypes.bfloat16)

    # Output buffer for per-row Σx² (fp32, padded to M_padded rows).
    partial_buf = np.zeros(M_padded, dtype=np.float32)

    # Numpy reference.
    a_ref     = np.asarray(a_bf16).astype(np.float32)
    w0_ref    = np.asarray(w0_bf16).astype(np.float32)
    h0        = a_ref.T @ w0_ref                # M x N, fp32
    h1        = h0                              # phase-1a: no residual
    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)
    d_ref     = np.asarray((h1 * gamma_ref[np.newaxis, :]).astype(ml_dtypes.bfloat16))
    sumsq_ref = np.sum(h1 ** 2, axis=1)        # per-row Σx², fp32

    sk_args      = compute_sk3_dp_args(M, N, K, solution)
    stagger_u    = solution.get("StaggerU", 0)
    su_map       = solution.get("StaggerUMapping", 0)
    ss_shift     = solution.get("_staggerStrideShift", 0)
    su_word      = (su_map << 13) | ((ss_shift << 8) & 0x1F00) | (stagger_u & 0xFF)
    kernel_info0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))
    wgmxcc       = solution.get("WorkGroupMappingXCC", 1)
    kernel_info1 = np.uint32((wgmxcc << 16) | (solution["WorkGroupMapping"] & 0xFFFF))

    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    # Argument layout (mirrors RMSNorm layout, with PartialRMS-specific tail):
    #   slots 0-29: identical to RMSNorm kernarg layout
    #   slot 30: RMSNormGamma (ptr, bf16, len N_hidden)
    #   slot 31: PartialBuf   (ptr, fp32, len M_padded) [InOutArray]
    args = [
        np.uint32(1),                           # 0: Gemm info
        kernel_info0,                           # 1: kernel_info0
        kernel_info1,                           # 2: kernel_info1
        np.uint32(numWG),                       # 3: numWG
        np.uint32(M),                           # 4: SizesFree0=M
        np.uint32(N),                           # 5: SizesFree1=N
        np.uint32(1),                           # 6: SizesFree2=batch
        np.uint32(K),                           # 7: SizesSum0=K
        amdgpu_exec.InOutArray(d_bf16),         # 8: D (bf16)
        amdgpu_exec.InputArray(c_bf16),         # 9: C (bf16, beta=0)
        amdgpu_exec.InputArray(a_bf16),         # 10: A (K×M col-major)
        amdgpu_exec.InputArray(w0_bf16),        # 11: B (K×N col-major)
        amdgpu_exec.InputArray(ws_dummy),       # 12: AddressWS (unused with ForceDPOnly)
        amdgpu_exec.InputArray(flags_dummy),    # 13: AddressFlags
        np.uint32(M), np.uint32(0),             # 14,15: strideD0=M, strideD1=0
        np.uint32(M), np.uint32(0),             # 16,17: strideC0=M, strideC1=0
        np.uint32(K), np.uint32(0),             # 18,19: strideA0=K, strideA1=0
        np.uint32(K), np.uint32(0),             # 20,21: strideB0=K, strideB1=0
        np.float32(1.0),                        # 22: alpha
        np.float32(0.0),                        # 23: beta
        sk_args["iters_per_tile"],              # 24: ItersPerTile
        sk_args["magic_iters_per_tile"],        # 25: MagicNumberItersPerTile
        sk_args["shift_iters_per_tile"],        # 26: MagicShiftItersPerTile
        sk_args["sk_iters_per_wg"],             # 27: SKItersPerWG
        sk_args["sk_grid"],                     # 28: skGrid
        sk_args["sk_tiles"],                    # 29: skTiles
        amdgpu_exec.InputArray(gamma_bf16),     # 30: RMSNormGamma (bf16)
        amdgpu_exec.InOutArray(partial_buf),    # 31: PartialBuf (fp32)
    ]

    result_holder = {}

    def verify(arguments):
        d_gpu_bf16  = np.asarray(arguments[8].array)
        d_gpu_f32   = d_gpu_bf16.astype(np.float32)
        d_ref_f32   = d_ref.astype(np.float32)
        pb_gpu      = np.asarray(arguments[31].array)

        # Check D (bf16): row-wise comparison.
        rtol, atol = 2e-2, 2e-2
        diff_d  = np.abs(d_gpu_f32[:M] - d_ref_f32[:M])
        tol_d   = atol + rtol * np.abs(d_ref_f32[:M])
        bad_d   = np.where(~np.isfinite(d_gpu_f32[:M]) | (diff_d > tol_d))
        d_ok    = len(bad_d[0]) == 0

        # Check partialBuf (fp32): first M entries.
        rtol_p, atol_p = 1e-4, 1e-4
        diff_p  = np.abs(pb_gpu[:M] - sumsq_ref)
        tol_p   = atol_p + rtol_p * np.abs(sumsq_ref)
        bad_p   = np.where(~np.isfinite(pb_gpu[:M]) | (diff_p > tol_p))
        p_ok    = len(bad_p[0]) == 0

        ok = d_ok and p_ok
        max_abs_d = float(np.nanmax(np.abs(d_gpu_f32[:M] - d_ref_f32[:M]))) if M > 0 else 0.0
        max_abs_p = float(np.nanmax(np.abs(pb_gpu[:M] - sumsq_ref))) if M > 0 else 0.0

        if ok:
            print(f"verification: PASSED  "
                  f"D max_abs={max_abs_d:.3e}  partialBuf max_abs={max_abs_p:.3e}")
        else:
            print(f"verification: FAILED")
            if not d_ok:
                print(f"  D: {len(bad_d[0])} elements out of tolerance, max_abs={max_abs_d:.3e}")
                r, c = bad_d[0][0], bad_d[1][0]
                print(f"    first bad D[{r},{c}]: gpu={d_gpu_f32[r,c]:.6f} "
                      f"ref={d_ref_f32[r,c]:.6f}")
            if not p_ok:
                print(f"  partialBuf: {len(bad_p[0])} rows out of tolerance, "
                      f"max_abs={max_abs_p:.3e}")
                i = bad_p[0][0]
                print(f"    first bad partialBuf[{i}]: gpu={pb_gpu[i]:.6f} "
                      f"ref={sumsq_ref[i]:.6f}")

        result_holder["ok"] = ok

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(numWG, 1, 1),
        block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=1,
        verify_fn=verify,
    )
    return result_holder.get("ok", False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TensileLite fused GEMM+PartialRMS example")
    p.add_argument("--phase", choices=["k1"], default="k1",
                   help="Phase to run: k1 = GEMM + PartialRMS epilogue")
    p.add_argument("--M",     type=int, default=2048, help="Output rows")
    p.add_argument("--K",     type=int, default=4096, help="Reduction dimension")
    p.add_argument("--wg-n",  type=int, default=1, dest="wg_n",
                   help="MIWaveGroup[1]: waves splitting N (1=single, >1=cross-wave LDS)")
    p.add_argument("--chip",  default=None, help="Target GPU (default: auto-detect)")
    return p.parse_args()


def main():
    args = parse_args()
    chip = args.chip or amdgpu_exec.get_chip()
    print(f"device     : {chip}")

    if not chip.startswith("gfx950"):
        print(f"WARNING: PartialRMS is only implemented for gfx950; current chip={chip}")

    print("Setting up TensileLite...")
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)

    if args.phase == "k1":
        print(f"Building K1 PartialRMS solution (wg_n={args.wg_n})...")
        solution = build_k1_solution(chip, assembler, isaInfoMap, wg_n=args.wg_n)
        N = solution["MacroTile1"]
        print(f"problem    : M={args.M}, N={N}, K={args.K}")
        print(f"MacroTile  : {solution['MacroTile0']}×{solution['MacroTile1']}")
        print(f"MIWaveGroup: {solution['MIWaveGroup']}")
        print(f"NumThreads : {solution['NumThreads']}")

        print("Generating assembly...")
        t0 = time.perf_counter()
        asm_str, kernel_name = generate_asm(solution, assembler, debugConfig)
        print(f"Gen time   : {time.perf_counter()-t0:.3f} s")
        print(f"Kernel     : {kernel_name}")

        print("Compiling to HSACO...")
        t0 = time.perf_counter()
        hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
        print(f"Compile    : {time.perf_counter()-t0:.3f} s")

        print("Running K1 kernel...")
        ok = run_k1(hsaco, kernel_name, solution, args.M, N, args.K)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
