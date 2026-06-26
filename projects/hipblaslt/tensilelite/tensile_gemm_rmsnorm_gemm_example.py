# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite fused GEMM + PartialRMS (K1 kernel) example.

Demonstrates the PartialRMS fused epilogue on gfx950:
  D = h1 * gamma           (bf16, M x N_hidden)
  partialBuf = Σx²         (fp32, M)   where h1 = A^T * W0

This is Phase 1 (K1) of a two-kernel RMSNorm pipeline. K1 computes the raw
per-row sum of squares (not divided by N) and writes to partialBuf. K2
reads partialBuf, computes rstd = rsqrt(Σx²/N + eps), and writes rstdBuf.

Row-containment constraint: N_hidden must equal MacroTile1 so each WG owns
exactly one complete output row. This is validated at launch time.

StreamKForceDPOnly=1 ensures every WG computes a complete tile (no K-split
partial fixup) so the accumulator is final at the PartialRMS epilogue hook.

Usage:
    python tensile_gemm_rmsnorm_gemm_example.py --phase k1
    python tensile_gemm_rmsnorm_gemm_example.py --phase k1 --wg-n 2
    python tensile_gemm_rmsnorm_gemm_example.py --phase k2 --N-hidden 64
    python tensile_gemm_rmsnorm_gemm_example.py --phase k2 --N-hidden 4096 --eps 1e-6
"""
import argparse
import functools
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
# K2: auxiliary reduction kernel (GCN assembly generated via amdclang++)
# ---------------------------------------------------------------------------
# Kernarg layout (packed, as seen by the HIP runtime):
#   offset  0: partialBuf ptr (fp32, 8B)
#   offset  8: rstdBuf    ptr (fp32, 8B)
#   offset 16: M          (u32, 4B)
#   offset 20: eps        (f32, 4B)
#   total = 24 bytes of user args; the gfx950 ABI pads the kernarg segment
#   further for hidden dispatch args (block dims etc.), so the full
#   kernarg_size is larger.
#
# N_hidden is NOT a runtime arg: 1.0/N_hidden is embedded as a compile-time
# float constant in the generated assembly.
# One thread per row: grid = (ceil(M/256), 1, 1), block = (256, 1, 1).

@functools.lru_cache(maxsize=None)
def build_aux_reduction_asm(chip: str, N_hidden: int) -> tuple:
    """Build, assemble and return (asm_str, kernel_name, hsaco) for K2.

    Uses amdclang++ to compile a HIP C++ device kernel to GCN assembly, then
    compile_asm_to_hsaco to assemble it. The result is cached per (chip,
    N_hidden) so repeated calls within a session compile only once.
    """
    import subprocess
    import tempfile

    kernel_name = f"aux_reduction_N{N_hidden}"
    gfx = chip.split(":")[0]
    # 1.0/N_hidden is a compile-time constant; use repr to get full precision.
    inv_n = repr(1.0 / N_hidden) + "f"

    hip_src = f"""\
#include <hip/hip_runtime.h>
// K2 auxiliary reduction: one thread per row, computes
//   rstd[tid] = rsqrt(partialBuf[tid] / N_hidden + eps)
// N_hidden is baked in as a compile-time constant for maximum throughput.
extern "C" __global__ void {kernel_name}(
    const float* __restrict__ partialBuf,
    float* __restrict__ rstdBuf,
    unsigned int M,
    float eps
) {{
    constexpr float invN = {inv_n};
    unsigned int tid = (unsigned int)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;
    float p = partialBuf[tid] * invN + eps;
    rstdBuf[tid] = __builtin_amdgcn_rsqf(p);
}}
"""

    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(hip_src)
        src_path = f.name

    s_path = src_path.replace(".cpp", ".s")
    try:
        r = subprocess.run(
            [
                "amdclang++",
                f"--offload-arch={gfx}",
                "-O2",
                "-x", "hip",
                "-S",
                "--cuda-device-only",
                "-o", s_path,
                src_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        with open(s_path) as sf:
            asm_str = sf.read()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"amdclang++ failed: {e.stderr[:500]}"
        ) from e
    finally:
        if os.path.exists(src_path):
            os.unlink(src_path)
        if os.path.exists(s_path):
            os.unlink(s_path)

    hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
    return asm_str, kernel_name, hsaco


def run_k2(hsaco: bytes, kernel_name: str, M: int, N_hidden: int, eps: float,
           partial_buf_in=None):
    """Execute the K2 auxiliary reduction kernel and verify outputs.

    Computes per-row rstd = rsqrt(partialBuf[m] / N_hidden + eps).

    If partial_buf_in is None, a random synthetic partialBuf is generated.
    Returns (ok, rstdBuf_gpu) where rstdBuf_gpu is a numpy fp32 array of
    length ceil(M/256)*256 (padded).
    """
    if partial_buf_in is None:
        rng = np.random.default_rng(42)
        partial_buf = rng.random(M, dtype=np.float32) * 10.0 + 1e-3
    else:
        partial_buf = np.asarray(partial_buf_in, dtype=np.float32)

    # Pad to multiple of 256 for the GPU buffer.
    M_padded = math.ceil(M / 256) * 256
    partial_buf_padded = np.zeros(M_padded, dtype=np.float32)
    partial_buf_padded[:M] = partial_buf[:M]

    rstd_buf = np.zeros(M_padded, dtype=np.float32)

    # Reference: fp32 rsqrt per row.
    rstd_ref = np.array(
        [1.0 / math.sqrt(float(partial_buf_padded[m]) / N_hidden + eps)
         for m in range(M)],
        dtype=np.float32,
    )

    args = [
        amdgpu_exec.InputArray(partial_buf_padded),
        amdgpu_exec.InOutArray(rstd_buf),
        np.uint32(M),
        np.float32(eps),
    ]

    grid_dim  = (math.ceil(M / 256), 1, 1)
    block_dim = (256, 1, 1)

    result_holder = {}

    def verify(arguments):
        rstd_gpu = np.asarray(arguments[1].array).copy()
        result_holder["rstd_gpu"] = rstd_gpu

        rtol, atol = 1e-4, 1e-4
        diff   = np.abs(rstd_gpu[:M] - rstd_ref)
        tol    = atol + rtol * np.abs(rstd_ref)
        bad    = np.where(~np.isfinite(rstd_gpu[:M]) | (diff > tol))[0]
        ok     = len(bad) == 0

        max_abs = float(np.nanmax(np.abs(rstd_gpu[:M] - rstd_ref))) if M > 0 else 0.0
        if ok:
            print(f"verification: PASSED  rstdBuf max_abs={max_abs:.3e}")
        else:
            print(f"verification: FAILED  rstdBuf max_abs={max_abs:.3e}  "
                  f"mismatches={len(bad)}")
            i = bad[0]
            print(f"    first bad rstdBuf[{i}]: gpu={rstd_gpu[i]:.6f} "
                  f"ref={rstd_ref[i]:.6f}")
        result_holder["ok"] = ok

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=grid_dim,
        block_dim=block_dim,
        num_iterations=1,
        verify_fn=verify,
    )
    return result_holder.get("ok", False), result_holder.get("rstd_gpu", rstd_buf)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Build K3: GEMM + RstdScale solution
# ---------------------------------------------------------------------------

def build_k3_solution(chip: str, assembler, isaInfoMap,
                      N_hidden: int, N_out: int, wg_n: int = 1):
    """Build a bf16 GEMM2 + RstdScale kernel for gfx950.

    GEMM2 operands: A = h2 (M x N_hidden, bf16), B = W1 (N_out x N_hidden, bf16).
    TN layout (TransposeA=True, TransposeB=False), contracts over N_hidden.
    N_out must equal MacroTile1.
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
        "RstdScale":             True,
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
        raise RuntimeError("MI parameter validation failed for K3")

    solution = Solution(
        config,
        splitGSU=False,
        printSolutionRejectionReason=True,
        printIndexAssignmentInfo=False,
        assembler=assembler,
        isaInfoMap=isaInfoMap,
    )
    if not solution["Valid"]:
        raise RuntimeError("K3 solution was rejected — see reason above")
    return solution


# ---------------------------------------------------------------------------
# Run K3: GEMM2 + RstdScale
# ---------------------------------------------------------------------------

def run_k3(hsaco: bytes, kernel_name: str, solution, M: int, N_hidden: int,
           N_out: int, rstd_ref: np.ndarray):
    """Execute the K3 fused GEMM2+RstdScale kernel and verify output y.

    Checks:
      y (slot 8): bf16 output, tol = 2e-2. y = (h2 @ W1.T) * rstd[:, None]

    rstd_ref: pre-computed numpy fp32 array of shape (M,) — drive from K2 output
    or a numpy reference.
    """
    import ml_dtypes

    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]

    if N_out != MT1:
        raise ValueError(
            f"Row-containment violated: N_out={N_out} must equal MacroTile1={MT1}."
        )

    M_padded = math.ceil(M / MT0) * MT0
    numWG    = math.ceil(M / MT0) * math.ceil(N_out / MT1)

    rng = np.random.default_rng(99)
    h2_f32  = np.asfortranarray(rng.random((N_hidden, M), dtype=np.float32) * 0.1)
    w1_f32  = np.asfortranarray(rng.random((N_hidden, N_out), dtype=np.float32) * 0.1)

    h2_bf16 = np.asfortranarray(h2_f32.astype(ml_dtypes.bfloat16))
    w1_bf16 = np.asfortranarray(w1_f32.astype(ml_dtypes.bfloat16))

    c_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')
    y_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')

    # Pad rstd to M_padded.
    rstd_padded = np.zeros(M_padded, dtype=np.float32)
    rstd_padded[:M] = rstd_ref[:M]

    # Numpy reference.
    h2_ref = np.asarray(h2_bf16).astype(np.float32)   # N_hidden x M
    w1_ref = np.asarray(w1_bf16).astype(np.float32)   # N_hidden x N_out
    h3     = h2_ref.T @ w1_ref                         # M x N_out, fp32
    y_ref  = (h3 * rstd_ref[:M, np.newaxis]).astype(ml_dtypes.bfloat16)

    sk_args      = compute_sk3_dp_args(M, N_out, N_hidden, solution)
    stagger_u    = solution.get("StaggerU", 0)
    su_map       = solution.get("StaggerUMapping", 0)
    ss_shift     = solution.get("_staggerStrideShift", 0)
    su_word      = (su_map << 13) | ((ss_shift << 8) & 0x1F00) | (stagger_u & 0xFF)
    kernel_info0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))
    wgmxcc       = solution.get("WorkGroupMappingXCC", 1)
    kernel_info1 = np.uint32((wgmxcc << 16) | (solution["WorkGroupMapping"] & 0xFFFF))

    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    # Argument layout (K3):
    #   slots 0-29: same GEMM scaffolding as K1
    #   slot 30: RstdBuf (ptr, fp32, len M_padded)
    args = [
        np.uint32(1),                           # 0: GemmInfo
        kernel_info0,                           # 1: kernel_info0
        kernel_info1,                           # 2: kernel_info1
        np.uint32(numWG),                       # 3: numWG
        np.uint32(M),                           # 4: SizesFree0=M
        np.uint32(N_out),                       # 5: SizesFree1=N_out
        np.uint32(1),                           # 6: SizesFree2=batch
        np.uint32(N_hidden),                    # 7: SizesSum0=N_hidden (K)
        amdgpu_exec.InOutArray(y_bf16),         # 8: D=y (bf16)
        amdgpu_exec.InputArray(c_bf16),         # 9: C (bf16, beta=0)
        amdgpu_exec.InputArray(h2_bf16),        # 10: A=h2 (N_hidden×M col-major)
        amdgpu_exec.InputArray(w1_bf16),        # 11: B=W1 (N_hidden×N_out col-major)
        amdgpu_exec.InputArray(ws_dummy),       # 12: AddressWS
        amdgpu_exec.InputArray(flags_dummy),    # 13: AddressFlags
        np.uint32(M), np.uint32(0),             # 14,15: strideD0=M, strideD1=0
        np.uint32(M), np.uint32(0),             # 16,17: strideC0=M, strideC1=0
        np.uint32(N_hidden), np.uint32(0),      # 18,19: strideA0=N_hidden, strideA1=0
        np.uint32(N_hidden), np.uint32(0),      # 20,21: strideB0=N_hidden, strideB1=0
        np.float32(1.0),                        # 22: alpha
        np.float32(0.0),                        # 23: beta
        sk_args["iters_per_tile"],              # 24: ItersPerTile
        sk_args["magic_iters_per_tile"],        # 25: MagicNumberItersPerTile
        sk_args["shift_iters_per_tile"],        # 26: MagicShiftItersPerTile
        sk_args["sk_iters_per_wg"],             # 27: SKItersPerWG
        sk_args["sk_grid"],                     # 28: skGrid
        sk_args["sk_tiles"],                    # 29: skTiles
        amdgpu_exec.InputArray(rstd_padded),    # 30: RstdBuf (fp32)
    ]

    result_holder = {}

    def verify(arguments):
        y_gpu_bf16 = np.asarray(arguments[8].array)
        y_gpu_f32  = y_gpu_bf16.astype(np.float32)
        y_ref_f32  = np.asarray(y_ref).astype(np.float32)

        rtol, atol = 2e-2, 2e-2
        diff     = np.abs(y_gpu_f32[:M] - y_ref_f32[:M])
        tol      = atol + rtol * np.abs(y_ref_f32[:M])
        bad      = np.where(~np.isfinite(y_gpu_f32[:M]) | (diff > tol))
        ok       = len(bad[0]) == 0
        max_abs  = float(np.nanmax(np.abs(y_gpu_f32[:M] - y_ref_f32[:M]))) if M > 0 else 0.0

        if ok:
            print(f"verification: PASSED  y max_abs={max_abs:.3e}")
        else:
            print(f"verification: FAILED  y max_abs={max_abs:.3e}  "
                  f"mismatches={len(bad[0])}")
            r, c = bad[0][0], bad[1][0]
            print(f"  first bad y[{r},{c}]: gpu={y_gpu_f32[r,c]:.6f} "
                  f"ref={y_ref_f32[r,c]:.6f}")

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


def run_pipeline(chip: str, M: int, K: int, N_hidden: int, N_out: int, eps: float, wg_n: int):
    """Run K1 → K2 → K3 in sequence on shared device buffers and verify y.

    K1 writes partialBuf and D (=h2).
    K2 reads partialBuf and writes rstdBuf.
    K3 reads h2 (K1's D) and rstdBuf and writes y.

    Returns True if y matches the reference within bf16 tolerance.
    """
    import ml_dtypes

    assembler, isaInfoMap, debugConfig = setup_tensile(chip)

    k1_sol = build_k1_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    k3_sol = build_k3_solution(chip, assembler, isaInfoMap,
                               N_hidden=N_hidden, N_out=N_out, wg_n=wg_n)

    k1_asm, k1_name = generate_asm(k1_sol, assembler, debugConfig)
    k3_asm, k3_name = generate_asm(k3_sol, assembler, debugConfig)
    _k2_asm, k2_name, k2_hsaco = build_aux_reduction_asm(chip, N_hidden)

    k1_hsaco = amdgpu_exec.compile_asm_to_hsaco(k1_asm, chip)
    k3_hsaco = amdgpu_exec.compile_asm_to_hsaco(k3_asm, chip)

    MT0_k1 = k1_sol["MacroTile0"]
    MT1_k1 = k1_sol["MacroTile1"]   # == N_hidden
    MT0_k3 = k3_sol["MacroTile0"]
    MT1_k3 = k3_sol["MacroTile1"]   # == N_out

    M_padded_k1 = math.ceil(M / MT0_k1) * MT0_k1
    M_padded_k3 = math.ceil(M / MT0_k3) * MT0_k3
    M_padded_k2 = math.ceil(M / 256) * 256

    numWG_k1 = math.ceil(M / MT0_k1) * math.ceil(N_hidden / MT1_k1)
    numWG_k3 = math.ceil(M / MT0_k3) * math.ceil(N_out / MT1_k3)

    rng = np.random.default_rng(42)

    # K1 inputs.
    a_f32  = np.asfortranarray(rng.random((K, M), dtype=np.float32) * 0.1)
    w0_f32 = np.asfortranarray(rng.random((K, N_hidden), dtype=np.float32) * 0.1)
    a_bf16  = np.asfortranarray(a_f32.astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(w0_f32.astype(ml_dtypes.bfloat16))
    gamma_f32  = rng.random(N_hidden, dtype=np.float32) + 0.5
    gamma_bf16 = gamma_f32.astype(ml_dtypes.bfloat16)

    # K3 inputs.
    w1_f32  = np.asfortranarray(rng.random((N_hidden, N_out), dtype=np.float32) * 0.1)
    w1_bf16 = np.asfortranarray(w1_f32.astype(ml_dtypes.bfloat16))

    # Shared device buffers written by K1, read by K2/K3.
    c_k1_bf16   = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order='F')
    # K1 writes D as (M, N_hidden) Fortran order (strideD0=M).
    d_bf16      = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order='F')
    partial_buf = np.zeros(M_padded_k1, dtype=np.float32)

    # K3 needs A in (N_hidden, M) Fortran order (strideA0=N_hidden). Allocated
    # separately; filled with d_bf16.T after K1 completes.
    h2_for_k3   = np.zeros((N_hidden, M), dtype=ml_dtypes.bfloat16, order='F')

    # K3 output.
    c_k3_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')
    y_bf16    = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')
    rstd_buf  = np.zeros(M_padded_k2, dtype=np.float32)

    # Numpy reference.
    a_ref   = np.asarray(a_bf16).astype(np.float32)
    w0_ref  = np.asarray(w0_bf16).astype(np.float32)
    h0      = a_ref.T @ w0_ref                     # M x N_hidden, fp32
    h1      = h0
    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)
    # h2 is K1's D output: bf16-rounded h1*gamma
    h2_ref  = np.asarray((h1 * gamma_ref[np.newaxis, :]).astype(ml_dtypes.bfloat16))
    sumsq   = np.sum(h1 ** 2, axis=1)              # M, fp32 raw sum
    rstd_ref = 1.0 / np.sqrt(sumsq / N_hidden + eps)
    # w1_f32 shape (N_hidden, N_out) → K3 computes h2^T (M x N_hidden) @ W1 (N_hidden x N_out).
    # h2_ref has shape (M, N_hidden) so the product is h2_ref @ w1_ref.
    w1_ref  = np.asarray(w1_bf16).astype(np.float32)   # N_hidden x N_out
    h3      = h2_ref.astype(np.float32) @ w1_ref        # M x N_out, fp32
    y_ref   = (h3 * rstd_ref[:, np.newaxis]).astype(ml_dtypes.bfloat16)

    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    # ---- K1 launch ----
    sk1 = compute_sk3_dp_args(M, N_hidden, K, k1_sol)
    su1 = k1_sol.get("StaggerU", 0)
    su_map1 = k1_sol.get("StaggerUMapping", 0)
    ss1 = k1_sol.get("_staggerStrideShift", 0)
    su_word1 = (su_map1 << 13) | ((ss1 << 8) & 0x1F00) | (su1 & 0xFF)
    ki0_k1 = np.uint32((su_word1 << 16) | (k1_sol["GlobalSplitU"] & 0x3FFF))
    ki1_k1 = np.uint32(
        (k1_sol.get("WorkGroupMappingXCC", 1) << 16) | (k1_sol["WorkGroupMapping"] & 0xFFFF)
    )

    d_inout  = amdgpu_exec.InOutArray(d_bf16)
    pb_inout = amdgpu_exec.InOutArray(partial_buf)

    args_k1 = [
        np.uint32(1), ki0_k1, ki1_k1, np.uint32(numWG_k1),
        np.uint32(M), np.uint32(N_hidden), np.uint32(1), np.uint32(K),
        d_inout,
        amdgpu_exec.InputArray(c_k1_bf16),
        amdgpu_exec.InputArray(a_bf16),
        amdgpu_exec.InputArray(w0_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk1["iters_per_tile"], sk1["magic_iters_per_tile"], sk1["shift_iters_per_tile"],
        sk1["sk_iters_per_wg"], sk1["sk_grid"], sk1["sk_tiles"],
        amdgpu_exec.InputArray(gamma_bf16),
        pb_inout,
    ]

    amdgpu_exec.execute_hsaco(
        hsaco=k1_hsaco,
        kernel_name=k1_name,
        arguments=args_k1,
        grid_dim=(numWG_k1, 1, 1),
        block_dim=(k1_sol["NumThreads"], 1, 1),
        num_iterations=1,
    )

    # Transpose K1's D output from (M, N_hidden) col-major to (N_hidden, M) col-major
    # for K3 which expects A in (N_hidden x M) col-major layout with strideA0=N_hidden.
    np.copyto(h2_for_k3, d_bf16.T)

    # ---- K2 launch ----
    partial_buf_padded_k2 = np.zeros(M_padded_k2, dtype=np.float32)
    partial_buf_padded_k2[:M] = partial_buf[:M]
    rstd_inout = amdgpu_exec.InOutArray(rstd_buf)

    args_k2 = [
        amdgpu_exec.InputArray(partial_buf_padded_k2),
        rstd_inout,
        np.uint32(M),
        np.float32(eps),
    ]

    amdgpu_exec.execute_hsaco(
        hsaco=k2_hsaco,
        kernel_name=k2_name,
        arguments=args_k2,
        grid_dim=(math.ceil(M / 256), 1, 1),
        block_dim=(256, 1, 1),
        num_iterations=1,
    )

    # ---- K3 launch ----
    rstd_padded_k3 = np.zeros(M_padded_k3, dtype=np.float32)
    rstd_padded_k3[:M] = rstd_buf[:M]

    sk3 = compute_sk3_dp_args(M, N_out, N_hidden, k3_sol)
    su3 = k3_sol.get("StaggerU", 0)
    su_map3 = k3_sol.get("StaggerUMapping", 0)
    ss3 = k3_sol.get("_staggerStrideShift", 0)
    su_word3 = (su_map3 << 13) | ((ss3 << 8) & 0x1F00) | (su3 & 0xFF)
    ki0_k3 = np.uint32((su_word3 << 16) | (k3_sol["GlobalSplitU"] & 0x3FFF))
    ki1_k3 = np.uint32(
        (k3_sol.get("WorkGroupMappingXCC", 1) << 16) | (k3_sol["WorkGroupMapping"] & 0xFFFF)
    )

    y_inout = amdgpu_exec.InOutArray(y_bf16)

    args_k3 = [
        np.uint32(1), ki0_k3, ki1_k3, np.uint32(numWG_k3),
        np.uint32(M), np.uint32(N_out), np.uint32(1), np.uint32(N_hidden),
        y_inout,
        amdgpu_exec.InputArray(c_k3_bf16),
        amdgpu_exec.InputArray(h2_for_k3),
        amdgpu_exec.InputArray(w1_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(N_hidden), np.uint32(0),
        np.uint32(N_hidden), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk3["iters_per_tile"], sk3["magic_iters_per_tile"], sk3["shift_iters_per_tile"],
        sk3["sk_iters_per_wg"], sk3["sk_grid"], sk3["sk_tiles"],
        amdgpu_exec.InputArray(rstd_padded_k3),
    ]

    result_holder = {}

    def verify_pipeline(arguments):
        y_gpu_bf16 = np.asarray(arguments[8].array)
        y_gpu_f32  = y_gpu_bf16.astype(np.float32)
        y_ref_f32  = np.asarray(y_ref).astype(np.float32)

        rtol, atol = 2e-2, 2e-2
        diff = np.abs(y_gpu_f32[:M] - y_ref_f32[:M])
        tol  = atol + rtol * np.abs(y_ref_f32[:M])
        bad  = np.where(~np.isfinite(y_gpu_f32[:M]) | (diff > tol))
        ok   = len(bad[0]) == 0
        max_abs = float(np.nanmax(np.abs(y_gpu_f32[:M] - y_ref_f32[:M]))) if M > 0 else 0.0

        if ok:
            print(f"pipeline verification: PASSED  y max_abs={max_abs:.3e}")
        else:
            print(f"pipeline verification: FAILED  y max_abs={max_abs:.3e}  "
                  f"mismatches={len(bad[0])}")
            r, c = bad[0][0], bad[1][0]
            print(f"  first bad y[{r},{c}]: gpu={y_gpu_f32[r,c]:.6f} "
                  f"ref={y_ref_f32[r,c]:.6f}")

        result_holder["ok"] = ok

    amdgpu_exec.execute_hsaco(
        hsaco=k3_hsaco,
        kernel_name=k3_name,
        arguments=args_k3,
        grid_dim=(numWG_k3, 1, 1),
        block_dim=(k3_sol["NumThreads"], 1, 1),
        num_iterations=1,
        verify_fn=verify_pipeline,
    )

    return result_holder.get("ok", False)


def parse_args():
    p = argparse.ArgumentParser(description="TensileLite fused GEMM+PartialRMS/RstdScale example")
    p.add_argument("--phase", choices=["k1", "k2", "k3", "pipeline"], default="k1",
                   help="Phase to run: k1=GEMM+PartialRMS, k2=aux reduction, "
                        "k3=GEMM+RstdScale, pipeline=K1→K2→K3 end-to-end")
    p.add_argument("--M",        type=int,   default=2048,  help="Output rows")
    p.add_argument("--K",        type=int,   default=4096,  help="Reduction dimension (K1 only)")
    p.add_argument("--wg-n",     type=int,   default=1,     dest="wg_n",
                   help="MIWaveGroup[1]: waves splitting N (1=single, >1=cross-wave LDS)")
    p.add_argument("--N-hidden", type=int,   default=64,    dest="N_hidden",
                   help="Hidden dimension N (K2: embedded as 1/N; K3: GEMM2 contraction dim)")
    p.add_argument("--N-out",    type=int,   default=None,  dest="N_out",
                   help="K3: output columns (default: 64*wg_n = MacroTile1)")
    p.add_argument("--eps",      type=float, default=1e-5,  help="Epsilon for K2 rstd")
    p.add_argument("--chip",     default=None, help="Target GPU (default: auto-detect)")
    p.add_argument("--pipeline-N-out", type=int, default=None, dest="pipeline_N_out",
                   help="Pipeline mode: N_out for K3 (default: same as N-hidden)")
    return p.parse_args()


def main():
    args = parse_args()
    chip = args.chip or amdgpu_exec.get_chip()
    print(f"device     : {chip}")

    if not chip.startswith("gfx950"):
        print(f"WARNING: PartialRMS/RstdScale is only implemented for gfx950; "
              f"current chip={chip}")

    if args.phase == "k1":
        print("Setting up TensileLite...")
        assembler, isaInfoMap, debugConfig = setup_tensile(chip)

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

    if args.phase == "k2":
        print(f"Building K2 aux-reduction kernel (N_hidden={args.N_hidden})...")
        t0 = time.perf_counter()
        _asm_str, kernel_name, hsaco = build_aux_reduction_asm(chip, args.N_hidden)
        print(f"Compile    : {time.perf_counter()-t0:.3f} s")
        print(f"Kernel     : {kernel_name}")
        print(f"problem    : M={args.M}, N_hidden={args.N_hidden}, eps={args.eps}")

        print("Running K2 kernel...")
        ok, _rstd_gpu = run_k2(hsaco, kernel_name, args.M, args.N_hidden, args.eps)
        sys.exit(0 if ok else 1)

    if args.phase == "k3":
        print("Setting up TensileLite...")
        assembler, isaInfoMap, debugConfig = setup_tensile(chip)

        N_out = args.N_out if args.N_out is not None else 64 * args.wg_n
        print(f"Building K3 RstdScale solution (wg_n={args.wg_n}, "
              f"N_hidden={args.N_hidden}, N_out={N_out})...")
        solution = build_k3_solution(chip, assembler, isaInfoMap,
                                     N_hidden=args.N_hidden, N_out=N_out, wg_n=args.wg_n)
        MT1 = solution["MacroTile1"]
        print(f"problem    : M={args.M}, N_hidden={args.N_hidden}, N_out={N_out}")
        print(f"MacroTile  : {solution['MacroTile0']}×{MT1}")
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

        # Generate synthetic rstd_ref for standalone K3 verification.
        rng = np.random.default_rng(42)
        rstd_ref = (rng.random(args.M, dtype=np.float32) * 0.5 + 0.5)

        print("Running K3 kernel...")
        ok = run_k3(hsaco, kernel_name, solution, args.M, args.N_hidden, MT1, rstd_ref)
        sys.exit(0 if ok else 1)

    if args.phase == "pipeline":
        N_hidden = args.N_hidden
        N_out = args.pipeline_N_out if args.pipeline_N_out is not None else N_hidden
        print(f"problem    : M={args.M}, K={args.K}, N_hidden={N_hidden}, N_out={N_out}")
        print(f"eps={args.eps}, wg_n={args.wg_n}")
        print("Running K1 → K2 → K3 pipeline...")
        ok = run_pipeline(chip, args.M, args.K, N_hidden, N_out, args.eps, args.wg_n)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
