# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Performance characterization of fused GEMM+SwiGLU vs. unfused baseline on gfx950.

The fused kernel computes D = up * silu(gate) in a single pass over global
memory, writing only the (M, N_out) output (N_out = N_gemm // 2).  The unfused
baseline is a plain bf16 GEMM of identical tile configuration that writes the
full (M, N_gemm) intermediate to global memory, representing the cost an
unfused pipeline must pay before a separate SwiGLU activation kernel.

Llama 3 FFN context:
  Llama 3 8B:  hidden_size=4096, intermediate_size=14336, N_gemm=2*14336=28672
  Llama 3 70B: hidden_size=8192, intermediate_size=28672, N_gemm=2*28672=57344

The fused kernel supports multi-tile N: N_gemm = n_tiles * MacroTile1 for any
positive integer n_tiles and any wg_n, so full Llama-3 FFN N dimensions can be
run by launching the grid across multiple N-tiles.  This script sweeps both a
representative (M, K) grid and the actual Llama-3 FFN N dimensions.

Usage:
    python tensile_swiglu_benchmark.py [--chip gfx950] [--wg-n 1 2]
                                       [--iterations 50]
"""
import argparse
import math
import os
import sys

import numpy as np
import amdgpu_exec

_TENSILE_DIR = os.path.dirname(os.path.abspath(__file__))
if _TENSILE_DIR not in sys.path:
    sys.path.insert(0, _TENSILE_DIR)

from tensile_swiglu_example import (
    setup_tensile,
    build_swiglu_solution,
    generate_asm,
    compute_sk3_dp_args,
)

# ---------------------------------------------------------------------------
# Representative benchmark shapes: (M, K, N_gemm).
#
# N_gemm is the full doubled GEMM-N width; N_out = N_gemm // 2 is written to D.
# N_gemm must be a positive multiple of MacroTile1 (64 * wg_n).  All N_gemm
# values here are multiples of 128 (the largest MT1 considered, at wg_n=2),
# so they are automatically valid for any wg_n in {1, 2}.
#
# Llama-3 FFN context (x @ [W_gate|W_up]):
#   8B:  K=hidden_size=4096, N_gemm=2*intermediate=2*14336=28672
#   70B: K=hidden_size=8192, N_gemm=2*intermediate=2*28672=57344
# ---------------------------------------------------------------------------

# Single-tile sweep: N_gemm == MacroTile1 (one tile, measures per-tile delta).
_SHAPES_SINGLE_TILE = [
    # (M, K, N_gemm=None) — N_gemm filled in at runtime as MT1
    # Llama-3 8B K
    (   1, 4096, None),
    ( 512, 4096, None),
    (2048, 4096, None),
    (4096, 4096, None),
    # Llama-3 70B K
    (   1, 8192, None),
    ( 512, 8192, None),
    (2048, 8192, None),
    (4096, 8192, None),
]

# Llama-3 FFN sweep: full N_gemm at real hidden/intermediate sizes.
# 28672 = 2*14336 (8B intermediate_size); 57344 = 2*28672 (70B intermediate_size).
# Both are exact multiples of 128, so valid for wg_n=1 (MT1=64) and wg_n=2 (MT1=128).
_SHAPES_LLAMA3 = [
    # (M, K, N_gemm)  — Llama-3 8B (K=4096, N_gemm=28672)
    (   1, 4096, 28672),
    ( 512, 4096, 28672),
    (2048, 4096, 28672),
    (4096, 4096, 28672),
    # (M, K, N_gemm)  — Llama-3 70B (K=8192, N_gemm=57344)
    (   1, 8192, 57344),
    ( 512, 8192, 57344),
    (2048, 8192, 57344),
    (4096, 8192, 57344),
]

_SHAPES = _SHAPES_SINGLE_TILE + _SHAPES_LLAMA3


# ---------------------------------------------------------------------------
# Plain GEMM baseline (same tile config as fused, SwiGLU=False).
# Writes the full (M, N_gemm) output to capture the extra global-store cost
# that an unfused pipeline must pay before a separate SwiGLU kernel.
# ---------------------------------------------------------------------------

def build_plain_gemm_solution(chip: str, assembler, isaInfoMap, wg_n: int = 1):
    """Build an identical tile config to the fused SwiGLU kernel, but with SwiGLU=False.

    The output N is the full N_gemm = MacroTile1 (no halving).  This is the
    minimum work an unfused pipeline must do before applying SwiGLU separately.
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
        "DataType":      "b",
        "DestDataType":  "b",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
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
        "SwiGLU":                False,  # plain GEMM — no epilogue gating
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
        raise RuntimeError("plain GEMM MI parameter validation failed")

    solution = Solution(
        config,
        splitGSU=False,
        printSolutionRejectionReason=True,
        printIndexAssignmentInfo=False,
        assembler=assembler,
        isaInfoMap=isaInfoMap,
    )
    if not solution["Valid"]:
        raise RuntimeError("plain GEMM solution was rejected — see reason above")
    return solution


# ---------------------------------------------------------------------------
# Kernel argument packing helpers
# ---------------------------------------------------------------------------

def _encode_kernel_infos(solution):
    """Return (kernel_info0, kernel_info1) uint32 packed scalars for a solution."""
    stagger_u         = solution.get("StaggerU", 0)
    stagger_u_mapping = solution.get("StaggerUMapping", 0)
    stagger_stride_shift = solution.get("_staggerStrideShift", 0)
    su_word      = (stagger_u_mapping << 13) | ((stagger_stride_shift << 8) & 0x1F00) | (stagger_u & 0xFF)
    kernel_info0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))
    wgmxcc       = solution.get("WorkGroupMappingXCC", 1)
    kernel_info1 = np.uint32((wgmxcc << 16) | (solution["WorkGroupMapping"] & 0xFFFF))
    return kernel_info0, kernel_info1


def _run_fused(hsaco, kernel_name, solution, M, K, N_gemm, num_iterations):
    """Run the fused SwiGLU kernel and return raw timing list (ns).

    N_gemm must be a positive multiple of MacroTile1; N_out = N_gemm // 2 is
    the output width written to D.  The grid covers ceil(M/MT0)*ceil(N_gemm/MT1)
    work-groups.
    """
    import ml_dtypes

    MT0   = solution["MacroTile0"]
    N_out = N_gemm // 2
    numWG = math.ceil(M / MT0) * math.ceil(N_gemm / solution["MacroTile1"])

    rng    = np.random.default_rng(42)
    a_bf16 = np.asfortranarray(
        (rng.random((K, M), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16)
    )
    b_bf16 = np.asfortranarray(
        (rng.random((K, N_gemm), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16)
    )
    c_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')
    d_bf16 = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')

    sk_args     = compute_sk3_dp_args(M, N_gemm, K, solution)
    ki0, ki1    = _encode_kernel_infos(solution)
    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    args = [
        np.uint32(1), ki0, ki1, np.uint32(numWG),
        np.uint32(M), np.uint32(N_gemm), np.uint32(1), np.uint32(K),
        amdgpu_exec.InOutArray(d_bf16),
        amdgpu_exec.InputArray(c_bf16),
        amdgpu_exec.InputArray(a_bf16),
        amdgpu_exec.InputArray(b_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.float32(1.0),
        sk_args["iters_per_tile"],
        sk_args["magic_iters_per_tile"],
        sk_args["shift_iters_per_tile"],
        sk_args["sk_iters_per_wg"],
        sk_args["sk_grid"],
        sk_args["sk_tiles"],
    ]

    return amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(numWG, 1, 1),
        block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=num_iterations,
    )


def _run_unfused(hsaco, kernel_name, solution, M, K, N_gemm, num_iterations):
    """Run the plain GEMM baseline and return raw timing list (ns).

    The baseline writes the full (M, N_gemm) output to global memory, matching
    the work an unfused pipeline must do before applying SwiGLU separately.
    N_gemm must be a positive multiple of MacroTile1.
    """
    import ml_dtypes

    MT0   = solution["MacroTile0"]
    numWG = math.ceil(M / MT0) * math.ceil(N_gemm / solution["MacroTile1"])

    rng    = np.random.default_rng(42)
    a_bf16 = np.asfortranarray(
        (rng.random((K, M), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16)
    )
    b_bf16 = np.asfortranarray(
        (rng.random((K, N_gemm), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16)
    )
    c_bf16 = np.zeros((M, N_gemm), dtype=ml_dtypes.bfloat16, order='F')
    d_bf16 = np.zeros((M, N_gemm), dtype=ml_dtypes.bfloat16, order='F')

    sk_args     = compute_sk3_dp_args(M, N_gemm, K, solution)
    ki0, ki1    = _encode_kernel_infos(solution)
    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    args = [
        np.uint32(1), ki0, ki1, np.uint32(numWG),
        np.uint32(M), np.uint32(N_gemm), np.uint32(1), np.uint32(K),
        amdgpu_exec.InOutArray(d_bf16),
        amdgpu_exec.InputArray(c_bf16),
        amdgpu_exec.InputArray(a_bf16),
        amdgpu_exec.InputArray(b_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.float32(1.0),
        sk_args["iters_per_tile"],
        sk_args["magic_iters_per_tile"],
        sk_args["shift_iters_per_tile"],
        sk_args["sk_iters_per_wg"],
        sk_args["sk_grid"],
        sk_args["sk_tiles"],
    ]

    return amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name=kernel_name,
        arguments=args,
        grid_dim=(numWG, 1, 1),
        block_dim=(solution["NumThreads"], 1, 1),
        num_iterations=num_iterations,
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _best_avg_ms(times_ns, drop_first):
    """Return (best_ms, avg_ms) from a nanosecond timing list.

    The first iteration is discarded when drop_first=True because the first
    kernel launch carries elevated one-time dispatch / HSACO load latency that
    is not representative of steady-state throughput.
    """
    samples = times_ns[1:] if (drop_first and len(times_ns) > 1) else times_ns
    if not samples:
        return float("nan"), float("nan")
    best_ms = min(samples) / 1e6
    avg_ms  = sum(samples) / len(samples) / 1e6
    return best_ms, avg_ms


# ---------------------------------------------------------------------------
# Per-wg_n benchmark
# ---------------------------------------------------------------------------

def run_benchmark(chip, wg_n, num_iterations, shapes):
    """Build kernels and benchmark all shapes for a given wg_n.

    Each entry in shapes is (M, K, N_gemm) where N_gemm=None means a single
    MacroTile1-wide tile (N_gemm = MT1).  N_gemm must be a multiple of MT1.

    Returns a list of row dicts with timing results.
    """
    import time

    assembler, isaInfoMap, debugConfig = setup_tensile(chip)

    print(f"\n=== wg_n={wg_n} ===")
    print("Building fused SwiGLU solution...")
    t0 = time.perf_counter()
    fused_sol = build_swiglu_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    print(f"  MacroTile={fused_sol['MacroTile0']}x{fused_sol['MacroTile1']}  "
          f"({time.perf_counter()-t0:.2f}s)")

    print("Generating + compiling fused kernel...")
    t0 = time.perf_counter()
    fused_asm, fused_name = generate_asm(fused_sol, assembler, debugConfig)
    fused_hsaco = amdgpu_exec.compile_asm_to_hsaco(fused_asm, chip)
    print(f"  {fused_name}  ({time.perf_counter()-t0:.2f}s)")

    print("Building plain GEMM baseline solution...")
    t0 = time.perf_counter()
    plain_sol = build_plain_gemm_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    print(f"  MacroTile={plain_sol['MacroTile0']}x{plain_sol['MacroTile1']}  "
          f"({time.perf_counter()-t0:.2f}s)")

    print("Generating + compiling plain GEMM kernel...")
    t0 = time.perf_counter()
    plain_asm, plain_name = generate_asm(plain_sol, assembler, debugConfig)
    plain_hsaco = amdgpu_exec.compile_asm_to_hsaco(plain_asm, chip)
    print(f"  {plain_name}  ({time.perf_counter()-t0:.2f}s)")

    MT1 = fused_sol["MacroTile1"]

    # Warmup via first-iteration discard; execute_hsaco has no warmup param.
    drop_first = num_iterations > 1

    rows = []
    for M, K, n_gemm_spec in shapes:
        # Resolve N_gemm: None means single tile (N_gemm = MT1).
        N_gemm = n_gemm_spec if n_gemm_spec is not None else MT1
        if N_gemm % MT1 != 0:
            raise ValueError(
                f"N_gemm={N_gemm} is not a multiple of MacroTile1={MT1} "
                f"(wg_n={wg_n}); adjust the shape or wg_n"
            )
        N_out  = N_gemm // 2

        fused_ns   = _run_fused(fused_hsaco, fused_name, fused_sol,
                                M, K, N_gemm, num_iterations)
        unfused_ns = _run_unfused(plain_hsaco, plain_name, plain_sol,
                                  M, K, N_gemm, num_iterations)

        f_best, f_avg = _best_avg_ms(fused_ns, drop_first)
        u_best, u_avg = _best_avg_ms(unfused_ns, drop_first)
        speedup       = u_best / f_best if f_best > 0 else float("nan")
        rows.append({
            "M":               M,
            "N_gemm":          N_gemm,
            "N_out":           N_out,
            "K":               K,
            "fused_best_ms":   f_best,
            "fused_avg_ms":    f_avg,
            "unfused_best_ms": u_best,
            "unfused_avg_ms":  u_avg,
            "speedup":         speedup,
        })
    return rows


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(rows, wg_n):
    """Print a formatted comparison table for one wg_n config."""
    hdr = (
        f"{'M':>6}  {'N_gemm':>6}  {'N_out':>5}  {'K':>5}  "
        f"{'fused_best':>10}  {'fused_avg':>9}  "
        f"{'unfused_best':>12}  {'unfused_avg':>11}  "
        f"{'speedup':>7}"
    )
    sep = "-" * len(hdr)
    print(f"\n--- Fused vs Unfused (wg_n={wg_n}, times in ms) ---")
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['M']:>6}  {r['N_gemm']:>6}  {r['N_out']:>5}  {r['K']:>5}  "
            f"{r['fused_best_ms']:>10.4f}  {r['fused_avg_ms']:>9.4f}  "
            f"{r['unfused_best_ms']:>12.4f}  {r['unfused_avg_ms']:>11.4f}  "
            + (f"{r['speedup']:>7.3f}x" if math.isfinite(r['speedup']) else f"{'N/A':>7} ")
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fused GEMM+SwiGLU vs unfused baseline benchmark on gfx950"
    )
    p.add_argument("--chip", default=None,
                   help="target GPU (default: auto-detect via amdgpu_exec.get_chip())")
    p.add_argument("--wg-n", type=int, nargs="+", default=[1, 2],
                   metavar="WG_N",
                   help="MIWaveGroup[1] values to benchmark (default: 1 2)")
    p.add_argument("--iterations", "-i", type=int, default=50,
                   help="kernel launches per shape (default: 50)")
    return p.parse_args()


def main():
    args = parse_args()

    chip = args.chip or amdgpu_exec.get_chip()
    print(f"device     : {chip}")

    if not chip.startswith("gfx950"):
        print(f"ERROR: this benchmark targets gfx950 only; detected chip={chip}")
        sys.exit(1)

    print(
        "\nFused kernel supports multi-tile N: N_gemm = n_tiles * MacroTile1.\n"
        "Shapes include single-tile (N_gemm=MT1) and full Llama-3 FFN dims:\n"
        "  Llama-3 8B:  K=4096, N_gemm=28672 (2*14336), N_out=14336\n"
        "  Llama-3 70B: K=8192, N_gemm=57344 (2*28672), N_out=28672\n"
    )

    for wg_n in args.wg_n:
        rows = run_benchmark(chip, wg_n, args.iterations, _SHAPES)
        print_table(rows, wg_n)


if __name__ == "__main__":
    main()
