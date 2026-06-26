# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite fused GEMM+RMSNorm pipeline benchmark.

Measures per-kernel throughput and combined pipeline throughput for:
  K1: GEMM + PartialRMS  (TFLOPS)
  K2: auxiliary rsqrt reduction (GB/s — memory-bound)
  K3: GEMM + RstdScale   (TFLOPS)
  Pipeline: K1 + K3 total flops / (K1 + K2 + K3 latency)

Usage:
    python tensile_gemm_rmsnorm_gemm_benchmark.py
    python tensile_gemm_rmsnorm_gemm_benchmark.py --M 2048 --N-hidden 64 --N-out 64 --K 4096
    python tensile_gemm_rmsnorm_gemm_benchmark.py --warmup 5 --iters 20
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

from tensile_gemm_rmsnorm_gemm_example import (
    setup_tensile,
    build_k1_solution,
    build_k3_solution,
    build_aux_reduction_asm,
    generate_asm,
    compute_sk3_dp_args,
)


def _build_k1_args(solution, M, N_hidden, K, a_bf16, w0_bf16, gamma_bf16,
                   c_bf16, d_bf16, partial_buf, ws_dummy, flags_dummy):
    """Assemble the kernel argument list for K1."""
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    M_padded = math.ceil(M / MT0) * MT0
    numWG = math.ceil(M / MT0) * math.ceil(N_hidden / MT1)

    sk = compute_sk3_dp_args(M, N_hidden, K, solution)
    su = solution.get("StaggerU", 0)
    su_map = solution.get("StaggerUMapping", 0)
    ss = solution.get("_staggerStrideShift", 0)
    su_word = (su_map << 13) | ((ss << 8) & 0x1F00) | (su & 0xFF)
    ki0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))
    ki1 = np.uint32(
        (solution.get("WorkGroupMappingXCC", 1) << 16) | (solution["WorkGroupMapping"] & 0xFFFF)
    )

    args = [
        np.uint32(1), ki0, ki1, np.uint32(numWG),
        np.uint32(M), np.uint32(N_hidden), np.uint32(1), np.uint32(K),
        amdgpu_exec.InOutArray(d_bf16),
        amdgpu_exec.InputArray(c_bf16),
        amdgpu_exec.InputArray(a_bf16),
        amdgpu_exec.InputArray(w0_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.uint32(K), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk["iters_per_tile"], sk["magic_iters_per_tile"], sk["shift_iters_per_tile"],
        sk["sk_iters_per_wg"], sk["sk_grid"], sk["sk_tiles"],
        amdgpu_exec.InputArray(gamma_bf16),
        amdgpu_exec.InOutArray(partial_buf),
    ]
    return args, numWG


def _build_k2_args(partial_buf_padded, rstd_buf, M, eps):
    """Assemble the kernel argument list for K2."""
    args = [
        amdgpu_exec.InputArray(partial_buf_padded),
        amdgpu_exec.InOutArray(rstd_buf),
        np.uint32(M),
        np.float32(eps),
    ]
    grid_dim = (math.ceil(M / 256), 1, 1)
    return args, grid_dim


def _build_k3_args(solution, M, N_hidden, N_out, d_bf16, w1_bf16,
                   c_bf16, y_bf16, rstd_padded, ws_dummy, flags_dummy):
    """Assemble the kernel argument list for K3."""
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    numWG = math.ceil(M / MT0) * math.ceil(N_out / MT1)

    sk = compute_sk3_dp_args(M, N_out, N_hidden, solution)
    su = solution.get("StaggerU", 0)
    su_map = solution.get("StaggerUMapping", 0)
    ss = solution.get("_staggerStrideShift", 0)
    su_word = (su_map << 13) | ((ss << 8) & 0x1F00) | (su & 0xFF)
    ki0 = np.uint32((su_word << 16) | (solution["GlobalSplitU"] & 0x3FFF))
    ki1 = np.uint32(
        (solution.get("WorkGroupMappingXCC", 1) << 16) | (solution["WorkGroupMapping"] & 0xFFFF)
    )

    args = [
        np.uint32(1), ki0, ki1, np.uint32(numWG),
        np.uint32(M), np.uint32(N_out), np.uint32(1), np.uint32(N_hidden),
        amdgpu_exec.InOutArray(y_bf16),
        amdgpu_exec.InputArray(c_bf16),
        amdgpu_exec.InputArray(d_bf16),
        amdgpu_exec.InputArray(w1_bf16),
        amdgpu_exec.InputArray(ws_dummy),
        amdgpu_exec.InputArray(flags_dummy),
        np.uint32(M), np.uint32(0),
        np.uint32(M), np.uint32(0),
        np.uint32(N_hidden), np.uint32(0),
        np.uint32(N_hidden), np.uint32(0),
        np.float32(1.0), np.float32(0.0),
        sk["iters_per_tile"], sk["magic_iters_per_tile"], sk["shift_iters_per_tile"],
        sk["sk_iters_per_wg"], sk["sk_grid"], sk["sk_tiles"],
        amdgpu_exec.InputArray(rstd_padded),
    ]
    return args, numWG


def benchmark(chip, M, N_hidden, N_out, K, wg_n, eps, warmup, iters):
    """Build all three kernels, warm up, and measure per-kernel and pipeline timing."""
    import ml_dtypes

    print(f"device     : {chip}")
    print(f"M={M}  N_hidden={N_hidden}  N_out={N_out}  K={K}  wg_n={wg_n}  eps={eps}")
    print(f"warmup={warmup}  iters={iters}\n")

    print("Setting up TensileLite...")
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)

    print("Building solutions...")
    k1_sol = build_k1_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    k3_sol = build_k3_solution(chip, assembler, isaInfoMap,
                               N_hidden=N_hidden, N_out=N_out, wg_n=wg_n)

    print("Generating assembly...")
    k1_asm, k1_name = generate_asm(k1_sol, assembler, debugConfig)
    k3_asm, k3_name = generate_asm(k3_sol, assembler, debugConfig)
    _k2_asm, k2_name, k2_hsaco = build_aux_reduction_asm(chip, N_hidden)

    print("Compiling to HSACO...")
    k1_hsaco = amdgpu_exec.compile_asm_to_hsaco(k1_asm, chip)
    k3_hsaco = amdgpu_exec.compile_asm_to_hsaco(k3_asm, chip)

    # Buffer dimensions.
    MT0_k1 = k1_sol["MacroTile0"]
    MT0_k3 = k3_sol["MacroTile0"]
    M_padded_k1 = math.ceil(M / MT0_k1) * MT0_k1
    M_padded_k3 = math.ceil(M / MT0_k3) * MT0_k3
    M_padded_k2 = math.ceil(M / 256) * 256

    rng = np.random.default_rng(0)

    a_bf16  = np.asfortranarray(rng.random((K, M),          dtype=np.float32).astype(ml_dtypes.bfloat16))
    w0_bf16 = np.asfortranarray(rng.random((K, N_hidden),   dtype=np.float32).astype(ml_dtypes.bfloat16))
    w1_bf16 = np.asfortranarray(rng.random((N_hidden, N_out), dtype=np.float32).astype(ml_dtypes.bfloat16))
    gamma_bf16 = (rng.random(N_hidden, dtype=np.float32) + 0.5).astype(ml_dtypes.bfloat16)

    c_k1_bf16  = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order='F')
    # K1 writes D as (M, N_hidden) col-major; K3 needs A as (N_hidden, M) col-major.
    d_bf16      = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order='F')
    h2_for_k3   = np.zeros((N_hidden, M), dtype=ml_dtypes.bfloat16, order='F')
    partial_buf = np.zeros(M_padded_k1, dtype=np.float32)

    c_k3_bf16  = np.zeros((M, N_out), dtype=ml_dtypes.bfloat16, order='F')
    y_bf16     = np.zeros((M, N_out),  dtype=ml_dtypes.bfloat16, order='F')
    rstd_buf   = np.zeros(M_padded_k2, dtype=np.float32)

    ws_dummy    = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    partial_buf_padded_k2 = np.zeros(M_padded_k2, dtype=np.float32)
    rstd_padded_k3 = np.zeros(M_padded_k3, dtype=np.float32)

    args_k1, numWG_k1 = _build_k1_args(
        k1_sol, M, N_hidden, K, a_bf16, w0_bf16, gamma_bf16,
        c_k1_bf16, d_bf16, partial_buf, ws_dummy, flags_dummy,
    )
    args_k2, grid_k2 = _build_k2_args(partial_buf_padded_k2, rstd_buf, M, eps)
    args_k3, numWG_k3 = _build_k3_args(
        k3_sol, M, N_hidden, N_out, h2_for_k3, w1_bf16,
        c_k3_bf16, y_bf16, rstd_padded_k3, ws_dummy, flags_dummy,
    )

    def run_k1():
        return amdgpu_exec.execute_hsaco(
            hsaco=k1_hsaco, kernel_name=k1_name, arguments=args_k1,
            grid_dim=(numWG_k1, 1, 1), block_dim=(k1_sol["NumThreads"], 1, 1),
            num_iterations=1,
        )

    def run_k2():
        return amdgpu_exec.execute_hsaco(
            hsaco=k2_hsaco, kernel_name=k2_name, arguments=args_k2,
            grid_dim=grid_k2, block_dim=(256, 1, 1),
            num_iterations=1,
        )

    def run_k3():
        return amdgpu_exec.execute_hsaco(
            hsaco=k3_hsaco, kernel_name=k3_name, arguments=args_k3,
            grid_dim=(numWG_k3, 1, 1), block_dim=(k3_sol["NumThreads"], 1, 1),
            num_iterations=1,
        )

    # Warmup: run all kernels in sequence without recording times.
    print("Warming up...")
    for _ in range(warmup):
        run_k1()
        run_k2()
        run_k3()

    # Timed runs.
    print("Benchmarking...")
    times_k1 = []
    times_k2 = []
    times_k3 = []
    times_pipeline = []

    for _ in range(iters):
        import time
        t0 = time.perf_counter_ns()
        r1 = run_k1()
        r2 = run_k2()
        r3 = run_k3()
        t1 = time.perf_counter_ns()

        # execute_hsaco returns a list of ns timings for each kernel iteration.
        if r1:
            times_k1.append(min(r1))
        if r2:
            times_k2.append(min(r2))
        if r3:
            times_k3.append(min(r3))
        times_pipeline.append(t1 - t0)

    k1_flops = 2 * M * N_hidden * K
    k3_flops = 2 * M * N_out * N_hidden
    # K2 reads M fp32 + writes M fp32.
    k2_bytes = 2 * M * 4

    def report_tflops(name, flops, times_ns):
        if not times_ns:
            print(f"  {name}: no timing data")
            return
        best_ns = min(times_ns)
        avg_ns  = sum(times_ns) / len(times_ns)
        tflops_best = flops / best_ns * 1e-3
        tflops_avg  = flops / avg_ns  * 1e-3
        print(f"  {name}: best={tflops_best:.3f} TFLOPS  avg={tflops_avg:.3f} TFLOPS"
              f"  (best={best_ns/1e6:.3f} ms  avg={avg_ns/1e6:.3f} ms)")

    def report_gbs(name, bytes_val, times_ns):
        if not times_ns:
            print(f"  {name}: no timing data")
            return
        best_ns = min(times_ns)
        avg_ns  = sum(times_ns) / len(times_ns)
        gbs_best = bytes_val / best_ns
        gbs_avg  = bytes_val / avg_ns
        print(f"  {name}: best={gbs_best:.3f} GB/s  avg={gbs_avg:.3f} GB/s"
              f"  (best={best_ns/1e6:.3f} ms  avg={avg_ns/1e6:.3f} ms)")

    pipeline_flops = k1_flops + k3_flops

    print("\n--- Per-kernel throughput ---")
    report_tflops("K1 (GEMM+PartialRMS)", k1_flops,  times_k1)
    report_gbs   ("K2 (aux rsqrt)       ", k2_bytes,  times_k2)
    report_tflops("K3 (GEMM+RstdScale) ", k3_flops,  times_k3)

    if times_pipeline:
        best_pipeline_ns = min(times_pipeline)
        avg_pipeline_ns  = sum(times_pipeline) / len(times_pipeline)
        tflops_p_best = pipeline_flops / best_pipeline_ns * 1e-3
        tflops_p_avg  = pipeline_flops / avg_pipeline_ns  * 1e-3
        print(f"\n--- Pipeline (K1+K2+K3) ---")
        print(f"  best={tflops_p_best:.3f} TFLOPS  avg={tflops_p_avg:.3f} TFLOPS"
              f"  (best={best_pipeline_ns/1e6:.3f} ms  avg={avg_pipeline_ns/1e6:.3f} ms)")


def parse_args():
    p = argparse.ArgumentParser(description="TensileLite fused GEMM+RMSNorm pipeline benchmark")
    p.add_argument("--M",        type=int,   default=2048, help="Output rows")
    p.add_argument("--N-hidden", type=int,   default=64,   dest="N_hidden",
                   help="Hidden dimension (K1 N and K3 contraction K)")
    p.add_argument("--N-out",    type=int,   default=None, dest="N_out",
                   help="K3 output columns (default: same as N-hidden)")
    p.add_argument("--K",        type=int,   default=4096, help="K1 reduction dimension")
    p.add_argument("--wg-n",     type=int,   default=1,    dest="wg_n",
                   help="MIWaveGroup[1]: waves splitting N")
    p.add_argument("--eps",      type=float, default=1e-5, help="Epsilon for K2 rstd")
    p.add_argument("--warmup",   type=int,   default=3,    help="Warmup iterations")
    p.add_argument("--iters",    type=int,   default=10,   help="Timed iterations")
    p.add_argument("--chip",     default=None, help="Target GPU (default: auto-detect)")
    return p.parse_args()


def main():
    args = parse_args()
    chip = args.chip or amdgpu_exec.get_chip()
    N_out = args.N_out if args.N_out is not None else args.N_hidden

    if not chip.startswith("gfx950"):
        print(f"WARNING: PartialRMS/RstdScale is only implemented for gfx950; "
              f"current chip={chip}")

    benchmark(chip, args.M, args.N_hidden, N_out, args.K,
              args.wg_n, args.eps, args.warmup, args.iters)


if __name__ == "__main__":
    main()
