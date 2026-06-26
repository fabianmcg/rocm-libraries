# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Combined GEMM+PartialRMS → RMSNorm pipeline benchmark.

Pipeline
--------
K1 (GEMM + PartialRMS):
  D[m, n]          = h1[m, n] * gamma[n]         (bf16, col-major M×N_hidden)
  partialBuf[m, t] = Σ_{n in tile t} h1[m, n]²  (f32,  row-major M_pad×N_tiles)
  where h1 = A^T @ W0.

  K1 writes D col-major (UseSubtileImpl hard-codes StrideD0I=1).

col_div (RMSNorm kernel, col-major C):
  D[m, n] /= sqrt(inv_d * sum_t(partialBuf[m, t]) + eps)  (bf16, in-place)
  where inv_d = 1 / N_hidden.

  col_div reads and writes C in col-major order, matching K1's output layout
  directly — no transpose required between stages.

Both kernels share the same D (col-major) and partialBuf GPU buffers.

Benchmark metrics
-----------------
  K1      : TFLOPS = 2*M*N_hidden*K / elapsed_s
  col_div : BW     = 2*M*N_hidden*2 / elapsed_s  (C read+write, bf16)
  Pipeline: TFLOPS = 2*M*N_hidden*K / (K1+col_div elapsed_s)

Usage
-----
  python bench_gemm_rms.py
  python bench_gemm_rms.py --M 4096 --N-hidden 4096 --K 4096
  python bench_gemm_rms.py --warmup 5 --iters 20 --no-verify
"""

import argparse
import math
import os
import sys

import numpy as np
import amdgpu_exec
from amdgpu_exec.runtime import GpuBuffer, GpuModule, MemoryManager, _create_kernel_args
from amdgpu_exec._runtime_module import (
    hip_init,
    hip_module_launch_kernel,
    hip_stream_synchronize,
    hip_stream_create,
    hip_stream_destroy,
    hip_event_create,
    hip_event_destroy,
    hip_event_record,
    hip_event_synchronize,
    hip_event_elapsed_time,
    Ptr,
)

_TENSILE_DIR = os.path.dirname(os.path.abspath(__file__))
if _TENSILE_DIR not in sys.path:
    sys.path.insert(0, _TENSILE_DIR)

from gemm_partialrms_colv2_helpers import (
    setup_tensile,
    build_k1_solution,
    generate_asm,
    compute_sk3_dp_args,
    _pack_kernel_info,
)
from Colv2Generator import build_colv2

COLV2_BLOCK = (256, 1, 1)
COLV2_ROWS_PER_BLOCK = 256  # fixed by the kernel: each block handles 256 rows


def colv2_launch_args(buf_c: GpuBuffer, buf_d: GpuBuffer,
                      M: int, N: int, n_d: int,
                      inv_d: float, eps: float):
    """Build kernel args for colv2.

    Argument layout (from .amdgpu_metadata offsets):
      0  (8B): ptr_c  — bf16, col-major M×N
      8  (8B): ptr_d  — f32,  row-major M×n_d
     16  (4B): M
     20  (4B): N
     24  (4B): n_d
     28  (4B): inv_d  — f32 scalar (1/N_hidden)
     32  (4B): eps    — f32 scalar
     36+     : hidden group/grid sizes

    Launch: block=(256,1,1), grid=(ceil(M/256), ceil(N/256), 1).
    wg_id_x (s2) selects a 256-row tile; wg_id_y (s3) selects a 256-col tile.
    """
    grid_x = -(-M // 256)   # ceil(M/256) — row tiles → wg_id_x
    grid_y = -(-N // 256)   # ceil(N/256) — col tiles → wg_id_y
    kp, ka, kpa = _create_kernel_args(
        buf_c, buf_d,
        np.int32(M), np.int32(N), np.int32(n_d),
        np.float32(inv_d), np.float32(eps),
        np.int32(COLV2_BLOCK[0]),
        np.int32(COLV2_BLOCK[1]),
        np.int32(COLV2_BLOCK[2]),
        np.int32(grid_x), np.int32(grid_y), np.int32(1),
    )
    return kp, ka, kpa, grid_x, grid_y


# ---------------------------------------------------------------------------
# K1 argument builder (standard col-major D, strideD0=M)
# ---------------------------------------------------------------------------


def k1_launch_args(solution, M, N_hidden, K, N_tiles_N,
                   buf_d, buf_c, buf_a, buf_w0, buf_gamma,
                   buf_partial, buf_ws, buf_flags):
    """Build kernel args for K1 with standard col-major D output (strideD0=M)."""
    MT0 = solution["MacroTile0"]
    numWG = math.ceil(M / MT0) * N_tiles_N
    sk = compute_sk3_dp_args(M, N_hidden, K, solution)
    ki0, ki1 = _pack_kernel_info(solution)
    kp, ka, kpa = _create_kernel_args(
        np.uint32(1),         ki0,              ki1,
        np.uint32(numWG),
        np.uint32(M),         np.uint32(N_hidden),
        np.uint32(1),         np.uint32(K),
        buf_d,                                  # D (bf16, col-major)
        buf_c,                                  # C (bf16, beta=0, unused)
        buf_a,                                  # A (K×M col-major)
        buf_w0,                                 # B (K×N_hidden col-major)
        buf_ws,                                 # AddressWS
        buf_flags,                              # AddressFlags
        np.uint32(M),        np.uint32(0),      # strideD0=M (col-major), strideD1=0
        np.uint32(M),        np.uint32(0),      # strideC0, strideC1
        np.uint32(K),        np.uint32(0),      # strideA0, strideA1
        np.uint32(K),        np.uint32(0),      # strideB0, strideB1
        np.float32(1.0),                        # alpha
        np.float32(0.0),                        # beta
        sk["iters_per_tile"],
        sk["magic_iters_per_tile"],
        sk["shift_iters_per_tile"],
        sk["sk_iters_per_wg"],
        sk["sk_grid"],
        sk["sk_tiles"],
        buf_gamma,                              # RMSNormGamma (bf16)
        buf_partial,                            # PartialBuf (fp32, row-major)
    )
    return kp, ka, kpa, numWG


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_pipeline(d_host, partial_host, M, N_hidden, N_tiles_N, MT1,
                    a_bf16, w0_bf16, gamma_bf16, inv_d: float, eps: float):
    """Verify the full K1 → col_div pipeline end-to-end.

    d_host      : (M, N_hidden) uint16, col-major (F-order) bf16.
    partial_host: (M_pad, N_tiles_N) f32, row-major.
    """
    import ml_dtypes

    a_ref  = np.asarray(a_bf16).astype(np.float32)
    w0_ref = np.asarray(w0_bf16).astype(np.float32)
    h1     = a_ref.T @ w0_ref                          # M×N_hidden, fp32

    gamma_ref = np.asarray(gamma_bf16).astype(np.float32)

    # Reference partialBuf: per-tile Σh1².
    sumsq_ref = np.zeros((M, N_tiles_N), dtype=np.float32)
    for t in range(N_tiles_N):
        lo, hi = t * MT1, min((t + 1) * MT1, N_hidden)
        sumsq_ref[:, t] = (h1[:, lo:hi] ** 2).sum(axis=1)

    pb     = partial_host[:M, :]
    diff_p = np.abs(pb - sumsq_ref)
    bad_p  = np.where(~np.isfinite(pb) | (diff_p > 1e-3 + 1e-3 * np.abs(sumsq_ref)))
    p_ok   = len(bad_p[0]) == 0
    max_p  = float(np.nanmax(diff_p)) if pb.size > 0 else 0.0

    # End-to-end reference: bf16(h1*gamma) / sqrt(inv_d * Σh1² + eps).
    h1_gamma_f32 = (h1 * gamma_ref[np.newaxis, :]).astype(ml_dtypes.bfloat16).astype(np.float32)
    row_sums     = sumsq_ref.sum(axis=1)
    rms_denom    = np.sqrt(inv_d * row_sums + eps)
    d_ref_f32    = h1_gamma_f32 / rms_denom[:, np.newaxis]

    # Decode GPU output (col-major uint16 → float32, preserving F-order).
    d_gpu_f32 = (d_host.astype(np.uint32) << 16).view(np.float32)

    diff_d = np.abs(d_gpu_f32 - d_ref_f32)
    bad_d  = np.where(~np.isfinite(d_gpu_f32) | (diff_d > 2e-2 + 2e-2 * np.abs(d_ref_f32)))
    d_ok   = len(bad_d[0]) == 0
    max_d  = float(np.nanmax(diff_d)) if d_gpu_f32.size > 0 else 0.0

    ok = p_ok and d_ok
    if ok:
        print(f"  verification PASSED  "
              f"partialBuf max_abs={max_p:.2e}  D max_abs={max_d:.2e}")
    else:
        print("  verification FAILED")
        if not p_ok:
            r, c = bad_p[0][0], bad_p[1][0]
            print(f"    partialBuf[{r},{c}]: gpu={pb[r,c]:.4g}  ref={sumsq_ref[r,c]:.4g}  "
                  f"({len(bad_p[0])} bad, max_abs={max_p:.2e})")
        if not d_ok:
            r, c = bad_d[0][0], bad_d[1][0]
            print(f"    D[{r},{c}]: gpu={d_gpu_f32[r,c]:.4g}  ref={d_ref_f32[r,c]:.4g}  "
                  f"({len(bad_d[0])} bad, max_abs={max_d:.2e})")
    return ok


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def benchmark(chip, M, N_hidden, K, eps, warmup, iters, do_verify, wg_n):
    import ml_dtypes

    inv_d = 1.0 / N_hidden

    print(f"chip            : {chip}")
    print(f"M={M}  N_hidden={N_hidden}  K={K}")
    print(f"inv_d={inv_d:.6g}  eps={eps}")
    print(f"warmup={warmup}  iters={iters}\n")

    print("Setting up TensileLite...")
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)

    print(f"Building K1 solution (wg_n={wg_n})...")
    k1_sol = build_k1_solution(chip, assembler, isaInfoMap, wg_n=wg_n)
    MT0 = k1_sol["MacroTile0"]
    MT1 = k1_sol["MacroTile1"]
    N_tiles_N = math.ceil(N_hidden / MT1)
    M_padded  = math.ceil(M / MT0) * MT0
    numWG_k1  = math.ceil(M / MT0) * N_tiles_N
    cd_grid_x = -(-N_hidden // 256)
    cd_grid_y = -(-M // 256)

    print(f"MacroTile       : {MT0}×{MT1}")
    print(f"N_tiles_N (n_d) : {N_tiles_N}")
    print(f"K1 grid         : ({numWG_k1}, 1, 1)  block: ({k1_sol['NumThreads']}, 1, 1)")
    print(f"colv2 grid      : ({cd_grid_x}, {cd_grid_y}, 1)  block: {COLV2_BLOCK}")
    print()

    print("Generating K1 assembly...")
    k1_asm, k1_name = generate_asm(k1_sol, assembler, debugConfig)
    print(f"K1 kernel       : {k1_name}")

    print("Compiling HSACO...")
    k1_hsaco  = amdgpu_exec.compile_asm_to_hsaco(k1_asm, chip)
    cd_asm, _ = build_colv2(chip, M=M)
    cd_hsaco  = amdgpu_exec.compile_asm_to_hsaco(cd_asm, chip)
    print()

    # Host arrays.
    rng    = np.random.default_rng(42)
    a_np   = np.asfortranarray(
        (rng.random((K, M),        dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16))
    w0_np  = np.asfortranarray(
        (rng.random((K, N_hidden), dtype=np.float32) * 0.1).astype(ml_dtypes.bfloat16))
    gam_np = (rng.random(N_hidden, dtype=np.float32) + 0.5).astype(ml_dtypes.bfloat16)

    # D col-major (M, N_hidden) bf16 — K1 writes this, col_div reads/writes it.
    d_host     = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order="F")
    c_k1_host  = np.zeros((M, N_hidden), dtype=ml_dtypes.bfloat16, order="F")
    pb_host    = np.zeros((M_padded, N_tiles_N), dtype=np.float32,  order="C")
    ws_dummy   = np.zeros(4, dtype=np.float32)
    flags_dummy = np.zeros(4, dtype=np.float32)

    hip_init()
    mm      = MemoryManager()
    buf_d   = mm.register(d_host,              upload=True)
    buf_c   = mm.register(c_k1_host,           upload=True)
    buf_a   = mm.register(np.asarray(a_np),    upload=True)
    buf_w0  = mm.register(np.asarray(w0_np),   upload=True)
    buf_gam = mm.register(np.asarray(gam_np),  upload=True)
    buf_pb  = mm.register(pb_host,             upload=True)
    buf_ws  = mm.register(ws_dummy,            upload=True)
    buf_flg = mm.register(flags_dummy,         upload=True)

    k1_mod  = GpuModule(k1_hsaco)
    k1_fn   = k1_mod.get_function(k1_name)
    cd_mod  = GpuModule(cd_hsaco)
    cd_fn   = cd_mod.get_function("colv2")

    stream = hip_stream_create()

    def launch_k1():
        kp, ka, kpa, _ = k1_launch_args(
            k1_sol, M, N_hidden, K, N_tiles_N,
            buf_d, buf_c, buf_a, buf_w0, buf_gam, buf_pb, buf_ws, buf_flg,
        )
        hip_module_launch_kernel(
            k1_fn.handle, numWG_k1, 1, 1,
            k1_sol["NumThreads"], 1, 1, kp,
        )

    def launch_col_div():
        kp, ka, kpa, gx, gy = colv2_launch_args(
            buf_d, buf_pb, M, N_hidden, N_tiles_N, inv_d, eps,
        )
        hip_module_launch_kernel(
            cd_fn.handle, gx, gy, 1, *COLV2_BLOCK, kp,
        )

    def reset_buffers():
        buf_d.memset(0)
        buf_pb.memset(0)

    # Verification.
    if do_verify:
        print("Verifying...")
        reset_buffers()
        launch_k1()
        launch_col_div()
        hip_stream_synchronize(stream)

        mm.sync_from_gpu(d_host)
        mm.sync_from_gpu(pb_host)

        ok = verify_pipeline(
            d_host.view(np.uint16), pb_host,
            M, N_hidden, N_tiles_N, MT1,
            a_np, w0_np, gam_np, inv_d, eps,
        )
        if not ok:
            mm.release_all()
            hip_stream_destroy(stream)
            sys.exit(1)
        print()

    # Warmup.
    for _ in range(warmup):
        reset_buffers()
        launch_k1()
        launch_col_div()
    hip_stream_synchronize(stream)

    # Timed iterations.
    ev0 = hip_event_create()
    ev1 = hip_event_create()
    ev2 = hip_event_create()

    times_k1 = []
    times_cd = []
    times_pip = []

    for _ in range(iters):
        reset_buffers()
        hip_stream_synchronize(stream)

        hip_event_record(ev0, Ptr(0))
        launch_k1()
        hip_event_record(ev1, Ptr(0))
        launch_col_div()
        hip_event_record(ev2, Ptr(0))
        hip_event_synchronize(ev2)

        times_k1.append(int(hip_event_elapsed_time(ev0, ev1) * 1_000_000))
        times_cd.append(int(hip_event_elapsed_time(ev1, ev2) * 1_000_000))
        times_pip.append(int(hip_event_elapsed_time(ev0, ev2) * 1_000_000))

    hip_event_destroy(ev0)
    hip_event_destroy(ev1)
    hip_event_destroy(ev2)
    hip_stream_destroy(stream)
    mm.release_all()

    k1_flops = 2 * M * N_hidden * K
    d_bytes  = M * N_hidden * 2

    med_k1  = float(np.median(times_k1))
    med_cd  = float(np.median(times_cd))
    med_pip = float(np.median(times_pip))

    print(f"{'':=<58}")
    print(f"{'Kernel':<20} {'median (us)':>12}  {'metric':>18}")
    print(f"{'-'*58}")
    print(f"{'K1 (GEMM+PartRMS)':<20} {med_k1/1e3:>12.1f}  "
          f"{k1_flops/med_k1*1e-3:>14.3f} TFLOPS")
    print(f"{'colv2 (RMSNorm)':<20} {med_cd/1e3:>12.1f}  "
          f"{2*d_bytes/(med_cd*1e-9)/1e9:>14.1f} GB/s")
    print(f"{'-'*58}")
    print(f"{'Pipeline':<20} {med_pip/1e3:>12.1f}  "
          f"{k1_flops/med_pip*1e-3:>14.3f} TFLOPS")
    print(f"{'':=<58}")


def parse_args():
    p = argparse.ArgumentParser(
        description="GEMM+PartialRMS → RMSNorm (col-major) pipeline benchmark"
    )
    p.add_argument("--M",          type=int,   default=4096)
    p.add_argument("--N-hidden",   type=int,   default=4096, dest="N_hidden")
    p.add_argument("--K",          type=int,   default=4096)
    p.add_argument("--wg-n",       type=int,   default=2, dest="wg_n",
                   help="MIWaveGroup[1] (default 2 → MT1=128)")
    p.add_argument("--eps",        type=float, default=1e-5)
    p.add_argument("--warmup",     type=int,   default=3)
    p.add_argument("--iters",      type=int,   default=10)
    p.add_argument("--no-verify",  action="store_true", dest="no_verify")
    p.add_argument("--chip",       default=None)
    return p.parse_args()


def main():
    args = parse_args()
    chip = args.chip or amdgpu_exec.get_chip()

    if not chip.startswith("gfx950"):
        print(f"WARNING: PartialRMS is only implemented for gfx950; chip={chip}")

    benchmark(
        chip=chip,
        M=args.M,
        N_hidden=args.N_hidden,
        K=args.K,
        eps=args.eps,
        warmup=args.warmup,
        iters=args.iters,
        do_verify=not args.no_verify,
        wg_n=args.wg_n,
    )


if __name__ == "__main__":
    main()
