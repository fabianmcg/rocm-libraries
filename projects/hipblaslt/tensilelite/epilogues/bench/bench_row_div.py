# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark for the row_div GPU assembly kernel.

The kernel computes C[i, j] /= sqrt(inv_d * sum(D[i, :]) + eps) where:
  C: (m, n)   bf16, row-major, in-place read+write.
  D: (m, n_d) f32,  row-major, read-only partial sums.
  inv_d, eps: f32 scalars.

The assembly source is read at runtime from ~/row_div.s.

Note: the .s file pins .amdgcn_target gfx950. If your local chip differs,
pass --chip gfx950 explicitly.

Usage
-----
  python bench_row_div.py
  python bench_row_div.py --m 4096 --n 4096 --n-d 4096
  python bench_row_div.py --chip gfx950 --block-size 256 --no-verify
"""

import argparse
import os
import sys

import ml_dtypes
import numpy as np
from amdgpu_exec import execute_hsaco, get_chip, compile_asm_to_hsaco
from amdgpu_exec.runtime import InputArray, InOutArray

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_TENSILE_DIR = os.path.dirname(os.path.dirname(_BENCH_DIR))
if _TENSILE_DIR not in sys.path:
    sys.path.insert(0, _TENSILE_DIR)

from epilogues.tensilelite.partialrms_helpers import buildRowDivArgs
from epilogues.tensilelite.numpy_helpers import rmsDenom

_KERNEL_DIR  = os.path.join(os.path.dirname(_BENCH_DIR), "kernels")
_KERNEL_PATH = os.path.join(_KERNEL_DIR, "row_div.s")
_KERNEL_NAME = "row_div"


def make_host_arrays(m, n, n_d):
    rng = np.random.default_rng(42)
    c_host = rng.standard_normal((m, n)).astype(np.float32).astype(ml_dtypes.bfloat16)
    d_host = rng.random((m, n_d), dtype=np.float32) * 0.01
    return c_host, d_host


def make_verify_fn(c_orig, d_host, inv_d, eps):
    """Return a verify_fn compatible with execute_hsaco's verify_fn parameter.

    execute_hsaco calls verify_fn(arguments) after copying results back to the
    host InOutArray, so c_orig.array already holds the GPU output at that point.
    """
    denom = rmsDenom(d_host.sum(axis=1), inv_d, eps)
    c_ref = c_orig.astype(np.float32) / denom[:, np.newaxis]

    def verify_fn(arguments):
        c_result = arguments[0].array.astype(np.float32)
        np.testing.assert_allclose(c_result, c_ref, rtol=2e-2, atol=2e-2)
        print("  verification PASSED")

    return verify_fn


def benchmark(chip, m, n, n_d, inv_d, eps, block_size, warmup, num_iters, do_verify):
    print(f"chip        : {chip}")
    print(f"m={m}  n={n}  n_d={n_d}  block_size={block_size}")
    print(f"inv_d={inv_d:.6g}  eps={eps}")
    print(f"warmup={warmup}  iters={num_iters}\n")

    with open(_KERNEL_PATH) as fh:
        asm_text = fh.read()
    hsaco = compile_asm_to_hsaco(asm_text, chip)

    c_host, d_host = make_host_arrays(m, n, n_d)

    n_split = n // block_size   # grid_dim_y: one block per column partition
    grid = (m, n_split, 1)
    block = (block_size, 1, 1)
    args = buildRowDivArgs(InOutArray(c_host), InputArray(d_host),
                           n, block_size, n_d, inv_d, eps)
    verify_fn = make_verify_fn(c_host.copy(), d_host, inv_d, eps) if do_verify else None

    times_ns = execute_hsaco(
        hsaco, _KERNEL_NAME, args,
        grid_dim=grid, block_dim=block,
        num_iterations=warmup + num_iters,
        verify_fn=verify_fn,
    )
    times_ns = times_ns[warmup:]

    avg_us = float(np.mean(times_ns)) / 1_000
    min_us = float(np.min(times_ns)) / 1_000
    # C read + write (bf16 = 2B each); D read (f32 = 4B).
    bytes_moved = 2 * m * n * 2 + m * n_d * 4
    bw_gbs = bytes_moved / (avg_us * 1e-6) / 1e9

    print(f"\n{'':=<52}")
    print(f"{'Kernel':<20} {'avg (us)':>10}  {'GB/s':>14}")
    print(f"{'-'*52}")
    print(f"{'row_div':<20} {avg_us:>10.2f}  {bw_gbs:>14.1f}")
    print(f"{'':=<52}")
    print(f"  min={min_us:.2f} us")


def parse_args():
    p = argparse.ArgumentParser(description="row_div kernel benchmark")
    p.add_argument("--m",           type=int,   default=4096)
    p.add_argument("--n",           type=int,   default=4096)
    p.add_argument("--n-d",         type=int,   default=4096, dest="n_d")
    p.add_argument("--num-iters",   type=int,   default=10,   dest="num_iters")
    p.add_argument("--warmup",      type=int,   default=3)
    p.add_argument("--block-size",  type=int,   default=128,  dest="block_size")
    p.add_argument("--eps",         type=float, default=1e-5)
    p.add_argument("--inv-d",       type=float, default=None, dest="inv_d",
                   help="inv_d scalar (default: 1/n_d)")
    p.add_argument("--chip",        default=None)
    p.add_argument("--no-verify",   action="store_true", dest="no_verify")
    return p.parse_args()


def main():
    args = parse_args()
    chip = args.chip or get_chip()
    inv_d = args.inv_d if args.inv_d is not None else 1.0 / args.n_d
    benchmark(
        chip=chip,
        m=args.m,
        n=args.n,
        n_d=args.n_d,
        inv_d=inv_d,
        eps=args.eps,
        block_size=args.block_size,
        warmup=args.warmup,
        num_iters=args.num_iters,
        do_verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
