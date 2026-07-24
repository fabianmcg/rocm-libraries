# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""SAXPY example: y[i] = a * x[i] + y[i] for all i.

Usage:
    python saxpy.py <n>         # n is the vector length
    python saxpy.py 1048576     # 1M elements
"""

import amdgpu_exec
import argparse
import sys

import numpy as np


def parse_args():
    """Parse command-line arguments for the SAXPY example."""
    parser = argparse.ArgumentParser(description="SAXPY GPU example")
    parser.add_argument("n", type=int, help="vector length")
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="timing iterations (default: 10)",
    )
    parser.add_argument(
        "--rocm-path",
        default="/opt/rocm",
        help="ROCm installation path (default: /opt/rocm)",
    )
    parser.add_argument(
        "--block-dim", type=int, default=256, help="threads per block (default: 256)"
    )
    parser.add_argument(
        "--chip",
        default=amdgpu_exec.get_chip(),
        help="target GPU (default: auto-detect)",
    )

    args = parser.parse_args()
    if args.n <= 0:
        raise ValueError("n must be a positive integer")

    if args.block_dim <= 0:
        raise ValueError("block_dim must be a positive integer")

    print(f"device: {args.chip}")
    print(f"n     : {args.n:,}")
    print(f"block : {args.block_dim}\n")
    return args


# ---------------------------------------------------------------------------
# SAXPY kernel in LLVM IR
# ---------------------------------------------------------------------------
#
# Signature: saxpy(float a, float* x, float* y, int n)
# gid = __ockl_get_group_id(0) * __ockl_get_local_size(0) + __ockl_get_local_id(0)
# Out-of-bounds threads exit early.

_SAXPY_IR = """\
; ModuleID = 'saxpy'

define amdgpu_kernel void @saxpy(float %a, ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %group_id   = call i64 @__ockl_get_group_id(i32 0)
  %local_size = call i64 @__ockl_get_local_size(i32 0)
  %local_id   = call i64 @__ockl_get_local_id(i32 0)
  %base       = mul i64 %group_id, %local_size
  %gid64      = add i64 %base, %local_id

  %n64        = sext i32 %n to i64
  %in_bounds  = icmp slt i64 %gid64, %n64
  br i1 %in_bounds, label %compute, label %exit

compute:
  %xptr   = getelementptr float, ptr addrspace(1) %x, i64 %gid64
  %yptr   = getelementptr float, ptr addrspace(1) %y, i64 %gid64
  %xval   = load float, ptr addrspace(1) %xptr, align 4
  %yval   = load float, ptr addrspace(1) %yptr, align 4
  %ax     = fmul float %a, %xval
  %result = fadd float %ax, %yval
  store float %result, ptr addrspace(1) %yptr, align 4
  br label %exit

exit:
  ret void
}

declare i64 @__ockl_get_group_id(i32)
declare i64 @__ockl_get_local_size(i32)
declare i64 @__ockl_get_local_id(i32)
"""


def compile_saxpy_kernel(chip: str, rocm_path: str) -> bytes:
    """Compile the SAXPY kernel from LLVM IR to a linked HSA code object."""
    opts = amdgpu_exec.CompileOptions.defaults()
    opts.rocm_path = rocm_path

    asm = amdgpu_exec.llvmir_to_asm(_SAXPY_IR, chip, opts=opts)

    print("=== generated assembly ===")
    print(asm)
    print("=" * 26)
    print()

    hsaco = amdgpu_exec.compile_asm_to_hsaco(asm, chip)
    return hsaco


def execute_saxpy_kernel(hsaco: bytes, n: int, block_dim: int, iterations: int):
    """Execute the SAXPY kernel and report timing."""
    # -----------------------------------------------------------------------
    # Initialise host data
    # -----------------------------------------------------------------------
    a = np.float32(2.0)
    rng = np.random.default_rng()
    x = rng.random(n, dtype=np.float32)
    y = rng.random(n, dtype=np.float32)

    # Keep a reference copy for verification.
    y_ref = a * x + y

    # -----------------------------------------------------------------------
    # Launch
    # -----------------------------------------------------------------------
    grid_x = (n + block_dim - 1) // block_dim

    def verify(arguments):
        result = arguments[2].array  # y is the third argument
        if not np.allclose(result, y_ref, rtol=1e-5, atol=1e-5):
            max_err = np.max(np.abs(result - y_ref))
            print(f"VERIFICATION FAILED  max_err={max_err:.3e}", file=sys.stderr)
            sys.exit(1)
        print("verification: PASSED")

    times_ns = amdgpu_exec.execute_hsaco(
        hsaco=hsaco,
        kernel_name="saxpy",
        arguments=[
            a,
            amdgpu_exec.InputArray(x),
            amdgpu_exec.InOutArray(y),
            np.int32(n),
        ],
        grid_dim=(grid_x, 1, 1),
        block_dim=(block_dim, 1, 1),
        num_iterations=iterations,
        verify_fn=verify,
    )

    # -----------------------------------------------------------------------
    # Report timing
    # -----------------------------------------------------------------------
    times_ms = [t / 1e6 for t in times_ns]
    best_ms = min(times_ms)
    avg_ms = sum(times_ms) / len(times_ms)
    nbytes = 3 * n * np.dtype(np.float32).itemsize  # read x, read+write y
    gbps = nbytes / (best_ms * 1e-3) / 1e9

    print("=== timing ===")
    print(f"iterations : {len(times_ms)}")
    print(f"best       : {best_ms:.3f} ms")
    print(f"avg        : {avg_ms:.3f} ms")
    print(f"bandwidth  : {gbps:.1f} GB/s  (3 * {n} * 4 bytes / best)")


def main():
    args = parse_args()

    hsaco = compile_saxpy_kernel(args.chip, args.rocm_path)

    execute_saxpy_kernel(
        hsaco=hsaco,
        n=args.n,
        block_dim=args.block_dim,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
