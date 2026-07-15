# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark for the row-major GEMM + RMSNorm two-kernel pipeline.

Stage K1 (PartialRMSAxis=0): bf16 TN GEMM fused with PartialRMS epilogue.
  D_row[m, i] = bf16(h1[m, i] * gamma[i])        (bf16, row-major M x N_hidden)
  partialBuf[m, t] = sum_{i in tile t} h1[m,i]^2 (f32, row-major M_pad x n_d)

Stage row_div: divides D_row in-place by sqrt(inv_d * sum_t(partialBuf[m,:]) + eps).

Usage
-----
  python bench_gemm_rmsnorm.py
  python bench_gemm_rmsnorm.py --M 4096 --N-hidden 4096 --K 4096
  python bench_gemm_rmsnorm.py --chip gfx950 --no-verify
"""

import argparse
import math
import os
import sys

import ml_dtypes
import numpy as np
import amdgpu_exec
from amdgpu_exec import execute_hsaco, compile_asm_to_hsaco
from amdgpu_exec.runtime import InputArray, InOutArray

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_TENSILE_DIR = os.path.dirname(os.path.dirname(_BENCH_DIR))
if _TENSILE_DIR not in sys.path:
    sys.path.insert(0, _TENSILE_DIR)

_DEFAULT_ROWMAJOR_YAML = os.path.join(_TENSILE_DIR, "epilogues", "yaml", "gemm_partial_rms_k1_rowmajor.yaml")
_ROW_DIV_PATH = os.path.join(_TENSILE_DIR, "epilogues", "kernels", "row_div.s")

from epilogues.tensilelite.partialrms_helpers import (
    setup_tensile, compileSolution, compute_sk3_dp_args, _pack_kernel_info,
    buildSubtileArgs, buildRowDivArgs,
)
from epilogues.tensilelite.yaml_solution_builder import solutionsFromYaml
from epilogues.tensilelite.numpy_helpers import randBf16, randGamma, rmsNormReference
_ROW_DIV_NAME = "row_div"
# Columns processed per row_div block; N_hidden must be a multiple.
# grid=(M, n_split, 1), block=(64, 1, 1): n_split = N_hidden // _RD_BLOCK.
_RD_BLOCK = 128

bf16 = ml_dtypes.bfloat16


# ---------------------------------------------------------------------------
# Host array construction.
# ---------------------------------------------------------------------------


def makeHostArrays(M, nHidden, K, MT0, MT1):
    """Allocate and initialise all host arrays for the pipeline.

    Tensile writes D col-major with free0=nHidden contiguous, producing a
    Fortran nHidden×M buffer that is bit-identical to row-major M×nHidden.
    We allocate dFortran as nHidden×M Fortran so the kernel addresses it
    correctly; callers read it back as dFortran.T (M×nHidden row-major view).

    A_tensile = bFortran (K×nHidden Fortran, strideA0=K).
    B_tensile = aFortranT = aRow.T as Fortran K×M (strideB0=K).
    """
    nD = math.ceil(nHidden / MT0)
    mPad = math.ceil(M / MT1) * MT1
    rng = np.random.default_rng(42)
    aRow = randBf16(rng, (M, K))                              # M×K logical
    bRow = randBf16(rng, (K, nHidden))                        # K×nHidden logical
    _gF32, gammaBf16 = randGamma(rng, nHidden)
    bFortran = np.asfortranarray(bRow)                        # K×nHidden Fortran (strideA0=K)
    aFortranT = np.asfortranarray(aRow.T)                     # K×M Fortran (strideB0=K)
    cFortran = np.zeros((nHidden, M), dtype=bf16, order="F")  # C dummy, same layout as D
    dFortran = np.zeros((nHidden, M), dtype=bf16, order="F")  # D: kernel writes nHidden×M Fortran
    partialBuf = np.zeros((mPad, nD), dtype=np.float32, order="C")
    return aRow, bRow, gammaBf16, bFortran, aFortranT, cFortran, dFortran, partialBuf



# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------


def runTimed(hsaco, kernelName, args, grid, block, numIters):
    """Execute hsaco with timing, return list of elapsed nanoseconds."""
    return execute_hsaco(
        hsaco=hsaco, kernel_name=kernelName, arguments=args,
        grid_dim=grid, block_dim=block,
        num_iterations=numIters,
    )


def printReport(k1AvgUs, rdAvgUs, M, nHidden, K, nD):
    """Print timing table."""
    k1Flops = 2 * M * nHidden * K
    pipeAvgUs = k1AvgUs + rdAvgUs
    k1Tflops = k1Flops / (k1AvgUs * 1e-6) / 1e12
    pipeTflops = k1Flops / (pipeAvgUs * 1e-6) / 1e12
    rdBytes = 2 * M * nHidden * 2 + M * nD * 4
    rdGbs = rdBytes / (rdAvgUs * 1e-6) / 1e9
    width = 58
    print(f"\n{'':=<{width}}")
    print(f"{'Kernel':<20} {'avg (us)':>10}  {'TFLOPS/GB/s':>14}")
    print(f"{'-'*width}")
    print(f"{'K1 (GEMM+RMS)  ':<20} {k1AvgUs:>10.2f}  {k1Tflops:>14.2f} TFLOPS")
    print(f"{'row_div':<20} {rdAvgUs:>10.2f}  {rdGbs:>14.1f} GB/s")
    print(f"{'Pipeline':<20} {pipeAvgUs:>10.2f}  {pipeTflops:>14.2f} TFLOPS")
    print(f"{'':=<{width}}")


# ---------------------------------------------------------------------------
# Main benchmark.
# ---------------------------------------------------------------------------


def buildK1Solution(chip, yamlPath, wgN):
    """Return the base no-residual K1 solution with MacroTile1 = 128 * wgN.

    Enumerates all no-residual groups (problemIdx=0) and picks the wg0=1 tile
    (smallest MacroTile0). Robust to future group additions — stops at the first
    missing group index.
    """
    assembler, isaInfoMap, debugConfig = setup_tensile(chip)
    wantMT1 = 128 * wgN
    matches = []
    groupIdx = 0
    while True:
        try:
            sols = solutionsFromYaml(
                yamlPath, assembler, isaInfoMap, debugConfig,
                problemIdx=0, groupIdx=groupIdx,
            )
        except (IndexError, KeyError):
            break
        matches.extend(s for s, _sid in sols if s["MacroTile1"] == wantMT1)
        groupIdx += 1
    if matches:
        return min(matches, key=lambda s: s["MacroTile0"])
    raise RuntimeError(f"no K1 solution with MacroTile1={wantMT1} in {yamlPath}")


def buildPipeline(chip, M, nHidden, K, wgN, config):
    """Set up and compile K1 kernel and row_div kernel."""
    yamlPath = config if config is not None else _DEFAULT_ROWMAJOR_YAML
    k1Sol = buildK1Solution(chip, yamlPath, wgN)
    k1Name, k1Hsaco, _chip = compileSolution(k1Sol)
    MT0 = k1Sol["MacroTile0"]
    MT1 = k1Sol["MacroTile1"]
    nD = math.ceil(nHidden / MT0)
    mPad = math.ceil(M / MT1) * MT1
    numWG = math.ceil(nHidden / MT0) * math.ceil(M / MT1)
    with open(_ROW_DIV_PATH) as fh:
        rdAsm = fh.read()
    rdHsaco = compile_asm_to_hsaco(rdAsm, chip)
    return k1Sol, nD, mPad, numWG, k1Hsaco, k1Name, rdHsaco


def benchmark(chip, M, nHidden, K, wgN, eps, warmup, numIters, doVerify, config):
    if not chip.startswith("gfx950"):
        print(f"warning: chip={chip!r} is not gfx950; kernel may not run correctly")

    k1Sol, nD, mPad, numWG, k1Hsaco, k1Name, rdHsaco = buildPipeline(chip, M, nHidden, K, wgN, config)
    MT0 = k1Sol["MacroTile0"]
    MT1 = k1Sol["MacroTile1"]
    invD = 1.0 / nHidden

    arrays = makeHostArrays(M, nHidden, K, MT0, MT1)
    aRow, bRow, gammaBf16, bFortran, aFortranT, cFortran, dFortran, partialBuf = arrays

    k1Grid = (numWG, 1, 1)
    k1Block = (k1Sol["NumThreads"], 1, 1)

    nSplit = nHidden // _RD_BLOCK  # grid_dim_y for row_div
    rdGrid = (M, nSplit, 1)
    rdBlock = (64, 1, 1)

    def makeArgs():
        """Build fresh argument lists for both kernels (dFortran and dRow are shared buffers)."""
        sk = compute_sk3_dp_args(M=nHidden, N=M, K=K, solution=k1Sol)
        ki0, ki1 = _pack_kernel_info(k1Sol)
        k1Args = buildSubtileArgs(
            nHidden, M, K, numWG,
            InOutArray(dFortran), InputArray(cFortran),
            InputArray(bFortran), InputArray(aFortranT),
            sk, ki0, ki1,
            [InputArray(gammaBf16), InOutArray(partialBuf)],
        )
        dRow = np.ascontiguousarray(dFortran.T)    # M×nHidden row-major bf16 (copy after K1)
        partialBufM = np.ascontiguousarray(partialBuf[:M, :])
        rdArgs = buildRowDivArgs(InOutArray(dRow), InputArray(partialBufM),
                                 nHidden, _RD_BLOCK, nD, invD, eps)
        return k1Args, dRow, rdArgs

    if doVerify:
        ref = rmsNormReference(aRow, bRow, gammaBf16, invD, eps)
        # Run pipeline once: K1 writes dFortran and partialBuf, then row_div divides dRow.
        k1Args, dRow, rdArgs = makeArgs()
        execute_hsaco(k1Hsaco, k1Name, k1Args, grid_dim=k1Grid, block_dim=k1Block, num_iterations=1)
        execute_hsaco(rdHsaco, _ROW_DIV_NAME, rdArgs,
                      grid_dim=rdGrid, block_dim=rdBlock, num_iterations=1)
        np.testing.assert_allclose(dRow.astype(np.float32), ref, rtol=2e-2, atol=2e-2)
        print("  pipeline verification PASSED")

    # Timed runs: warmup then measured.
    k1Args, dRow, rdArgs = makeArgs()
    k1Times = runTimed(k1Hsaco, k1Name, k1Args, k1Grid, k1Block, warmup + numIters)
    rdTimes = runTimed(rdHsaco, _ROW_DIV_NAME, rdArgs, rdGrid, rdBlock, warmup + numIters)

    k1AvgUs = float(np.mean(k1Times[warmup:])) / 1_000
    rdAvgUs = float(np.mean(rdTimes[warmup:])) / 1_000
    printReport(k1AvgUs, rdAvgUs, M, nHidden, K, nD)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parseArgs():
    p = argparse.ArgumentParser(description="Row-major GEMM + RMSNorm pipeline benchmark")
    p.add_argument("--M",          type=int,   default=4096)
    p.add_argument("--N-hidden",   type=int,   default=4096, dest="N_hidden")
    p.add_argument("--K",          type=int,   default=4096)
    p.add_argument("--wg-n",       type=int,   default=2,    dest="wg_n")
    p.add_argument("--eps",        type=float, default=1e-5)
    p.add_argument("--warmup",     type=int,   default=3)
    p.add_argument("--num-iters",  type=int,   default=10,   dest="num_iters")
    p.add_argument("--no-verify",  action="store_true", dest="no_verify")
    p.add_argument("--chip",       default=None)
    p.add_argument("--config",     default=None,
                   help="path to K1 YAML config (default: gemm_partial_rms_k1_rowmajor.yaml)")
    return p.parse_args()


def main():
    args = parseArgs()
    chip = args.chip or amdgpu_exec.get_chip()
    benchmark(
        chip=chip,
        M=args.M,
        nHidden=args.N_hidden,
        K=args.K,
        wgN=args.wg_n,
        eps=args.eps,
        warmup=args.warmup,
        numIters=args.num_iters,
        doVerify=not args.no_verify,
        config=args.config,
    )


if __name__ == "__main__":
    main()
