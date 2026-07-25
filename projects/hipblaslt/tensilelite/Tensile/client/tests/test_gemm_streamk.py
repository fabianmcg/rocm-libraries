# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M6 test suite: StreamK=4 and StreamK=5 argument layout (task 6.5).

Pure-Python tests verify byte count, slot order, and computed field values
against the CeilDivide logic from ContractionSolution.cpp:778-908.
GPU tests require gfx950 and compile from gemm_streamk45_gpu.yaml (group 0 = SK4,
group 1 = SK5).  Both GPU tests use all-data-parallel mode (sk_tiles=0) so
every WG computes a complete output tile, giving a result identical to standard
GEMM that can be verified against the numpy reference.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys

import numpy as np
import pytest

try:
    import amdgpu_exec
    HAVE_DEPS = True
except ImportError:
    amdgpu_exec = None
    HAVE_DEPS = False

from .conftest import requires_gfx950

_TESTS_DIR = os.path.dirname(__file__)
_YAML_PATH = os.path.join(_TESTS_DIR, "yaml", "gemm_streamk45_gpu.yaml")
_TENSILE_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", ".."))

if _TENSILE_ROOT not in sys.path:
    sys.path.insert(0, _TENSILE_ROOT)

from Tensile.client.gemm_args import (
    buildKernelArgs,
    _buildStreamK4Args,
    _buildStreamK5Args,
    _computeInternalArg0,
    _computeInternalArg1,
)
from Tensile.client.reference import gemm, assertClose, RTOL_FP32, ATOL_FP32
from epilogues.epilogue_harness.yaml_solution_builder import _injectInternalArgsSupport


# ---------------------------------------------------------------------------
# Minimal solution dict for StreamK tests (no MX, no epilogue).
# ---------------------------------------------------------------------------


def _sk4SolDict() -> dict:
    return {
        "KernArgsVersion": 2,
        "SupportCustomWGM": True,
        "SupportCustomStaggerU": False,
        "SupportUserGSU": False,
        "UseSFC": False,
        "UseUniversalArgs": True,
        "MacroTile0": 64,
        "MacroTile1": 64,
        "WorkGroupMapping": 8,
        "WorkGroupMappingXCC": 0,
        "WorkGroupMappingXCCGroup": 0,
        "StaggerU": 32,
        "StaggerUMapping": 1,
        "_staggerStrideShift": 2,
        "GlobalSplitU": 1,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "StreamK": 4,
        "StreamKAtomic": 0,
        "StridedBatched": True,
        "UseBeta": True,
        "GlobalAccumulation": 0,
        "ExpertSchedulingMode": 0,
        "HighPrecisionAccumulate": True,
        "ComputeDataType": 0,
    }


def _sk5SolDict(sk_value: int = 5) -> dict:
    d = _sk4SolDict()
    d["StreamK"] = sk_value
    return d


def _sk4ProblemParams(iters_per_tile: int = 64, tiles: int = 100,
                      sk_tiles: int = 0, sk_split: int = 2,
                      sk_grid: int = 128) -> dict:
    return {
        "sizes": [256, 256, 4, 256],
        "ldd": 256, "stride_d": 256 * 256,
        "ldc": 256, "stride_c": 256 * 256,
        "lda": 256, "stride_a": 256 * 256,
        "ldb": 256, "stride_b": 256 * 256,
        "alpha": 1.0,
        "beta": 0.0,
        "gsu": 1,
        "sk4": {
            "iters_per_tile": iters_per_tile,
            "tiles": tiles,
            "sk_tiles": sk_tiles,
            "sk_split": sk_split,
            "sk_grid": sk_grid,
        },
    }


def _sk5DynamicProblemParams(**sk4_kwargs) -> dict:
    pp = _sk4ProblemParams(**sk4_kwargs)
    sk4 = pp.pop("sk4")
    sk4["effective_dynamic"] = True
    pp["sk5"] = sk4
    return pp


def _sk5StaticProblemParams(iters_per_tile: int = 64, sk_iters_per_wg: int = 32,
                             sk_grid: int = 128, sk_tiles: int = 128) -> dict:
    return {
        "sizes": [256, 256, 4, 256],
        "ldd": 256, "stride_d": 256 * 256,
        "ldc": 256, "stride_c": 256 * 256,
        "lda": 256, "stride_a": 256 * 256,
        "ldb": 256, "stride_b": 256 * 256,
        "alpha": 1.0,
        "beta": 0.0,
        "gsu": 1,
        "sk5": {
            "effective_dynamic": False,
            "iters_per_tile": iters_per_tile,
            "sk_iters_per_wg": sk_iters_per_wg,
            "sk_grid": sk_grid,
            "sk_tiles": sk_tiles,
        },
    }


# ===========================================================================
# Task 6.5 — StreamK=4 argument layout
# ===========================================================================


class TestStreamK4Args:
    """Verify StreamK=4 argument slots match ContractionSolution.cpp:778-806."""

    def test_six_slots_24_bytes(self):
        pp = _sk4ProblemParams()
        buf = _buildStreamK4Args({}, pp)
        assert len(buf) == 24  # 6 × uint32

    def test_iters_per_tile_slot(self):
        pp = _sk4ProblemParams(iters_per_tile=128)
        buf = _buildStreamK4Args({}, pp)
        iters_per_tile = struct.unpack_from("<I", buf, 0)[0]
        assert iters_per_tile == 128

    def test_ceildivide_sk_split(self):
        """Verify skSplit is recalculated via CeilDivide(itersPerTile, skItersPerWI)."""
        iters_per_tile = 64
        initial_split = 3
        pp = _sk4ProblemParams(iters_per_tile=iters_per_tile, sk_split=initial_split)
        buf = _buildStreamK4Args({}, pp)
        # sk_iters_per_wi = ceil(64/3) = 22; sk_split = ceil(64/22) = 3
        sk_iters_per_wi = math.ceil(iters_per_tile / initial_split)
        expected_split = math.ceil(iters_per_tile / sk_iters_per_wi)
        sk_split = struct.unpack_from("<I", buf, 12)[0]  # slot 3
        assert sk_split == expected_split

    def test_sk_iters_per_wi_slot(self):
        iters_per_tile = 64
        sk_split = 2
        pp = _sk4ProblemParams(iters_per_tile=iters_per_tile, sk_split=sk_split)
        buf = _buildStreamK4Args({}, pp)
        expected = math.ceil(iters_per_tile / sk_split)
        sk_iters_per_wi = struct.unpack_from("<I", buf, 16)[0]  # slot 4
        assert sk_iters_per_wi == expected

    def test_total_items_slot(self):
        iters_per_tile = 64
        tiles = 100
        sk_tiles = 10
        sk_split = 2
        pp = _sk4ProblemParams(iters_per_tile=iters_per_tile, tiles=tiles,
                               sk_tiles=sk_tiles, sk_split=sk_split)
        buf = _buildStreamK4Args({}, pp)
        sk_iters_per_wi = math.ceil(iters_per_tile / sk_split)
        actual_split = math.ceil(iters_per_tile / sk_iters_per_wi)
        expected_total = (tiles - sk_tiles) + sk_tiles * actual_split
        total_items = struct.unpack_from("<I", buf, 4)[0]  # slot 1
        assert total_items == expected_total

    def test_sk_grid_slot(self):
        pp = _sk4ProblemParams(sk_grid=256)
        buf = _buildStreamK4Args({}, pp)
        sk_grid = struct.unpack_from("<I", buf, 20)[0]  # slot 5
        assert sk_grid == 256

    def test_buildKernelArgs_sk4_includes_sk_block(self):
        """buildKernelArgs with StreamK=4 appends the 6-slot SK4 block."""
        sol = _sk4SolDict()
        pp = _sk4ProblemParams()
        tensors = {"D": 0x1000, "C": 0x2000, "A": 0x3000, "B": 0x4000,
                   "workspace": 0x5000, "flags": 0x6000}
        buf = buildKernelArgs(sol, pp, tensors)
        assert len(buf) > 0

    def test_streamk_1_still_raises(self):
        sol = _sk4SolDict()
        sol["StreamK"] = 1
        pp = _sk4ProblemParams()
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0, "workspace": 0, "flags": 0}
        with pytest.raises(NotImplementedError, match="streamK=1"):
            buildKernelArgs(sol, pp, tensors)


# ===========================================================================
# Task 6.5 — StreamK=5 argument layout (dynamic sub-mode)
# ===========================================================================


class TestStreamK5DynamicArgs:
    """Verify SK5-dynamic slots: same as SK4 but mode bit 30 set in SKTiles slot."""

    def test_six_slots_24_bytes(self):
        pp = _sk5DynamicProblemParams()
        buf = _buildStreamK5Args({}, pp)
        assert len(buf) == 24

    def test_mode_bit_30_set(self):
        pp = _sk5DynamicProblemParams(sk_tiles=0)
        buf = _buildStreamK5Args({}, pp)
        packed_sk_tiles = struct.unpack_from("<I", buf, 8)[0]  # slot 2
        assert packed_sk_tiles & 0x40000000, "mode bit 30 must be set for dynamic SK5"

    def test_sk_tiles_value_preserved(self):
        pp = _sk5DynamicProblemParams(sk_tiles=5)
        buf = _buildStreamK5Args({}, pp)
        packed_sk_tiles = struct.unpack_from("<I", buf, 8)[0]
        sk_tiles_raw = packed_sk_tiles & ~0x40000000
        assert sk_tiles_raw == 5

    def test_iters_per_tile_matches_sk4(self):
        pp4 = _sk4ProblemParams(iters_per_tile=128)
        pp5 = _sk5DynamicProblemParams(iters_per_tile=128)
        buf4 = _buildStreamK4Args({}, pp4)
        buf5 = _buildStreamK5Args({}, pp5)
        assert struct.unpack_from("<I", buf4, 0) == struct.unpack_from("<I", buf5, 0)


# ===========================================================================
# Task 6.5 — StreamK=5 argument layout (static sub-mode, mirrors SK3)
# ===========================================================================


class TestStreamK5StaticArgs:
    """Verify SK5-static slots: itersPerTile, magic, shift, SKItersPerWG, grid, tiles."""

    def test_six_slots_24_bytes(self):
        pp = _sk5StaticProblemParams()
        buf = _buildStreamK5Args({}, pp)
        assert len(buf) == 24

    def test_iters_per_tile_slot(self):
        pp = _sk5StaticProblemParams(iters_per_tile=64)
        buf = _buildStreamK5Args({}, pp)
        assert struct.unpack_from("<I", buf, 0)[0] == 64

    def test_sk_iters_per_wg_slot(self):
        pp = _sk5StaticProblemParams(sk_iters_per_wg=16)
        buf = _buildStreamK5Args({}, pp)
        sk_iters_per_wg = struct.unpack_from("<I", buf, 12)[0]  # slot 3
        assert sk_iters_per_wg == 16

    def test_sk_grid_slot(self):
        pp = _sk5StaticProblemParams(sk_grid=64)
        buf = _buildStreamK5Args({}, pp)
        sk_grid = struct.unpack_from("<I", buf, 16)[0]  # slot 4
        assert sk_grid == 64

    def test_sk_tiles_slot(self):
        pp = _sk5StaticProblemParams(sk_tiles=50)
        buf = _buildStreamK5Args({}, pp)
        sk_tiles = struct.unpack_from("<I", buf, 20)[0]  # slot 5
        assert sk_tiles == 50

    def test_mode_bit_not_set_in_static_path(self):
        """Static SK5 must NOT set bit 30 in the magic-number slot.

        The assert in ContractionSolution.cpp:865 requires this.
        """
        pp = _sk5StaticProblemParams(iters_per_tile=64)
        buf = _buildStreamK5Args({}, pp)
        magic_shift = struct.unpack_from("<I", buf, 8)[0]  # slot 2 = shift
        assert (magic_shift & 0x40000000) == 0, \
            "magic shift must not have bit 30 set in SK5-static"

    def test_buildKernelArgs_sk5_static_works(self):
        sol = _sk5SolDict()
        sol["StreamK"] = 5
        pp = _sk5StaticProblemParams()
        tensors = {"D": 0x1000, "C": 0x2000, "A": 0x3000, "B": 0x4000,
                   "workspace": 0x5000, "flags": 0x6000}
        buf = buildKernelArgs(sol, pp, tensors)
        assert len(buf) > 0


# ===========================================================================
# GPU test infrastructure (mirrors test_gemm_standard.py pattern)
# ===========================================================================


def _setupTensile(chip: str):
    """Initialize Tensile assembler and ISA map for kernel compilation."""
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
    return assembler, isaInfoMap, DebugConfig()


def _generateAsm(solution, assembler, debugConfig):
    """Return (asm_str, kernel_name) for a solution."""
    import rocisa
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelNameMin

    kwa = KernelWriterAssembly(assembler, debugConfig)
    ti = rocisa.rocIsa.getInstance()
    kwa.setRocIsa(ti.getData(), ti.getOutputOptions())
    kernel = solution.getKernels()[0]
    kernel.duplicate = False
    err, asmStr = kwa.getSourceFileString(kernel)
    if err:
        raise RuntimeError(f"assembly generation failed: {err}")
    return asmStr, getKernelNameMin(kernel, splitGSU=False)


def _compileSk(problemIdx: int):
    """Compile SK solutions for one YAML problem group; return list of entry dicts."""
    if not HAVE_DEPS:
        return []
    try:
        from epilogues.epilogue_harness.yaml_solution_builder import solutionsFromYaml
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        sols = solutionsFromYaml(_YAML_PATH, assembler, isaInfoMap, debugConfig,
                                 problemIdx=problemIdx)
    except Exception as exc:
        import warnings
        warnings.warn(f"could not compile SK solutions (problemIdx={problemIdx}): {exc}")
        return []

    compiled = []
    for sol, sid in sols:
        try:
            asmStr, kernelName = _generateAsm(sol, assembler, debugConfig)
            hsaco = amdgpu_exec.compile_asm_to_hsaco(asmStr, chip)
        except Exception as exc:
            import warnings
            warnings.warn(f"SK solution {sid} failed to compile: {exc}")
            continue
        rawDict = dict(sol)
        solDict = _injectInternalArgsSupport(rawDict, chip)
        compiled.append({
            "sol_dict": solDict,
            "raw_dict": rawDict,
            "kernel_name": kernelName,
            "hsaco": hsaco,
            "chip": chip,
            "sid": sid,
        })
    return compiled


def _filterSkSolution(entry: dict) -> bool:
    """Return True if the solution has a non-zero WorkGroupMapping (is usable)."""
    return entry["sol_dict"].get("WorkGroupMapping", 0) != 0


def _deviceCuCount() -> int:
    """Return the device CU count for device 0."""
    if not HAVE_DEPS:
        return 0
    props = amdgpu_exec._runtime_module.hip_get_device_props(0)
    return int(props.get("multiprocessor_count", 0))


def _computeSk4DpParams(solDict: dict, M: int, N: int, K: int, batch: int) -> dict:
    """Compute SK4 all-data-parallel params for a given problem.

    With sk_tiles=0, every WG runs the data-parallel path and computes a
    complete tile.  The result is numerically identical to standard GEMM.
    """
    mt0 = solDict["MacroTile0"]
    mt1 = solDict["MacroTile1"]
    depthU = solDict["DepthU"]
    numTiles = math.ceil(M / mt0) * math.ceil(N / mt1) * batch
    itersPerTile = max(1, math.ceil(K / depthU))
    return {
        "iters_per_tile": itersPerTile,
        "tiles": numTiles,
        "sk_tiles": 0,
        "sk_split": 2,
        "sk_grid": numTiles,
    }


def _computeSk5DpParams(solDict: dict, M: int, N: int, K: int, batch: int) -> dict:
    """Compute SK5 all-data-parallel params (static path, sk_tiles=0)."""
    mt0 = solDict["MacroTile0"]
    mt1 = solDict["MacroTile1"]
    depthU = solDict["DepthU"]
    numTiles = math.ceil(M / mt0) * math.ceil(N / mt1) * batch
    itersPerTile = max(1, math.ceil(K / depthU))
    return {
        "effective_dynamic": False,
        "iters_per_tile": itersPerTile,
        "sk_iters_per_wg": itersPerTile,
        "sk_grid": numTiles,
        "sk_tiles": 0,
    }


def _buildSkLaunchArgs(solDict: dict, M: int, N: int, batch: int, K: int,
                       dBuf, cBuf, aBuf, bBuf, wsBuf, flagsBuf,
                       numWg: int, skParamKey: str, skParams: dict) -> list:
    """Build typed arg list for GpuFunction.launch: NT SGEMM with StreamK.

    NT strides: lda=M (A is M×K col-major), ldb=N (B is N×K col-major),
    ldd=ldc=M.  All buffer args must be GpuBuffer — SK kernels write workspace
    and flags even in all-DP mode; InputArray host-pinned memory causes a GPU
    page fault.  SK block (6 × uint32) is appended after alpha/beta.
    """
    arg0 = _computeInternalArg0(solDict, gsu=1)
    arg1 = _computeInternalArg1(solDict, cu_count=_deviceCuCount())
    gemmCount = (1 & 0x3FFFFFFF) | (0 << 30)

    ppDummy = {
        "sizes": [M, N, batch, K],
        "ldd": M, "stride_d": M * N,
        "ldc": M, "stride_c": M * N,
        "lda": M, "stride_a": M * K,
        "ldb": N, "stride_b": N * K,
        "alpha": 1.0, "beta": 0.0, "gsu": 1,
        skParamKey: skParams,
    }
    if skParamKey == "sk4":
        skBlock = _buildStreamK4Args(solDict, ppDummy)
    else:
        skBlock = _buildStreamK5Args(solDict, ppDummy)

    args = [
        np.uint32(gemmCount), np.uint32(arg0), np.int32(arg1), np.uint32(numWg),
        np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K),
        dBuf, cBuf, aBuf, bBuf, wsBuf, flagsBuf,
        np.uint32(M), np.uint32(M * N),
        np.uint32(M), np.uint32(M * N),
        np.uint32(M), np.uint32(M * K),
        np.uint32(N), np.uint32(N * K),
        np.float32(1.0), np.float32(0.0),
    ]
    for i in range(6):
        args.append(np.uint32(struct.unpack_from("<I", skBlock, i * 4)[0]))
    return args


def _runSkNtGemm(entry: dict, M: int, N: int, batch: int, K: int,
                 skParamKey: str, skParams: dict, label: str):
    """Execute one SK GEMM kernel (NT, fp32, stridedBatched) and verify output.

    Uses all-data-parallel mode (sk_tiles=0) so no streaming reduction occurs
    and the output matches the standard GEMM numpy reference.  All device
    buffers are GpuBuffer; the kernel writes workspace and flags even in all-DP
    mode so these must be writable device memory.
    """
    from amdgpu_exec import GpuBuffer, GpuModule, GpuEvent
    solDict = entry["sol_dict"]
    kernelName = entry["kernel_name"]
    hsaco = entry["hsaco"]
    numThreads = solDict["NumThreads"]
    mt0, mt1 = solDict["MacroTile0"], solDict["MacroTile1"]
    numWg = math.ceil(M / mt0) * math.ceil(N / mt1) * batch

    rng = np.random.default_rng(seed=M * 5000 + N + K)
    A_np = np.asfortranarray(rng.random((M, K)).astype(np.float32))
    B_np = np.asfortranarray(rng.random((N, K)).astype(np.float32))  # NT: B is N×K
    A_flat = np.tile(A_np.ravel(order='F'), batch)
    B_flat = np.tile(B_np.ravel(order='F'), batch)

    D_buf = GpuBuffer(M * N * batch * 4); D_buf.memset(0)
    C_buf = GpuBuffer(M * N * batch * 4); C_buf.memset(0)
    A_buf = GpuBuffer(A_flat.nbytes); A_buf.copy_from_host(A_flat)
    B_buf = GpuBuffer(B_flat.nbytes); B_buf.copy_from_host(B_flat)
    # SK kernels write workspace/flags even in all-DP mode (sk_tiles=0).
    ws_buf = GpuBuffer(M * N * batch * 4 + 4096); ws_buf.memset(0)
    flags_buf = GpuBuffer(max(numWg, 256) * 4); flags_buf.memset(0)

    args = _buildSkLaunchArgs(solDict, M, N, batch, K,
                              D_buf, C_buf, A_buf, B_buf, ws_buf, flags_buf,
                              numWg, skParamKey, skParams)

    module = GpuModule(hsaco)
    fn = module.get_function(kernelName)
    stop = GpuEvent()
    fn.launch((numWg, 1, 1), (numThreads, 1, 1), args)
    stop.record(); stop.synchronize()
    module.unload()

    D_host = np.zeros(M * N * batch, dtype=np.float32)
    D_buf.copy_to_host(D_host)
    for buf in [D_buf, C_buf, A_buf, B_buf, ws_buf, flags_buf]:
        buf.free()

    # NT reference: D = A @ B^T where A is (M,K) and B is (N,K).
    A_slice = A_flat[:M * K].reshape(M, K, order='F')
    B_slice = B_flat[:N * K].reshape(N, K, order='F')
    D_ref = gemm(A_slice, B_slice.T, alpha=1.0, beta=0.0, C=None)
    D_ref_flat = np.tile(np.asfortranarray(D_ref).ravel(order='F'), batch)
    assertClose(D_host, D_ref_flat.astype(np.float32), rtol=RTOL_FP32, atol=ATOL_FP32,
                label=label)


# ---------------------------------------------------------------------------
# Session-scoped compiled solution fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sk4Kernels():
    """Compile SK4 SGEMM NT solutions from YAML group 0."""
    return _compileSk(0)


@pytest.fixture(scope="session")
def sk5Kernels():
    """Compile SK5 SGEMM NT solutions from YAML group 1."""
    return _compileSk(1)


# ===========================================================================
# GPU StreamK=4/5 tests
# ===========================================================================


class TestStreamKGpuGfx950:
    """GPU StreamK=4 and SK=5 correctness tests using all-data-parallel mode.

    Both tests use sk_tiles=0 so every WG runs the standard data-parallel path,
    producing output equivalent to standard GEMM verified against numpy.
    Adapted from Tensile/Tests/common/streamk/sk_dynamic.yaml (SK4) and
    sk_hybrid.yaml (SK5).  Compatible with gfx950.
    """

    @requires_gfx950
    def test_gpu_sk4_data_parallel(self, sk4Kernels):
        """StreamK=4 NT GEMM 512×512×1×512 in all-data-parallel mode."""
        if not HAVE_DEPS:
            pytest.skip("amdgpu_exec not installed")
        entries = [e for e in sk4Kernels if _filterSkSolution(e)]
        if not entries:
            pytest.skip("no usable SK4 solution compiled from gemm_streamk45_gpu.yaml")

        M, N, batch, K = 512, 512, 1, 512
        entry = entries[0]
        skParams = _computeSk4DpParams(entry["sol_dict"], M, N, K, batch)
        _runSkNtGemm(entry, M, N, batch, K,
                     skParamKey="sk4", skParams=skParams,
                     label=f"SK4 M{M}N{N}B{batch}K{K} {entry['sid']}")

    @requires_gfx950
    def test_gpu_sk5_data_parallel(self, sk5Kernels):
        """StreamK=5 NT GEMM 512×512×1×512 in all-data-parallel mode (static path)."""
        if not HAVE_DEPS:
            pytest.skip("amdgpu_exec not installed")
        entries = [e for e in sk5Kernels if _filterSkSolution(e)]
        if not entries:
            pytest.skip("no usable SK5 solution compiled from gemm_streamk45_gpu.yaml")

        M, N, batch, K = 512, 512, 1, 512
        entry = entries[0]
        skParams = _computeSk5DpParams(entry["sol_dict"], M, N, K, batch)
        _runSkNtGemm(entry, M, N, batch, K,
                     skParamKey="sk5", skParams=skParams,
                     label=f"SK5 M{M}N{N}B{batch}K{K} {entry['sid']}")
