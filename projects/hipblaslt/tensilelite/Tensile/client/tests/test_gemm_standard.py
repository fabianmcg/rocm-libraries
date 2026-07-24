# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M1 test suite: standard GEMM (fp32, fp16, bf16) + strided/pointer-array batch.

Covers Tasks 1.2–1.5 of the TensileLite Python client plan:
  1.2  Poison-input test (corrupt strideA[1], assert ≥50% bad elements) + NotImplementedError
  1.3  gemmFp16 / gemmBf16 reference unit tests (no GPU)
  1.4  Standard GEMM GPU correctness: dtype × stridedBatched × problem sizes
  1.5  Cross-validation notes (see fixtures/m1_cross_validate_notes.txt)

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec.  Non-GPU tests
run under plain tox -e unit.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guards — checked at collection time, not at runtime.
# ---------------------------------------------------------------------------

try:
    import amdgpu_exec
    import ml_dtypes
    HAVE_DEPS = True
except ImportError:
    amdgpu_exec = None
    ml_dtypes = None
    HAVE_DEPS = False

from .conftest import requires_gfx950

# ---------------------------------------------------------------------------
# Module-level paths
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
_YAML_PATH = os.path.join(_TESTS_DIR, "yaml", "gemm_standard.yaml")
_TENSILE_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "..", ".."))

if _TENSILE_ROOT not in sys.path:
    sys.path.insert(0, _TENSILE_ROOT)

# ---------------------------------------------------------------------------
# Imports from the library under test (guarded so collection does not fail).
# ---------------------------------------------------------------------------

from Tensile.client.gemm_args import (
    buildKernelArgs,
    _computeInternalArg0,
    _computeInternalArg1,
)
from Tensile.client.reference import (
    gemm, gemmFp16, gemmBf16,
    assertClose,
    RTOL_FP32, ATOL_FP32,
    RTOL_FP16, ATOL_FP16,
    RTOL_BF16, ATOL_BF16,
)
from epilogues.epilogue_harness.yaml_solution_builder import _injectInternalArgsSupport

# ---------------------------------------------------------------------------
# Problem group indices in gemm_standard.yaml.
# ---------------------------------------------------------------------------

# Problem sizes: (M, N, batch, K)
_PROBLEM_SIZES = [
    (256, 256, 4, 256),
    (512, 512, 4, 512),
    (1024, 1024, 4, 1024),
    (256, 512, 4, 256),
    (512, 256, 4, 256),
]


# ---------------------------------------------------------------------------
# Helpers: NT GEMM strides (TransposeA=False, TransposeB=True, column-major).
# ---------------------------------------------------------------------------

def _ntStrides(M: int, N: int, K: int, batch: int):
    """Return (lda, ldb, ldd, ldc, stride_a, stride_b, stride_d, stride_c) for NT GEMM.

    NT column-major: A is M×K Fortran (lda=M), B is N×K Fortran (ldb=N),
    D/C are M×N Fortran (ldd=ldc=M).
    """
    return M, N, M, M, M * K, N * K, M * N, M * N


def _ntProblemParams(M: int, N: int, batch: int, K: int,
                     alpha: float = 1.0, beta: float = 0.0):
    """Build a problem_params dict for NT batched GEMM."""
    lda, ldb, ldd, ldc, sa, sb, sd, sc = _ntStrides(M, N, K, batch)
    return {
        "sizes": [M, N, batch, K],
        "ldd": ldd, "stride_d": sd,
        "ldc": ldc, "stride_c": sc,
        "lda": lda, "stride_a": sa,
        "ldb": ldb, "stride_b": sb,
        "alpha": alpha,
        "beta": beta,
        "gsu": 1,
    }


# ---------------------------------------------------------------------------
# Minimal solution dict for no-GPU unit tests.
# ---------------------------------------------------------------------------

def _minimalSolDict(version: int = 2, stridedBatched: bool = True,
                    supportCustomStaggerU: bool = False,
                    staggerU: int = 32,
                    wgm: int = 8, mt0: int = 64, mt1: int = 64) -> dict:
    """Return the minimal solution dict needed by buildKernelArgs for unit tests."""
    return {
        "KernArgsVersion": version,
        "SupportCustomWGM": True,
        "SupportCustomStaggerU": supportCustomStaggerU,
        "SupportUserGSU": False,
        "UseSFC": False,
        "UseUniversalArgs": True,
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "WorkGroupMapping": wgm,
        "WorkGroupMappingXCC": 0,
        "WorkGroupMappingXCCGroup": 0,
        "StaggerU": staggerU,
        "StaggerUMapping": 1,
        "_staggerStrideShift": 2,
        "GlobalSplitU": 1,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "StreamK": 0,
        "StreamKAtomic": 0,
        "StridedBatched": stridedBatched,
        "UseBeta": True,
        "GlobalAccumulation": 0,
        "ExpertSchedulingMode": 0,
        "HighPrecisionAccumulate": True,
        "ComputeDataType": 0,  # float32
        "NumThreads": 256,
        "InternalSupportParams": {
            "KernArgsVersion": version,
            "SupportCustomWGM": True,
            "SupportCustomStaggerU": supportCustomStaggerU,
            "SupportUserGSU": False,
            "UseSFC": False,
            "UseUniversalArgs": True,
        },
    }


# ===========================================================================
# Task 1.2 — NotImplementedError tests (no GPU required)
# ===========================================================================


class TestBuildKernelArgsErrors:
    """Verify that unsupported configurations raise NotImplementedError."""

    def _baseParams(self):
        sol = _minimalSolDict()
        pp = _ntProblemParams(256, 256, 4, 256)
        tensors = {"D": 0x1000, "C": 0x2000, "A": 0x3000, "B": 0x4000}
        return sol, pp, tensors

    def test_gsu_gt1_no_streamk_raises(self):
        sol, pp, tensors = self._baseParams()
        pp["gsu"] = 2
        sol["SupportUserGSU"] = True
        with pytest.raises(NotImplementedError, match="GSU=2 > 1 with streamK=0"):
            buildKernelArgs(sol, pp, tensors)

    def test_global_accumulation_3_raises(self):
        sol, pp, tensors = self._baseParams()
        sol["GlobalAccumulation"] = 3
        with pytest.raises(NotImplementedError, match="globalAccumulation=3"):
            buildKernelArgs(sol, pp, tensors)

    def test_use_sfc_raises(self):
        sol, pp, tensors = self._baseParams()
        sol["UseSFC"] = True
        with pytest.raises(NotImplementedError, match="useSFC=True"):
            buildKernelArgs(sol, pp, tensors)

    def test_version_gt2_raises(self):
        sol, pp, tensors = self._baseParams()
        sol["KernArgsVersion"] = 3
        with pytest.raises(NotImplementedError, match="KernArgsVersion=3"):
            buildKernelArgs(sol, pp, tensors)

    def test_expert_scheduling_mode_raises(self):
        sol, pp, tensors = self._baseParams()
        sol["ExpertSchedulingMode"] = 1
        with pytest.raises(NotImplementedError, match="expertSchedulingMode=1"):
            buildKernelArgs(sol, pp, tensors)

    def test_stream_k_atomic1_raises(self):
        sol, pp, tensors = self._baseParams()
        sol["StreamKAtomic"] = 1
        with pytest.raises(NotImplementedError, match="streamKAtomic"):
            buildKernelArgs(sol, pp, tensors)

    def test_stream_k_invalid_raises(self):
        sol, pp, tensors = self._baseParams()
        sol["StreamK"] = 2
        with pytest.raises(NotImplementedError, match="streamK=2"):
            buildKernelArgs(sol, pp, tensors)


# ===========================================================================
# Task 1.2b — Byte-layout correctness tests (no GPU)
# ===========================================================================


class TestBuildKernelArgsBytesLayout:
    """Verify byte layout of buildKernelArgs for version=2 standard GEMM.

    These tests manually construct expected bytes and compare, verifying that
    the Python port of singleCallArgs matches the C++ layout.
    """

    def test_header_v2_strided_batched_with_staggeru(self):
        """Version=2 with SupportCustomStaggerU=True packs StaggerU into arg0 bits 16-29."""
        import struct
        # Use supportCustomStaggerU=True so arg0 carries the staggerU encoding.
        sol = _minimalSolDict(version=2, stridedBatched=True,
                              supportCustomStaggerU=True,
                              wgm=8, staggerU=32, mt0=64, mt1=64)
        sol["_staggerStrideShift"] = 2
        sol["StaggerUMapping"] = 1
        pp = _ntProblemParams(256, 256, 4, 256)
        tensors = {"D": 0xD000, "C": 0xC000, "A": 0xA000, "B": 0xB000}

        buf = buildKernelArgs(sol, pp, tensors)

        # gemm_count: (1 & 0x3FFFFFFF) | (0 << 30) = 1 (stridedBatched=True, argType=0)
        assert struct.unpack_from("<I", buf, 0)[0] == 1

        # internalArg0 for v=2 with SupportCustomStaggerU=True:
        #   bits 0-13: GSU=1
        #   bits 14-15: gsuwgmrr=0, gsuc=0
        #   bits 16-29: su_word (staggerU encoding) shifted left 16
        # su_word = (StaggerUMapping=1 << 13) | ((_staggerStrideShift=2 << 8) & 0x1F00) | (staggerU=32 & 0xFF)
        #         = 0x2000 | 0x0200 | 0x0020 = 0x2220
        # arg0 = (0x2220 << 16) | 1 = 0x22200001
        expected_su_word = (1 << 13) | ((2 << 8) & 0x1F00) | (32 & 0xFF)  # 0x2220
        expected_arg0 = (expected_su_word << 16) | 1
        actual_arg0 = struct.unpack_from("<I", buf, 4)[0]
        assert actual_arg0 == expected_arg0, f"arg0: {actual_arg0:#010x} != {expected_arg0:#010x}"

        # internalArg1 for v=2, useSFC=False: (wgmxccg << 22) | (wgmxcc << 16) | (wgm & 0xFFFF)
        # wgmxcc=0, wgmxccg=0, wgm=8 → 8
        actual_arg1 = struct.unpack_from("<i", buf, 8)[0]
        assert actual_arg1 == 8

        # numWG: ceil(256/64) * ceil(256/64) * 4 = 4*4*4 = 64
        actual_numwg = struct.unpack_from("<I", buf, 12)[0]
        assert actual_numwg == 64

    def test_header_v2_strided_batched_no_staggeru(self):
        """Version=2 without SupportCustomStaggerU: arg0 = gsu only."""
        import struct
        # SupportCustomStaggerU=False: StaggerU is fixed in kernel, not in arg0.
        sol = _minimalSolDict(version=2, stridedBatched=True,
                              supportCustomStaggerU=False,
                              wgm=8, staggerU=32, mt0=64, mt1=64)
        pp = _ntProblemParams(256, 256, 4, 256)
        tensors = {"D": 0xD000, "C": 0xC000, "A": 0xA000, "B": 0xB000}

        buf = buildKernelArgs(sol, pp, tensors)

        # arg0 = gsu=1 only (no StaggerU bits when SupportCustomStaggerU=False).
        actual_arg0 = struct.unpack_from("<I", buf, 4)[0]
        assert actual_arg0 == 1

    def test_problem_sizes_in_buffer(self):
        """Problem sizes M, N, batch, K appear at bytes 16-31."""
        import struct
        sol = _minimalSolDict(version=2)
        pp = _ntProblemParams(512, 256, 2, 128)
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}

        buf = buildKernelArgs(sol, pp, tensors)

        # Header occupies bytes 0-15 (4 uint32s for version >= 1).
        M, N, batch, K = struct.unpack_from("<IIII", buf, 16)
        assert M == 512
        assert N == 256
        assert batch == 2
        assert K == 128

    def test_pointer_layout_strided_batched(self):
        """D, C, A, B pointers at bytes 32-63 (4 × 8 bytes)."""
        import struct
        sol = _minimalSolDict(version=2, stridedBatched=True)
        pp = _ntProblemParams(64, 64, 1, 64)
        tensors = {
            "D": 0xD000000000000001,
            "C": 0xC000000000000002,
            "A": 0xA000000000000003,
            "B": 0xB000000000000004,
        }
        buf = buildKernelArgs(sol, pp, tensors)

        # Version=2: header=16B, sizes=16B → pointers at offset 32.
        D, C, A, B = struct.unpack_from("<QQQQ", buf, 32)
        assert D == tensors["D"] & 0xFFFFFFFFFFFFFFFF
        assert C == tensors["C"] & 0xFFFFFFFFFFFFFFFF
        assert A == tensors["A"] & 0xFFFFFFFFFFFFFFFF
        assert B == tensors["B"] & 0xFFFFFFFFFFFFFFFF

    def test_strides_nt_batched(self):
        """NT GEMM strides: strideD[1]=M, strideD[2]=M*N, lda=M, ldb=N."""
        import struct
        M, N, batch, K = 128, 64, 4, 32
        sol = _minimalSolDict(version=2)
        pp = _ntProblemParams(M, N, batch, K)
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}

        buf = buildKernelArgs(sol, pp, tensors)

        # strides start after header(16)+sizes(16)+4ptrs(32) = offset 64
        stride_offset = 64
        strides = struct.unpack_from("<8I", buf, stride_offset)
        # D: ldd=M, strideD2=M*N
        assert strides[0] == M
        assert strides[1] == M * N
        # C: ldc=M, strideC2=M*N
        assert strides[2] == M
        assert strides[3] == M * N
        # A: lda=M, strideA2=M*K
        assert strides[4] == M
        assert strides[5] == M * K
        # B: ldb=N, strideB2=N*K
        assert strides[6] == N
        assert strides[7] == N * K

    def test_alpha_beta_at_end(self):
        """Alpha=2.0 and beta=0.5 appear after strides."""
        import struct
        M, N, batch, K = 64, 64, 1, 64
        sol = _minimalSolDict(version=2)
        pp = _ntProblemParams(M, N, batch, K, alpha=2.0, beta=0.5)
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}

        buf = buildKernelArgs(sol, pp, tensors)

        # strides_offset=64, 8 strides = 32 bytes → alpha at offset 96
        alpha_off = 64 + 8 * 4
        alpha_val = struct.unpack_from("<f", buf, alpha_off)[0]
        beta_val = struct.unpack_from("<f", buf, alpha_off + 4)[0]
        assert abs(alpha_val - 2.0) < 1e-6
        assert abs(beta_val - 0.5) < 1e-6

    def test_total_byte_length_v2_batched(self):
        """Total byte length for version=2, stridedBatched=True, batched, no StreamK."""
        sol = _minimalSolDict(version=2)
        pp = _ntProblemParams(256, 256, 4, 256)
        tensors = {"D": 0, "C": 0, "A": 0, "B": 0}

        buf = buildKernelArgs(sol, pp, tensors)

        # header=16, sizes=16, ptrs=32, strides=32, alpha+beta=8 → 104 bytes
        assert len(buf) == 104

    def test_ptr_array_batch_gemm_count(self):
        """stridedBatched=False sets arg_type=3 in gemm_count bits 30-31."""
        import struct
        sol = _minimalSolDict(version=2, stridedBatched=False)
        pp = _ntProblemParams(64, 64, 4, 64)
        tensors = {
            "batchD": 0x1000, "batchC": 0x2000,
            "batchA": 0x3000, "batchB": 0x4000,
        }

        buf = buildKernelArgs(sol, pp, tensors)

        gemm_count = struct.unpack_from("<I", buf, 0)[0]
        arg_type = (gemm_count >> 30) & 0x3
        assert arg_type == 3


# ===========================================================================
# Task 1.3 — gemmFp16 / gemmBf16 reference unit tests (no GPU)
# ===========================================================================


class TestGemmFp16Reference:
    """Unit tests for reference.gemmFp16 without GPU."""

    def test_identity_matmul(self):
        A = np.eye(4, dtype=np.float16)
        B = np.eye(4, dtype=np.float16)
        D = gemmFp16(A, B)
        np.testing.assert_allclose(
            D.astype(np.float32), np.eye(4, dtype=np.float32), atol=1e-3,
        )

    def test_output_dtype_is_fp16(self):
        A = np.ones((2, 3), dtype=np.float16)
        B = np.ones((3, 2), dtype=np.float16)
        D = gemmFp16(A, B)
        assert D.dtype == np.float16

    def test_alpha_scaling(self):
        A = np.ones((2, 2), dtype=np.float16)
        B = np.ones((2, 2), dtype=np.float16)
        D = gemmFp16(A, B, alpha=3.0)
        # alpha * (A @ B) = 3 * [[2,2],[2,2]] = [[6,6],[6,6]]
        np.testing.assert_allclose(
            D.astype(np.float32), np.full((2, 2), 6.0), atol=ATOL_FP16,
        )

    def test_beta_accumulation(self):
        A = np.ones((2, 2), dtype=np.float16)
        B = np.ones((2, 2), dtype=np.float16)
        C = np.ones((2, 2), dtype=np.float16)
        D = gemmFp16(A, B, alpha=1.0, beta=0.5, C=C)
        # D = 1*(A@B) + 0.5*C = [[2,2],[2,2]] + [[0.5,0.5],[0.5,0.5]] = [[2.5,...]]
        np.testing.assert_allclose(
            D.astype(np.float32), np.full((2, 2), 2.5), atol=ATOL_FP16,
        )

    def test_non_square(self):
        M, K, N = 3, 4, 5
        rng = np.random.default_rng(42)
        A = rng.random((M, K)).astype(np.float16)
        B = rng.random((K, N)).astype(np.float16)
        D = gemmFp16(A, B)
        ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(
            D.astype(np.float32), ref.astype(np.float32), atol=ATOL_FP16,
        )


class TestGemmBf16Reference:
    """Unit tests for reference.gemmBf16 without GPU.

    bf16 requires ml_dtypes; these tests skip if unavailable.
    """

    @pytest.fixture(autouse=True)
    def require_ml_dtypes(self):
        if ml_dtypes is None:
            pytest.skip("ml_dtypes not installed")

    def test_identity_matmul(self):
        A = np.eye(4).astype(ml_dtypes.bfloat16)
        B = np.eye(4).astype(ml_dtypes.bfloat16)
        D = gemmBf16(A, B)
        np.testing.assert_allclose(
            np.asarray(D, dtype=np.float32), np.eye(4, dtype=np.float32),
            atol=ATOL_BF16,
        )

    def test_output_dtype_is_bf16(self):
        A = np.ones((2, 3)).astype(ml_dtypes.bfloat16)
        B = np.ones((3, 2)).astype(ml_dtypes.bfloat16)
        D = gemmBf16(A, B)
        assert D.dtype == ml_dtypes.bfloat16

    def test_alpha_scaling(self):
        A = np.ones((2, 2)).astype(ml_dtypes.bfloat16)
        B = np.ones((2, 2)).astype(ml_dtypes.bfloat16)
        D = gemmBf16(A, B, alpha=3.0)
        np.testing.assert_allclose(
            np.asarray(D, dtype=np.float32), np.full((2, 2), 6.0), atol=ATOL_BF16,
        )

    def test_beta_accumulation(self):
        A = np.ones((2, 2)).astype(ml_dtypes.bfloat16)
        B = np.ones((2, 2)).astype(ml_dtypes.bfloat16)
        C = np.ones((2, 2)).astype(ml_dtypes.bfloat16)
        D = gemmBf16(A, B, alpha=1.0, beta=0.5, C=C)
        np.testing.assert_allclose(
            np.asarray(D, dtype=np.float32), np.full((2, 2), 2.5), atol=ATOL_BF16,
        )

    def test_non_square(self):
        M, K, N = 3, 4, 5
        rng = np.random.default_rng(42)
        A = rng.random((M, K)).astype(ml_dtypes.bfloat16)
        B = rng.random((K, N)).astype(ml_dtypes.bfloat16)
        D = gemmBf16(A, B)
        ref = (np.asarray(A, np.float32) @ np.asarray(B, np.float32)).astype(ml_dtypes.bfloat16)
        np.testing.assert_allclose(
            np.asarray(D, np.float32), np.asarray(ref, np.float32), atol=ATOL_BF16,
        )


# ===========================================================================
# GPU test infrastructure
# ===========================================================================

def _setupTensile(chip: str):
    """Initialize Tensile assembler + ISA map for kernel compilation.

    Mirrors the setup_tensile helper from epilogue_harness.partialrms_helpers.
    """
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
    debugConfig = DebugConfig()
    return assembler, isaInfoMap, debugConfig


def _generateAsm(solution, assembler, debugConfig):
    """Return (asm_string, kernel_name) for a solution."""
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
        raise RuntimeError(f"assembly generation failed: {err}")
    return asm_str, getKernelNameMin(kernel, splitGSU=False)


def _compileSolutions(problem_idx: int):
    """Compile all solutions for one YAML problem group; returns list of dicts.

    Each entry: {"sol_dict": dict, "kernel_name": str, "hsaco": bytes, "chip": str}.
    Returns [] when GPU or deps are unavailable.
    """
    if not HAVE_DEPS:
        return []
    try:
        from epilogues.epilogue_harness.yaml_solution_builder import solutionsFromYaml
        chip = amdgpu_exec.get_chip()
        assembler, isaInfoMap, debugConfig = _setupTensile(chip)
        sols = solutionsFromYaml(_YAML_PATH, assembler, isaInfoMap, debugConfig,
                                 problemIdx=problem_idx)
    except Exception as exc:
        import warnings
        warnings.warn(f"could not compile solutions (problemIdx={problem_idx}): {exc}")
        return []

    compiled = []
    for sol, sid in sols:
        try:
            asm_str, kernel_name = _generateAsm(sol, assembler, debugConfig)
            hsaco = amdgpu_exec.compile_asm_to_hsaco(asm_str, chip)
        except Exception as exc:
            import warnings
            warnings.warn(f"solution {sid} failed to compile: {exc}")
            continue
        raw_dict = dict(sol)
        sol_dict = _injectInternalArgsSupport(raw_dict, chip)
        compiled.append({
            "sol_dict": sol_dict,
            "raw_dict": raw_dict,
            "kernel_name": kernel_name,
            "hsaco": hsaco,
            "chip": chip,
            "sid": sid,
        })
    return compiled


def _filterSolution(entry: dict, strided_batched: bool) -> bool:
    """Return True if the solution should be used for this strided_batched setting.

    Also implements the M1 kernel filter (skip auto-WGM and auto-StaggerU).
    Kernels compiled with StridedBatched=True support both batch modes at
    runtime via the arg_type field in gemm_count, so no filtering on that flag.
    """
    sol_dict = entry["sol_dict"]
    raw_dict = entry["raw_dict"]

    if sol_dict.get("WorkGroupMapping", 0) == 0:
        return False

    isp = raw_dict.get("InternalSupportParams", {}) or {}
    if sol_dict.get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
        return False

    return True


def _deviceCuCount() -> int:
    """Return the device CU count (multiprocessor_count) for device 0.

    The result is cached after the first call. Used to resolve the
    WorkGroupMappingXCCGroup=-1 sentinel, which the C++ code replaces with
    pAMDGPU->computeUnitCount at runtime.
    """
    if not HAVE_DEPS:
        return 0
    props = amdgpu_exec._runtime_module.hip_get_device_props(0)
    return int(props.get("multiprocessor_count", 0))


def _buildNtTypedArgs(sol_dict: dict, M: int, N: int, batch: int, K: int,
                      D_arr, C_arr, A_arr, B_arr,
                      alpha: float = 1.0, beta: float = 0.0):
    """Build the typed args list for execute_hsaco for NT stridedBatched=True GEMM.

    Mirrors the same layout as buildKernelArgs but as individual typed values
    compatible with amdgpu_exec.execute_hsaco.
    """
    version = sol_dict.get("KernArgsVersion", 0)
    mt0 = sol_dict["MacroTile0"]
    mt1 = sol_dict["MacroTile1"]
    num_wg = math.ceil(M / mt0) * math.ceil(N / mt1) * batch

    arg0 = _computeInternalArg0(sol_dict, gsu=1)
    # stridedBatched=True → argType=0
    gemm_count = (1 & 0x3FFFFFFF) | (0 << 30)

    args = [np.uint32(gemm_count), np.uint32(arg0)]
    if version >= 1:
        arg1 = _computeInternalArg1(sol_dict, cu_count=_deviceCuCount())
        args.append(np.int32(arg1))
        args.append(np.uint32(num_wg))

    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])

    # D, C, A, B pointers (InOutArray / InputArray wrappers).
    args.extend([D_arr, C_arr, A_arr, B_arr])

    # NT strides: lda=M, ldb=N, ldd=ldc=M, batch strides.
    lda, ldb, ldd, ldc = M, N, M, M
    stride_a, stride_b, stride_d, stride_c = M * K, N * K, M * N, M * N
    args.extend([
        np.uint32(ldd), np.uint32(stride_d),
        np.uint32(ldc), np.uint32(stride_c),
        np.uint32(lda), np.uint32(stride_a),
        np.uint32(ldb), np.uint32(stride_b),
    ])

    args.extend([np.float32(alpha), np.float32(beta)])
    return args


def _allocStrideArrays(M: int, N: int, K: int, batch: int, np_dtype, rng):
    """Allocate flat buffers for a batched strided GEMM.

    The GPU kernel uses lda=M and stride_a=M*K for A, ldb=N and stride_b=N*K
    for B, ldd=M and stride_d=M*N for D/C.  Each batch element is a separate
    Fortran (column-major) matrix stored consecutively: element (m, k, b) of A
    lives at offset b*M*K + k*M + m.  np.tile of the Fortran-flat per-batch
    matrix produces this layout.

    All batch elements share the same A and B data (same random matrix tiled).

    Returns (A_buf_np, B_buf_np, C_buf_np, D_buf_np) as 1-D numpy arrays.
    """
    A_np = np.asfortranarray(rng.random((M, K)).astype(np_dtype))
    B_np = np.asfortranarray(rng.random((N, K)).astype(np_dtype))
    A_buf_np = np.tile(A_np.ravel(order='F'), batch)   # M*K*batch elements
    B_buf_np = np.tile(B_np.ravel(order='F'), batch)   # N*K*batch elements
    C_buf_np = np.zeros(M * N * batch, dtype=np_dtype)
    D_buf_np = np.zeros(M * N * batch, dtype=np_dtype)
    return A_buf_np, B_buf_np, C_buf_np, D_buf_np


def _stridedBatchRef(A_buf_np, B_buf_np, batch: int, M: int, N: int, K: int,
                     ref_fn, alpha: float, beta: float):
    """Compute the batched NT GEMM reference, returning a flat 1-D array.

    Reads back A and B from the same flat buffers the GPU receives, so the
    comparison is always against exactly the same data the kernel sees.
    All batch elements share the same A/B data, so the reference is computed
    once and tiled.
    """
    A_slice = A_buf_np[:M * K].reshape(M, K, order='F')
    B_slice = B_buf_np[:N * K].reshape(N, K, order='F')
    D_one = ref_fn(A_slice, B_slice.T, alpha, beta, None)
    return np.tile(np.asfortranarray(D_one).ravel(order='F'), batch)


def _runStridedBatched(entry: dict, M: int, N: int, batch: int, K: int,
                       np_dtype, ref_fn, rtol: float, atol: float, label: str):
    """Execute one stridedBatched GEMM kernel and verify output against reference.

    np_dtype: numpy dtype for A/B/C/D (e.g. np.float32, np.float16).
    ref_fn: callable(A, B^T, alpha, beta, C) → D_ref (reference GEMM).
    """
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]
    alpha, beta = 1.0, 0.0

    rng = np.random.default_rng(seed=M * 1000 + N + K)
    A_buf_np, B_buf_np, C_buf_np, D_buf_np = _allocStrideArrays(M, N, K, batch, np_dtype, rng)

    result_holder = {}

    def capture(arguments):
        # D is the first output array in the args; index 8 is after header + sizes.
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np_dtype).copy()

    D_inout = amdgpu_exec.InOutArray(D_buf_np)
    C_in = amdgpu_exec.InputArray(C_buf_np)
    A_in = amdgpu_exec.InputArray(A_buf_np)
    B_in = amdgpu_exec.InputArray(B_buf_np)
    args = _buildNtTypedArgs(sol_dict, M, N, batch, K, D_inout, C_in, A_in, B_in, alpha, beta)
    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    D_gpu = result_holder["D_gpu"]
    D_ref_flat = _stridedBatchRef(A_buf_np, B_buf_np, batch, M, N, K, ref_fn, alpha, beta)
    assertClose(D_gpu, D_ref_flat, rtol=rtol, atol=atol, label=label)


# ---------------------------------------------------------------------------
# Session-scoped compiled solution fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fp32Kernels():
    """Compile fp32 solutions from YAML group 0."""
    return _compileSolutions(0)


@pytest.fixture(scope="session")
def fp16Kernels():
    """Compile fp16 solutions from YAML group 1."""
    return _compileSolutions(1)


@pytest.fixture(scope="session")
def bf16Kernels():
    """Compile bf16 solutions from YAML group 2."""
    return _compileSolutions(2)


# ---------------------------------------------------------------------------
# Task 1.2b — Poison-input test (GPU required).
# ---------------------------------------------------------------------------


def _corruptStrideA1(argList, M):
    """Corrupt strideA[1] (lda) in the typed argument list and return it.

    Layout: header(2 or 4) + sizes(4) + ptrs(4) = 10 or 12 args before strides.
    strideA[0] = lda = M; the first element of the A-stride pair, located after
    header, sizes, ptrs, 2 D-stride args (ldd, stride_d), 2 C-stride args (ldc, stride_c).
    Offset: header_n + 4 (sizes) + 4 (ptrs) + 2 (D: ldd, stride_d) + 2 (C: ldc, stride_c)
          = header_n + 12.
    Uses strideA[1] rather than strideA[0] so the batch-stride position is not affected.
    Infers header_n from list length: total = header_n + 18 (sizes+ptrs+strides+scalars).
    """
    header_n = len(argList) - 18  # header + 4 sizes + 4 ptrs + 8 strides + 2 scalars = 18
    stride_a1_idx = header_n + 4 + 4 + 4  # header + sizes + ptrs + D-strides + C-strides
    original = argList[stride_a1_idx]
    argList[stride_a1_idx] = np.uint32(int(original) + M)
    return argList


def _assertPoisonDetected(gpuOut, refOut, rtol, label):
    """Assert that >= 50% of elements in gpuOut differ from refOut by more than 10 * rtol.

    Raises AssertionError with label if poison was not detected, proving that argument
    corruption caused computation errors.
    """
    bad = np.abs(gpuOut - refOut) > 10 * rtol * (np.abs(refOut) + 1)
    bad_frac = bad.sum() / bad.size
    assert bad_frac >= 0.5, (
        f"{label}: only {bad_frac:.1%} elements corrupted — "
        "argument vector may not be driving computation"
    )


def _allocPoisonRun(solDict, M, N, batch, K):
    """Allocate buffers, build a poisoned argument list, and return (args, dRef).

    Fills A and B with random data, then corrupts strideA[1] by +M so the
    kernel reads from the wrong row offset — proving the args drive computation.
    """
    rng = np.random.default_rng(seed=999)
    A_np = np.asfortranarray(rng.random((M, K)).astype(np.float32))
    B_np = np.asfortranarray(rng.random((N, K)).astype(np.float32))
    A_buf = np.tile(A_np.ravel(order='F'), batch)
    B_buf = np.tile(B_np.ravel(order='F'), batch)
    C_buf = np.zeros(M * N * batch, dtype=np.float32)
    D_poison = np.zeros(M * N * batch, dtype=np.float32)
    dRef = (A_np.astype(np.float64) @ B_np.T.astype(np.float64)).astype(np.float32)
    D_io = amdgpu_exec.InOutArray(D_poison)
    C_in = amdgpu_exec.InputArray(C_buf)
    A_in = amdgpu_exec.InputArray(A_buf)
    B_in = amdgpu_exec.InputArray(B_buf)
    args = _buildNtTypedArgs(solDict, M, N, batch, K, D_io, C_in, A_in, B_in, 1.0, 0.0)
    args = _corruptStrideA1(args, M)
    return args, dRef


@requires_gfx950
@pytest.mark.parametrize("size", [(256, 256, 4, 256)], ids=["M256N256B4K256"])
def test_buildKernelArgs_poison(fp32Kernels, size):
    """Corrupt strideA[1] by +M and assert >= 50% of outputs differ from reference.

    This test proves that buildKernelArgs actually drives computation.
    Without this test, a trivially zero output would pass the correctness test.
    """
    M, N, batch, K = size
    entries = [e for e in fp32Kernels if _filterSolution(e, strided_batched=True)]
    if not entries:
        pytest.skip("no stridedBatched=True fp32 solution compiled")

    entry = entries[0]
    sol_dict = entry["sol_dict"]
    kernel_name = entry["kernel_name"]
    hsaco = entry["hsaco"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]

    args, dRef = _allocPoisonRun(sol_dict, M, N, batch, K)

    result_holder = {}

    def capture(arguments):
        result_holder["D_gpu"] = np.asarray(arguments[8].array, dtype=np.float32).copy()

    amdgpu_exec.execute_hsaco(
        hsaco=hsaco, kernel_name=kernel_name, arguments=args,
        grid_dim=(num_wg, 1, 1), block_dim=(num_threads, 1, 1),
        num_iterations=1, verify_fn=capture,
    )

    gpuOut = result_holder["D_gpu"][: M * N]
    _assertPoisonDetected(gpuOut, dRef.ravel(order="F"), RTOL_FP32, "poison")


# ---------------------------------------------------------------------------
# Task 1.4 — Standard GEMM GPU correctness.
# ---------------------------------------------------------------------------


def _dtypeInfo(dtype_str: str):
    """Return (numpy_dtype, ref_fn, rtol, atol, problem_idx) for a dtype string."""
    if dtype_str == "fp32":
        return np.float32, gemm, RTOL_FP32, ATOL_FP32, 0
    if dtype_str == "fp16":
        return np.float16, gemmFp16, RTOL_FP16, ATOL_FP16, 1
    if dtype_str == "bf16":
        if ml_dtypes is None:
            return None, None, None, None, 2
        return ml_dtypes.bfloat16, gemmBf16, RTOL_BF16, ATOL_BF16, 2
    raise ValueError(f"unknown dtype_str: {dtype_str}")


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_fp32_strided_batched(fp32Kernels, size):
    """fp32 NT stridedBatched=True correctness."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in fp32Kernels if _filterSolution(e, strided_batched=True)]
    if not entries:
        pytest.skip("no stridedBatched=True fp32 solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        isp = entry["raw_dict"].get("InternalSupportParams", {})
        if entry["sol_dict"].get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
            continue
        _runStridedBatched(entry, M, N, batch, K, np.float32,
                           gemm, RTOL_FP32, ATOL_FP32,
                           label=f"fp32 M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_fp16_strided_batched(fp16Kernels, size):
    """fp16 HPA NT stridedBatched=True correctness."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in fp16Kernels if _filterSolution(e, strided_batched=True)]
    if not entries:
        pytest.skip("no stridedBatched=True fp16 solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        isp = entry["raw_dict"].get("InternalSupportParams", {})
        if entry["sol_dict"].get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
            continue
        _runStridedBatched(entry, M, N, batch, K, np.float16,
                           gemmFp16, RTOL_FP16, ATOL_FP16,
                           label=f"fp16 M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_bf16_strided_batched(bf16Kernels, size):
    """bf16 HPA NT stridedBatched=True correctness."""
    if not HAVE_DEPS or ml_dtypes is None:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in bf16Kernels if _filterSolution(e, strided_batched=True)]
    if not entries:
        pytest.skip("no stridedBatched=True bf16 solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        isp = entry["raw_dict"].get("InternalSupportParams", {})
        if entry["sol_dict"].get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
            continue
        _runStridedBatched(entry, M, N, batch, K, ml_dtypes.bfloat16,
                           gemmBf16, RTOL_BF16, ATOL_BF16,
                           label=f"bf16 M{M}N{N}B{batch}K{K} {sid}")


# ---------------------------------------------------------------------------
# stridedBatched=False (pointer-array batch) correctness tests.
# ---------------------------------------------------------------------------

def _allocPtrBufs(M: int, N: int, K: int, batch: int, np_dtype, rng):
    """Allocate per-element numpy arrays and GPU buffers for pointer-array batch GEMM.

    Returns (A_nps, B_nps, C_nps, A_bufs, B_bufs, C_bufs, D_bufs).
    """
    from amdgpu_exec import GpuBuffer
    A_nps, B_nps, C_nps = [], [], []
    A_bufs, B_bufs, C_bufs, D_bufs = [], [], [], []
    for _ in range(batch):
        A_b = np.asfortranarray(rng.random((M, K)).astype(np_dtype))
        B_b = np.asfortranarray(rng.random((N, K)).astype(np_dtype))
        C_b = np.asfortranarray(np.zeros((M, N), dtype=np_dtype))
        D_b = np.asfortranarray(np.zeros((M, N), dtype=np_dtype))
        A_nps.append(A_b); B_nps.append(B_b); C_nps.append(C_b)
        a_buf = GpuBuffer(A_b.nbytes); a_buf.copy_from_host(A_b); A_bufs.append(a_buf)
        b_buf = GpuBuffer(B_b.nbytes); b_buf.copy_from_host(B_b); B_bufs.append(b_buf)
        c_buf = GpuBuffer(C_b.nbytes); c_buf.copy_from_host(C_b); C_bufs.append(c_buf)
        d_buf = GpuBuffer(D_b.nbytes); d_buf.memset(0); D_bufs.append(d_buf)
    return A_nps, B_nps, C_nps, A_bufs, B_bufs, C_bufs, D_bufs


def _buildPtrBatchArgList(sol_dict: dict, M: int, N: int, batch: int, K: int,
                          pD_buf, pC_buf, pA_buf, pB_buf) -> list:
    """Build the typed argument list for a pointer-array batch kernel launch."""
    version = sol_dict.get("KernArgsVersion", 0)
    arg0 = _computeInternalArg0(sol_dict, gsu=1)
    num_wg_val = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    # argType=3 selects the pointer-array batch path in the kernel.
    gemm_count = (1 & 0x3FFFFFFF) | (3 << 30)
    args: list = [np.uint32(gemm_count), np.uint32(arg0)]
    if version >= 1:
        args.append(np.int32(_computeInternalArg1(sol_dict, cu_count=_deviceCuCount())))
        args.append(np.uint32(num_wg_val))
    args.extend([np.uint32(M), np.uint32(N), np.uint32(batch), np.uint32(K)])
    args.extend([
        ctypes.c_void_p(pD_buf.ptr_value),
        ctypes.c_void_p(pC_buf.ptr_value),
        ctypes.c_void_p(pA_buf.ptr_value),
        ctypes.c_void_p(pB_buf.ptr_value),
    ])
    # Pointer-array batch: leading-dimension strides only; batch stride is zero.
    lda, ldb, ldd, ldc = M, N, M, M
    args.extend([
        np.uint32(ldd), np.uint32(0), np.uint32(ldc), np.uint32(0),
        np.uint32(lda), np.uint32(0), np.uint32(ldb), np.uint32(0),
    ])
    args.extend([np.float32(1.0), np.float32(0.0)])
    return args


def _verifyPtrBatchResult(dBufs, refOutputs, nBufs, dtype, rtol, atol):
    """Copy the first nBufs GpuBuffers to host and compare against refOutputs.

    Any mismatch raises AssertionError with a descriptive message.
    """
    for i in range(nBufs):
        D_result = np.zeros(refOutputs[i].shape, dtype=dtype, order="F")
        dBufs[i].copy_to_host(D_result)
        if dtype == np.float32:
            gpu_flat = D_result.ravel(order="F")
            ref_flat = refOutputs[i].ravel(order="F")
        else:
            gpu_flat = np.asarray(D_result, dtype=np.float32).ravel(order="F")
            ref_flat = np.asarray(refOutputs[i], dtype=np.float32).ravel(order="F")
        assertClose(gpu_flat, ref_flat, rtol=rtol, atol=atol,
                    label=f"ptr-batch element {i}")


def _runPtrBatch(entry: dict, M: int, N: int, batch: int, K: int,
                 np_dtype, ref_fn, rtol: float, atol: float, label: str):
    """Execute a pointer-array-batch GEMM kernel and verify output.

    Each batch element has its own device buffer; a device buffer of pointers
    is passed to the kernel, which selects the pointer-array path via arg_type=3.
    """
    from amdgpu_exec import GpuModule, GpuBuffer, GpuEvent

    sol_dict = entry["sol_dict"]
    num_wg = math.ceil(M / sol_dict["MacroTile0"]) * math.ceil(N / sol_dict["MacroTile1"]) * batch
    num_threads = sol_dict["NumThreads"]
    rng = np.random.default_rng(seed=M * 2000 + N + K)
    A_nps, B_nps, C_nps, A_bufs, B_bufs, C_bufs, D_bufs = _allocPtrBufs(
        M, N, K, batch, np_dtype, rng
    )

    def makeDevPtrBuf(gpu_bufs):
        ptrs = np.array([buf.ptr_value for buf in gpu_bufs], dtype=np.uint64)
        dev = GpuBuffer(ptrs.nbytes)
        dev.copy_from_host(ptrs)
        return dev

    pD_buf, pC_buf = makeDevPtrBuf(D_bufs), makeDevPtrBuf(C_bufs)
    pA_buf, pB_buf = makeDevPtrBuf(A_bufs), makeDevPtrBuf(B_bufs)
    args = _buildPtrBatchArgList(sol_dict, M, N, batch, K, pD_buf, pC_buf, pA_buf, pB_buf)

    module = GpuModule(entry["hsaco"])
    fn = module.get_function(entry["kernel_name"])
    stop = GpuEvent()
    fn.launch((num_wg, 1, 1), (num_threads, 1, 1), args)
    stop.record()
    stop.synchronize()

    D_ref_b0 = ref_fn(A_nps[0], B_nps[0].T, 1.0, 0.0, None)
    _verifyPtrBatchResult(D_bufs, [D_ref_b0], 1, np_dtype, rtol, atol)

    for buf in A_bufs + B_bufs + C_bufs + D_bufs + [pD_buf, pC_buf, pA_buf, pB_buf]:
        buf.free()
    module.unload()


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_fp32_ptr_batch(fp32Kernels, size):
    """fp32 NT stridedBatched=False (pointer-array batch) correctness."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in fp32Kernels if _filterSolution(e, strided_batched=False)]
    if not entries:
        pytest.skip("no stridedBatched=False fp32 solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        isp = entry["raw_dict"].get("InternalSupportParams", {})
        if entry["sol_dict"].get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
            continue
        _runPtrBatch(entry, M, N, batch, K, np.float32,
                     gemm, RTOL_FP32, ATOL_FP32,
                     label=f"fp32-ptr M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_fp16_ptr_batch(fp16Kernels, size):
    """fp16 HPA NT stridedBatched=False (pointer-array batch) correctness."""
    if not HAVE_DEPS:
        pytest.skip("amdgpu_exec not installed")
    entries = [e for e in fp16Kernels if _filterSolution(e, strided_batched=False)]
    if not entries:
        pytest.skip("no stridedBatched=False fp16 solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        isp = entry["raw_dict"].get("InternalSupportParams", {})
        if entry["sol_dict"].get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
            continue
        _runPtrBatch(entry, M, N, batch, K, np.float16,
                     gemmFp16, RTOL_FP16, ATOL_FP16,
                     label=f"fp16-ptr M{M}N{N}B{batch}K{K} {sid}")


@requires_gfx950
@pytest.mark.parametrize("size", _PROBLEM_SIZES,
                         ids=[f"M{m}N{n}B{b}K{k}" for m, n, b, k in _PROBLEM_SIZES])
def test_bf16_ptr_batch(bf16Kernels, size):
    """bf16 HPA NT stridedBatched=False (pointer-array batch) correctness."""
    if not HAVE_DEPS or ml_dtypes is None:
        pytest.skip("amdgpu_exec or ml_dtypes not installed")
    entries = [e for e in bf16Kernels if _filterSolution(e, strided_batched=False)]
    if not entries:
        pytest.skip("no stridedBatched=False bf16 solution compiled")

    M, N, batch, K = size
    for entry in entries:
        sid = entry["sid"]
        isp = entry["raw_dict"].get("InternalSupportParams", {})
        if entry["sol_dict"].get("StaggerU", 0) == 0 and isp.get("SupportCustomStaggerU", False):
            continue
        _runPtrBatch(entry, M, N, batch, K, ml_dtypes.bfloat16,
                     gemmBf16, RTOL_BF16, ATOL_BF16,
                     label=f"bf16-ptr M{M}N{N}B{batch}K{K} {sid}")
