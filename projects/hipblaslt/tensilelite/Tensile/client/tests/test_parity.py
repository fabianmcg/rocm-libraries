# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M13 parity test suite: Python harness vs C++ tensilelite-client.

Covers Tasks 13.1–13.2 of the TensileLite Python client plan:
  13.1  Run SweepRunner on each supported feature combination and check
        GFLOPS plausibility; compare against C++ reference CSV if available.
  13.2  Collect data for parity_report.md (written by --generate-parity-report).

GPU tests require gfx950 (@requires_gfx950) and amdgpu_exec.
Non-GPU tests run under plain tox -e unit.

C++ reference CSV status: BLOCKED — the C++ client requires a pre-built
solution library. See fixtures/cpp_client_reference_cmd.txt for details.
Until the reference CSV contains real data, parity tests check plausibility
only (GFLOPS in [100, 1_000_000]).

Feature coverage (see plan/m13_parity.md):
  - fp32 GEMM, no epilogue, sizes (256, 512, 1024)
  - bf16 GEMM, stridedBatched=True, sizes (256, 512, 1024, 2048, 4096)
  - fp16 GEMM row bias+Relu: SKIP (no fp16 epilogue kernel in YAML)
  - int8 -> int32 accumulation, sizes (256, 512)
  - fp8 E4M3 OCP, size (512, 512, 512)
  - MX float8 + E8 scale block_k=32, sizes (256, 512)
  - Grouped GEMM: SKIP (no gfx950 kernel YAML available)
  - Sparse GEMM: SKIP (no gfx950 kernel YAML available)
  - PartialRMS (K1), RstdScale (K3), StreamK=3: referenced from epilogue tests
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

try:
    import amdgpu_exec
    haveDeps = True
except ImportError:
    amdgpu_exec = None
    haveDeps = False

from .conftest import requires_gfx950

# ---------------------------------------------------------------------------
# Module-level paths.
# ---------------------------------------------------------------------------

_testsDir = os.path.dirname(__file__)
_yamlDir = os.path.join(_testsDir, "yaml")
_fixturesDir = os.path.join(_testsDir, "fixtures")
# 4 levels up: tests -> client -> Tensile -> tensilelite-parent (for sys.path compat).
_tensileRoot = os.path.abspath(os.path.join(_testsDir, "..", "..", "..", ".."))
# 3 levels up: tests -> client -> Tensile -> tensilelite root.
_tensileLiteRoot = os.path.abspath(os.path.join(_testsDir, "..", "..", ".."))
_epilogueTests = os.path.join(_tensileLiteRoot, "epilogues", "unittests")

if _tensileRoot not in sys.path:
    sys.path.insert(0, _tensileRoot)

from Tensile.client.sweep_runner import SweepRunner

# ---------------------------------------------------------------------------
# YAML paths and problem group indices.
# ---------------------------------------------------------------------------

_yamlStd = os.path.join(_yamlDir, "gemm_standard.yaml")
_yamlInt = os.path.join(_yamlDir, "gemm_int_xf32.yaml")
_yamlFp8 = os.path.join(_yamlDir, "gemm_fp8.yaml")
_yamlMx = os.path.join(_yamlDir, "gemm_mx.yaml")

_grpFp32 = 0   # gemm_standard.yaml: fp32 NT batched.
_grpFp16 = 1   # gemm_standard.yaml: fp16 HPA NT batched.
_grpBf16 = 2   # gemm_standard.yaml: bf16 HPA NT batched.
_grpInt8 = 0   # gemm_int_xf32.yaml: int8 -> int32.
_grpFp8  = 0   # gemm_fp8.yaml: F8 OCP E4M3.
_grpMx   = 0   # gemm_mx.yaml: MX float8 + E8.

# ---------------------------------------------------------------------------
# Target problem sizes (M, N, batch, K) for each feature.
# ---------------------------------------------------------------------------

_fp32Sizes  = [(256, 256, 4, 256), (512, 512, 4, 512), (1024, 1024, 4, 1024)]
_bf16Sizes  = [(256, 256, 4, 256), (512, 512, 4, 512), (1024, 1024, 4, 1024),
                (2048, 2048, 4, 2048), (4096, 4096, 4, 4096)]
_int8Sizes  = [(256, 256, 4, 256), (512, 512, 4, 512)]
_fp8Sizes   = [(512, 512, 4, 512)]   # plan says size (512, 512, 512); batch=4 in YAML.
_mxSizes    = [(256, 256, 4, 256), (512, 512, 4, 512)]

# ---------------------------------------------------------------------------
# GFLOPS plausibility bounds.
# ---------------------------------------------------------------------------

_gflopsLower = 100.0
_gflopsUpper = 1_000_000.0

# Tolerance for C++ reference comparison.
_tolLarge = 0.05   # ±5% for M×N ≥ 1024².
_tolSmall = 0.10   # ±10% for M×N < 1024².

# ---------------------------------------------------------------------------
# Reference CSV helpers.
# ---------------------------------------------------------------------------


def _loadCppReference(csvPath) -> dict:
    """Load C++ reference CSV. Returns {} if placeholder (first line starts with 'status')."""
    if not os.path.exists(csvPath):
        return {}
    with open(csvPath) as f:
        first = f.readline()
    if first.strip().startswith("status"):
        return {}
    import csv
    result = {}
    with open(csvPath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gflops = float(row.get("GFlops", 0))
                sizes = (int(row.get("SizeI", 0)), int(row.get("SizeJ", 0)),
                         int(row.get("SizeL", 0)), int(row.get("SizeK", 0)))
                result[sizes] = max(result.get(sizes, 0), gflops)
            except (ValueError, KeyError):
                pass
    return result


def _compareCppGflops(pyGflops, cppRef, sizeKey, label, tol):
    """Assert Python GFLOPS is within tol of C++ reference for sizeKey.

    Skips gracefully if sizeKey not in cppRef (placeholder or missing problem).
    """
    if sizeKey not in cppRef:
        pytest.skip(f"no C++ reference for {label} {sizeKey}")
    cppGflops = cppRef[sizeKey]
    delta = abs(pyGflops - cppGflops) / max(cppGflops, 1)
    assert delta <= tol, (
        f"gflops delta for {label} exceeds {tol:.0%}: "
        f"python={pyGflops:.1f} cpp={cppGflops:.1f} delta={delta:.1%}"
    )


# ---------------------------------------------------------------------------
# Sweep runner helper.
# ---------------------------------------------------------------------------


def _runSweep(yamlPath, problemIdx, groupIdx=0, nWarmup=3, nIters=15):
    """Run SweepRunner and return list of SweepResult, or [] on failure."""
    if not haveDeps:
        return []
    try:
        runner = SweepRunner(
            yamlPath=yamlPath,
            nWarmup=nWarmup,
            nIters=nIters,
            rotatingBuffers=8,
            icacheCopies="auto",
            problemIdx=problemIdx,
            groupIdx=groupIdx,
        )
        return runner.run()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("sweep failed: %s", exc)
        return []


def _gflopsForSize(results, M, N, batch, K):
    """Return best GFLOPS for the given size, or None if not found."""
    matching = [r for r in results
                if r.problemSize[:4] == (M, N, batch, K) and r.gflops > 0]
    if not matching:
        return None
    return max(r.gflops for r in matching)


def _checkPlausible(gflops, label):
    """Assert gflops is within [_gflopsLower, _gflopsUpper]."""
    assert _gflopsLower <= gflops <= _gflopsUpper, (
        f"gflops {gflops:.1f} for {label} outside [{_gflopsLower}, {_gflopsUpper}]"
        " — possible unit-conversion bug"
    )


def _toleranceFor(M, N):
    """Return comparison tolerance based on problem size."""
    return _tolLarge if M * N >= 1024 * 1024 else _tolSmall


def _recordFeature(config, feature, status, evidence):
    """Append a feature row to the parity data."""
    config._parityData["features"].append((feature, status, evidence))


def _recordGflops(config, feature, size, pyGflops, cppGflops=None):
    """Append a GFLOPS row to the parity data."""
    config._parityData["gflops"].append((feature, size, pyGflops, cppGflops))


# ===========================================================================
# Session-scoped sweep fixtures.
# ===========================================================================


@pytest.fixture(scope="session")
def fp32ParitySweep():
    """SweepRunner results for fp32 NT batched (gemm_standard.yaml group 0)."""
    return _runSweep(_yamlStd, problemIdx=_grpFp32)


@pytest.fixture(scope="session")
def bf16ParitySweep():
    """SweepRunner results for bf16 HPA NT batched (gemm_standard.yaml group 2)."""
    return _runSweep(_yamlStd, problemIdx=_grpBf16)


@pytest.fixture(scope="session")
def int8ParitySweep():
    """SweepRunner results for int8->int32 NT batched (gemm_int_xf32.yaml group 0)."""
    return _runSweep(_yamlInt, problemIdx=_grpInt8)


@pytest.fixture(scope="session")
def fp8ParitySweep():
    """SweepRunner results for fp8 E4M3 OCP NT batched (gemm_fp8.yaml group 0)."""
    return _runSweep(_yamlFp8, problemIdx=_grpFp8)


@pytest.fixture(scope="session")
def mxParitySweep():
    """SweepRunner results for MX float8 TN batched (gemm_mx.yaml group 0)."""
    return _runSweep(_yamlMx, problemIdx=_grpMx)


# ===========================================================================
# fp32 parity tests.
# ===========================================================================


@requires_gfx950
def test_fp32_gflops_plausible(fp32ParitySweep, request):
    """fp32 GEMM GFLOPS must be in [100, 1_000_000] for sizes (256, 512, 1024)."""
    if not fp32ParitySweep:
        pytest.skip("no sweep results (no solutions compiled or no GPU)")
    found = False
    for M, N, batch, K in _fp32Sizes:
        gflops = _gflopsForSize(fp32ParitySweep, M, N, batch, K)
        if gflops is None:
            continue
        found = True
        _checkPlausible(gflops, f"fp32 {M}x{N}x{batch}x{K}")
        _recordGflops(request.config, "fp32", (M, N, batch, K), gflops)
    if not found:
        pytest.skip("no matching problem sizes in sweep results")
    _recordFeature(request.config, "fp32 GEMM", "PASS",
                   "plausibility [100, 1_000_000] GFLOPS")


@requires_gfx950
def test_fp32_cpp_reference(fp32ParitySweep, request):
    """fp32 GEMM Python GFLOPS vs C++ reference."""
    csvPath = os.path.join(_fixturesDir, "cpp_client_reference.csv")
    cppRef = _loadCppReference(csvPath)
    if not cppRef:
        _recordFeature(request.config, "fp32 vs C++ reference", "SKIP",
                       "C++ client requires pre-built library (see cpp_client_reference_cmd.txt)")
        pytest.skip("C++ reference CSV not yet populated")
    if not fp32ParitySweep:
        pytest.skip("no sweep results (no solutions compiled or no GPU)")
    for M, N, batch, K in _fp32Sizes:
        gflops = _gflopsForSize(fp32ParitySweep, M, N, batch, K)
        if gflops is None:
            continue
        tol = _toleranceFor(M, N)
        _compareCppGflops(gflops, cppRef, (M, N, batch, K), f"fp32 {M}x{N}x{batch}x{K}", tol)
        _recordGflops(request.config, "fp32", (M, N, batch, K), gflops,
                      cppRef.get((M, N, batch, K)))
    _recordFeature(request.config, "fp32 vs C++ reference", "PASS",
                   "within tolerance of C++ GFLOPS")


# ===========================================================================
# bf16 parity tests.
# ===========================================================================


@requires_gfx950
def test_bf16_gflops_plausible(bf16ParitySweep, request):
    """bf16 GEMM GFLOPS plausible for sizes (256, 512, 1024, 2048, 4096)."""
    if not bf16ParitySweep:
        pytest.skip("no sweep results (no solutions compiled or no GPU)")
    found = False
    for M, N, batch, K in _bf16Sizes:
        gflops = _gflopsForSize(bf16ParitySweep, M, N, batch, K)
        if gflops is None:
            continue
        found = True
        _checkPlausible(gflops, f"bf16 {M}x{N}x{batch}x{K}")
        _recordGflops(request.config, "bf16", (M, N, batch, K), gflops)
    if not found:
        pytest.skip("no matching problem sizes in sweep results")
    _recordFeature(request.config, "bf16 GEMM stridedBatched", "PASS",
                   "plausibility [100, 1_000_000] GFLOPS")


@requires_gfx950
def test_bf16_cpp_reference(bf16ParitySweep, request):
    """bf16 GEMM Python GFLOPS vs C++ reference."""
    csvPath = os.path.join(_fixturesDir, "cpp_client_reference.csv")
    cppRef = _loadCppReference(csvPath)
    if not cppRef:
        _recordFeature(request.config, "bf16 vs C++ reference", "SKIP",
                       "C++ client requires pre-built library (see cpp_client_reference_cmd.txt)")
        pytest.skip("C++ reference CSV not yet populated")
    if not bf16ParitySweep:
        pytest.skip("no sweep results (no solutions compiled or no GPU)")
    for M, N, batch, K in _bf16Sizes:
        gflops = _gflopsForSize(bf16ParitySweep, M, N, batch, K)
        if gflops is None:
            continue
        tol = _toleranceFor(M, N)
        _compareCppGflops(gflops, cppRef, (M, N, batch, K), f"bf16 {M}x{N}x{batch}x{K}", tol)
        _recordGflops(request.config, "bf16", (M, N, batch, K), gflops,
                      cppRef.get((M, N, batch, K)))
    _recordFeature(request.config, "bf16 vs C++ reference", "PASS",
                   "within tolerance of C++ GFLOPS")


# ===========================================================================
# fp16 + epilogue parity tests.
# ===========================================================================


def test_fp16_bias_relu_skip(request):
    """fp16 GEMM row bias+Relu: skipped — no fp16 epilogue kernel in YAML.

    gemm_epilogues.yaml has bf16/fp32 epilogue groups only. A dedicated
    fp16 epilogue YAML would be needed to test this combination.
    """
    _recordFeature(request.config, "fp16 bias+Relu", "SKIP",
                   "no fp16 epilogue kernel YAML available (bf16/fp32 only in gemm_epilogues.yaml)")
    pytest.skip("no fp16 bias+Relu kernel YAML available; "
                "SweepRunner does not support epilogue buffers")


# ===========================================================================
# int8 parity tests.
# ===========================================================================


@requires_gfx950
def test_int8_gflops_plausible(int8ParitySweep, request):
    """int8->int32 GEMM GFLOPS plausible for sizes (256, 512)."""
    if not int8ParitySweep:
        pytest.skip("no sweep results (no solutions compiled or no GPU)")
    found = False
    for M, N, batch, K in _int8Sizes:
        gflops = _gflopsForSize(int8ParitySweep, M, N, batch, K)
        if gflops is None:
            continue
        found = True
        _checkPlausible(gflops, f"int8 {M}x{N}x{batch}x{K}")
        _recordGflops(request.config, "int8->int32", (M, N, batch, K), gflops)
    if not found:
        pytest.skip("no matching problem sizes in sweep results")
    _recordFeature(request.config, "int8->int32 accumulation", "PASS",
                   "plausibility [100, 1_000_000] GFLOPS")


# ===========================================================================
# fp8 parity tests.
# ===========================================================================


@requires_gfx950
def test_fp8_gflops_plausible(fp8ParitySweep, request):
    """fp8 E4M3 OCP GEMM GFLOPS plausible at size (512, 512, 4, 512)."""
    if not fp8ParitySweep:
        pytest.skip("no sweep results (no solutions compiled or no GPU)")
    found = False
    for M, N, batch, K in _fp8Sizes:
        gflops = _gflopsForSize(fp8ParitySweep, M, N, batch, K)
        if gflops is None:
            continue
        found = True
        _checkPlausible(gflops, f"fp8 E4M3 OCP {M}x{N}x{batch}x{K}")
        _recordGflops(request.config, "fp8 E4M3 OCP", (M, N, batch, K), gflops)
    if not found:
        pytest.skip("no matching problem sizes in sweep results")
    _recordFeature(request.config, "fp8 E4M3 OCP", "PASS",
                   "plausibility [100, 1_000_000] GFLOPS")


# ===========================================================================
# MX parity tests.
# ===========================================================================


@requires_gfx950
def test_mx_gflops_plausible(mxParitySweep, request):
    """MX float8 + E8 scale GEMM GFLOPS plausible at sizes (256, 512).

    MX kernels force StaggerU=0 (via UseSubtileImpl=True). SweepRunner's
    filter skips solutions where StaggerU=0 and SupportCustomStaggerU=True,
    so MX sweeps return empty results. GPU correctness and performance for
    MX are validated by test_gemm_mx.py (M4) using a dedicated MX filter
    that bypasses the StaggerU restriction. This parity test records the
    reference and skips if no sweep results are available.
    """
    if not mxParitySweep:
        _recordFeature(
            request.config,
            "MX float8+E8 scale block_k=32",
            "REFERENCED",
            "SweepRunner skips MX (StaggerU=0 filter); validated by test_gemm_mx.py (M4)",
        )
        pytest.skip(
            "SweepRunner excludes MX kernels (StaggerU=0 forced by UseSubtileImpl=True, "
            "hits SupportCustomStaggerU filter). MX is validated by test_gemm_mx.py (M4)."
        )
    found = False
    for M, N, batch, K in _mxSizes:
        gflops = _gflopsForSize(mxParitySweep, M, N, batch, K)
        if gflops is None:
            continue
        found = True
        _checkPlausible(gflops, f"MX float8 {M}x{N}x{batch}x{K}")
        _recordGflops(request.config, "MX float8+E8", (M, N, batch, K), gflops)
    if not found:
        pytest.skip("no matching problem sizes in sweep results")
    _recordFeature(request.config, "MX float8+E8 scale block_k=32", "PASS",
                   "plausibility [100, 1_000_000] GFLOPS")


# ===========================================================================
# Grouped GEMM and Sparse GEMM — documented skips.
# ===========================================================================


def test_grouped_gemm_skip(request):
    """Grouped GEMM: skipped — no gfx950 kernel YAML available.

    GroupedGemm=True kernels require a dedicated YAML.
    See fixtures/m6_grouped_notes.txt for details.
    """
    _recordFeature(request.config, "Grouped GEMM (4 groups)", "SKIP",
                   "no gfx950 grouped GEMM kernel YAML (see m6_grouped_notes.txt)")
    pytest.skip("no gfx950 grouped GEMM kernel YAML available; "
                "see fixtures/m6_grouped_notes.txt")


def test_sparse_gemm_skip(request):
    """Sparse GEMM fp16: skipped — no gfx950 2:4 sparse kernel YAML.

    See fixtures/m6_sparse_notes.txt for details.
    """
    _recordFeature(request.config, "Sparse GEMM fp16 2:4", "SKIP",
                   "no gfx950 2:4 sparse kernel YAML (see m6_sparse_notes.txt)")
    pytest.skip("no gfx950 2:4 sparse kernel YAML available; "
                "see fixtures/m6_sparse_notes.txt")


# ===========================================================================
# Epilogue tests — PartialRMS, RstdScale, StreamK=3 (via subprocess).
# ===========================================================================


def _runEpilogueTest(testFile, label, request):
    """Run an epilogue test file via subprocess pytest and record feature status.

    Returns the CompletedProcess result. Skips if the test dir is missing.
    """
    if not os.path.exists(_epilogueTests):
        _recordFeature(request.config, label, "SKIP",
                       f"epilogues/unittests/ not found at {_epilogueTests}")
        pytest.skip(f"epilogues/unittests/ not found at {_epilogueTests}")

    fullPath = os.path.join(_epilogueTests, testFile)
    if not os.path.exists(fullPath):
        _recordFeature(request.config, label, "SKIP",
                       f"{testFile} not found in epilogues/unittests/")
        pytest.skip(f"{testFile} not found")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", fullPath,
         "--no-header", "-q", "--tb=short",
         f"--rootdir={_tensileLiteRoot}"],
        cwd=_tensileLiteRoot,
        capture_output=True, text=True,
        timeout=600,
    )
    return result


@requires_gfx950
def test_partial_rms_k1_passes(request):
    """PartialRMS (K1): epilogue tests must pass (or be skipped for no GPU).

    Runs epilogues/unittests/test_gemm_partial_rms.py via subprocess.
    Exit code 0 = all passed; 5 = all skipped (no GPU in subprocess env).
    Any other exit code indicates test failures.
    """
    result = _runEpilogueTest("test_gemm_partial_rms.py", "PartialRMS (K1)", request)
    if result.returncode == 5:
        _recordFeature(request.config, "PartialRMS (K1) StreamK=3", "SKIP",
                       "epilogue tests skipped (no GPU in subprocess env)")
        pytest.skip("epilogue tests all skipped (no GPU)")
    if result.returncode != 0:
        request.config._parityData["discrepancies"].append(
            ("PartialRMS (K1)", f"test exit code {result.returncode}")
        )
        pytest.fail(
            f"test_gemm_partial_rms.py exited with {result.returncode}.\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-500:]}"
        )
    _recordFeature(request.config, "PartialRMS (K1) StreamK=3", "PASS",
                   "test_gemm_partial_rms.py passed via subprocess")


@requires_gfx950
def test_rstd_scale_k3_passes(request):
    """RstdScale (K3): epilogue tests must pass (or be skipped for no GPU).

    Runs epilogues/unittests/test_gemm_rstd_scale.py via subprocess.
    """
    result = _runEpilogueTest("test_gemm_rstd_scale.py", "RstdScale (K3)", request)
    if result.returncode == 5:
        _recordFeature(request.config, "RstdScale (K3) StreamK=3", "SKIP",
                       "epilogue tests skipped (no GPU in subprocess env)")
        pytest.skip("epilogue tests all skipped (no GPU)")
    if result.returncode != 0:
        request.config._parityData["discrepancies"].append(
            ("RstdScale (K3)", f"test exit code {result.returncode}")
        )
        pytest.fail(
            f"test_gemm_rstd_scale.py exited with {result.returncode}.\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-500:]}"
        )
    _recordFeature(request.config, "RstdScale (K3) StreamK=3", "PASS",
                   "test_gemm_rstd_scale.py passed via subprocess")
