# Plan: Full Python Replacement of `tensilelite-client`

## Overview

This plan replaces the `tensilelite-client` C++ binary with a Python harness built on
`amdgpu_exec`, numpy, `ml_dtypes`, and nanobind bindings. The foundation is the existing
`epilogues/` approach, which already handles YAML-driven solution enumeration, in-process
compilation, HIP-event timing, and numpy-based correctness validation for PartialRMS and
RstdScale.

Each milestone is executed by a **fresh implementor agent** given only this plan directory
and the relevant source files as context. After each milestone, a **reviewer agent** audits
the implementation against the milestone's acceptance criteria before the next begins.
No milestone starts until the previous one passes review.

## Implementation branch

**This plan is NOT implemented on the epilogue development branch (`users/fabianmcg/gemm_rms`).** M0 task 0.0 creates a dedicated branch from `develop` before any other work begins. All milestones M0–M14 land on `users/<github-username>/python-client-replacement`.

## Sub-plans

| File | Milestones | Topic |
|---|---|---|
| [m00_baseline.md](m00_baseline.md) | M0 | **Branch setup**, baseline audit, infrastructure setup, dependency pinning |
| [m01_standard_gemm.md](m01_standard_gemm.md) | M1 | `build_kernel_args` core, fp32/fp16/bf16 + strided-batched |
| [m02_int8_xf32.md](m02_int8_xf32.md) | M2 | Int8/Int32 and XFloat32 (TF32) |
| [m03_fp8.md](m03_fp8.md) | M3 | Float8 (OCP and fnuz variants) |
| [m04_mx.md](m04_mx.md) | M4 | MX block-scaled types (E8, E5M3, Float4/6) |
| [m05_epilogues.md](m05_epilogues.md) | M5 | Fused epilogues (bias, activations, scales, AmaxD, E tensor) |
| [m06_grouped_sparse.md](m06_grouped_sparse.md) | M6 | Grouped GEMM, sparse GEMM, StreamK=4/5 |
| [m07_harness.md](m07_harness.md) | M7 | Rotating buffers, I-cache simulation, ELF binding |
| [m08_bounds.md](m08_bounds.md) | M8 | Bounds checking (guard-page buffers) |
| [m09_hw_monitor.md](m09_hw_monitor.md) | M9 | Hardware monitoring (pyamdsmi) |
| [m10_runtime_bindings.md](m10_runtime_bindings.md) | M10 | nanobind: MasterSolutionLibrary, predicates, Formocast, calculateAuto* |
| [m11_rocprofiler.md](m11_rocprofiler.md) | M11 | ROCprofiler-SDK bindings |
| [m12_sweep.md](m12_sweep.md) | M12 | CSV output, SweepRunner, full benchmark pipeline |
| [m13_parity.md](m13_parity.md) | M13 | End-to-end parity validation |
| [m14_clientwriter.md](m14_clientwriter.md) | M14 | Replace `ClientWriter.py` subprocess with `SweepRunner` |
| [review_protocol.md](review_protocol.md) | — | Review protocol and dependency graph |
| [architectural_decisions.md](architectural_decisions.md) | — | Key architectural decisions (read before any milestone) |
| [amdgpu_exec_reference.md](amdgpu_exec_reference.md) | — | `amdgpu_exec` API reference (GpuModule, GpuBuffer, GpuFunction, execute_hsaco, compile_asm_to_hsaco, low-level execution pattern) |

## Reading order

Every implementor and reviewer agent must read:
1. `architectural_decisions.md` — resolved design decisions that all milestones depend on
2. `review_protocol.md` — review rules and dependency graph
3. The specific milestone sub-plan for the work being done

## Code location

- Production harness: `Tensile/client/`
- Nanobind modules: alongside `rocisa/`, following its CMake pattern
- Epilogue-specific tests: `epilogues/` — imported from `users/fabianmcg/gemm_rms` by M0 task 0.0; `epilogues/tensilelite/` renamed to `epilogues/epilogue_harness/` in M0 task 0.2
- `amdgpu_exec`: installed as a read-only binary wheel — cannot be modified; all new GPU primitives go into standalone nanobind modules
