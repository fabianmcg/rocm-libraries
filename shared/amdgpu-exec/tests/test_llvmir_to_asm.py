# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for the llvmir_to_asm binding."""

import os

import pytest
import amdgpu_exec

# A minimal AMDGPU kernel in LLVM IR. The amdgpu_kernel calling convention is
# required so the backend emits kernel prologue/epilogue code.
_SIMPLE_IR = """\
; ModuleID = 'add_kernel'
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @add(ptr addrspace(1) %a) {
entry:
  %val = load i32, ptr addrspace(1) %a, align 4
  %inc = add i32 %val, 1
  store i32 %inc, ptr addrspace(1) %a, align 4
  ret void
}
"""

_CHIP = "gfx90a"

# IR that calls an __ocml_ builtin, so device libraries are required.
_IR_WITH_OCML = """\
; ModuleID = 'cos_kernel'
target triple = "amdgcn-amd-amdhsa"

declare float @__ocml_cos_f32(float)

define amdgpu_kernel void @cos_kernel(ptr addrspace(1) %a) {
entry:
  %val = load float, ptr addrspace(1) %a, align 4
  %r   = call float @__ocml_cos_f32(float %val)
  store float %r, ptr addrspace(1) %a, align 4
  ret void
}
"""


def test_llvmir_to_asm_returns_string():
    """llvmir_to_asm should return a non-empty string."""
    result = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP)
    assert isinstance(result, str)
    assert len(result) > 0


def test_llvmir_to_asm_contains_text_section():
    """Output should contain the ELF .text section marker."""
    result = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP)
    assert ".text" in result


def test_llvmir_to_asm_contains_kernel_name():
    """Output should reference the kernel symbol name."""
    result = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP)
    # Match the kernel as a symbol/label, not the "add" substring of v_add etc.
    assert ("\nadd:" in result) or (".amdhsa_kernel add" in result)


def test_llvmir_to_asm_contains_amdgpu_instruction():
    """Output should contain at least one recognisable AMDGPU instruction."""
    result = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP)
    assert any(instr in result for instr in ("global_load", "flat_load", "s_endpgm"))


def test_llvmir_to_asm_bad_ir_raises():
    """Malformed IR should raise RuntimeError, not crash."""
    with pytest.raises(RuntimeError):
        amdgpu_exec.llvmir_to_asm("this is not valid llvm ir", _CHIP)


def test_llvmir_to_asm_bad_triple_raises():
    """An unknown target triple should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP, triple="unknown-unknown-unknown")


def test_compile_asm_returns_elf():
    """compile_asm should assemble ASM into an ELF object (starts with magic)."""
    asm = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP)
    obj = amdgpu_exec.compile_asm(asm, _CHIP, "")
    assert isinstance(obj, bytes)
    assert obj[:4] == b"\x7fELF"


def test_compile_asm_bad_asm_raises():
    """Malformed assembly should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        amdgpu_exec.compile_asm("this is not valid amdgpu asm", _CHIP, "")


def test_link_binary_produces_hsaco():
    """Full pipeline: ASM -> ELF object -> linked HSA code object."""
    asm = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP)
    obj = amdgpu_exec.compile_asm(asm, _CHIP, "")
    hsaco = amdgpu_exec.link_binary(obj)
    assert isinstance(hsaco, bytes)
    assert hsaco[:4] == b"\x7fELF"


def test_link_binary_bad_input_raises():
    """Linking non-ELF input should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        amdgpu_exec.link_binary(b"not an elf object")


# ---------------------------------------------------------------------------
# CompileOptions / OptLevel / AbiVersion
# ---------------------------------------------------------------------------


def test_compile_options_defaults():
    """CompileOptions.defaults() should produce a usable options object."""
    opts = amdgpu_exec.CompileOptions.defaults()
    assert opts.opt_level == amdgpu_exec.OptLevel.O3
    assert opts.abi_version == amdgpu_exec.AbiVersion.V6
    assert opts.rocm_path is None


def test_llvmir_to_asm_opt_level_o0():
    """OptLevel.O0 should still produce valid assembly."""
    opts = amdgpu_exec.CompileOptions.defaults()
    opts.opt_level = amdgpu_exec.OptLevel.O0
    result = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP, opts=opts)
    assert isinstance(result, str)
    assert ".text" in result


def test_llvmir_to_asm_opt_level_o3():
    """OptLevel.O3 should produce valid assembly."""
    opts = amdgpu_exec.CompileOptions.defaults()
    opts.opt_level = amdgpu_exec.OptLevel.O3
    result = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP, opts=opts)
    assert isinstance(result, str)
    assert ".text" in result


def test_llvmir_to_asm_no_rocm_path_succeeds():
    """Without rocm_path, simple IR should still compile (no device libs needed)."""
    opts = amdgpu_exec.CompileOptions.defaults()
    assert opts.rocm_path is None
    result = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP, opts=opts)
    assert isinstance(result, str)
    assert len(result) > 0


def test_llvmir_to_asm_abi_version_applied_without_rocm_path():
    """abi_version must affect the emitted code object even when no device
    libraries are linked (rocm_path unset)."""
    opts_v5 = amdgpu_exec.CompileOptions.defaults()
    opts_v5.abi_version = amdgpu_exec.AbiVersion.V5
    assert opts_v5.rocm_path is None
    asm_v5 = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP, opts=opts_v5)
    assert ".amdhsa_code_object_version 5" in asm_v5

    opts_v6 = amdgpu_exec.CompileOptions.defaults()
    opts_v6.abi_version = amdgpu_exec.AbiVersion.V6
    asm_v6 = amdgpu_exec.llvmir_to_asm(_SIMPLE_IR, _CHIP, opts=opts_v6)
    assert ".amdhsa_code_object_version 6" in asm_v6


def test_llvmir_to_asm_bad_rocm_path_raises():
    """A non-existent rocm_path with IR that needs device libs should raise."""
    opts = amdgpu_exec.CompileOptions.defaults()
    opts.rocm_path = "/nonexistent/rocm/path"
    with pytest.raises(RuntimeError):
        amdgpu_exec.llvmir_to_asm(_IR_WITH_OCML, _CHIP, opts=opts)


@pytest.mark.skipif(
    not os.path.isdir("/opt/rocm/amdgcn/bitcode"),
    reason="/opt/rocm bitcode libraries not available",
)
def test_llvmir_to_asm_with_ocml_device_lib():
    """IR calling __ocml_cos_f32 should compile when rocm_path points to ROCm."""
    opts = amdgpu_exec.CompileOptions.defaults()
    opts.rocm_path = "/opt/rocm"
    result = amdgpu_exec.llvmir_to_asm(_IR_WITH_OCML, _CHIP, opts=opts)
    assert isinstance(result, str)
    assert ".text" in result
