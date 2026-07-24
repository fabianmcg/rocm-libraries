# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compilation helpers: LLVM IR -> HSA code object."""

from pathlib import Path
from typing import Optional

from ._compile_asm import compile_asm, link_binary, llvmir_to_asm
from .runtime import get_chip


def compile_hsaco(llvm_ir: str, chip: Optional[str] = None) -> bytes:
    """Compile LLVM IR to a linked HSA code object."""
    if chip is None:
        chip = get_chip()
    asm = llvmir_to_asm(llvm_ir, chip)
    obj = compile_asm(asm, chip)
    return link_binary(obj)


def compile_asm_to_hsaco(asm: str, chip: Optional[str] = None) -> bytes:
    """Compile assembly to a linked HSA code object."""
    if chip is None:
        chip = get_chip()
    obj = compile_asm(asm, chip)
    return link_binary(obj)


def compile_asm_from_file(path: Path, chip: Optional[str] = None) -> bytes:
    """Read an assembly file and compile it to a linked HSA code object."""
    asm = path.read_text()
    return compile_asm_to_hsaco(asm, chip)
