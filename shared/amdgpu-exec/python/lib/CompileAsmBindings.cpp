//===- CompileAsmBindings.cpp - nanobind bindings for CompileAsm --*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "API.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <stdexcept>

namespace nb = nanobind;

NB_MODULE(_compile_asm, m) {
  m.doc() = "AMDGPU ASM compilation and HSA code object linking";

  nb::enum_<OptLevel>(m, "OptLevel")
      .value("O0", OptLevel::O0)
      .value("O1", OptLevel::O1)
      .value("O2", OptLevel::O2)
      .value("O3", OptLevel::O3);

  nb::enum_<AbiVersion>(m, "AbiVersion")
      .value("V5", AbiVersion::V5)
      .value("V6", AbiVersion::V6);

  nb::class_<CompileOptions>(m, "CompileOptions")
      .def(nb::init<>())
      .def_rw("rocm_path", &CompileOptions::rocmPath)
      .def_rw("opt_level", &CompileOptions::optLevel)
      .def_rw("abi_version", &CompileOptions::abiVersion)
      .def_rw("wave64", &CompileOptions::wave64)
      .def_rw("daz", &CompileOptions::daz)
      .def_rw("finite_only", &CompileOptions::finiteOnly)
      .def_rw("unsafe_math", &CompileOptions::unsafeMath)
      .def_rw("fast_math", &CompileOptions::fastMath)
      .def_rw("correct_sqrt", &CompileOptions::correctSqrt)
      .def_static("defaults", &CompileOptions::defaults);

  m.def(
      "compile_asm",
      [](std::string_view asmSrc, std::string_view chip,
         std::string_view features, std::string_view triple) {
        llvm::FailureOr<llvm::SmallVector<char>> result =
            compileAsm(asmSrc, chip, features, triple);
        if (llvm::failed(result))
          throw std::runtime_error("compile_asm failed");
        return nb::bytes(result->data(), result->size());
      },
      nb::arg("asm_src"), nb::arg("chip"), nb::arg("features") = "",
      nb::arg("triple") = "amdgcn-amd-amdhsa",
      "Assemble AMDGPU ISA source to an ELF binary.");

  m.def(
      "link_binary",
      [](nb::bytes binary) {
        llvm::FailureOr<llvm::SmallVector<char>> result =
            linkBinary(llvm::ArrayRef<char>(
                reinterpret_cast<const char *>(binary.data()), binary.size()));
        if (llvm::failed(result))
          throw std::runtime_error("link_binary failed");
        return nb::bytes(result->data(), result->size());
      },
      nb::arg("binary"), "Link an ELF binary to an HSA code object.");

  m.def(
      "llvmir_to_asm",
      [](std::string_view irSrc, std::string_view chip,
         std::string_view features, std::string_view triple,
         CompileOptions opts) {
        llvm::FailureOr<std::string> result =
            llvmIrToAsm(irSrc, chip, features, triple, opts);
        if (llvm::failed(result))
          throw std::runtime_error("llvmir_to_asm failed");
        return std::move(*result);
      },
      nb::arg("ir_src"), nb::arg("chip"), nb::arg("features") = "",
      nb::arg("triple") = "amdgcn-amd-amdhsa",
      nb::arg("opts") = CompileOptions::defaults(),
      "Compile LLVM IR source to AMDGPU assembly text.");
}
