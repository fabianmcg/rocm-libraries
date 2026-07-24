//===- API.h - AMDGPU compilation public API --------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_EXEC_API_H
#define AMDGPU_EXEC_API_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>
#include <string>
#include <string_view>

enum class OptLevel { O0, O1, O2, O3 };

// Code-object ABI version. Only V5 (500) and V6 (600) are supported.
enum class AbiVersion { V5 = 500, V6 = 600 };

struct CompileOptions {
  // If nullopt, no device libraries are linked.
  std::optional<std::string> rocmPath;
  OptLevel optLevel = OptLevel::O3;
  AbiVersion abiVersion = AbiVersion::V6;
  bool wave64 = true;
  bool daz = false;
  bool finiteOnly = false;
  bool unsafeMath = false;
  bool fastMath = false;
  bool correctSqrt = true;

  static CompileOptions defaults() { return {}; }
};

// Assembles AMDGPU ISA source to an ELF binary using LLVM MC.
llvm::FailureOr<llvm::SmallVector<char>> compileAsm(std::string_view asmCode,
                                                    std::string_view chip,
                                                    std::string_view features,
                                                    std::string_view triple);

// Compiles LLVM IR source to AMDGPU assembly text. Optionally links device
// libraries and runs the LLVM optimization pipeline.
llvm::FailureOr<std::string>
llvmIrToAsm(std::string_view irCode, std::string_view chip,
            std::string_view features, std::string_view triple,
            CompileOptions opts = CompileOptions::defaults());

// Links an ELF binary to an HSA code object using LLD.
llvm::FailureOr<llvm::SmallVector<char>>
linkBinary(llvm::ArrayRef<char> binary);

#endif // AMDGPU_EXEC_API_H
