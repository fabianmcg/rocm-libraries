include(AddLLVM)

set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN "ON")

set(LLVM_OPTIONAL_SOURCES CompileAsm.cpp CompileAsmBindings.cpp RuntimeModule.cpp)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)

set(SPURIOUS_LLVM_DEPS
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info
  SPIRVDesc
  SPIRVCodeGen
  SPIRVInfo
)

llvm_add_library(compile_asm_llvm STATIC
  CompileAsm.cpp
  DISABLE_LLVM_LINK_LLVM_DYLIB
  LINK_COMPONENTS
  AMDGPUAsmParser
  AMDGPUCodeGen
  AMDGPUUtils
  AMDGPUDesc
  AMDGPUInfo
  BitReader
  IPO
  IRReader
  Linker
  MC
  MCParser
  DTLTO
  LTO
  Option
  Passes
  Support
  Target
  TargetParser
  TransformUtils
  # These are spurious deps introduced by lldElf from the rock.
  ${SPURIOUS_LLVM_DEPS}
  LINK_LIBS
  lldELF
  lldCommon
)

nanobind_add_module(_compile_asm
  NB_DOMAIN amdgpu_exec_compile_asm
  CompileAsmBindings.cpp
)
target_link_directories(_compile_asm PRIVATE ${LLVM_LIBRARY_DIR})
target_link_libraries(_compile_asm PRIVATE compile_asm_llvm)
install(TARGETS _compile_asm LIBRARY DESTINATION amdgpu_exec)

nanobind_add_stub(_compile_asm_stub
  MODULE _compile_asm
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_compile_asm.pyi"
  DEPENDS _compile_asm
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_compile_asm.pyi"
  DESTINATION amdgpu_exec
)

nanobind_add_module(_runtime_module
  NB_DOMAIN amdgpu_exec_runtime_module
  RuntimeModule.cpp
)
target_include_directories(_runtime_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(_runtime_module PRIVATE dl)
install(TARGETS _runtime_module LIBRARY DESTINATION amdgpu_exec)

nanobind_add_stub(_runtime_module_stub
  MODULE _runtime_module
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_runtime_module.pyi"
  DEPENDS _runtime_module
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_runtime_module.pyi"
  DESTINATION amdgpu_exec
)
