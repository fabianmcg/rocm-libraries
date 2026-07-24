cmake_minimum_required(VERSION 3.21)
project(amdgpu-exec C CXX)

# Default to an optimized build (-O3 -DNDEBUG) unless the user overrides it.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Locate ROCm LLVM cmake packages and add them to CMAKE_PREFIX_PATH.
#
# EXTERNAL_LLVM=ON  — CMAKE_PREFIX_PATH already contains what is needed;
#                     nothing is auto-detected.
# EXTERNAL_LLVM=OFF — auto-detect via rocm-sdk, then fall back to /opt/rocm.
# ---------------------------------------------------------------------------
option(EXTERNAL_LLVM "Use LLVM provided via CMAKE_PREFIX_PATH (skip auto-detection)" OFF)

if(NOT EXTERNAL_LLVM)
  execute_process(
    COMMAND rocm-sdk path --root
    OUTPUT_VARIABLE _rocm_sdk_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rocm_sdk_result
    ERROR_QUIET
  )
  if(_rocm_sdk_result EQUAL 0 AND _rocm_sdk_root)
    set(_rocm_llvm_cmake "${_rocm_sdk_root}/llvm/lib/cmake")
    message(STATUS "ROCm LLVM cmake dir (rocm-sdk): ${_rocm_llvm_cmake}")
  else()
    set(_rocm_llvm_cmake "/opt/rocm/llvm/lib/cmake")
    message(STATUS "ROCm LLVM cmake dir (fallback): ${_rocm_llvm_cmake}")
  endif()
  if(NOT EXISTS "${_rocm_llvm_cmake}")
    message(WARNING "ROCm LLVM cmake dir does not exist: ${_rocm_llvm_cmake}")
  endif()
  list(PREPEND CMAKE_PREFIX_PATH "${_rocm_llvm_cmake}")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(python/lib)
