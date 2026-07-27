# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Support both standalone (scikit-build-core / tox) and subdirectory
# (hipblaslt CMake) builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)
    project(tensilelite_runtime LANGUAGES CXX)
    set(TENSILELITE_RUNTIME_STANDALONE ON)
else()
    set(TENSILELITE_RUNTIME_STANDALONE OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Client sources are one level up; locate them relative to this file so the
# standalone (scikit-build-core) build and the subdirectory build use the
# same path.
set(_CLIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../client")

# Pre-built libraries live in build_tmp/, which is one level above us.
set(_BUILD_TMP "${CMAKE_CURRENT_SOURCE_DIR}/../build_tmp")

# TensileLite and shared library source roots.
set(_TL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(_ORIGAMI_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/origami")

if(TENSILELITE_RUNTIME_STANDALONE)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind REQUIRED CONFIG)

    # HIP provides hip::host for hipGetDevice* host-side calls.
    set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation prefix")
    find_package(hip REQUIRED CONFIG PATHS "${ROCM_PATH}/lib/cmake/hip" "${ROCM_PATH}")

    # tensilelitehost static library — no cmake config installed, use IMPORTED.
    add_library(tensilelitehost STATIC IMPORTED)
    set_target_properties(tensilelitehost PROPERTIES
        IMPORTED_LOCATION "${_BUILD_TMP}/tensilelite/libtensilelite-host.a"
        INTERFACE_INCLUDE_DIRECTORIES
            "${_TL_ROOT}/include;${_BUILD_TMP}/tensilelite/include;${_TL_ROOT}/rocisa"
        INTERFACE_COMPILE_DEFINITIONS
            "TENSILE_YAML;ORIGAMI_STATIC;__HIP_HCC_COMPAT_MODE__=1;__HIP_PLATFORM_AMD__=1"
    )

    # origami — use IMPORTED target; headers span both src and build dirs.
    add_library(roc::origami STATIC IMPORTED)
    set_target_properties(roc::origami PROPERTIES
        IMPORTED_LOCATION "${_BUILD_TMP}/origami/liborigami.a"
        INTERFACE_INCLUDE_DIRECTORIES
            "${_ORIGAMI_SRC}/include;${_BUILD_TMP}/origami/include"
        INTERFACE_COMPILE_DEFINITIONS "ORIGAMI_STATIC"
    )

    # LLVM static libraries — required at link time by libtensilelite-host.a.
    set(_LLVM_LIB "/opt/rocm/core-7.14/lib/llvm/lib" CACHE PATH "LLVM lib directory")
    set(_LLVM_LIBS
        "${_LLVM_LIB}/libLLVMObjectYAML.a"
        "${_LLVM_LIB}/libLLVMObject.a"
        "${_LLVM_LIB}/libLLVMIRReader.a"
        "${_LLVM_LIB}/libLLVMBitReader.a"
        "${_LLVM_LIB}/libLLVMAsmParser.a"
        "${_LLVM_LIB}/libLLVMCore.a"
        "${_LLVM_LIB}/libLLVMRemarks.a"
        "${_LLVM_LIB}/libLLVMBitstreamReader.a"
        "${_LLVM_LIB}/libLLVMMCParser.a"
        "${_LLVM_LIB}/libLLVMTextAPI.a"
        "${_LLVM_LIB}/libLLVMDebugInfoCodeView.a"
        "${_LLVM_LIB}/libLLVMMC.a"
        "${_LLVM_LIB}/libLLVMDebugInfoDWARFLowLevel.a"
        "${_LLVM_LIB}/libLLVMBinaryFormat.a"
        "${_LLVM_LIB}/libLLVMTargetParser.a"
        "${_LLVM_LIB}/libLLVMSupport.a"
        "${_LLVM_LIB}/libLLVMDemangle.a"
    )
endif()

nanobind_add_module(_tensilelite_runtime
    NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/LibraryBindings.cpp"
    "${_CLIENT_DIR}/src/ElfUtils.cpp"
)

target_include_directories(_tensilelite_runtime PRIVATE
    "${_CLIENT_DIR}/include"
)

set_target_properties(_tensilelite_runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tensilelite_runtime"
)

if(TENSILELITE_RUNTIME_STANDALONE)
    target_link_libraries(_tensilelite_runtime PRIVATE
        tensilelitehost
        roc::origami
        hip::host
        ${_LLVM_LIBS}
        # amd_smi is a runtime dep of libamdhip64 on ROCm platforms.
        "${ROCM_PATH}/lib/libamd_smi.so"
        dl rt pthread m z zstd
    )

    target_compile_definitions(_tensilelite_runtime PRIVATE
        TENSILE_YAML
        ORIGAMI_STATIC
        __HIP_HCC_COMPAT_MODE__=1
        __HIP_PLATFORM_AMD__=1
        HIP_HAS_CLUSTER_LAUNCH
    )

    # TensileLite headers (DataTypes_Float6.hpp etc.) include HIP OCP headers
    # that require the -x hip mode for amdclang++. Apply this only to the
    # files that pull in those headers, not to nanobind's own source files.
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/LibraryBindings.cpp"
        PROPERTIES COMPILE_OPTIONS "-x;hip;--offload-arch=gfx950"
    )
    target_link_options(_tensilelite_runtime PRIVATE
        --hip-link --offload-arch=gfx950
    )

    install(TARGETS _tensilelite_runtime DESTINATION tensilelite_runtime)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tensilelite_runtime/__init__.py"
            DESTINATION tensilelite_runtime)
endif()
