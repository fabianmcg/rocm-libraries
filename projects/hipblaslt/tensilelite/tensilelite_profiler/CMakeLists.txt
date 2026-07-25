# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Support both standalone (scikit-build-core / tox) and subdirectory
# (hipblaslt CMake) builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)
    project(tensilelite_profiler LANGUAGES CXX)
    set(TENSILELITE_PROFILER_STANDALONE ON)
else()
    set(TENSILELITE_PROFILER_STANDALONE OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The module is only useful when ROCprofiler-SDK is present.
option(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK "Enable ROCprofiler-SDK counter bindings" OFF)

if(NOT TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
    if(TENSILELITE_PROFILER_STANDALONE)
        message(FATAL_ERROR
            "Pass -DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK=ON to build tensilelite_profiler")
    else()
        return()
    endif()
endif()

if(TENSILELITE_PROFILER_STANDALONE)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind REQUIRED CONFIG)

    set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation prefix")
    find_package(hip REQUIRED CONFIG
        PATHS "${ROCM_PATH}/lib/cmake/hip" "${ROCM_PATH}")
    find_package(rocprofiler-sdk REQUIRED CONFIG)
endif()

nanobind_add_module(_tensilelite_profiler
    NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
)

target_link_libraries(_tensilelite_profiler PRIVATE
    rocprofiler-sdk::rocprofiler-sdk
    hip::host
)

# hipDeviceGetPCIBusId is a host API; no device compilation needed.
# Set __HIP_PLATFORM_AMD__ so that hip/hip_runtime_api.h resolves AMD symbols.
target_compile_definitions(_tensilelite_profiler PRIVATE
    __HIP_PLATFORM_AMD__=1
    __HIP_HCC_COMPAT_MODE__=1
)

set_target_properties(_tensilelite_profiler PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tensilelite_profiler"
)

if(TENSILELITE_PROFILER_STANDALONE)
    install(TARGETS _tensilelite_profiler DESTINATION tensilelite_profiler)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tensilelite_profiler/__init__.py"
            DESTINATION tensilelite_profiler)
endif()
