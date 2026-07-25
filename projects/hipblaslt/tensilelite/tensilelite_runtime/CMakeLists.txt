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

if(TENSILELITE_RUNTIME_STANDALONE)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind REQUIRED CONFIG)
endif()

nanobind_add_module(_tensilelite_runtime
    NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
    "${_CLIENT_DIR}/src/ElfUtils.cpp"
)

target_include_directories(_tensilelite_runtime PRIVATE
    "${_CLIENT_DIR}/include"
)

set_target_properties(_tensilelite_runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tensilelite_runtime"
)

if(TENSILELITE_RUNTIME_STANDALONE)
    install(TARGETS _tensilelite_runtime DESTINATION tensilelite_runtime)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tensilelite_runtime/__init__.py"
            DESTINATION tensilelite_runtime)
endif()
