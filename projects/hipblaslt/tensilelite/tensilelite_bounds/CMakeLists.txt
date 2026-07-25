# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Support both standalone (scikit-build-core / tox) and subdirectory
# (hipblaslt CMake) builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)
    project(tensilelite_bounds LANGUAGES CXX)
    set(TENSILELITE_BOUNDS_STANDALONE ON)
else()
    set(TENSILELITE_BOUNDS_STANDALONE OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(TENSILELITE_BOUNDS_STANDALONE)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind REQUIRED CONFIG)
endif()

nanobind_add_module(_tensilelite_bounds
    NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
)

set_target_properties(_tensilelite_bounds PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tensilelite_bounds"
)

if(TENSILELITE_BOUNDS_STANDALONE)
    install(TARGETS _tensilelite_bounds DESTINATION tensilelite_bounds)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tensilelite_bounds/__init__.py"
            DESTINATION tensilelite_bounds)
endif()
