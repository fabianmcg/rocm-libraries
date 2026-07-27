// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Register the M10 library/solution/hardware/problem bindings into module m.
void bindLibraryTypes(nb::module_& m);
