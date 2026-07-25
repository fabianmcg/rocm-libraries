// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#if defined(__linux__)
#include <cstdint>
#include <string>

namespace TensileLite
{
    namespace Client
    {
        // Parse the AMDGPU ELF code object at coPath and return the smallest
        // distance (bytes) from any FUNC/GLOBAL kernel-entry symbol to its
        // corresponding label_GW_End LOCAL symbol.
        // Returns 0 when parsing fails or no valid pairing is found.
        std::uintmax_t getMinKernelSizeToGwEnd(std::string const& coPath);
    } // namespace Client
} // namespace TensileLite

#endif // defined(__linux__)
