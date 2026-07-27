// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#if defined(__linux__)
#include "ElfUtils.hpp"
#endif

#include "LibraryBindings.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

NB_MODULE(_tensilelite_runtime, m)
{
    m.doc() = "TensileLite runtime Python bindings.";

#if defined(__linux__)
    // Recommended number of independent GpuModule copies for I-cache rotation.
    // Parses the ELF symbol table of the compiled code object at co_path to
    // find the smallest kernel's hot-path size, then divides the default I-cache
    // budget (icache-rotate-size=64 -> 128 KB) by that size. Returns 1 when the
    // ELF cannot be parsed or the kernel already covers the full budget.
    m.def(
        "get_icache_module_copies",
        [](std::string const& coPath) -> int {
            std::uintmax_t sz = TensileLite::Client::getMinKernelSizeToGwEnd(coPath);
            if(sz == 0)
                return 1;
            // Default --icache-rotate-size=64 gives 128 KB budget.
            constexpr std::uintmax_t kBudget = 64UL * 2 * 1024;
            if(kBudget <= sz)
                return 1;
            return static_cast<int>(kBudget / sz);
        },
        nb::arg("co_path"),
        "Return the recommended number of GpuModule copies for I-cache rotation.");
#endif

    bindLibraryTypes(m);
}
