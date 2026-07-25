// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#if defined(__linux__)
#include "ElfUtils.hpp"

#include <algorithm>
#include <cstring>
#include <elf.h>
#include <fstream>
#include <limits>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        std::uintmax_t getMinKernelSizeToGwEnd(std::string const& coPath)
        {
            std::ifstream f(coPath, std::ios::binary);
            if(!f)
                return 0;

            Elf64_Ehdr eh{};
            f.read(reinterpret_cast<char*>(&eh), sizeof(eh));
            if(!f
               || std::memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0
               || eh.e_ident[EI_CLASS] != ELFCLASS64)
                return 0;

            // Read all section headers.
            std::vector<Elf64_Shdr> shdrs(eh.e_shnum);
            f.seekg(eh.e_shoff);
            f.read(reinterpret_cast<char*>(shdrs.data()),
                   static_cast<std::streamsize>(eh.e_shnum * sizeof(Elf64_Shdr)));
            if(!f)
                return 0;

            // Find .symtab and its linked string table (sh_link).
            Elf64_Shdr const* symSh = nullptr;
            for(auto const& sh : shdrs)
                if(sh.sh_type == SHT_SYMTAB)
                {
                    symSh = &sh;
                    break;
                }
            if(!symSh
               || symSh->sh_link >= shdrs.size()
               || symSh->sh_size == 0
               || symSh->sh_size % sizeof(Elf64_Sym) != 0)
                return 0;

            auto const& strSh = shdrs[symSh->sh_link];

            // Read string table.
            std::vector<char> strs(strSh.sh_size);
            f.seekg(strSh.sh_offset);
            f.read(strs.data(), static_cast<std::streamsize>(strSh.sh_size));
            if(!f)
                return 0;

            // Read symbol table.
            auto const             symCount = symSh->sh_size / sizeof(Elf64_Sym);
            std::vector<Elf64_Sym> syms(symCount);
            f.seekg(symSh->sh_offset);
            f.read(reinterpret_cast<char*>(syms.data()),
                   static_cast<std::streamsize>(symSh->sh_size));
            if(!f)
                return 0;

            // Collect kernel entry addresses (FUNC + GLOBAL) and label_GW_End
            // addresses (we match by name; some toolchains emit different binds).
            std::vector<std::uint64_t> kernelStarts;
            std::vector<std::uint64_t> gwEndAddrs;
            for(auto const& s : syms)
            {
                if(s.st_name >= strs.size())
                    continue;
                char const*   name = &strs[s.st_name];
                unsigned char type = ELF64_ST_TYPE(s.st_info);
                unsigned char bind = ELF64_ST_BIND(s.st_info);

                if(type == STT_FUNC && bind == STB_GLOBAL)
                    kernelStarts.push_back(s.st_value);
                else if(std::strcmp(name, "label_GW_End") == 0)
                    gwEndAddrs.push_back(s.st_value);
            }

            if(kernelStarts.empty() || gwEndAddrs.empty())
                return 0;

            std::sort(kernelStarts.begin(), kernelStarts.end());

            // For each label_GW_End, the owning kernel is the one with the
            // largest start address <= this end address.
            std::uintmax_t minSize = std::numeric_limits<std::uintmax_t>::max();
            for(auto end : gwEndAddrs)
            {
                auto it = std::upper_bound(kernelStarts.begin(),
                                           kernelStarts.end(),
                                           end);
                if(it == kernelStarts.begin())
                    continue;
                std::uint64_t start = *(it - 1);
                if(end <= start)
                    continue;
                auto sz = static_cast<std::uintmax_t>(end - start);
                if(sz < minSize)
                    minSize = sz;
            }

            return (minSize == std::numeric_limits<std::uintmax_t>::max())
                       ? std::uintmax_t{0}
                       : minSize;
        }
    } // namespace Client
} // namespace TensileLite

#endif // defined(__linux__)
