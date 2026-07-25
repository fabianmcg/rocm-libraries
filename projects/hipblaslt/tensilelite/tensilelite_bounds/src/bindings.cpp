// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

//===----------------------------------------------------------------------===//
// Minimal HIP type forward declarations
//===----------------------------------------------------------------------===//

enum hipError_t : int
{
    hipSuccess = 0
};

enum hipMemcpyKind
{
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
};

//===----------------------------------------------------------------------===//
// Minimal HIP API table — only the symbols BoundedBuffer needs.
//===----------------------------------------------------------------------===//

struct HipApi
{
    hipError_t (*malloc)(void **ptr, size_t size)                                   = nullptr;
    hipError_t (*free)(void *ptr)                                                   = nullptr;
    hipError_t (*memcpy)(void *dst, const void *src, size_t bytes, hipMemcpyKind k) = nullptr;
    const char *(*getErrorString)(hipError_t err)                                   = nullptr;

    // Library handle, closed on destruction.
    void *lib = nullptr;

    ~HipApi()
    {
        if(lib)
            ::dlclose(lib);
    }

    // Opens libamdhip64.so, resolves required symbols, and returns a pointer to
    // a static instance. Returns nullptr on failure (each missing symbol is
    // reported to stderr).
    static const HipApi *load();
};

const HipApi *HipApi::load()
{
    void *handle = ::dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
    if(!handle)
        return nullptr;

    static HipApi api;
    bool ok = true;

    // Resolve one symbol; set ok=false and print to stderr if absent.
    auto resolve = [&](void **fp, const char *sym) {
        *fp = ::dlsym(handle, sym);
        if(!*fp) {
            std::fprintf(stderr, "HIP runtime: symbol '%s' not found in libamdhip64.so\n", sym);
            ok = false;
        }
    };

    resolve(reinterpret_cast<void **>(&api.malloc), "hipMalloc");
    resolve(reinterpret_cast<void **>(&api.free), "hipFree");
    resolve(reinterpret_cast<void **>(&api.memcpy), "hipMemcpy");
    resolve(reinterpret_cast<void **>(&api.getErrorString), "hipGetErrorString");

    if(!ok)
    {
        ::dlclose(handle);
        return nullptr;
    }

    api.lib = handle;
    return &api;
}

//===----------------------------------------------------------------------===//
// Error checking helpers
//===----------------------------------------------------------------------===//

static const HipApi &getHip()
{
    static const HipApi *api = HipApi::load();
    if(!api)
        throw std::runtime_error("HIP runtime not available: could not load libamdhip64.so");
    return *api;
}

static void hipCheck(hipError_t err)
{
    if(err == hipSuccess)
        return;
    throw std::runtime_error(std::string("HIP error: ") + getHip().getErrorString(err));
}

//===----------------------------------------------------------------------===//
// BoundedBuffer — device allocation with trailing sentinel region
//===----------------------------------------------------------------------===//

// The sentinel value written past the valid region.
static constexpr uint32_t kSentinel = 0xDEADBEEFu;

class BoundedBuffer
{
public:
    // Allocates sizeBytes + sentinelSlots*4 bytes and fills the sentinel region
    // with 0xDEADBEEF via synchronous H→D copy.
    BoundedBuffer(size_t sizeBytes, int sentinelSlots)
        : _ptr(nullptr)
        , _sizeBytes(sizeBytes)
        , _sentinelSlots(sentinelSlots)
        , _freed(false)
    {
        if(sentinelSlots <= 0)
            throw std::invalid_argument("sentinel_slots must be positive");

        const HipApi &h = getHip();
        size_t total    = sizeBytes + static_cast<size_t>(sentinelSlots) * sizeof(uint32_t);
        hipCheck(h.malloc(&_ptr, total));
        _fillSentinel(h);
    }

    ~BoundedBuffer()
    {
        _doFree();
    }

    // Device pointer to the start of the valid region.
    uintptr_t ptrValue() const
    {
        return reinterpret_cast<uintptr_t>(_ptr);
    }

    // Device pointer to the first sentinel slot (ptrValue + sizeBytes).
    uintptr_t sentinelPtr() const
    {
        return reinterpret_cast<uintptr_t>(static_cast<char *>(_ptr) + _sizeBytes);
    }

    // Returns true if all sentinel slots still contain 0xDEADBEEF.
    // Must only be called after the device is synchronized by the caller.
    bool checkSentinel()
    {
        if(_freed)
            throw std::runtime_error("BoundedBuffer already freed");
        const HipApi &h = getHip();
        std::vector<uint32_t> host(static_cast<size_t>(_sentinelSlots));
        char *sentinelAddr = static_cast<char *>(_ptr) + _sizeBytes;
        hipCheck(h.memcpy(host.data(), sentinelAddr,
                          static_cast<size_t>(_sentinelSlots) * sizeof(uint32_t),
                          hipMemcpyDeviceToHost));
        for(uint32_t v : host)
        {
            if(v != kSentinel)
                return false;
        }
        return true;
    }

    void free()
    {
        _doFree();
    }

private:
    // Fills the sentinel region from a host-side pattern buffer.
    void _fillSentinel(const HipApi &h)
    {
        std::vector<uint32_t> pattern(static_cast<size_t>(_sentinelSlots), kSentinel);
        char *sentinelAddr = static_cast<char *>(_ptr) + _sizeBytes;
        hipCheck(h.memcpy(sentinelAddr, pattern.data(),
                          static_cast<size_t>(_sentinelSlots) * sizeof(uint32_t),
                          hipMemcpyHostToDevice));
    }

    void _doFree() noexcept
    {
        if(_freed || !_ptr)
            return;
        try
        {
            getHip().free(_ptr);
        }
        catch(...)
        {
        }
        _ptr   = nullptr;
        _freed = true;
    }

    void *_ptr;
    size_t _sizeBytes;
    int _sentinelSlots;
    bool _freed;
};

//===----------------------------------------------------------------------===//
// Python module
//===----------------------------------------------------------------------===//

NB_MODULE(_tensilelite_bounds, m)
{
    m.doc() = "TensileLite sentinel-based bounds-check Python bindings.";

    nb::class_<BoundedBuffer>(m, "BoundedBuffer",
                              "Device allocation with a sentinel region past the valid area.")
        .def(nb::init<size_t, int>(), nb::arg("size_bytes"), nb::arg("sentinel_slots") = 1,
             "Allocate size_bytes bytes plus sentinel_slots*4 sentinel bytes on the device.")
        .def_prop_ro("ptrValue",
                     [](const BoundedBuffer &self) -> int64_t {
                         return static_cast<int64_t>(self.ptrValue());
                     },
                     "Device pointer (int) to the start of the valid region.")
        .def_prop_ro("dataPtr",
                     [](const BoundedBuffer &self) -> int64_t {
                         return static_cast<int64_t>(self.ptrValue());
                     },
                     "Alias for ptrValue.")
        .def_prop_ro("sentinelPtr",
                     [](const BoundedBuffer &self) -> int64_t {
                         return static_cast<int64_t>(self.sentinelPtr());
                     },
                     "Device pointer (int) to the first sentinel slot.")
        .def("checkSentinel", &BoundedBuffer::checkSentinel,
             "Return True if all sentinel slots still contain 0xDEADBEEF.")
        .def("free", &BoundedBuffer::free, "Release device memory.")
        .def("__del__", [](BoundedBuffer &self) { self.free(); });
}
