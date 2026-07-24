//===- RuntimeModule.cpp -------------------------------------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// File adapted from https://github.com/iree-org/aster
//
//===----------------------------------------------------------------------===//

#include "hip.h"
#include <cstdio>
#include <dlfcn.h>
#include <mutex>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

//===----------------------------------------------------------------------===//
// Ptr — opaque handle exchanged with Python as uintptr_t
//===----------------------------------------------------------------------===//

// Wraps an opaque HIP pointer as a uintptr_t so Python sees a plain integer
// rather than a ctypes void* that requires explicit dereferencing.
struct Ptr {
  void *value;

  Ptr() : value(nullptr) {}
  explicit Ptr(uintptr_t v) : value(reinterpret_cast<void *>(v)) {}

  template <typename T>
  explicit Ptr(T *p) : value(static_cast<void *>(p)) {}

  template <typename T>
  T *as() const {
    return static_cast<T *>(value);
  }

  explicit operator bool() const { return value != nullptr; }
};

//===----------------------------------------------------------------------===//
// Dynamic library loading
//===----------------------------------------------------------------------===//

HipApi::~HipApi() {
  if (lib)
    ::dlclose(lib);
}

const HipApi *HipApi::load() {
  void *handle = ::dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
  if (!handle)
    return nullptr;

  static HipApi api;
  bool ok = true;

  // Resolves one symbol into a typed function pointer; prints to stderr and
  // sets ok=false if the symbol is absent.
  auto loadSym = [&ok, handle](auto &fp, const char *sym) {
    void *p = ::dlsym(handle, sym);
    if (!p) {
      std::fprintf(stderr,
                   "HIP runtime: symbol '%s' not found in libamdhip64.so\n",
                   sym);
      ok = false;
      return;
    }
    static_assert(sizeof(p) == sizeof(fp), "function pointer size mismatch");
    *reinterpret_cast<void **>(&fp) = p;
  };

  loadSym(api.init, "hipInit");
  loadSym(api.setDevice, "hipSetDevice");
  loadSym(api.getDevice, "hipGetDevice");
  loadSym(api.getDeviceCount, "hipGetDeviceCount");
  loadSym(api.getDeviceProperties, "hipGetDevicePropertiesR0600");
  loadSym(api.deviceReset, "hipDeviceReset");
  loadSym(api.moduleLoadData, "hipModuleLoadData");
  loadSym(api.moduleGetFunction, "hipModuleGetFunction");
  loadSym(api.moduleUnload, "hipModuleUnload");
  loadSym(api.moduleLaunchKernel, "hipModuleLaunchKernel");
  loadSym(api.malloc, "hipMalloc");
  loadSym(api.free, "hipFree");
  loadSym(api.memcpyAsync, "hipMemcpyAsync");
  loadSym(api.memsetAsync, "hipMemsetAsync");
  loadSym(api.deviceSynchronize, "hipDeviceSynchronize");
  loadSym(api.peekAtLastError, "hipPeekAtLastError");
  loadSym(api.getLastError, "hipGetLastError");
  loadSym(api.getErrorString, "hipGetErrorString");
  loadSym(api.eventCreate, "hipEventCreate");
  loadSym(api.eventDestroy, "hipEventDestroy");
  loadSym(api.eventRecord, "hipEventRecord");
  loadSym(api.eventSynchronize, "hipEventSynchronize");
  loadSym(api.eventElapsedTime, "hipEventElapsedTime");
  loadSym(api.streamCreate, "hipStreamCreate");
  loadSym(api.streamDestroy, "hipStreamDestroy");
  loadSym(api.streamSynchronize, "hipStreamSynchronize");
  loadSym(api.moduleOccupancyMaxActiveBlocksPerMultiprocessor,
          "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");

  if (!ok) {
    ::dlclose(handle);
    return nullptr;
  }

  api.lib = handle;
  return &api;
}

//===----------------------------------------------------------------------===//
// Error checking helper
//===----------------------------------------------------------------------===//

// Returns the HIP API table, throwing if libamdhip64.so was not loaded.
static const HipApi &getHip() {
  static const HipApi *api = HipApi::load();
  if (!api)
    throw std::runtime_error(
        "HIP runtime not available: could not load libamdhip64.so");
  return *api;
}

static void hipCheckImpl(hipError_t err, const char *file, int line) {
  if (err == hipSuccess)
    return;
  throw std::runtime_error(std::string("HIP error at ") + file + ":" +
                           std::to_string(line) + " - " +
                           getHip().getErrorString(err));
}

#define hipCheck(call) hipCheckImpl((call), __FILE__, __LINE__)

//===----------------------------------------------------------------------===//
// Python module
//===----------------------------------------------------------------------===//

NB_MODULE(_runtime_module, m) {
  m.doc() = "Python bindings for HIP runtime module";

  nb::class_<Ptr>(m, "Ptr")
      .def(nb::init<uintptr_t>())
      .def("__int__",
           [](Ptr p) { return reinterpret_cast<uintptr_t>(p.value); })
      .def("__bool__", [](Ptr p) { return static_cast<bool>(p); })
      .def("__repr__", [](Ptr p) {
        return "Ptr(" + std::to_string(reinterpret_cast<uintptr_t>(p.value)) +
               ")";
      });

  m.def("hip_init", []() {
    const HipApi &h = getHip();
    static std::once_flag flag;
    std::call_once(flag, [&h]() {
      hipCheck(h.init(0));
      hipCheck(h.setDevice(0));
    });
  });

  // Clear any sticky HIP error from a previous failed call. Without this,
  // a failed hipModuleLoadData leaves a deferred error that the next
  // hipCheck (e.g. on hipMalloc) picks up, cascading failures across
  // configs in the same subprocess pool.
  m.def("hip_peek_at_last_error", []() -> std::string {
    const HipApi &h = getHip();
    hipError_t err = h.peekAtLastError();
    if (err != hipSuccess)
      return std::string(h.getErrorString(err));
    return "";
  });

  m.def("hip_clear_last_error", []() { (void)getHip().getLastError(); });

  // Query all device properties needed for occupancy/resource checks.
  // Returns a dict with the hardware constants that target.py hardcodes.
  // Source: clr/rocclr/device/rocm/rocdevice.cpp (lines 1593-1610).
  m.def("hip_get_device_props", [](int device_id) -> nb::dict {
    hipDeviceProp_t props;
    hipCheck(getHip().getDeviceProperties(&props, device_id));
    nb::dict d;
    d["name"] = std::string(props.name);
    d["gcn_arch_name"] = std::string(props.gcnArchName);
    d["warp_size"] = props.warpSize;
    // LDS per CU (bytes).
    d["lds_per_cu"] = props.sharedMemPerMultiprocessor;
    // Register file: regsPerMultiprocessor = vgprsPerSimd * simdPerCU *
    // warpSize e.g. 512 * 4 * 64 = 131072 on gfx942.
    d["regs_per_multiprocessor"] = props.regsPerMultiprocessor;
    // CU count.
    d["multiprocessor_count"] = props.multiProcessorCount;
    // Max threads per block.
    d["max_threads_per_block"] = props.maxThreadsPerBlock;
    // Max threads per multiprocessor (= max waves per CU * warpSize).
    d["max_threads_per_multiprocessor"] = props.maxThreadsPerMultiProcessor;
    return d;
  });

  m.def("hip_module_load_data", [](const nb::bytes &binary) -> Ptr {
    hipModule_t result = nullptr;
    hipCheck(getHip().moduleLoadData(&result, binary.data()));
    return Ptr(result);
  });

  m.def("hip_module_get_function",
        [](Ptr module, const nb::bytes &name) -> Ptr {
          hipFunction_t result = nullptr;
          hipCheck(getHip().moduleGetFunction(
              &result, module.as<ihipModule_t>(),
              reinterpret_cast<const char *>(name.data())));
          return Ptr(result);
        });

  m.def("hip_module_unload", [](Ptr module) {
    if (!module)
      return;
    hipCheck(getHip().moduleUnload(module.as<ihipModule_t>()));
  });

  m.def(
      "hip_module_launch_kernel",
      [](Ptr function, int64_t gx, int64_t gy, int64_t gz, int64_t bx,
         int64_t by, int64_t bz, Ptr kernelParams) {
        hipCheck(getHip().moduleLaunchKernel(
            function.as<ihipFunction_t>(), static_cast<uint32_t>(gx),
            static_cast<uint32_t>(gy), static_cast<uint32_t>(gz),
            static_cast<uint32_t>(bx), static_cast<uint32_t>(by),
            static_cast<uint32_t>(bz),
            /*sharedMem=*/0,
            /*stream=*/nullptr, kernelParams.as<void *>(),
            /*extra=*/nullptr));
      },
      nb::arg("function"), nb::arg("gx"), nb::arg("gy"), nb::arg("gz"),
      nb::arg("bx"), nb::arg("by"), nb::arg("bz"),
      nb::arg("kernelParams") = nb::none());

  m.def("hip_device_synchronize",
        []() { hipCheck(getHip().deviceSynchronize()); });

  m.def("hip_malloc", [](int64_t size) -> Ptr {
    void *ptr = nullptr;
    hipCheck(getHip().malloc(&ptr, static_cast<size_t>(size)));
    return Ptr(ptr);
  });

  m.def("hip_free", [](Ptr ptr) {
    if (!ptr)
      return;
    hipCheck(getHip().free(ptr.as<void>()));
  });

  m.def(
      "hip_memcpy_host_to_device_async",
      [](Ptr dst, Ptr src, int64_t size, nb::object stream) {
        Ptr s = stream.is_none() ? Ptr() : nb::cast<Ptr>(stream);
        hipCheck(getHip().memcpyAsync(
            dst.as<void>(), src.as<void>(), static_cast<size_t>(size),
            hipMemcpyHostToDevice, s.as<ihipStream_t>()));
      },
      nb::arg("dst"), nb::arg("src"), nb::arg("size"),
      nb::arg("stream") = nb::none());

  m.def(
      "hip_memcpy_device_to_host_async",
      [](Ptr dst, Ptr src, int64_t size, nb::object stream) {
        Ptr s = stream.is_none() ? Ptr() : nb::cast<Ptr>(stream);
        hipCheck(getHip().memcpyAsync(
            dst.as<void>(), src.as<void>(), static_cast<size_t>(size),
            hipMemcpyDeviceToHost, s.as<ihipStream_t>()));
      },
      nb::arg("dst"), nb::arg("src"), nb::arg("size"),
      nb::arg("stream") = nb::none());

  m.def(
      "hip_memcpy_device_to_device_async",
      [](Ptr dst, Ptr src, int64_t size, nb::object stream) {
        Ptr s = stream.is_none() ? Ptr() : nb::cast<Ptr>(stream);
        hipCheck(getHip().memcpyAsync(
            dst.as<void>(), src.as<void>(), static_cast<size_t>(size),
            hipMemcpyDeviceToDevice, s.as<ihipStream_t>()));
      },
      nb::arg("dst"), nb::arg("src"), nb::arg("size"),
      nb::arg("stream") = nb::none());

  m.def(
      "hip_memset_async",
      [](Ptr dst, int value, int64_t size, nb::object stream) {
        Ptr s = stream.is_none() ? Ptr() : nb::cast<Ptr>(stream);
        hipCheck(getHip().memsetAsync(dst.as<void>(), value,
                                      static_cast<size_t>(size),
                                      s.as<ihipStream_t>()));
      },
      nb::arg("dst"), nb::arg("value"), nb::arg("size"),
      nb::arg("stream") = nb::none());

  m.def("hip_get_device_count", []() -> int {
    int count = 0;
    hipCheck(getHip().getDeviceCount(&count));
    return count;
  });

  m.def("hip_set_device",
        [](int device_id) { hipCheck(getHip().setDevice(device_id)); });

  m.def("hip_device_reset", []() { hipCheck(getHip().deviceReset()); });

  m.def("hip_get_device", []() -> int {
    int device_id = 0;
    hipCheck(getHip().getDevice(&device_id));
    return device_id;
  });

  m.def("hip_event_create", []() -> Ptr {
    hipEvent_t result = nullptr;
    hipCheck(getHip().eventCreate(&result));
    return Ptr(result);
  });

  m.def("hip_event_destroy", [](Ptr event) {
    if (!event)
      return;
    hipCheck(getHip().eventDestroy(event.as<ihipEvent_t>()));
  });

  m.def(
      "hip_event_record",
      [](Ptr event, nb::object stream) {
        Ptr s = stream.is_none() ? Ptr() : nb::cast<Ptr>(stream);
        hipCheck(getHip().eventRecord(event.as<ihipEvent_t>(),
                                      s.as<ihipStream_t>()));
      },
      nb::arg("event"), nb::arg("stream") = nb::none());

  m.def("hip_event_synchronize", [](Ptr event) {
    const HipApi &h = getHip();
    hipCheck(h.eventSynchronize(event.as<ihipEvent_t>()));
  });

  m.def("hip_event_elapsed_time", [](Ptr start, Ptr stop) -> double {
    float ms = 0.0f;
    hipCheck(getHip().eventElapsedTime(&ms, start.as<ihipEvent_t>(),
                                       stop.as<ihipEvent_t>()));
    return ms;
  });

  m.def("hip_stream_create", []() -> Ptr {
    hipStream_t result = nullptr;
    hipCheck(getHip().streamCreate(&result));
    return Ptr(result);
  });

  m.def("hip_stream_destroy", [](Ptr stream) {
    if (!stream)
      return;
    hipCheck(getHip().streamDestroy(stream.as<ihipStream_t>()));
  });

  m.def("hip_stream_synchronize", [](Ptr stream) {
    hipCheck(getHip().streamSynchronize(stream.as<ihipStream_t>()));
  });

  m.def(
      "hip_occupancy_max_active_blocks_per_multiprocessor",
      [](Ptr function, int blockSize, int64_t dynSharedMemPerBlk) -> int {
        if (dynSharedMemPerBlk < 0)
          throw std::invalid_argument(
              "dyn_shared_mem_per_blk must be non-negative");
        int numBlocks = 0;
        hipCheck(getHip().moduleOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, function.as<ihipFunction_t>(), blockSize,
            static_cast<size_t>(dynSharedMemPerBlk)));
        return numBlocks;
      },
      nb::arg("function"), nb::arg("block_size"),
      nb::arg("dyn_shared_mem_per_blk") = 0);
}
