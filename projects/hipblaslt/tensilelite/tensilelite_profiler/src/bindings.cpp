// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Self-contained nanobind module wrapping ROCprofiler-SDK counter collection.
 * rocprofiler_configure is exported with default visibility so that
 * rocprofiler_force_configure (called at module import) triggers context setup
 * synchronously before any HIP call initialises HSA.
 */
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <hip/hip_runtime_api.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstdio>
#include <future>
#include <iomanip>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nb = nanobind;

// ──────────────────────────────────────────────────────────────────────────
// Profiler state singleton
// ──────────────────────────────────────────────────────────────────────────

namespace {

struct DimInfo {
    size_t      total;  // product of all dim instance_sizes.
    std::vector<rocprofiler_counter_record_dimension_info_t> dims;
    std::vector<std::string> dimNames;
    std::vector<size_t>      strides;
};

using CounterVal = std::variant<double, std::vector<double>>;

struct PyProfiler {
    bool doProfile     = false;
    bool initialized   = false;
    bool contextStarted = false;
    rocprofiler_context_id_t        context{};
    rocprofiler_agent_v0_t          agent{};
    rocprofiler_counter_config_id_t agentProfile{};
    std::set<std::string>           counterNames;
    std::unordered_map<std::string, rocprofiler_counter_id_t> name2Id;
    std::unordered_map<uint64_t, DimInfo>     dimInfos;
    std::mutex                                mutex;
    std::promise<void>                        promise;
    std::future<void>                         future;
    std::unordered_map<uint64_t, CounterVal>  record;
};

PyProfiler& profiler()
{
    static PyProfiler p;
    return p;
}

// ──────────────────────────────────────────────────────────────────────────
// Small helpers
// ──────────────────────────────────────────────────────────────────────────

static uint32_t locationIdFromDevice(int deviceIdx)
{
    char pci[32] = {};
    if(hipDeviceGetPCIBusId(pci, sizeof(pci), deviceIdx) != hipSuccess)
        throw std::runtime_error("failed to get PCI bus id for device");
    int dom = 0, bus = 0, dev = 0, fnc = 0;
    bool ok = (std::sscanf(pci, "%x:%x:%x.%u", &dom, &bus, &dev, &fnc) == 4);
    if(!ok)
        ok = (std::sscanf(pci, "%x:%x.%u", &bus, &dev, &fnc) == 3);
    if(!ok)
        throw std::runtime_error("failed to parse PCI bus id");
    return ((static_cast<uint32_t>(bus) & 0xFF) << 8)
         | ((static_cast<uint32_t>(dev) & 0x1F) << 3)
         | (static_cast<uint32_t>(fnc) & 0x07);
}

static std::string dimAbbrev(const char* name)
{
    std::string s = name ? name : "";
    if(s == "DIMENSION_SHADER_ENGINE") return "SE";
    if(s == "DIMENSION_SHADER_ARRAY")  return "SA";
    if(s == "DIMENSION_INSTANCE")      return "INST";
    if(s.size() > 10 && s.compare(0, 10, "DIMENSION_") == 0)
        return s.substr(10);
    return s;
}

// Format a scalar counter value, omitting the decimal point for whole numbers.
static void appendScalar(std::ostringstream& ss, double v)
{
    if(v == static_cast<double>(static_cast<int64_t>(v)))
        ss << std::fixed << std::setprecision(0) << v;
    else
        ss << std::defaultfloat << v;
}

// Format one multi-dim counter instance at linear index i.
static void appendInstance(std::ostringstream&      ss,
                            const std::string&       name,
                            const DimInfo&           info,
                            const std::vector<double>& vals,
                            size_t                   i)
{
    ss << name << '(';
    size_t pos = i, nd = info.strides.size();
    for(size_t j = 0; j < nd; ++j) {
        size_t k      = nd - 1 - j;
        size_t stride = info.strides[k];
        size_t idx    = pos / stride;
        ss << info.dimNames[k] << '=' << idx;
        if(k != 0) ss << ',';
        pos %= stride;
    }
    ss << "): ";
    appendScalar(ss, vals[i]);
}

// Build the formatted counter string from the current dispatch record.
static std::string formatRecord()
{
    PyProfiler& p = profiler();
    std::ostringstream ss;
    bool first = true;
    for(auto const& [name, id] : p.name2Id) {
        auto cit = p.record.find(id.handle);
        if(cit == p.record.end()) continue;
        DimInfo& info = p.dimInfos.at(id.handle);
        if(!first) ss << ',';
        first = false;
        if(info.total <= 1) {
            ss << name << ": ";
            appendScalar(ss, std::get<double>(cit->second));
        } else {
            std::vector<double>& vals = std::get<std::vector<double>>(cit->second);
            for(size_t i = 0; i < info.total; ++i) {
                if(i != 0) ss << ',';
                appendInstance(ss, name, info, vals, i);
            }
        }
    }
    return ss.str();
}

// ──────────────────────────────────────────────────────────────────────────
// ROCprofiler-SDK callbacks
// ──────────────────────────────────────────────────────────────────────────

// Profile any kernel when armed; no kernel-ID filtering for Python use.
static void dispatchCallback(rocprofiler_dispatch_counting_service_data_t dispatchData,
                             rocprofiler_counter_config_id_t*             config,
                             rocprofiler_user_data_t*                     userData,
                             void*                                        /*callbackData*/)
{
    PyProfiler& p = profiler();
    std::lock_guard<std::mutex> lock(p.mutex);
    if(p.doProfile) {
        *config         = p.agentProfile;
        userData->value = 0;
    } else {
        *config = rocprofiler_counter_config_id_t{0};
    }
}

static void storeCounterInstance(PyProfiler&                         p,
                                 const rocprofiler_counter_record_t& rec,
                                 uint64_t                            handle,
                                 const DimInfo&                      info)
{
    if(info.total <= 1) {
        p.record[handle] = rec.counter_value;
        return;
    }
    auto it = p.record.find(handle);
    if(it == p.record.end()) {
        p.record[handle] = std::vector<double>(info.total, 0.0);
        it               = p.record.find(handle);
    }
    size_t linearIdx = 0;
    for(size_t j = 0; j < info.dims.size(); ++j) {
        size_t pos = 0;
        rocprofiler_query_record_dimension_position(rec.id, info.dims[j].id, &pos);
        linearIdx += pos * info.strides[j];
    }
    std::get<std::vector<double>>(it->second)[linearIdx] = rec.counter_value;
}

static void recordCallback(rocprofiler_dispatch_counting_service_data_t /*dispatchData*/,
                           rocprofiler_counter_record_t*                recordData,
                           unsigned long                                recordCount,
                           rocprofiler_user_data_t                      /*userData*/,
                           void*                                        /*callbackData*/)
{
    PyProfiler& p = profiler();
    std::lock_guard<std::mutex> lock(p.mutex);
    p.record.clear();
    for(size_t i = 0; i < recordCount; ++i) {
        rocprofiler_counter_id_t cid{.handle = 0};
        rocprofiler_query_record_counter_id(recordData[i].id, &cid);
        auto dit = p.dimInfos.find(cid.handle);
        if(dit == p.dimInfos.end()) continue;
        storeCounterInstance(p, recordData[i], cid.handle, dit->second);
    }
    p.promise.set_value();
}

static int toolInitImpl(rocprofiler_client_finalize_t /*finiFunc*/, void* /*toolData*/)
{
    PyProfiler& p  = profiler();
    rocprofiler_status_t rc = rocprofiler_create_context(&p.context);
    if(rc != ROCPROFILER_STATUS_SUCCESS) return -1;
    rc = rocprofiler_configure_callback_dispatch_counting_service(
        p.context, dispatchCallback, nullptr, recordCallback, nullptr);
    return rc == ROCPROFILER_STATUS_SUCCESS ? 0 : -1;
}

}  // namespace

extern "C" {
    __attribute__((visibility("default")))
    rocprofiler_tool_configure_result_t*
    rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
    {
        static rocprofiler_tool_configure_result_t result{
            .size       = sizeof(rocprofiler_tool_configure_result_t),
            .initialize = toolInitImpl,
            .finalize   = nullptr,
            .tool_data  = nullptr,
        };
        return &result;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Initialize helpers (query agent, build counter profile)
// ──────────────────────────────────────────────────────────────────────────

namespace {

static rocprofiler_agent_v0_t queryAgent(int deviceIdx)
{
    uint32_t locationId = locationIdFromDevice(deviceIdx);
    struct Ctx { uint32_t loc; rocprofiler_agent_v0_t agent{}; bool found = false; } ctx{locationId};
    auto cb = [](rocprofiler_agent_version_t, const void** agents, size_t n, void* data) {
        auto* c = static_cast<Ctx*>(data);
        for(size_t i = 0; i < n; ++i) {
            const auto* a = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
            if(a->type == ROCPROFILER_AGENT_TYPE_GPU && a->location_id == c->loc) {
                c->agent = *a;
                c->found = true;
            }
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };
    rocprofiler_status_t rc = rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0, cb, sizeof(rocprofiler_agent_v0_t), &ctx);
    if(rc != ROCPROFILER_STATUS_SUCCESS || !ctx.found)
        throw std::runtime_error("no GPU agent found for given device index");
    return ctx.agent;
}

static DimInfo buildDimInfo(const rocprofiler_counter_record_dimension_info_t* dims,
                             size_t dimCount)
{
    DimInfo info;
    info.dims.assign(dims, dims + dimCount);
    info.strides.push_back(1);
    for(size_t j = 0; j + 1 < dimCount; ++j)
        info.strides.push_back(info.dims[j].instance_size * info.strides.back());
    info.total = dimCount > 0 ? info.dims.back().instance_size * info.strides.back() : 1;
    for(auto& d : info.dims)
        info.dimNames.push_back(dimAbbrev(d.name));
    return info;
}

static void buildProfiles(PyProfiler& p, rocprofiler_agent_id_t agentId)
{
    struct Ctx { PyProfiler* p; std::vector<rocprofiler_counter_id_t> ids; bool failed = false; };
    Ctx ctx{&p};
    auto cb = [](rocprofiler_agent_id_t,
                 rocprofiler_counter_id_t* counters,
                 size_t n, void* data) {
        auto* c = static_cast<Ctx*>(data);
        rocprofiler_counter_info_v1_t info{};
        for(size_t i = 0; i < n; ++i) {
            if(rocprofiler_query_counter_info(counters[i],
                    ROCPROFILER_COUNTER_INFO_VERSION_1, &info) != ROCPROFILER_STATUS_SUCCESS)
                continue;
            if(!c->p->counterNames.count(info.name)) continue;
            c->ids.push_back(counters[i]);
            c->p->name2Id[info.name] = counters[i];
            const auto* dims = *info.dimensions;
            c->p->dimInfos[counters[i].handle] = buildDimInfo(dims, info.dimensions_count);
        }
        for(auto const& name : c->p->counterNames) {
            if(!c->p->name2Id.count(name)) {
                c->failed = true;
                break;
            }
        }
        if(c->failed) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        return ROCPROFILER_STATUS_SUCCESS;
    };
    rocprofiler_status_t rc = rocprofiler_iterate_agent_supported_counters(agentId, cb, &ctx);
    if(rc != ROCPROFILER_STATUS_SUCCESS || ctx.failed)
        throw std::runtime_error("one or more requested counters not available on this agent");
    rocprofiler_counter_config_id_t profile{};
    rc = rocprofiler_create_counter_config(
        agentId, ctx.ids.data(), ctx.ids.size(), &profile);
    if(rc != ROCPROFILER_STATUS_SUCCESS)
        throw std::runtime_error("failed to create counter config");
    p.agentProfile = profile;
}

static void doStart()
{
    PyProfiler& p = profiler();
    std::lock_guard<std::mutex> lock(p.mutex);
    if(p.contextStarted || p.context.handle == 0) return;
    rocprofiler_start_context(p.context);
    p.contextStarted = true;
}

static void doStop()
{
    PyProfiler& p = profiler();
    std::lock_guard<std::mutex> lock(p.mutex);
    if(!p.contextStarted || p.context.handle == 0) return;
    rocprofiler_stop_context(p.context);
    p.contextStarted = false;
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────
// NB_MODULE helpers
// ──────────────────────────────────────────────────────────────────────────

// Trigger rocprofiler tool registration and set the module docstring.
static void _registerPyInit(nb::module_& m)
{
    // Must happen before HSA is initialised (i.e. before any HIP call).
    rocprofiler_status_t status = rocprofiler_force_configure(rocprofiler_configure);
    if(status == ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED)
        throw std::runtime_error("import tensilelite_profiler before any HIP call");
    if(status != ROCPROFILER_STATUS_SUCCESS) {
        std::ostringstream msg;
        msg << "rocprofiler_force_configure failed with status " << static_cast<int>(status);
        throw std::runtime_error(msg.str());
    }
    m.doc() = "TensileLite ROCprofiler-SDK hardware counter collection bindings.";
}

// Register all counter-collection Python bindings on the module.
static void _registerCounterBindings(nb::module_& m)
{
    m.def(
        "initialize",
        [](int deviceIdx, std::vector<std::string> const& names) {
            PyProfiler& p = profiler();
            if(p.initialized) return;
            p.counterNames = {names.begin(), names.end()};
            p.agent        = queryAgent(deviceIdx);
            buildProfiles(p, p.agent.id);
            p.initialized  = true;
        },
        nb::arg("device_idx"),
        nb::arg("counter_names"),
        "Initialise the profiler for device_idx with the named hardware counters.");

    m.def("start", &doStart, "Start the rocprofiler context (call after initialize).");
    m.def("stop",  &doStop,  "Stop the rocprofiler context.");

    m.def(
        "enable",
        []() {
            PyProfiler& p = profiler();
            std::lock_guard<std::mutex> lock(p.mutex);
            p.promise   = std::promise<void>{};
            p.future    = p.promise.get_future();
            p.doProfile = true;
        },
        "Reset the promise and arm the profiler for the next kernel dispatch.");

    m.def(
        "disable",
        []() {
            PyProfiler& p = profiler();
            std::lock_guard<std::mutex> lock(p.mutex);
            p.doProfile = false;
        },
        "Disarm the profiler after a dispatch.");

    m.def(
        "fetch",
        [](int /*solutionIdx*/) -> std::string {
            PyProfiler& p = profiler();
            {
                // Release GIL while blocking: recordCallback fires from a
                // C++ HSA signal thread and must not compete with Python.
                nb::gil_scoped_release release;
                p.future.get();
            }
            std::lock_guard<std::mutex> lock(p.mutex);
            return formatRecord();
        },
        nb::arg("solution_idx"),
        "Block until the counter record arrives and return a formatted string.");
}

// ──────────────────────────────────────────────────────────────────────────
// NB_MODULE
// ──────────────────────────────────────────────────────────────────────────

NB_MODULE(_tensilelite_profiler, m)
{
    _registerPyInit(m);
    _registerCounterBindings(m);
}
