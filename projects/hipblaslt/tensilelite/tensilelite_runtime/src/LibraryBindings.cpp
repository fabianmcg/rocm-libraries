// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "LibraryBindings.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/UtilsOrigami.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>
#include <origami/simulator/tensilelite/formocast_simulator.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <stdexcept>

namespace nb = nanobind;
namespace TL = TensileLite;

using CPG    = TL::ContractionProblemGemm;
using CS     = TL::ContractionSolution;
using SolLib = TL::SolutionLibrary<CPG, CS>;
using LibPtr = std::shared_ptr<SolLib>;
using SolPtr = std::shared_ptr<CS>;
using HWPtr  = std::shared_ptr<TL::Hardware>;

// ---------------------------------------------------------------------------
// Origami adapter helpers
// ---------------------------------------------------------------------------

// Translate a TensileLite SizeMapping to an origami Formocast::SizeMapping.
static origami::Formocast::SizeMapping toOrigamiSizeMapping(TL::SizeMapping const& sm)
{
    origami::Formocast::SizeMapping fsm{};
    fsm.waveNum            = sm.waveNum;
    fsm.macroTile          = {static_cast<int>(sm.macroTile.x),
                              static_cast<int>(sm.macroTile.y),
                              static_cast<int>(sm.macroTile.z)};
    fsm.matrixInstruction  = sm.matrixInstruction;
    fsm.grvwA              = sm.grvwA;
    fsm.grvwB              = sm.grvwB;
    fsm.gwvwC              = sm.gwvwC;
    fsm.gwvwD              = sm.gwvwD;
    fsm.depthU             = sm.depthU;
    fsm.globalSplitU       = sm.globalSplitU;
    fsm.workGroupMapping   = sm.workGroupMapping;
    fsm.globalAccumulation = sm.globalAccumulation;
    fsm.workGroupMappingXCC               = sm.workGroupMappingXCC;
    fsm.workGroupMappingXCCGroup          = sm.workGroupMappingXCCGroup;
    fsm.globalSplitUCoalesced             = sm.globalSplitUCoalesced;
    fsm.globalSplitUWorkGroupMappingRoundRobin = sm.globalSplitUWorkGroupMappingRoundRobin;
    fsm.CUOccupancy            = sm.CUOccupancy;
    fsm.PrefetchGlobalRead     = sm.PrefetchGlobalRead;
    fsm.MathClocksUnrolledLoop = sm.MathClocksUnrolledLoop;
    fsm.DirectToVgprA          = sm.DirectToVgprA;
    fsm.DirectToVgprB          = sm.DirectToVgprB;
    fsm.NumLoadsCoalescedA     = sm.NumLoadsCoalescedA;
    fsm.NumLoadsCoalescedB     = sm.NumLoadsCoalescedB;
    fsm.VectorWidthA           = sm.VectorWidthA;
    fsm.VectorWidthB           = sm.VectorWidthB;
    fsm.LocalSplitU            = sm.LocalSplitU;
    fsm.waveGroup              = sm.waveGroup;
    fsm.DirectToLdsA           = sm.DirectToLdsA;
    fsm.DirectToLdsB           = sm.DirectToLdsB;
    return fsm;
}

// Populate origami ProblemInfo from a ContractionProblemGemm + ContractionSolution.
static origami::Formocast::ProblemInfo toOrigamiProblemInfo(CS const& sol, CPG const& prob)
{
    double k = 1.0;
    for(size_t i = 0; i < prob.boundIndices().size(); ++i)
        k *= static_cast<double>(prob.boundSize(i));

    origami::Formocast::ProblemInfo pi{};
    pi.M          = sol.calculateDimensionM(prob);
    pi.N          = sol.calculateDimensionN(prob);
    pi.NumBatches = sol.calculateNumBatches(prob);
    pi.K          = k;
    pi.bpeA       = static_cast<uint32_t>(TL::DataTypeInfo::Get(prob.a().dataType()).elementSize);
    pi.bpeB       = static_cast<uint32_t>(TL::DataTypeInfo::Get(prob.b().dataType()).elementSize);
    pi.bpeD       = static_cast<uint32_t>(TL::DataTypeInfo::Get(prob.d().dataType()).elementSize);
    pi.bpeCompute = static_cast<uint32_t>(
        TL::DataTypeInfo::Get(prob.computeInputTypeA()).elementSize);
    pi.transA         = prob.transA();
    pi.transB         = prob.transB();
    pi.swizzleTensorA = sol.problemType.swizzleTensorA;
    pi.swizzleTensorB = sol.problemType.swizzleTensorB;
    pi.dataType       = sol.getOrigamiDatatype(prob);
    return pi;
}

// Run Formocast simulation and return predicted GFLOPS (0 if simulation fails).
static double computeGflops(CS const&                         sol,
                             CPG const&                        prob,
                             origami::hardware_t::architecture_t arch)
{
    origami::Formocast fc;
    fc.setHardware(arch);
    fc.setProblem(toOrigamiProblemInfo(sol, prob));
    fc.setSolution(toOrigamiSizeMapping(sol.getSizeMapping()));

    auto perf = fc.predictedPerformance();
    if(perf.microSeconds <= 0.0)
        return 0.0;

    auto const& pi    = fc.problem;
    double      flops = 2.0 * pi.M * pi.N * pi.K * pi.NumBatches;
    return flops / (perf.microSeconds * 1e-6) / 1e9;
}

// Get origami architecture enum from a Hardware shared_ptr (via archName()).
// hw->archName() may include feature suffixes like "gfx950:sramecc+:xnack-";
// origami::arch_name_to_enum requires the bare name ("gfx950"), so strip at ":".
static origami::hardware_t::architecture_t archFromHardware(HWPtr const& hw)
{
    if(!hw)
        throw std::runtime_error("null hardware pointer");
    std::string name = hw->archName();
    auto        sep  = name.find(':');
    if(sep != std::string::npos)
        name.resize(sep);
    return origami::hardware_t::arch_name_to_enum(name);
}

// ---------------------------------------------------------------------------
// Python Problem factory
// ---------------------------------------------------------------------------

// Build a ContractionProblemGemm from Python-supplied arguments using the
// GEMM_Strides factory with column-major defaults (lda = leading row dim).
// computeInputTypeA/B default to the matrix data types (the common case);
// highPrecisionAccumulate must be set explicitly for Half-input kernels that
// accumulate in Float.
static CPG makeProblem(size_t      m,
                       size_t      n,
                       size_t      k,
                       std::string dtypeA,
                       std::string dtypeB,
                       std::string dtypeC,
                       std::string dtypeD,
                       std::string computeInputTypeA,
                       std::string computeInputTypeB,
                       bool        transA,
                       bool        transB,
                       bool        highPrecisionAccumulate,
                       size_t      batchSize,
                       double      beta)
{
    rocisa::DataType aType   = TL::DataTypeInfo::Get(dtypeA).dataType;
    rocisa::DataType bType   = TL::DataTypeInfo::Get(dtypeB).dataType;
    rocisa::DataType cType   = TL::DataTypeInfo::Get(dtypeC).dataType;
    rocisa::DataType dType   = TL::DataTypeInfo::Get(dtypeD).dataType;
    // Empty string means "same as the matrix data type" (the common case).
    rocisa::DataType ciTypeA = computeInputTypeA.empty()
                                   ? aType
                                   : TL::DataTypeInfo::Get(computeInputTypeA).dataType;
    rocisa::DataType ciTypeB = computeInputTypeB.empty()
                                   ? bType
                                   : TL::DataTypeInfo::Get(computeInputTypeB).dataType;

    // Column-major leading dimensions: non-transposed A is M×K so lda=M;
    // non-transposed B is K×N so ldb=K.
    size_t lda = transA ? k : m;
    size_t ldb = transB ? n : k;
    size_t ldc = m;
    size_t ldd = m;

    size_t strideA = lda * (transA ? m : k);
    size_t strideB = ldb * (transB ? k : n);
    size_t strideC = ldc * n;
    size_t strideD = ldd * n;

    CPG problem = CPG::GEMM_Strides(transA, transB, aType, bType, cType, dType,
                                    m, n, k, batchSize,
                                    lda, strideA, ldb, strideB,
                                    ldc, strideC, ldd, strideD,
                                    beta);
    problem.setComputeInputTypeA(ciTypeA);
    problem.setComputeInputTypeB(ciTypeB);
    problem.setHighPrecisionAccumulate(highPrecisionAccumulate);
    return problem;
}

// ---------------------------------------------------------------------------
// Binding helpers
// ---------------------------------------------------------------------------

static void bindProblem(nb::module_& m)
{
    nb::class_<CPG>(m, "Problem")
        .def(nb::init_implicit<CPG>())
        .def("__init__",
             [](CPG*        self,
                size_t      M,
                size_t      N,
                size_t      K,
                std::string dtypeA,
                std::string dtypeB,
                std::string dtypeC,
                std::string dtypeD,
                std::string computeInputTypeA,
                std::string computeInputTypeB,
                bool        transA,
                bool        transB,
                bool        highPrecisionAccumulate,
                size_t      batchSize,
                double      beta) {
                 new(self)
                     CPG(makeProblem(M, N, K,
                                     std::move(dtypeA), std::move(dtypeB),
                                     std::move(dtypeC), std::move(dtypeD),
                                     std::move(computeInputTypeA),
                                     std::move(computeInputTypeB),
                                     transA, transB, highPrecisionAccumulate,
                                     batchSize, beta));
             },
             nb::arg("M"),
             nb::arg("N"),
             nb::arg("K"),
             nb::arg("dtype_a") = "Float",
             nb::arg("dtype_b") = "Float",
             nb::arg("dtype_c") = "Float",
             nb::arg("dtype_d") = "Float",
             nb::arg("compute_input_type_a") = "",
             nb::arg("compute_input_type_b") = "",
             nb::arg("trans_a")                   = false,
             nb::arg("trans_b")                   = false,
             nb::arg("high_precision_accumulate")  = false,
             nb::arg("batch_size")                 = size_t{1},
             nb::arg("beta")                       = 0.0,
             "Construct a GEMM problem with column-major layout defaults.\n\n"
             "compute_input_type_a/b default to dtype_a/b when empty.");
}

static void bindSolution(nb::module_& m)
{
    // In nanobind the holder type is not a template arg (unlike pybind11).
    // Including <nanobind/stl/shared_ptr.h> handles shared_ptr<CS> automatically.
    nb::class_<CS>(m, "Solution")
        .def_prop_ro("kernel_name",
                     [](CS const& s) { return s.kernelName; },
                     "Kernel function name for this solution.")
        .def_prop_ro("code_object_path",
                     [](CS const& s) { return s.codeObjectFilename.load(); },
                     "Path to the compiled code object file for this solution.")
        .def("eval_hardware_predicate",
             [](CS const& s, HWPtr const& hw) {
                 if(!hw)
                     throw std::runtime_error("null hardware pointer");
                 return (*s.hardwarePredicate)(*hw);
             },
             nb::arg("hw"),
             "Return True if this solution targets the given hardware.")
        .def("eval_task_predicate",
             [](CS const& s, HWPtr const& hw, CPG const& prob) {
                 if(!hw)
                     throw std::runtime_error("null hardware pointer");
                 TL::Task task(*hw, prob, s);
                 return (*s.taskPredicate)(task);
             },
             nb::arg("hw"),
             nb::arg("prob"),
             "Return True if this solution satisfies the task predicate for hw+prob.")
        .def("calculate_auto_wgm",
             [](CS const& s, CPG const& prob, HWPtr const& hw, int skgrid) -> int {
                 if(!hw)
                     throw std::runtime_error("null hardware pointer");
                 auto [wgm, xcc, xccChunk] = s.calculateAutoWGM(prob, hw.get(),
                                                                 static_cast<uint32_t>(skgrid));
                 return static_cast<int>(wgm);
             },
             nb::arg("prob"), nb::arg("hw"), nb::arg("skgrid") = 0,
             "Compute the auto work-group mapping value for this solution.")
        .def("calculate_auto_gsu",
             [](CS const& s, CPG const& prob, HWPtr const& hw) -> int {
                 if(!hw)
                     throw std::runtime_error("null hardware pointer");
                 return static_cast<int>(s.calculateAutoGSU(prob, hw.get()));
             },
             nb::arg("prob"), nb::arg("hw"),
             "Compute the auto global-split-U value for this solution.")
        .def("calculate_auto_stagger_u",
             [](CS const& s, CPG const& prob, HWPtr const& hw, int skgrid, int autoWgm) -> int {
                 if(!hw)
                     throw std::runtime_error("null hardware pointer");
                 auto [staggerU, mapping, shift]
                     = s.calculateAutoStaggerU(prob, hw.get(),
                                               static_cast<uint32_t>(skgrid),
                                               static_cast<int32_t>(autoWgm));
                 return static_cast<int>(staggerU);
             },
             nb::arg("prob"), nb::arg("hw"), nb::arg("skgrid") = 0, nb::arg("auto_wgm") = 0,
             "Compute the auto stagger-U value for this solution.");
}

static void bindHardware(nb::module_& m)
{
    nb::class_<TL::Hardware>(m, "Hardware")
        .def_prop_ro("arch_name",
                     [](TL::Hardware const& hw) { return hw.archName(); },
                     "GPU architecture name, e.g. 'gfx950'.");
}

// Sort solutions by descending Formocast GFLOPS prediction.
// Handles zero/negative predictions gracefully by placing them last.
static void sortByFormocast(TL::SolutionVector<CS>& solutions,
                             CPG const&              prob,
                             origami::hardware_t::architecture_t arch)
{
    std::stable_sort(solutions.begin(), solutions.end(),
                     [&](SolPtr const& a, SolPtr const& b) {
                         double ga = a ? computeGflops(*a, prob, arch) : 0.0;
                         double gb = b ? computeGflops(*b, prob, arch) : 0.0;
                         return ga > gb;
                     });
}

static void bindLibrary(nb::module_& m)
{
    nb::class_<SolLib>(m, "Library")
        // nb::keep_alive<0, 1>: nurse=0 (returned Solution) keeps patient=1
        // (self/Library) alive. Library-owned data must outlive any Solution
        // that references it.
        .def("find_best_solution",
             [](SolLib const& lib, HWPtr const& hw, CPG const& prob) -> SolPtr {
                 if(!hw)
                     throw std::runtime_error("null hardware pointer");
                 // GIL must remain held: SingleSolutionLibrary.hpp:170-173 has
                 // a lazy requiredHostWorkspaceSizePerProblem init that is not
                 // mutex-protected against concurrent re-entry. Releasing the
                 // GIL here would allow a second Python thread to race through
                 // that init, producing a data race. When that path gains its
                 // own mutex this keep-GIL constraint can be relaxed.
                 return lib.findBestSolution(prob, *hw);
             },
             nb::keep_alive<0, 1>(),
             nb::arg("hw"), nb::arg("prob"),
             "Return the best solution for (hw, prob), keeping Library alive.")
        // find_top_solutions returns a list of Solutions that reference Library-owned data.
        // nb::keep_alive cannot be applied to individual list elements in nanobind;
        // the caller (LibraryRunner) is responsible for keeping the Library alive as
        // long as any returned Solution is in use — LibraryRunner stores both.
        //
        // Uses findAllSolutions (not findTopSolutions) because SingleSolutionLibrary
        // — the leaf type used in both test and production YAML files — does not
        // override findTopSolutions and inherits the no-op base implementation.
        // findAllSolutions is properly overridden and returns all predicate-matching
        // solutions. We then sort by Formocast and truncate to n.
        .def("find_top_solutions",
             [](SolLib const& lib, HWPtr const& hw, CPG const& prob, int n)
                 -> std::vector<SolPtr> {
                 if(!hw)
                     throw std::runtime_error("null hardware pointer");
                 auto set = lib.findAllSolutions(prob, *hw);
                 TL::SolutionVector<CS> solutions(set.begin(), set.end());
                 // Sort by Formocast GFLOPS prediction (descending).
                 // This is the binding contract — callers rely on the order.
                 sortByFormocast(solutions, prob, archFromHardware(hw));
                 int count = std::min(n, static_cast<int>(solutions.size()));
                 return std::vector<SolPtr>(solutions.begin(), solutions.begin() + count);
             },
             nb::arg("hw"), nb::arg("prob"), nb::arg("n"),
             "Return up to n solutions sorted by Formocast prediction (best first).");
}

static void bindFreeFunctions(nb::module_& m)
{
    m.def(
        "load_library",
        [](std::string const& path) -> LibPtr {
            return TL::LoadLibraryFile<CPG, CS>(path);
        },
        nb::arg("path"),
        "Load a TensileLibrary YAML/msgpack file and return a Library.");

    m.def(
        "get_hardware",
        [](int deviceId) -> HWPtr {
            return TL::hip::GetDevice(deviceId);
        },
        nb::arg("device_id") = 0,
        "Return a Hardware descriptor for the given HIP device index.");

    m.def(
        "grouped_gemm_workspace_size",
        [](SolPtr const& sol, CPG const& prob) -> size_t {
            if(!sol)
                throw std::runtime_error("null solution pointer");
            // Use the current HIP device for workspace computation.
            auto hw = TL::hip::GetCurrentDevice();
            return sol->requiredHostSizeGroupedGemmSingle(prob, *hw);
        },
        nb::arg("solution"), nb::arg("prob"),
        "Return the per-problem host workspace size required for grouped GEMM.");

    m.def(
        "formocast_predict",
        [](SolPtr const& sol, CPG const& prob) -> double {
            if(!sol)
                throw std::runtime_error("null solution pointer");
            // Derive architecture from the current HIP device so the caller
            // does not need to pass hardware explicitly.
            auto hw   = TL::hip::GetCurrentDevice();
            auto arch = archFromHardware(hw);
            return computeGflops(*sol, prob, arch);
        },
        nb::arg("solution"), nb::arg("prob"),
        "Return predicted throughput in GFLOPS for the given solution and problem.");
}

// ---------------------------------------------------------------------------
// Entry point called from bindings.cpp NB_MODULE block
// ---------------------------------------------------------------------------

void bindLibraryTypes(nb::module_& m)
{
    bindProblem(m);
    bindHardware(m);
    bindSolution(m);
    bindLibrary(m);
    bindFreeFunctions(m);
}
