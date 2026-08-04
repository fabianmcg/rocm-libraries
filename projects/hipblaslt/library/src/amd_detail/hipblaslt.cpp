/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "hipblaslt/hipblaslt.h"
#include "UserDrivenTuningParser.hpp"
#include "check_numerics_matrix.hpp"
#include "exceptions.hpp"
#include "handle.h"
#include "hipblaslt/hipblaslt-ext-op.h"
#include "hipblaslt/hipblaslt_float8.h"
#include "hipblaslt_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocblaslt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "Debug.hpp"

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

bool override_path_compare_git_version(OverrideSingleton& override, hipblasLtHandle_t& handle)
{
    char git_version[128];
    hipblasLtGetGitRevision(handle, &git_version[0]);
    static std::string cached_firstline;
    static std::string cached_path;
    static bool        cached = false;
    std::string        firstline;

    if(!cached || cached_path != override.file_path)
    {
        std::ifstream file_read(override.file_path);
        std::getline(file_read, firstline);
        cached_firstline = firstline;
        cached_path      = override.file_path;
        cached           = true;
    }
    else
    {
        firstline = cached_firstline;
    }

    std::string header = "Git Version: ";
    size_t      pos    = firstline.find(header);
    if(pos != std::string::npos)
    {
        std::string file_version = firstline.substr(pos + header.length());
        if(file_version == git_version)
            return true;
    }

    override.env_mode = false;

    return false;
}

hipblasStatus_t RocBlasLtStatusToHIPStatus(rocblaslt_status_ status)
{
    switch(status)
    {
    case rocblaslt_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblaslt_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblaslt_status_not_implemented:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblaslt_status_invalid_pointer:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_invalid_size:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblaslt_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblaslt_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_arch_mismatch:
        return HIPBLAS_STATUS_ARCH_MISMATCH;
    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_HIPBLASLT_ERROR(INPUT_STATUS_FOR_CHECK)              \
    {                                                                  \
        hipblasStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != HIPBLAS_STATUS_SUCCESS)             \
        {                                                              \
            return TMP_STATUS_FOR_CHECK;                               \
        }                                                              \
    }

#define RETURN_IF_ROCBLASLT_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                   \
        rocblaslt_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblaslt_status_success)            \
        {                                                               \
            return RocBlasLtStatusToHIPStatus(TMP_STATUS_FOR_CHECK);    \
        }                                                               \
    }

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtCreate");

    // Check if handle is valid
    if(handle == nullptr)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int             deviceId;
    hipError_t      err;
    hipblasStatus_t retval = HIPBLAS_STATUS_SUCCESS;
    // TODO: Synchronizer size pass into predicate SynchronizerSizeCheck
    // 1K just for small size now, need to cal corner case if support all situations
    void* d_Synchronizer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_Synchronizer, 16 * 409600 * sizeof(int)));
    CHECK_HIP_ERROR(hipMemset(d_Synchronizer, 0, sizeof(int) * 16 * 409600));

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = RocBlasLtStatusToHIPStatus(rocblaslt_create((rocblaslt_handle*)handle));
        (*(rocblaslt_handle*)handle)->Synchronizer = d_Synchronizer;
    }
    rocblaslt::Debug::Instance().markerStop();
    return retval;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtDestroy");
    if(handle != nullptr and (*(rocblaslt_handle)handle).Synchronizer != nullptr)
    {
        CHECK_HIP_ERROR(hipFree((*(rocblaslt_handle)handle).Synchronizer));
    }

    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_destroy((const rocblaslt_handle)handle));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtSetSmCountTarget(hipblasLtHandle_t handle, int32_t smCountTarget)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtSetSmCountTarget");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_set_sm_count_target((rocblaslt_handle)handle, smCountTarget));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtGetSmCountTarget(hipblasLtHandle_t handle, int32_t* smCountTarget)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtGetSmCountTarget");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_get_sm_count_target((rocblaslt_handle)handle, smCountTarget));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtCheckNumericsDrain(hipblasLtHandle_t handle, uint32_t* first_nan_call_id)
try
{
    if(handle == nullptr)
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    const uint32_t first_nan = hipblaslt_check_numerics_drain_handle((rocblaslt_handle)handle);
    if(first_nan_call_id)
        *first_nan_call_id = first_nan;
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* matDescr,
                                            hipDataType              valueType,
                                            uint64_t                 rows,
                                            uint64_t                 cols,
                                            int64_t                  ld)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixLayoutCreate");
    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matrix_layout_create(
        (rocblaslt_matrix_layout*)matDescr, valueType, rows, cols, ld));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t descr)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixLayoutDestroy");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_destroy((const rocblaslt_matrix_layout)descr));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t* matmulDesc,
                                          hipblasComputeType_t   computeType,
                                          hipDataType            scaleType)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescCreate");
    char* override = std::getenv("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32");
    if(override && (computeType == hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_TF32)
       && (std::string(override) != ""))
    {
        switch(std::stoi(std::string(override)))
        {
        case 0:
            computeType = hipblasComputeType_t::HIPBLAS_COMPUTE_32F;
            break;
        case 2:
            computeType = hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_16BF;
            break;
        case 1:
        default:
            break;
        }
    }
    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul_desc_create(
        (rocblaslt_matmul_desc*)matmulDesc, (rocblaslt_compute_type)computeType, scaleType));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t          matLayout,
                                                  hipblasLtMatrixLayoutAttribute_t attr,
                                                  const void*                      buf,
                                                  size_t                           sizeInBytes)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixLayoutSetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_set_attribute((rocblaslt_matrix_layout)matLayout,
                                              (rocblaslt_matrix_layout_attribute)attr,
                                              buf,
                                              sizeInBytes));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(hipblasLtMatrixLayout_t          matLayout,
                                                  hipblasLtMatrixLayoutAttribute_t attr,
                                                  void*                            buf,
                                                  size_t                           sizeInBytes,
                                                  size_t*                          sizeWritten)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixLayoutGetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_get_attribute((rocblaslt_matrix_layout)matLayout,
                                              (rocblaslt_matrix_layout_attribute)attr,
                                              buf,
                                              sizeInBytes,
                                              sizeWritten));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t descr)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescDestroy");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_destroy((const rocblaslt_matmul_desc)descr));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Per-call handoff from the RMSNorm producer GEMM (K1) to the cross-tile reduction (K2).
// K1 writes partialBuf through the normal Tensile problem inputs; K2 consumes that buffer plus
// the by-value arguments below. Full RMSNorm reduces and applies immediately, while the
// decomposed flow has K2 write the final row scale into the descriptor below for GEMM2.
struct RmsNormHandoff
{
    void*   partialBuf = nullptr; // f32 partial sums, row-major [M_padded, nTilesN]
    int32_t M          = 0;       // logical rows; padded rows in partialBuf are ignored
    int32_t N          = 0;       // feature dimension reduced by RMSNorm
    int32_t nTilesN    = 0;       // columns of partialBuf, ceil(N / MacroTile1)
    float   invD       = 0.f;     // 1 / N
    float   eps        = 0.f;     // RMSNorm epsilon
};

// Cross-call state for the decomposed RMSNorm flow. The producer reduction materializes the
// finalized rstd here, and the later GEMM2 scale-apply epilogue consumes it. The full flow never
// creates this handle because its reduction applies rstd to D in the same matmul call.
struct hipblasLtFusedEpilogueRMSNormDescriptor
{
    // FP32 rstd, tightly packed [M * batch]. M and batch are implicit in the consumer GEMM2
    // problem, which must match the producer for the decomposed flow.
    void* per_row_scale = nullptr;
    // Set true once a requant-path producer (per-row or MX block) has populated the handoff
    // scales; the plain reduce path leaves per_row_scale null and only fills host_scale.
    bool populated = false;

    // CPU-shim storage for the finalized per-row scale (rstd), laid out as
    // [batch * rows + row], FP32. The producer (partial RMSNorm stats) fills this; the
    // consumer (RMSNorm scale-apply) reads it. Not part of the eventual on-device layout.
    std::vector<float> host_scale;
    // CPU-shim storage for the per-block UE8M0 MX scales, laid out row-major as
    // [(batch * rows + row) * blocks + block], uint8_t. Populated by the MX-block producer and
    // surfaced for the eventual GPU MXGEMM consumer; the CPU-shim consumer does not read it.
    std::vector<uint8_t> host_mx_scales;
    // Block size (elements per MX block along the contraction) that produced host_mx_scales; the
    // MX consumer reads it to dequantize A before GEMM2.
    int32_t mx_block_size = 0;
    // Rows per MX quantization tile that produced host_mx_scales (default 1 = per-row blocks).
    int32_t mx_tile_rows = 1;
    uint64_t           rows  = 0;
    int32_t            batch = 1;
};

// Definition of the opaque handle declared in hipblaslt.h. Owns the composed list of
// epilogue stages plus their parameters. Attached, non-owning, to a matmul descriptor via
// HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE.
struct hipblasLtFusedEpilogueDescriptor
{
    std::vector<hipblasLtFuseableEpilogue_t> stages;

    // Residual-add parameters. residual_output is optional; if unset, the residual input
    // tensor is updated in place with the post-add residual stream.
    void* residual        = nullptr;
    void* residual_output = nullptr;

    // RMSNorm parameters (shared by the full RMSNorm and partial-RMSNorm-stats stages).
    void* rmsnorm_gamma = nullptr;
    float rmsnorm_eps   = 0.f;
    bool  eps_set       = false;

    // Decomposed-flow handoff descriptor, set on both the producer and consumer handles.
    hipblasLtFusedEpilogueRMSNormDescriptor* rmsnorm_stats = nullptr;

    // Requant parameters and policy.
    void*                              requant_scale        = nullptr;
    void*                              requant_amax         = nullptr;
    hipblasLtRequantScaleComputeMode_t requant_compute_mode = HIPBLASLT_REQUANT_SCALE_STATIC;
    hipblasLtRequantScaleGranularity_t requant_granularity  = HIPBLASLT_REQUANT_SCALE_PER_TENSOR;
    // MX-block requant: elements per block along N. Power-of-two >= 1; default 32.
    int32_t requant_block_size     = 32;
    bool    requant_block_size_set = false;
    // MX-tile requant: rows per tile. Power-of-two >= 1; default 1 (per-row blocks).
    int32_t requant_tile_rows     = 1;
    bool    requant_tile_rows_set = false;
};

namespace
{
    // Supported RMSNorm-chain rank. A legal chain is an order-preserving subsequence of the
    // supported order
    //   residual add -> {RMSNorm | partial RMSNorm stats | RMSNorm scale-apply} -> AMax -> requant
    // with each stage appearing at most once. The three normalization stages share rank 1 so
    // that at most one of them can appear in a single chain (full vs decomposed are mutually
    // exclusive). Returns -1 for unrecognized stages or stages reserved for other epilogue
    // families (e.g. SwiGLU).
    int rmsnorm_chain_rank(hipblasLtFuseableEpilogue_t e)
    {
        switch(e)
        {
        case HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD:
            return 0;
        case HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM:
        case HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS:
        case HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY:
            return 1;
        case HIPBLASLT_FUSEABLE_EPILOGUE_AMAX:
            return 2;
        case HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT:
            return 3;
        default:
            return -1;
        }
    }

    // Classify a stage into a chain family so full and decomposed stages cannot be mixed in a
    // single chain (section 4.3): 0 = shared/neutral, 1 = full RMSNorm family, 2 = decomposed.
    // Requant is neutral: full RMSNorm uses it as an output epilogue, while the decomposed
    // producer can use it for the CODA dynamic-quantized handoff.
    int rmsnorm_chain_family(hipblasLtFuseableEpilogue_t e)
    {
        switch(e)
        {
        case HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM:
        case HIPBLASLT_FUSEABLE_EPILOGUE_AMAX:
            return 1;
        case HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS:
        case HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY:
            return 2;
        default:
            return 0;
        }
    }

    bool fused_epilogue_has_stage(const hipblasLtFusedEpilogueDescriptor* d,
                                  hipblasLtFuseableEpilogue_t             e)
    {
        for(auto s : d->stages)
            if(s == e)
                return true;
        return false;
    }

    bool requant_compute_mode_valid(hipblasLtRequantScaleComputeMode_t mode)
    {
        return mode == HIPBLASLT_REQUANT_SCALE_STATIC
               || mode == HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX;
    }

    bool requant_granularity_valid(hipblasLtRequantScaleGranularity_t granularity)
    {
        return granularity == HIPBLASLT_REQUANT_SCALE_PER_TENSOR
               || granularity == HIPBLASLT_REQUANT_SCALE_PER_ROW
               || granularity == HIPBLASLT_REQUANT_SCALE_MX_BLOCK;
    }

    bool fused_epilogue_has_requant(const hipblasLtFusedEpilogueDescriptor* d)
    {
        return fused_epilogue_has_stage(d, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT);
    }
}

hipblasStatus_t hipblasLtFusedEpilogueCreate(hipblasLtFusedEpilogueDescriptor_t* desc)
try
{
    if(desc == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;
    *desc = new hipblasLtFusedEpilogueDescriptor();
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueAdd(hipblasLtFusedEpilogueDescriptor_t desc,
                                          hipblasLtFuseableEpilogue_t        epilogue)
try
{
    if(desc == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;

    const int rank = rmsnorm_chain_rank(epilogue);
    if(rank < 0)
        return HIPBLAS_STATUS_INVALID_VALUE; // unrecognized or unsupported epilogue

    // Reject duplicates and out-of-order additions: the accumulated chain must stay an
    // order-preserving subsequence of the supported RMSNorm chain.
    if(!desc->stages.empty())
    {
        const int prev_rank = rmsnorm_chain_rank(desc->stages.back());
        if(rank <= prev_rank)
            return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Reject mixing full and decomposed RMSNorm stages in one chain: a single chain
    // uses either HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM or the decomposed producer/consumer
    // stages, never both.
    if(epilogue == HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT
       && fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY))
        return HIPBLAS_STATUS_INVALID_VALUE; // requant is producer/full-flow only

    const int family = rmsnorm_chain_family(epilogue);
    if(family != 0)
    {
        for(auto s : desc->stages)
        {
            const int existing = rmsnorm_chain_family(s);
            if(existing != 0 && existing != family)
                return HIPBLAS_STATUS_INVALID_VALUE;
        }
    }

    desc->stages.push_back(epilogue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueSetAttribute(hipblasLtFusedEpilogueDescriptor_t desc,
                                                   hipblasLtFusedEpilogueAttribute_t  attr,
                                                   const void*                        value,
                                                   size_t                             sizeInBytes)
try
{
    if(desc == nullptr || value == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;

    switch(attr)
    {
    case HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->rmsnorm_gamma, value, sizeof(void*));
        if(desc->rmsnorm_gamma == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS:
        if(sizeInBytes < sizeof(float))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->rmsnorm_eps, value, sizeof(float));
        desc->eps_set = true;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->residual, value, sizeof(void*));
        if(desc->residual == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->residual_output, value, sizeof(void*));
        break;
    case HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->rmsnorm_stats, value, sizeof(void*));
        if(desc->rmsnorm_stats == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_scale, value, sizeof(void*));
        if(desc->requant_scale == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_AMAX_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_amax, value, sizeof(void*));
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE:
        if(sizeInBytes < sizeof(hipblasLtRequantScaleComputeMode_t))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_compute_mode, value, sizeof(hipblasLtRequantScaleComputeMode_t));
        if(!requant_compute_mode_valid(desc->requant_compute_mode))
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY:
        if(sizeInBytes < sizeof(hipblasLtRequantScaleGranularity_t))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_granularity, value, sizeof(hipblasLtRequantScaleGranularity_t));
        if(!requant_granularity_valid(desc->requant_granularity))
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_BLOCK_SIZE:
    {
        if(sizeInBytes < sizeof(int32_t))
            return HIPBLAS_STATUS_INVALID_VALUE;
        int32_t block_size = 0;
        memcpy(&block_size, value, sizeof(int32_t));
        if(block_size <= 0 || (block_size & (block_size - 1)) != 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        desc->requant_block_size     = block_size;
        desc->requant_block_size_set = true;
        break;
    }
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_TILE_ROWS:
    {
        if(sizeInBytes < sizeof(int32_t))
            return HIPBLAS_STATUS_INVALID_VALUE;
        int32_t tile_rows = 0;
        memcpy(&tile_rows, value, sizeof(int32_t));
        if(tile_rows <= 0 || (tile_rows & (tile_rows - 1)) != 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        desc->requant_tile_rows     = tile_rows;
        desc->requant_tile_rows_set = true;
        break;
    }
    default:
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueDestroy(hipblasLtFusedEpilogueDescriptor_t desc)
try
{
    delete desc;
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtFusedEpilogueRMSNormDescriptorCreate(hipblasLtFusedEpilogueRMSNormDescriptor_t* desc)
try
{
    if(desc == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;
    *desc = new hipblasLtFusedEpilogueRMSNormDescriptor();
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(hipblasLtFusedEpilogueRMSNormDescriptor_t desc)
try
{
    delete desc;
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t           matmulDesc,
                                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                                const void*                     buf,
                                                size_t                          sizeInBytes)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescSetAttribute");

    // Validate a fused-epilogue handle before it is attached to the matmul descriptor.
    // This is the API-call-time gate for stage-specific required inputs.
    if(matmulAttr == HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE)
    {
        if(buf == nullptr || sizeInBytes < sizeof(void*))
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
        memcpy(&fused, buf, sizeof(void*));
        if(fused != nullptr
           && fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD)
           && fused->residual == nullptr)
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        // gamma and eps back both the full RMSNorm stage and the decomposed producer
        // (partial RMSNorm stats) stage.
        if(fused != nullptr
           && (fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM)
               || fused_epilogue_has_stage(fused,
                                           HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS)))
        {
            if(fused->rmsnorm_gamma == nullptr || !fused->eps_set)
            {
                rocblaslt::Debug::Instance().markerStop();
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
        }
        // Both decomposed stages require the opaque RMSNorm handoff descriptor to be set so
        // the producer and consumer calls share the same object.
        if(fused != nullptr
           && (fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS)
               || fused_epilogue_has_stage(fused,
                                           HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY))
           && fused->rmsnorm_stats == nullptr)
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        if(fused != nullptr && fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT))
        {
            if(fused->requant_scale == nullptr
               || !requant_compute_mode_valid(fused->requant_compute_mode)
               || !requant_granularity_valid(fused->requant_granularity))
            {
                rocblaslt::Debug::Instance().markerStop();
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            if(fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS))
            {
                const bool per_row
                    = fused->requant_granularity == HIPBLASLT_REQUANT_SCALE_PER_ROW;
                const bool mx_block
                    = fused->requant_granularity == HIPBLASLT_REQUANT_SCALE_MX_BLOCK;
                if(fused->requant_compute_mode != HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX
                   || (!per_row && !mx_block)
                   || (mx_block && !fused->requant_block_size_set))
                {
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
            }
        }
    }

    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_set_attribute((rocblaslt_matmul_desc)matmulDesc,
                                            (rocblaslt_matmul_desc_attributes)matmulAttr,
                                            buf,
                                            sizeInBytes));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t           matmulDesc,
                                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                                void*                           buf,
                                                size_t                          sizeInBytes,
                                                size_t*                         sizeWritten)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescGetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_get_attribute((rocblaslt_matmul_desc)matmulDesc,
                                            (rocblaslt_matmul_desc_attributes)matmulAttr,
                                            buf,
                                            sizeInBytes,
                                            sizeWritten));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t* pref)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulPreferenceCreate");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_create((rocblaslt_matmul_preference*)pref));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulPreferenceDestroy");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_destroy((const rocblaslt_matmul_preference)pref));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t           pref,
                                          hipblasLtMatmulPreferenceAttributes_t attribute,
                                          const void*                           data,
                                          size_t                                dataSize)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulPreferenceSetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_set_attribute((rocblaslt_matmul_preference)pref,
                                                  (rocblaslt_matmul_preference_attributes)attribute,
                                                  data,
                                                  dataSize));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulPreferenceGetAttribute(hipblasLtMatmulPreference_t           pref,
                                          hipblasLtMatmulPreferenceAttributes_t attribute,
                                          void*                                 data,
                                          size_t                                sizeInBytes,
                                          size_t*                               sizeWritten)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulPreferenceGetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_get_attribute((rocblaslt_matmul_preference)pref,
                                                  (rocblaslt_matmul_preference_attributes)attribute,
                                                  data,
                                                  sizeInBytes,
                                                  sizeWritten));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t                handle,
                                    hipblasLtMatmulDesc_t            matmulDesc,
                                    hipblasLtMatrixLayout_t          Adesc,
                                    hipblasLtMatrixLayout_t          Bdesc,
                                    hipblasLtMatrixLayout_t          Cdesc,
                                    hipblasLtMatrixLayout_t          Ddesc,
                                    hipblasLtMatmulPreference_t      pref,
                                    int                              requestedAlgoCount,
                                    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                    int*                             returnAlgoCount)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulAlgoGetHeuristic");

    OverrideSingleton& override = OverrideSingleton::getInstance();
    if(override.env_mode)
    {
        bool override_success = override_path_compare_git_version(override, handle);
        if(override_success)
            log_info(__func__, "HIPBLASLT_TUNING_OVERRIDE_FILE is the correct setting.");
        else
            log_error(
                __func__,
                "The hipBLASLt git version and the override file git version are not the same.");
    }

    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul_algo_get_heuristic(
        (rocblaslt_handle)handle,
        (rocblaslt_matmul_desc)matmulDesc,
        (rocblaslt_matrix_layout)Adesc,
        (rocblaslt_matrix_layout)Bdesc,
        (rocblaslt_matrix_layout)Cdesc,
        (rocblaslt_matrix_layout)Ddesc,
        (rocblaslt_matmul_preference)pref,
        requestedAlgoCount,
        (rocblaslt_matmul_heuristic_result*)heuristicResultsArray,
        returnAlgoCount));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

#ifdef __cplusplus
} // extern "C" -- the shim helpers below need C++ linkage (overloads + templates).
#endif

namespace
{
    // CPU "shim" for the fused-epilogue chain. Until the TensileLite fused kernels
    // (AIHPBLAS-3856) land, hipblasLtMatmul runs the base GEMM on device, then applies the
    // fused epilogue on the host so the advertised API returns correct answers. The shim
    // emulates the full flow (residual add, RMSNorm), the full flow plus static per-tensor FP8
    // requant, and the decomposed flow (partial RMSNorm stats producer + RMSNorm scale-apply
    // consumer), including the dynamic per-row quantized producer, on column-major tensors.
    // Unsupported stages/policies fall through to NOT_SUPPORTED.
    bool fused_epilogue_cpu_shim_supported(const hipblasLtFusedEpilogueDescriptor* d)
    {
        const bool has_requant = fused_epilogue_has_requant(d);
        for(auto s : d->stages)
        {
            if(s == HIPBLASLT_FUSEABLE_EPILOGUE_AMAX
               || s == HIPBLASLT_FUSEABLE_EPILOGUE_SWIGLU)
                return false;
        }
        if(has_requant)
        {
            if(fused_epilogue_has_stage(d, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM))
                return d->requant_compute_mode == HIPBLASLT_REQUANT_SCALE_STATIC
                       && d->requant_granularity == HIPBLASLT_REQUANT_SCALE_PER_TENSOR;
            if(fused_epilogue_has_stage(d, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS))
                return d->requant_compute_mode == HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX
                       && (d->requant_granularity == HIPBLASLT_REQUANT_SCALE_PER_ROW
                           || d->requant_granularity == HIPBLASLT_REQUANT_SCALE_MX_BLOCK);
            return false;
        }
        return true;
    }

    // FP32-accumulation load/store helpers for the supported storage types.
    inline float shim_load(const float& v)
    {
        return v;
    }
    inline float shim_load(const _Float16& v)
    {
        return static_cast<float>(v);
    }
    inline float shim_load(const hip_bfloat16& v)
    {
        return static_cast<float>(v);
    }
    inline void shim_store(float& d, float v)
    {
        d = v;
    }
    inline void shim_store(_Float16& d, float v)
    {
        d = static_cast<_Float16>(v);
    }
    inline void shim_store(hip_bfloat16& d, float v)
    {
        d = hip_bfloat16(v);
    }

    template <typename QuantT>
    constexpr float shim_quant_max()
    {
        return 0.0f;
    }

    template <>
    constexpr float shim_quant_max<hipblaslt_f8>()
    {
        // OCP E4M3 maximum finite magnitude.
        return 448.0f;
    }

    // UE8M0 is an exponent-only 8-bit scale: value = 2^(e - 127). Encoding rounds the exponent up
    // (ceil of log2) so the decoded scale never underestimates x; this keeps block-quantized values
    // at or below the FP8 max and avoids saturation. Zero, negative, non-finite (inf/NaN) inputs map to 0.
    inline uint8_t encode_ue8m0(float x)
    {
        if(!std::isfinite(x) || !(x > 0.0f))
            return 0;
        int         exponent = 0;
        const float m        = std::frexp(x, &exponent); // x = m * 2^exponent, m in [0.5, 1).
        // ceil(log2(x)) + 127: exact powers of two (m == 0.5) round to themselves, others up.
        const int biased = (m == 0.5f ? exponent - 1 : exponent) + 127;
        if(biased < 0)
            return 0;
        if(biased > 254)
            return 254;
        return static_cast<uint8_t>(biased);
    }

    inline float decode_ue8m0(uint8_t e)
    {
        return std::ldexp(1.0f, static_cast<int>(e) - 127); // 2^(e - 127).
    }

    // Reduce path over each row of a column-major [M,N] tensor (leading dimension ld), using
    // FP32 accumulation. Applies residual add (optional), then per-row RMSNorm reducing over N:
    //  - has_residual: z = d + residual, written to residual_out and back to d.
    //  - do_norm:      rstd = rsqrt(mean(z^2) + eps); the downstream value is z*gamma, scaled by
    //                  rstd when apply_scale is set (full flow) or left unscaled when it is not
    //                  (decomposed producer, which defers the scale to the consuming GEMM).
    //  - scale_out:    when non-null, receives rstd per (batch,row) as [batch*m + row].
    template <typename T>
    void fused_epilogue_reduce_host(T*       d_data,
                                   const T* residual_data,
                                   T*       residual_out_data,
                                   const T* gamma_data,
                                   uint64_t m,
                                   uint64_t n,
                                   int64_t  ld,
                                   int32_t  batch_count,
                                   int64_t  batch_stride,
                                   float    eps,
                                   bool     has_residual,
                                   bool     do_norm,
                                   bool     apply_scale,
                                   float*   scale_out)
    {
        for(int32_t b = 0; b < batch_count; ++b)
        {
            const int64_t base = static_cast<int64_t>(b) * batch_stride;
            for(uint64_t i = 0; i < m; ++i)
            {
                float sum_sq = 0.0f;
                for(uint64_t j = 0; j < n; ++j)
                {
                    const int64_t e
                        = base + static_cast<int64_t>(j) * ld + static_cast<int64_t>(i);
                    float z = shim_load(d_data[e]);
                    if(has_residual)
                    {
                        z += shim_load(residual_data[e]);
                        // z is the updated residual stream and the RMSNorm input.
                        shim_store(residual_out_data[e], z);
                        shim_store(d_data[e], z);
                    }
                    sum_sq += z * z;
                }
                if(do_norm)
                {
                    const float rstd = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
                    for(uint64_t j = 0; j < n; ++j)
                    {
                        const int64_t e
                            = base + static_cast<int64_t>(j) * ld + static_cast<int64_t>(i);
                        float v = shim_load(d_data[e]) * shim_load(gamma_data[j]);
                        if(apply_scale)
                            v *= rstd;
                        shim_store(d_data[e], v);
                    }
                    if(scale_out)
                        scale_out[static_cast<size_t>(b) * m + i] = rstd;
                }
            }
        }
    }

    // Consumer path: multiply each row of a column-major [M,N] tensor by the deferred per-row
    // scale carried in the handoff descriptor (broadcast across the N columns).
    template <typename T>
    void fused_epilogue_apply_scale_host(T*           d_data,
                                         const float* scale,
                                         uint64_t     m,
                                         uint64_t     n,
                                         int64_t      ld,
                                         int32_t      batch_count,
                                         int64_t      batch_stride)
    {
        for(int32_t b = 0; b < batch_count; ++b)
        {
            const int64_t base = static_cast<int64_t>(b) * batch_stride;
            for(uint64_t i = 0; i < m; ++i)
            {
                const float s = scale[static_cast<size_t>(b) * m + i];
                for(uint64_t j = 0; j < n; ++j)
                {
                    const int64_t e
                        = base + static_cast<int64_t>(j) * ld + static_cast<int64_t>(i);
                    shim_store(d_data[e], shim_load(d_data[e]) * s);
                }
            }
        }
    }

    // Full flow (RMSNorm) and decomposed producer (partial RMSNorm stats): residual add + the
    // per-row RMSNorm reduction. The full flow applies the scale to D; the producer instead
    // stashes the per-row scale into the handoff descriptor and defers the multiply.
    template <typename T>
    hipblasStatus_t fused_epilogue_cpu_shim_reduce_typed(
        const hipblasLtFusedEpilogueDescriptor* fused,
        void*                                   d_device,
        uint64_t                                m,
        uint64_t                                n,
        int64_t                                 ld,
        int32_t                                 batch_count,
        int64_t                                 batch_stride,
        hipStream_t                             stream)
    {
        const bool has_residual
            = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD);
        const bool has_full
            = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM);
        const bool has_producer
            = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS);
        const bool do_norm     = has_full || has_producer;
        const bool apply_scale = has_full;

        if(batch_count < 1)
            batch_count = 1;

        // Element span that covers every referenced (row, col, batch) position so untouched
        // leading-dimension padding round-trips unchanged.
        const size_t span
            = static_cast<size_t>(batch_count - 1) * static_cast<size_t>(batch_stride)
              + static_cast<size_t>(n - 1) * static_cast<size_t>(ld) + static_cast<size_t>(m);

        std::vector<T> h_d(span);
        std::vector<T> h_res, h_res_out, h_gamma;

        if(hipStreamSynchronize(stream) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        if(hipMemcpy(h_d.data(), d_device, span * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        if(has_residual)
        {
            h_res.resize(span);
            if(hipMemcpy(h_res.data(), fused->residual, span * sizeof(T), hipMemcpyDeviceToHost)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            // Seed the residual-out buffer with residual so untouched padding round-trips.
            h_res_out = h_res;
        }
        if(do_norm)
        {
            h_gamma.resize(n);
            if(hipMemcpy(
                   h_gamma.data(), fused->rmsnorm_gamma, n * sizeof(T), hipMemcpyDeviceToHost)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        std::vector<float> scale;
        if(has_producer)
            scale.resize(static_cast<size_t>(batch_count) * m);

        fused_epilogue_reduce_host<T>(h_d.data(),
                                     has_residual ? h_res.data() : nullptr,
                                     has_residual ? h_res_out.data() : nullptr,
                                     do_norm ? h_gamma.data() : nullptr,
                                     m,
                                     n,
                                     ld,
                                     batch_count,
                                     batch_stride,
                                     fused->rmsnorm_eps,
                                     has_residual,
                                     do_norm,
                                     apply_scale,
                                     has_producer ? scale.data() : nullptr);

        if(hipMemcpy(d_device, h_d.data(), span * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        if(has_residual)
        {
            void* res_out = fused->residual_output ? fused->residual_output : fused->residual;
            if(hipMemcpy(res_out, h_res_out.data(), span * sizeof(T), hipMemcpyHostToDevice)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
        if(has_producer)
        {
            // Hand the finalized per-row scale to the consuming GEMM via the handoff descriptor.
            auto* stats        = fused->rmsnorm_stats;
            stats->host_scale  = std::move(scale);
            stats->rows        = m;
            stats->batch       = batch_count;
            stats->populated   = true;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    template <typename QuantT, typename WorkT>
    hipblasStatus_t fused_epilogue_cpu_shim_requant_typed(
        const hipblasLtFusedEpilogueDescriptor* fused,
        void*                                   work_device,
        void*                                   d_device,
        uint64_t                                m,
        uint64_t                                n,
        int64_t                                 ld,
        int32_t                                 batch_count,
        int64_t                                 batch_stride,
        hipStream_t                             stream)
    {
        if(batch_count < 1)
            batch_count = 1;

        const bool has_residual
            = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD);
        const bool has_producer
            = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS);
        const size_t span
            = static_cast<size_t>(batch_count - 1) * static_cast<size_t>(batch_stride)
              + static_cast<size_t>(n - 1) * static_cast<size_t>(ld) + static_cast<size_t>(m);

        if(hipStreamSynchronize(stream) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        std::vector<WorkT> h_work(span);
        std::vector<WorkT> h_res, h_res_out, h_gamma;
        if(hipMemcpy(h_work.data(), work_device, span * sizeof(WorkT), hipMemcpyDeviceToHost)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        if(has_residual)
        {
            h_res.resize(span);
            if(hipMemcpy(h_res.data(), fused->residual, span * sizeof(WorkT), hipMemcpyDeviceToHost)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            h_res_out = h_res;
        }

        h_gamma.resize(n);
        if(hipMemcpy(h_gamma.data(), fused->rmsnorm_gamma, n * sizeof(WorkT), hipMemcpyDeviceToHost)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        std::vector<float> rstd;
        if(has_producer)
            rstd.resize(static_cast<size_t>(batch_count) * m);

        fused_epilogue_reduce_host<WorkT>(h_work.data(),
                                          has_residual ? h_res.data() : nullptr,
                                          has_residual ? h_res_out.data() : nullptr,
                                          h_gamma.data(),
                                          m,
                                          n,
                                          ld,
                                          batch_count,
                                          batch_stride,
                                          fused->rmsnorm_eps,
                                          has_residual,
                                          /*do_norm=*/true,
                                          /*apply_scale=*/!has_producer,
                                          has_producer ? rstd.data() : nullptr);

        if(has_residual)
        {
            void* res_out = fused->residual_output ? fused->residual_output : fused->residual;
            if(hipMemcpy(res_out, h_res_out.data(), span * sizeof(WorkT), hipMemcpyHostToDevice)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        std::vector<QuantT> h_quant(span);
        if(has_producer)
        {
            const float qmax = shim_quant_max<QuantT>();
            if(qmax == 0.0f)
                return HIPBLAS_STATUS_NOT_SUPPORTED;

            std::vector<float> rho(static_cast<size_t>(batch_count) * m, 0.0f);
            std::vector<float> amax(static_cast<size_t>(batch_count) * m, 0.0f);
            for(int32_t b = 0; b < batch_count; ++b)
            {
                const int64_t base = static_cast<int64_t>(b) * batch_stride;
                for(uint64_t i = 0; i < m; ++i)
                {
                    float h2_amax = 0.0f;
                    for(uint64_t j = 0; j < n; ++j)
                    {
                        const int64_t e
                            = base + static_cast<int64_t>(j) * ld + static_cast<int64_t>(i);
                        h2_amax = std::max(h2_amax, std::fabs(shim_load(h_work[e])));
                    }

                    const size_t row = static_cast<size_t>(b) * m + i;
                    const float  h2_scale = h2_amax == 0.0f ? 1.0f : h2_amax / qmax;
                    rho[row]              = rstd[row] * (h2_amax == 0.0f ? 0.0f : h2_scale);
                    amax[row]             = rstd[row] * h2_amax;

                    for(uint64_t j = 0; j < n; ++j)
                    {
                        const int64_t e
                            = base + static_cast<int64_t>(j) * ld + static_cast<int64_t>(i);
                        h_quant[e] = QuantT(shim_load(h_work[e]) / h2_scale);
                    }
                }
            }

            if(hipMemcpy(fused->requant_scale,
                         rho.data(),
                         rho.size() * sizeof(float),
                         hipMemcpyHostToDevice)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            if(fused->requant_amax != nullptr
               && hipMemcpy(fused->requant_amax,
                            amax.data(),
                            amax.size() * sizeof(float),
                            hipMemcpyHostToDevice)
                      != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;

            auto* stats        = fused->rmsnorm_stats;
            stats->host_scale  = std::move(rho);
            stats->rows        = m;
            stats->batch       = batch_count;
            stats->populated   = true;
            stats->per_row_scale = fused->requant_scale;
        }
        else
        {
            float dequant_scale = 0.0f;
            if(hipMemcpy(
                   &dequant_scale, fused->requant_scale, sizeof(float), hipMemcpyDeviceToHost)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            if(dequant_scale == 0.0f)
                return HIPBLAS_STATUS_INVALID_VALUE;

            float amax = 0.0f;
            for(size_t idx = 0; idx < span; ++idx)
            {
                const float x = shim_load(h_work[idx]);
                amax          = std::max(amax, std::fabs(x));
                h_quant[idx]  = QuantT(x / dequant_scale);
            }

            if(fused->requant_amax != nullptr)
            {
                if(hipMemcpy(fused->requant_amax, &amax, sizeof(float), hipMemcpyHostToDevice)
                   != hipSuccess)
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
            }
        }

        if(hipMemcpy(d_device, h_quant.data(), span * sizeof(QuantT), hipMemcpyHostToDevice)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Quantize a 2D tile of rows [i0, i1) x all column-tiles into FP8 codes and UE8M0 scales.
    // The scale for tile (rt, ct) lives at index rt * n_col_tiles + ct, where rt is the caller's
    // batch-inclusive row-tile index.
    template <typename QuantT, typename WorkT>
    void mxquant_tile(const std::vector<WorkT>& h_work,
                      std::vector<QuantT>&      h_quant,
                      std::vector<uint8_t>&     mx_scales,
                      int64_t                   base,
                      int64_t                   i0,
                      int64_t                   i1,
                      int64_t                   n,
                      int64_t                   ld,
                      int64_t                   rt,
                      int64_t                   n_col_tiles,
                      int32_t                   tile_cols,
                      float                     qmax)
    {
        for(int64_t ct = 0; ct < n_col_tiles; ++ct)
        {
            const int64_t j0        = ct * tile_cols;
            const int64_t j1        = std::min<int64_t>(j0 + tile_cols, n);
            float         amax_tile = 0.0f;
            for(int64_t j = j0; j < j1; ++j)
                for(int64_t i = i0; i < i1; ++i)
                    amax_tile = std::max(amax_tile, std::fabs(shim_load(h_work[base + j * ld + i])));
            const uint8_t s = encode_ue8m0(amax_tile / qmax);
            mx_scales[static_cast<size_t>(rt * n_col_tiles + ct)] = s;
            const float dec = decode_ue8m0(s);
            for(int64_t j = j0; j < j1; ++j)
                for(int64_t i = i0; i < i1; ++i)
                {
                    const int64_t e = base + j * ld + i;
                    h_quant[e]      = QuantT(shim_load(h_work[e]) / dec);
                }
        }
    }

    // Decomposed producer with MX 2D-tile quantization: residual add + gamma + partial RMSNorm
    // stats, then FP8 quantization where each [tile_rows, tile_cols] tile of the producer output
    // receives one UE8M0 scale; tile_cols comes from requant_block_size and tile_rows from
    // requant_tile_rows (default 1 = per-row tiles). The handoff carries the per-row rstd in
    // host_scale; the per-tile UE8M0 scales go to the requant scale pointer and host_mx_scales.
    template <typename QuantT, typename WorkT>
    hipblasStatus_t fused_epilogue_cpu_shim_mxquant_typed(
        const hipblasLtFusedEpilogueDescriptor* fused,
        void*                                   work_device,
        void*                                   d_device,
        uint64_t                                m,
        uint64_t                                n,
        int64_t                                 ld,
        int32_t                                 batch_count,
        int64_t                                 batch_stride,
        hipStream_t                             stream)
    {
        if(batch_count < 1)
            batch_count = 1;

        // MX block quantization is only defined for the decomposed producer chain.
        if(!fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS))
            return HIPBLAS_STATUS_NOT_SUPPORTED;

        const int32_t tile_cols = fused->requant_block_size;
        if(tile_cols <= 0 || (tile_cols & (tile_cols - 1)) != 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        const int32_t tile_rows = fused->requant_tile_rows;
        if(tile_rows <= 0 || (tile_rows & (tile_rows - 1)) != 0)
            return HIPBLAS_STATUS_INVALID_VALUE;

        const float qmax = shim_quant_max<QuantT>();
        if(qmax == 0.0f)
            return HIPBLAS_STATUS_NOT_SUPPORTED;

        const bool has_residual
            = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD);
        const int64_t n_col_tiles = (static_cast<int64_t>(n) + tile_cols - 1) / tile_cols;
        const int64_t n_row_tiles = (static_cast<int64_t>(m) + tile_rows - 1) / tile_rows;
        const size_t  span
            = static_cast<size_t>(batch_count - 1) * static_cast<size_t>(batch_stride)
              + static_cast<size_t>(n - 1) * static_cast<size_t>(ld) + static_cast<size_t>(m);

        if(hipStreamSynchronize(stream) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        std::vector<WorkT> h_work(span);
        std::vector<WorkT> h_res, h_res_out, h_gamma;
        if(hipMemcpy(h_work.data(), work_device, span * sizeof(WorkT), hipMemcpyDeviceToHost)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        if(has_residual)
        {
            h_res.resize(span);
            if(hipMemcpy(h_res.data(), fused->residual, span * sizeof(WorkT), hipMemcpyDeviceToHost)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            h_res_out = h_res;
        }

        h_gamma.resize(n);
        if(hipMemcpy(h_gamma.data(), fused->rmsnorm_gamma, n * sizeof(WorkT), hipMemcpyDeviceToHost)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        std::vector<float> rstd(static_cast<size_t>(batch_count) * m);
        fused_epilogue_reduce_host<WorkT>(h_work.data(),
                                          has_residual ? h_res.data() : nullptr,
                                          has_residual ? h_res_out.data() : nullptr,
                                          h_gamma.data(),
                                          m,
                                          n,
                                          ld,
                                          batch_count,
                                          batch_stride,
                                          fused->rmsnorm_eps,
                                          has_residual,
                                          /*do_norm=*/true,
                                          /*apply_scale=*/false,
                                          rstd.data());

        if(has_residual)
        {
            void* res_out = fused->residual_output ? fused->residual_output : fused->residual;
            if(hipMemcpy(res_out, h_res_out.data(), span * sizeof(WorkT), hipMemcpyHostToDevice)
               != hipSuccess)
                return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        std::vector<QuantT>  h_quant(span);
        std::vector<uint8_t> mx_scales(
            static_cast<size_t>(batch_count) * n_row_tiles * n_col_tiles, 0);

        std::vector<float> rho(static_cast<size_t>(batch_count) * n_row_tiles, 0.0f);
        for(int32_t b = 0; b < batch_count; ++b)
        {
            const int64_t base = static_cast<int64_t>(b) * batch_stride;
            for(int64_t rt = 0; rt < n_row_tiles; ++rt)
            {
                const int64_t i0  = rt * tile_rows;
                const int64_t i1  = std::min<int64_t>(i0 + tile_rows, static_cast<int64_t>(m));
                const size_t  row = static_cast<size_t>(b) * n_row_tiles + rt;
                mxquant_tile<QuantT, WorkT>(h_work,
                                            h_quant,
                                            mx_scales,
                                            base,
                                            i0,
                                            i1,
                                            static_cast<int64_t>(n),
                                            ld,
                                            static_cast<int64_t>(row),
                                            n_col_tiles,
                                            tile_cols,
                                            qmax);
                // Carry the rstd of the tile's first row as the CODA consumer scale. For
                // tile_rows==1 this is exactly the per-row rstd; for tile_rows>1 it is an
                // acknowledged approximation (a real GPU kernel would write per-row rstd).
                rho[row] = rstd[static_cast<size_t>(b) * m + i0];
            }
        }

        // Expand rho from n_row_tiles to M entries so the consumer scale-apply stays unchanged,
        // repeating each tile-row's rstd across the rows it covers.
        std::vector<float> rho_expanded(static_cast<size_t>(batch_count) * m, 0.0f);
        for(int32_t b = 0; b < batch_count; ++b)
            for(int64_t rt = 0; rt < n_row_tiles; ++rt)
            {
                const float   v  = rho[static_cast<size_t>(b) * n_row_tiles + rt];
                const int64_t i0 = rt * tile_rows;
                const int64_t i1 = std::min<int64_t>(i0 + tile_rows, static_cast<int64_t>(m));
                for(int64_t i = i0; i < i1; ++i)
                    rho_expanded[static_cast<size_t>(b) * m + i] = v;
            }

        if(hipMemcpy(fused->requant_scale,
                     mx_scales.data(),
                     mx_scales.size() * sizeof(uint8_t),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        if(hipMemcpy(d_device, h_quant.data(), span * sizeof(QuantT), hipMemcpyHostToDevice)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        hipblasLtFusedEpilogueRMSNormDescriptor* stats = fused->rmsnorm_stats;
        stats->host_scale     = std::move(rho_expanded);
        stats->host_mx_scales = std::move(mx_scales);
        stats->mx_block_size  = tile_cols;
        stats->mx_tile_rows   = tile_rows;
        stats->rows           = m;
        stats->batch          = batch_count;
        stats->populated      = true;
        stats->per_row_scale  = nullptr;
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Decomposed consumer (RMSNorm scale-apply): multiply GEMM2's output by the deferred per-row
    // scale carried in the handoff descriptor.
    template <typename T>
    hipblasStatus_t fused_epilogue_cpu_shim_apply_typed(
        const hipblasLtFusedEpilogueDescriptor* fused,
        void*                                   d_device,
        uint64_t                                m,
        uint64_t                                n,
        int64_t                                 ld,
        int32_t                                 batch_count,
        int64_t                                 batch_stride,
        hipStream_t                             stream)
    {
        if(batch_count < 1)
            batch_count = 1;

        const auto* stats = fused->rmsnorm_stats;
        // A scale-apply consuming a stats descriptor no producer populated is an error.
        if(stats == nullptr || !stats->populated)
            return HIPBLAS_STATUS_INVALID_VALUE;
        // The deferred per-row scale must match the consuming GEMM's row/batch counts.
        if(stats->rows != m || stats->batch != batch_count
           || stats->host_scale.size() < static_cast<size_t>(batch_count) * m)
            return HIPBLAS_STATUS_INVALID_VALUE;

        const size_t span
            = static_cast<size_t>(batch_count - 1) * static_cast<size_t>(batch_stride)
              + static_cast<size_t>(n - 1) * static_cast<size_t>(ld) + static_cast<size_t>(m);

        std::vector<T> h_d(span);
        if(hipStreamSynchronize(stream) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        if(hipMemcpy(h_d.data(), d_device, span * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        fused_epilogue_apply_scale_host<T>(
            h_d.data(), stats->host_scale.data(), m, n, ld, batch_count, batch_stride);

        if(hipMemcpy(d_device, h_d.data(), span * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t fused_epilogue_cpu_shim(const hipblasLtFusedEpilogueDescriptor* fused,
                                            const _rocblaslt_matrix_layout*         layoutD,
                                            void*                                   d_device,
                                            hipStream_t                             stream)
    {
        const uint64_t m            = layoutD->m;
        const uint64_t n            = layoutD->n;
        const int64_t  ld           = layoutD->ld;
        const int32_t  batch_count  = layoutD->batch_count;
        const int64_t  batch_stride = layoutD->batch_stride;

        if(m == 0 || n == 0)
            return HIPBLAS_STATUS_SUCCESS;

        // The decomposed consumer applies a deferred per-row scale; every other supported chain
        // goes through the residual/RMSNorm reduce path.
        const bool is_consumer
            = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY);

        switch(layoutD->type)
        {
        case HIP_R_32F:
            return is_consumer ? fused_epilogue_cpu_shim_apply_typed<float>(
                       fused, d_device, m, n, ld, batch_count, batch_stride, stream)
                               : fused_epilogue_cpu_shim_reduce_typed<float>(
                                   fused, d_device, m, n, ld, batch_count, batch_stride, stream);
        case HIP_R_16F:
            return is_consumer ? fused_epilogue_cpu_shim_apply_typed<_Float16>(
                       fused, d_device, m, n, ld, batch_count, batch_stride, stream)
                               : fused_epilogue_cpu_shim_reduce_typed<_Float16>(
                                   fused, d_device, m, n, ld, batch_count, batch_stride, stream);
        case HIP_R_16BF:
            return is_consumer ? fused_epilogue_cpu_shim_apply_typed<hip_bfloat16>(
                       fused, d_device, m, n, ld, batch_count, batch_stride, stream)
                               : fused_epilogue_cpu_shim_reduce_typed<hip_bfloat16>(
                                   fused, d_device, m, n, ld, batch_count, batch_stride, stream);
        default:
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
    }

    // Download an FP8 column-major [rows, cols] tensor (leading dim ld), dequantize each element,
    // and upload it as BF16 to dst. When mxScales is non-null each element is scaled by its UE8M0
    // tile scale indexed as (b*nRowTiles + i/tileRows)*nColTiles + j/tileCols; otherwise it
    // performs a plain FP8 to BF16 cast.
    hipblasStatus_t shimDequantF8ToBf16(const void*    src,
                                            void*          dst,
                                            const uint8_t* mxScales,
                                            uint64_t       rows,
                                            uint64_t       cols,
                                            int64_t        ld,
                                            int32_t        batchCount,
                                            int64_t        batchStride,
                                            int32_t        tileRows,
                                            int32_t        tileCols,
                                            int64_t        nRowTiles,
                                            int64_t        nColTiles,
                                            hipStream_t    stream)
    {
        if(batchCount < 1)
            batchCount = 1;
        const size_t span
            = static_cast<size_t>(batchCount - 1) * static_cast<size_t>(batchStride)
              + static_cast<size_t>(cols - 1) * static_cast<size_t>(ld) + static_cast<size_t>(rows);

        if(hipStreamSynchronize(stream) != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        std::vector<hipblaslt_f8> hSrc(span);
        if(hipMemcpy(hSrc.data(), src, span * sizeof(hipblaslt_f8), hipMemcpyDeviceToHost)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        std::vector<hip_bfloat16> hDst(span);
        for(int32_t b = 0; b < batchCount; ++b)
        {
            const int64_t base = static_cast<int64_t>(b) * batchStride;
            for(uint64_t j = 0; j < cols; ++j)
                for(uint64_t i = 0; i < rows; ++i)
                {
                    const int64_t e
                        = base + static_cast<int64_t>(j) * ld + static_cast<int64_t>(i);
                    float val = static_cast<float>(static_cast<_Float16>(hSrc[e]));
                    if(mxScales != nullptr)
                    {
                        const int64_t rt  = static_cast<int64_t>(i) / tileRows;
                        const int64_t ct  = static_cast<int64_t>(j) / tileCols;
                        const size_t  idx = (static_cast<size_t>(b) * static_cast<size_t>(nRowTiles)
                                             + static_cast<size_t>(rt))
                                                * static_cast<size_t>(nColTiles)
                                            + static_cast<size_t>(ct);
                        val *= decode_ue8m0(mxScales[idx]);
                    }
                    hDst[e] = hip_bfloat16(val);
                }
        }

        if(hipMemcpy(dst, hDst.data(), span * sizeof(hip_bfloat16), hipMemcpyHostToDevice)
           != hipSuccess)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Create a BF16 column-major matrix layout, setting batch attributes when batched. On failure
    // the caller owns and must destroy *out.
    hipblasStatus_t makeBf16Layout(hipblasLtMatrixLayout_t* out,
                                     uint64_t                 rows,
                                     uint64_t                 cols,
                                     int64_t                  ld,
                                     int32_t                  batchCount,
                                     int64_t                  batchStride)
    {
        if(hipblasLtMatrixLayoutCreate(out, HIP_R_16BF, rows, cols, ld) != HIPBLAS_STATUS_SUCCESS)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        if(batchCount == 1)
            return HIPBLAS_STATUS_SUCCESS;
        if(hipblasLtMatrixLayoutSetAttribute(
               *out, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount))
               != HIPBLAS_STATUS_SUCCESS
           || hipblasLtMatrixLayoutSetAttribute(*out,
                                                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                &batchStride,
                                                sizeof(batchStride))
                  != HIPBLAS_STATUS_SUCCESS)
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Derived GEMM2 dimensions for the MX consumer path.
    struct MxConsumerDims
    {
        uint64_t m         = 0;
        uint64_t k         = 0;
        uint64_t n         = 0;
        int64_t  nBlocks   = 0;
        int32_t  batchA    = 1;
        int32_t  batchB    = 1;
        int32_t  blockSize = 0;
        int32_t  tileRows  = 1;
        int64_t  nRowTiles = 0;
    };

    // Free MX consumer BF16 scratch buffers and layouts, nulling each handle. Null-safe.
    void mxConsumerFreeScratch(void**                   tmpA,
                               void**                   tmpB,
                               hipblasLtMatrixLayout_t* tmpLayoutA,
                               hipblasLtMatrixLayout_t* tmpLayoutB)
    {
        if(*tmpA)
            static_cast<void>(hipFree(*tmpA));
        if(*tmpB)
            static_cast<void>(hipFree(*tmpB));
        if(*tmpLayoutA)
            hipblasLtMatrixLayoutDestroy(*tmpLayoutA);
        if(*tmpLayoutB)
            hipblasLtMatrixLayoutDestroy(*tmpLayoutB);
        *tmpA       = nullptr;
        *tmpB       = nullptr;
        *tmpLayoutA = nullptr;
        *tmpLayoutB = nullptr;
    }

    // Allocate BF16 scratch for A and B and build their column-major layouts. On failure frees any
    // partial state and leaves all out-params null; the caller owns the handles on success.
    hipblasStatus_t mxConsumerAllocBf16Scratch(const _rocblaslt_matrix_layout* layoutA,
                                               const _rocblaslt_matrix_layout* layoutB,
                                               const MxConsumerDims&           dims,
                                               void**                          tmpA,
                                               void**                          tmpB,
                                               hipblasLtMatrixLayout_t*        tmpLayoutA,
                                               hipblasLtMatrixLayout_t*        tmpLayoutB)
    {
        const size_t spanA
            = static_cast<size_t>(dims.batchA - 1) * static_cast<size_t>(layoutA->batch_stride)
              + static_cast<size_t>(dims.k - 1) * static_cast<size_t>(layoutA->ld)
              + static_cast<size_t>(dims.m);
        const size_t spanB
            = static_cast<size_t>(dims.batchB - 1) * static_cast<size_t>(layoutB->batch_stride)
              + static_cast<size_t>(dims.n - 1) * static_cast<size_t>(layoutB->ld)
              + static_cast<size_t>(dims.k);

        if(hipMalloc(tmpA, spanA * sizeof(hip_bfloat16)) != hipSuccess
           || hipMalloc(tmpB, spanB * sizeof(hip_bfloat16)) != hipSuccess
           || makeBf16Layout(
                  tmpLayoutA, dims.m, dims.k, layoutA->ld, dims.batchA, layoutA->batch_stride)
                  != HIPBLAS_STATUS_SUCCESS
           || makeBf16Layout(
                  tmpLayoutB, dims.k, dims.n, layoutB->ld, dims.batchB, layoutB->batch_stride)
                  != HIPBLAS_STATUS_SUCCESS)
        {
            mxConsumerFreeScratch(tmpA, tmpB, tmpLayoutA, tmpLayoutB);
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Validate an MX consumer chain and derive GEMM2 dimensions from the FP8 input layouts.
    hipblasStatus_t mxConsumerValidate(const _rocblaslt_matrix_layout*                layoutA,
                                       const _rocblaslt_matrix_layout*                layoutB,
                                       const hipblasLtFusedEpilogueRMSNormDescriptor* stats,
                                       MxConsumerDims*                                dims)
    {
        if(layoutA->type != HIP_R_8F_E4M3 || layoutB->type != HIP_R_8F_E4M3)
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        dims->blockSize = stats->mx_block_size;
        if(dims->blockSize <= 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        dims->tileRows = stats->mx_tile_rows;
        if(dims->tileRows <= 0)
            return HIPBLAS_STATUS_INVALID_VALUE;

        dims->m       = layoutA->m; // GEMM2 rows.
        dims->k       = layoutA->n; // contraction dim (== producer N).
        dims->n       = layoutB->n; // GEMM2 cols.
        dims->nBlocks = (static_cast<int64_t>(dims->k) + dims->blockSize - 1) / dims->blockSize;
        dims->nRowTiles = (static_cast<int64_t>(dims->m) + dims->tileRows - 1) / dims->tileRows;
        dims->batchA  = layoutA->batch_count < 1 ? 1 : layoutA->batch_count;
        dims->batchB  = layoutB->batch_count < 1 ? 1 : layoutB->batch_count;
        if(stats->rows != dims->m || stats->batch != dims->batchA
           || stats->host_mx_scales.size()
                  < static_cast<size_t>(dims->batchA) * static_cast<size_t>(dims->nRowTiles)
                        * static_cast<size_t>(dims->nBlocks))
            return HIPBLAS_STATUS_INVALID_VALUE;
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Build BF16 GEMM2 inputs: allocate scratch, then dequantize A (with per-block MX scales) and B
    // (plain cast) into it. Frees all scratch and nulls the out-params on failure.
    hipblasStatus_t mxConsumerPrepareInputs(const void*                                    A,
                                            const _rocblaslt_matrix_layout*                layoutA,
                                            const void*                                    B,
                                            const _rocblaslt_matrix_layout*                layoutB,
                                            const hipblasLtFusedEpilogueRMSNormDescriptor* stats,
                                            const MxConsumerDims&                          dims,
                                            void**                                         tmpA,
                                            void**                                         tmpB,
                                            hipblasLtMatrixLayout_t*                       tmpLayoutA,
                                            hipblasLtMatrixLayout_t*                       tmpLayoutB,
                                            hipStream_t                                    stream)
    {
        hipblasStatus_t status = mxConsumerAllocBf16Scratch(
            layoutA, layoutB, dims, tmpA, tmpB, tmpLayoutA, tmpLayoutB);
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;

        status = shimDequantF8ToBf16(A,
                                     *tmpA,
                                     stats->host_mx_scales.data(),
                                     dims.m,
                                     dims.k,
                                     layoutA->ld,
                                     dims.batchA,
                                     layoutA->batch_stride,
                                     dims.tileRows,
                                     dims.blockSize,
                                     dims.nRowTiles,
                                     dims.nBlocks,
                                     stream);
        if(status == HIPBLAS_STATUS_SUCCESS)
            status = shimDequantF8ToBf16(B,
                                         *tmpB,
                                         nullptr,
                                         dims.k,
                                         dims.n,
                                         layoutB->ld,
                                         dims.batchB,
                                         layoutB->batch_stride,
                                         dims.tileRows,
                                         dims.blockSize,
                                         dims.nRowTiles,
                                         dims.nBlocks,
                                         stream);
        if(status != HIPBLAS_STATUS_SUCCESS)
            mxConsumerFreeScratch(tmpA, tmpB, tmpLayoutA, tmpLayoutB);
        return status;
    }

    // Run the BF16 x BF16 GEMM2 for the MX consumer using a freshly derived heuristic algo.
    hipblasStatus_t mxConsumerRunGemm(hipblasLtHandle_t       handle,
                                      hipblasLtMatmulDesc_t   matmul_descr,
                                      const void*             alpha,
                                      void*                   tmpA,
                                      hipblasLtMatrixLayout_t tmpLayoutA,
                                      void*                   tmpB,
                                      hipblasLtMatrixLayout_t tmpLayoutB,
                                      const void*             beta,
                                      const void*             C,
                                      hipblasLtMatrixLayout_t matC,
                                      void*                   D,
                                      hipblasLtMatrixLayout_t matD,
                                      void*                   workspace,
                                      size_t                  workspaceSizeInBytes,
                                      hipStream_t             stream)
    {
        hipblasLtMatmulPreference_t pref = nullptr;
        if(hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS
           || hipblasLtMatmulPreferenceSetAttribute(pref,
                                                    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspaceSizeInBytes,
                                                    sizeof(workspaceSizeInBytes))
                  != HIPBLAS_STATUS_SUCCESS)
        {
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        hipblasLtMatmulHeuristicResult_t heuristic[1];
        int                              algoCount = 0;
        hipblasStatus_t                  status
            = hipblasLtMatmulAlgoGetHeuristic(handle,
                                              matmul_descr,
                                              tmpLayoutA,
                                              tmpLayoutB,
                                              matC,
                                              matD,
                                              pref,
                                              1,
                                              heuristic,
                                              &algoCount);
        hipblasLtMatmulPreferenceDestroy(pref);
        if(status != HIPBLAS_STATUS_SUCCESS || algoCount == 0)
            return algoCount == 0 ? HIPBLAS_STATUS_NOT_SUPPORTED : status;

        return RocBlasLtStatusToHIPStatus(
            rocblaslt_matmul((rocblaslt_handle)handle,
                             (rocblaslt_matmul_desc)matmul_descr,
                             alpha,
                             tmpA,
                             (rocblaslt_matrix_layout)tmpLayoutA,
                             tmpB,
                             (rocblaslt_matrix_layout)tmpLayoutB,
                             beta,
                             C,
                             (rocblaslt_matrix_layout)matC,
                             D,
                             (rocblaslt_matrix_layout)matD,
                             (const rocblaslt_matmul_algo*)&heuristic[0].algo,
                             workspace,
                             workspaceSizeInBytes,
                             stream));
    }

    // MX decomposed consumer: dequantize the FP8 codes of A (and B) to BF16 using the per-block
    // UE8M0 scales from the handoff, run a BF16 x BF16 GEMM2 into D, then apply the per-row rstd.
    // This models y = rstd * (h2_approx @ W1) where h2_approx is A round-tripped through FP8 block
    // quantization -- the closest the CPU shim gets without a true MXGEMM.
    hipblasStatus_t runMxConsumer(hipblasLtHandle_t                       handle,
                                  hipblasLtMatmulDesc_t                   matmul_descr,
                                  const hipblasLtFusedEpilogueDescriptor* fused,
                                  const void*                             alpha,
                                  const void*                             A,
                                  hipblasLtMatrixLayout_t                 matA,
                                  const void*                             B,
                                  hipblasLtMatrixLayout_t                 matB,
                                  const void*                             beta,
                                  const void*                             C,
                                  hipblasLtMatrixLayout_t                 matC,
                                  void*                                   D,
                                  hipblasLtMatrixLayout_t                 matD,
                                  void*                                   workspace,
                                  size_t                                  workspaceSizeInBytes,
                                  hipStream_t                             stream)
    {
        auto* layoutA = (rocblaslt_matrix_layout)matA;
        auto* layoutB = (rocblaslt_matrix_layout)matB;
        auto* layoutD = (rocblaslt_matrix_layout)matD;

        MxConsumerDims  dims;
        hipblasStatus_t status = mxConsumerValidate(layoutA, layoutB, fused->rmsnorm_stats, &dims);
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;

        void*                   tmpA       = nullptr;
        void*                   tmpB       = nullptr;
        hipblasLtMatrixLayout_t tmpLayoutA = nullptr;
        hipblasLtMatrixLayout_t tmpLayoutB = nullptr;
        status                             = mxConsumerPrepareInputs(A,
                                        layoutA,
                                        B,
                                        layoutB,
                                        fused->rmsnorm_stats,
                                        dims,
                                        &tmpA,
                                        &tmpB,
                                        &tmpLayoutA,
                                        &tmpLayoutB,
                                        stream);
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;

        status = mxConsumerRunGemm(handle,
                                   matmul_descr,
                                   alpha,
                                   tmpA,
                                   tmpLayoutA,
                                   tmpB,
                                   tmpLayoutB,
                                   beta,
                                   C,
                                   matC,
                                   D,
                                   matD,
                                   workspace,
                                   workspaceSizeInBytes,
                                   stream);
        if(status == HIPBLAS_STATUS_SUCCESS)
            status = fused_epilogue_cpu_shim(fused, layoutD, D, stream);

        mxConsumerFreeScratch(&tmpA, &tmpB, &tmpLayoutA, &tmpLayoutB);
        return status;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t            handle,
                                hipblasLtMatmulDesc_t        matmul_descr,
                                const void*                  alpha,
                                const void*                  A,
                                hipblasLtMatrixLayout_t      matA,
                                const void*                  B,
                                hipblasLtMatrixLayout_t      matB,
                                const void*                  beta,
                                const void*                  C,
                                hipblasLtMatrixLayout_t      matC,
                                void*                        D,
                                hipblasLtMatrixLayout_t      matD,
                                const hipblasLtMatmulAlgo_t* algo,
                                void*                        workspace,
                                size_t                       workspaceSizeInBytes,
                                hipStream_t                  stream)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmul");
    hipblasStatus_t return_status = HIPBLAS_STATUS_SUCCESS;

    // Fused-epilogue path: a composable fused epilogue (e.g. RMSNorm) attached via
    // HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE has no TensileLite kernels yet (AIHPBLAS-3856).
    // A CPU shim runs the base GEMM and applies supported chains (full residual+RMSNorm,
    // full residual+RMSNorm+requant, or the decomposed producer/consumer stages) on the host
    // so the API returns correct answers; unsupported chains/types still return NOT_SUPPORTED.
    if(auto* desc = (rocblaslt_matmul_desc)matmul_descr)
    {
        if(desc->fused_epilogue != nullptr)
        {
            const auto* fused   = desc->fused_epilogue;
            auto*       layoutD = (rocblaslt_matrix_layout)matD;
            const bool  has_requant = fused_epilogue_has_requant(fused);
            if(!fused_epilogue_cpu_shim_supported(fused) || layoutD == nullptr
               || layoutD->order != HIPBLASLT_ORDER_COL
               || (!has_requant && !(layoutD->type == HIP_R_32F || layoutD->type == HIP_R_16F
                    || layoutD->type == HIP_R_16BF))
               || (has_requant && layoutD->type != HIP_R_8F_E4M3))
            {
                rocblaslt::Debug::Instance().markerStop();
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            }

            if(has_requant)
            {
                const int32_t batch_count
                    = layoutD->batch_count < 1 ? 1 : layoutD->batch_count;
                const int64_t batch_stride = layoutD->batch_stride;
                const size_t  span
                    = static_cast<size_t>(batch_count - 1) * static_cast<size_t>(batch_stride)
                      + static_cast<size_t>(layoutD->n - 1) * static_cast<size_t>(layoutD->ld)
                      + static_cast<size_t>(layoutD->m);

                void*                  tmpD       = nullptr;
                hipblasLtMatrixLayout_t tmpLayoutD = nullptr;
                hipblasLtMatmulPreference_t tmpPref = nullptr;
                auto cleanup = [&]() {
                    if(tmpD)
                        static_cast<void>(hipFree(tmpD));
                    if(tmpLayoutD)
                        hipblasLtMatrixLayoutDestroy(tmpLayoutD);
                    if(tmpPref)
                        hipblasLtMatmulPreferenceDestroy(tmpPref);
                };

                if(hipMalloc(&tmpD, span * sizeof(hip_bfloat16)) != hipSuccess
                   || hipblasLtMatrixLayoutCreate(
                          &tmpLayoutD, HIP_R_16BF, layoutD->m, layoutD->n, layoutD->ld)
                          != HIPBLAS_STATUS_SUCCESS)
                {
                    cleanup();
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                }
                if(batch_count != 1)
                {
                    if(hipblasLtMatrixLayoutSetAttribute(tmpLayoutD,
                                                         HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                         &batch_count,
                                                         sizeof(batch_count))
                           != HIPBLAS_STATUS_SUCCESS
                       || hipblasLtMatrixLayoutSetAttribute(
                              tmpLayoutD,
                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                              &batch_stride,
                              sizeof(batch_stride))
                              != HIPBLAS_STATUS_SUCCESS)
                    {
                        cleanup();
                        rocblaslt::Debug::Instance().markerStop();
                        return HIPBLAS_STATUS_INTERNAL_ERROR;
                    }
                }

                if(hipblasLtMatmulPreferenceCreate(&tmpPref) != HIPBLAS_STATUS_SUCCESS
                   || hipblasLtMatmulPreferenceSetAttribute(
                          tmpPref,
                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                          &workspaceSizeInBytes,
                          sizeof(workspaceSizeInBytes))
                          != HIPBLAS_STATUS_SUCCESS)
                {
                    cleanup();
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                }

                hipblasLtMatmulHeuristicResult_t heuristic[1];
                int                              algo_count = 0;
                return_status = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                                matmul_descr,
                                                                matA,
                                                                matB,
                                                                matC,
                                                                tmpLayoutD,
                                                                tmpPref,
                                                                1,
                                                                heuristic,
                                                                &algo_count);
                if(return_status != HIPBLAS_STATUS_SUCCESS || algo_count == 0)
                {
                    cleanup();
                    rocblaslt::Debug::Instance().markerStop();
                    return algo_count == 0 ? HIPBLAS_STATUS_NOT_SUPPORTED : return_status;
                }

                return_status = RocBlasLtStatusToHIPStatus(
                    rocblaslt_matmul((rocblaslt_handle)handle,
                                     (rocblaslt_matmul_desc)matmul_descr,
                                     alpha,
                                     A,
                                     (rocblaslt_matrix_layout)matA,
                                     B,
                                     (rocblaslt_matrix_layout)matB,
                                     beta,
                                     C,
                                     (rocblaslt_matrix_layout)matC,
                                     tmpD,
                                     (rocblaslt_matrix_layout)tmpLayoutD,
                                     (const rocblaslt_matmul_algo*)&heuristic[0].algo,
                                     workspace,
                                     workspaceSizeInBytes,
                                     stream));
                if(return_status == HIPBLAS_STATUS_SUCCESS)
                {
                    if(fused->requant_granularity == HIPBLASLT_REQUANT_SCALE_MX_BLOCK)
                        return_status
                            = fused_epilogue_cpu_shim_mxquant_typed<hipblaslt_f8, hip_bfloat16>(
                                fused,
                                tmpD,
                                D,
                                layoutD->m,
                                layoutD->n,
                                layoutD->ld,
                                batch_count,
                                batch_stride,
                                stream);
                    else
                        return_status
                            = fused_epilogue_cpu_shim_requant_typed<hipblaslt_f8, hip_bfloat16>(
                                fused,
                                tmpD,
                                D,
                                layoutD->m,
                                layoutD->n,
                                layoutD->ld,
                                batch_count,
                                batch_stride,
                                stream);
                }
                cleanup();
                rocblaslt::Debug::Instance().markerStop();
                return return_status;
            }

            // MX decomposed consumer: when the handoff carries per-block MX scales, dequantize the
            // FP8 codes before GEMM2 instead of contracting the raw codes.
            const bool is_mx_consumer
                = fused_epilogue_has_stage(fused,
                                           HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY)
                  && fused->rmsnorm_stats != nullptr
                  && !fused->rmsnorm_stats->host_mx_scales.empty();
            if(is_mx_consumer)
            {
                return_status = runMxConsumer(handle,
                                                matmul_descr,
                                                fused,
                                                alpha,
                                                A,
                                                matA,
                                                B,
                                                matB,
                                                beta,
                                                C,
                                                matC,
                                                D,
                                                matD,
                                                workspace,
                                                workspaceSizeInBytes,
                                                stream);
                rocblaslt::Debug::Instance().markerStop();
                return return_status;
            }

            // Compute the base GEMM first (rocblaslt_matmul ignores the fused-epilogue handle),
            // then apply the fused epilogue chain on the host.
            return_status = RocBlasLtStatusToHIPStatus(
                rocblaslt_matmul((rocblaslt_handle)handle,
                                 (rocblaslt_matmul_desc)matmul_descr,
                                 alpha,
                                 A,
                                 (rocblaslt_matrix_layout)matA,
                                 B,
                                 (rocblaslt_matrix_layout)matB,
                                 beta,
                                 C,
                                 (rocblaslt_matrix_layout)matC,
                                 D,
                                 (rocblaslt_matrix_layout)matD,
                                 (const rocblaslt_matmul_algo*)algo,
                                 workspace,
                                 workspaceSizeInBytes,
                                 stream));
            if(return_status != HIPBLAS_STATUS_SUCCESS)
            {
                rocblaslt::Debug::Instance().markerStop();
                return return_status;
            }

            return_status = fused_epilogue_cpu_shim(fused, layoutD, D, stream);
            rocblaslt::Debug::Instance().markerStop();
            return return_status;
        }
    }

    return_status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul((rocblaslt_handle)handle,
                                                                (rocblaslt_matmul_desc)matmul_descr,
                                                                alpha,
                                                                A,
                                                                (rocblaslt_matrix_layout)matA,
                                                                B,
                                                                (rocblaslt_matrix_layout)matB,
                                                                beta,
                                                                C,
                                                                (rocblaslt_matrix_layout)matC,
                                                                D,
                                                                (rocblaslt_matrix_layout)matD,
                                                                (const rocblaslt_matmul_algo*)algo,
                                                                workspace,
                                                                workspaceSizeInBytes,
                                                                stream));
    rocblaslt::Debug::Instance().markerStop();
    return return_status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixTransformDescCreate(hipblasLtMatrixTransformDesc_t* transformDesc,
                                                   hipDataType                     scaleType)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransformDescCreate");
    static_assert(sizeof(rocblaslt_matrix_transform_desc)
                      <= sizeof(hipblasLtMatrixTransformDescOpaque_t),
                  "hipblasLtMatrixTransformDescOpaque_t must have enough space");
    rocblaslt_matrix_transform_desc desc;
    desc.scaleType = scaleType;
    *transformDesc = new hipblasLtMatrixTransformDescOpaque_t;
    memcpy((*transformDesc)->data, &desc, sizeof(desc));
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasLtMatrixTransformDescDestroy(hipblasLtMatrixTransformDesc_t transformDesc)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransformDescDestroy");
    if(transformDesc)
        delete transformDesc;
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t
    hipblasLtMatrixTransformDescSetAttribute(hipblasLtMatrixTransformDesc_t           transformDesc,
                                             hipblasLtMatrixTransformDescAttributes_t attr,
                                             const void*                              buf,
                                             size_t                                   sizeInBytes)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransformDescSetAttribute");
    if(!buf || sizeInBytes != sizeof(int32_t))
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    rocblaslt_matrix_transform_desc* desc
        = reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]);
    // all possible values should be int32_t
    assert(sizeInBytes == sizeof(int32_t));
    int32_t value{};
    memcpy(&value, buf, sizeInBytes);

    switch(attr)
    {
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
    {
        desc->scaleType = static_cast<hipDataType>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
    {
        desc->pointerMode = static_cast<hipblasLtPointerMode_t>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
    {
        desc->opA = static_cast<hipblasOperation_t>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
    {
        desc->opB = static_cast<hipblasOperation_t>(value);
        break;
    }
    default:
        assert(false && "Unknown attribute");
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    }
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t
    hipblasLtMatrixTransformDescGetAttribute(hipblasLtMatrixTransformDesc_t           transformDesc,
                                             hipblasLtMatrixTransformDescAttributes_t attr,
                                             void*                                    buf,
                                             size_t                                   sizeInBytes,
                                             size_t*                                  sizeWritten)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransformDescGetAttribute");
    if(!sizeInBytes && !sizeWritten)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    if(sizeInBytes && !sizeWritten)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    if(sizeInBytes != sizeof(int32_t))
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    rocblaslt_matrix_transform_desc* desc
        = reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]);
    int32_t value{};

    switch(attr)
    {
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
    {
        value = static_cast<int32_t>(desc->scaleType);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
    {
        value = static_cast<int32_t>(desc->pointerMode);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
    {
        value = static_cast<int32_t>(desc->opA);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
    {
        value = static_cast<int32_t>(desc->opB);
        break;
    }
    default:
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
        assert(false && "Unknown attribute");
        break;
    }

    memcpy(buf, &value, sizeInBytes);
    *sizeWritten = sizeof(int32_t);
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasLtMatrixTransform(hipblasLtHandle_t              lightHandle,
                                         hipblasLtMatrixTransformDesc_t transformDesc,
                                         const void*             alpha, /* host or device pointer */
                                         const void*             A,
                                         hipblasLtMatrixLayout_t Adesc,
                                         const void*             beta, /* host or device pointer */
                                         const void*             B,
                                         hipblasLtMatrixLayout_t Bdesc,
                                         void*                   C,
                                         hipblasLtMatrixLayout_t Cdesc,
                                         hipStream_t             stream)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransform");
    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matrix_transform(
        (rocblaslt_handle)lightHandle,
        reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]),
        alpha,
        A,
        (rocblaslt_matrix_layout)Adesc,
        beta,
        B,
        (rocblaslt_matrix_layout)Bdesc,
        C,
        (rocblaslt_matrix_layout)Cdesc,
        stream));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}

// Other Utilities
hipblasStatus_t hipblasLtGetVersion(hipblasLtHandle_t handle, int* version)
try
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }

    *version = HIPBLASLT_VERSION_MAJOR * 100000 + HIPBLASLT_VERSION_MINOR * 100
               + HIPBLASLT_VERSION_PATCH;

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtGetGitRevision(hipblasLtHandle_t handle, char* rev)
try
{
    // Get hipBLASLt revision
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }

    if(rev == nullptr)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    static constexpr char v[] = TO_STR(HIPBLASLT_VERSION_TWEAK);

    memcpy(rev, v, sizeof(v));

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtGetArchName(char** archName)
try
{
    *archName        = nullptr;
    std::string arch = rocblaslt_internal_get_arch_name();
    *archName        = (char*)malloc(arch.size() + 1);
    memcpy(*archName, arch.c_str(), arch.size() + 1);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    if(archName != nullptr)
    {
        free(*archName);
        *archName = nullptr;
    }
    return exception_to_hipblas_status();
}

#ifdef __cplusplus
}
#endif
