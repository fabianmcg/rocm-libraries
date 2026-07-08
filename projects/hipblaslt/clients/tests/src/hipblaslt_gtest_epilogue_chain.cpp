/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Tests for the handle-based composable fused-epilogue API (RMSNorm focus).
//
// These tests validate the RMSNorm reference math plus composition rules at the API level,
// without any fused-epilogue kernels:
//  - The CPU reference implements y = x * rsqrt(mean(x^2) + eps) * gamma.
//  - The create/add/set/destroy lifecycle returns SUCCESS.
//  - An unrecognized or unsupported epilogue and an illegal/duplicate ordering are rejected
//    by hipblasLtFusedEpilogueAdd (INVALID_VALUE).
//  - Attaching an incomplete residual-add handle (unset residual pointer) is rejected at
//    descriptor-set time (INVALID_VALUE). If no residual-output pointer is provided, or if
//    it is explicitly set to NULL, the residual input is the in-place write-back target for
//    the updated residual stream.
//  - Attaching an incomplete RMSNorm handle (unset gamma or unset eps) is rejected at
//    descriptor-set time (INVALID_VALUE).
//  - The decomposed flow (partial RMSNorm stats producer + RMSNorm scale-apply consumer) is
//    linked by an opaque, library-populated RMSNorm handoff descriptor. Its create/destroy
//    lifecycle returns SUCCESS, full and decomposed RMSNorm stages cannot be mixed in one
//    chain, and attaching a decomposed handle without the handoff descriptor (or, for the
//    producer, without gamma/eps) is rejected at descriptor-set time (INVALID_VALUE).
//  - A chain the CPU shim does not implement (decomposed stages, AMax, FP8 requant) is
//    rejected by hipblasLtMatmul with NOT_SUPPORTED before kernel selection/launch.
//  - The CPU shim (until the TensileLite fused kernels land) makes hipblasLtMatmul return
//    correct answers for the full residual+RMSNorm flow; an end-to-end test compares device
//    output against the CPU reference on a real GEMM (skipped when no HIP device is present).

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>

namespace
{
    void cpuRmsNorm(float*       out,
                    const float* in,
                    const float* gamma,
                    std::size_t  rows,
                    std::size_t  cols,
                    float        eps)
    {
        for(std::size_t row = 0; row < rows; ++row)
        {
            const auto offset = row * cols;
            float      sum_sq = 0.0f;
            for(std::size_t col = 0; col < cols; ++col)
                sum_sq += in[offset + col] * in[offset + col];

            const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
            for(std::size_t col = 0; col < cols; ++col)
                out[offset + col] = in[offset + col] * inv_rms * gamma[col];
        }
    }

    // Runs D = alpha * (A @ B) on device with a fused-epilogue chain attached (column-major,
    // op N, BF16 storage, FP32 compute). Sets *algoCount to the number of heuristic solutions
    // found and returns the hipblasLtMatmul status (or a setup failure). Used by the decomposed
    // end-to-end test to drive the producer (GEMM1) and consumer (GEMM2) calls.
    hipblasStatus_t runBf16GemmWithFusedEpilogue(hipblasLtHandle_t                   handle,
                                                 int64_t                            m,
                                                 int64_t                            n,
                                                 int64_t                            k,
                                                 void*                              dA,
                                                 void*                              dB,
                                                 void*                              dC,
                                                 void*                              dD,
                                                 hipblasLtFusedEpilogueDescriptor_t fused,
                                                 void*                              dWorkspace,
                                                 size_t                             maxWorkspace,
                                                 int*                               algoCount)
    {
        *algoCount = 0;
        hipblasLtMatrixLayout_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
        hipblasLtMatrixLayoutCreate(&matA, HIP_R_16BF, m, k, m);
        hipblasLtMatrixLayoutCreate(&matB, HIP_R_16BF, k, n, k);
        hipblasLtMatrixLayoutCreate(&matC, HIP_R_16BF, m, n, m);
        hipblasLtMatrixLayoutCreate(&matD, HIP_R_16BF, m, n, m);

        hipblasLtMatmulDesc_t matmul = nullptr;
        hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused));

        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulPreferenceCreate(&pref);
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspace, sizeof(maxWorkspace));

        hipblasLtMatmulHeuristicResult_t heuristic[1];
        hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul, matA, matB, matC, matD, pref, 1, heuristic, algoCount);

        float           alpha = 1.0f, beta = 0.0f;
        hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
        if(*algoCount > 0)
            status = hipblasLtMatmul(handle,
                                     matmul,
                                     &alpha,
                                     dA,
                                     matA,
                                     dB,
                                     matB,
                                     &beta,
                                     dC,
                                     matC,
                                     dD,
                                     matD,
                                     &heuristic[0].algo,
                                     dWorkspace,
                                     maxWorkspace,
                                     hipStream_t(0));

        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(matmul);
        hipblasLtMatrixLayoutDestroy(matA);
        hipblasLtMatrixLayoutDestroy(matB);
        hipblasLtMatrixLayoutDestroy(matC);
        hipblasLtMatrixLayoutDestroy(matD);
        return status;
    }

    // Value snapped to the BF16 grid, returned as float (mirrors a BF16 store/reload).
    inline float refRndBf(float f)
    {
        return static_cast<float>(hip_bfloat16(f));
    }

    // CPU GEMM over column-major inputs, returning rndBf(A @ B) as column-major [rows, cols]
    // with FP32 accumulation and a BF16-rounded result, matching the device GEMM writing D as
    // BF16. A is [rows, kk], B is [kk, cols]; accepts BF16 or FP32 element types.
    template <typename TA, typename TB>
    std::vector<float> cpuGemmBf16(const std::vector<TA>& A,
                                   const std::vector<TB>& B,
                                   int64_t                rows,
                                   int64_t                cols,
                                   int64_t                kk)
    {
        std::vector<float> out(static_cast<size_t>(rows) * cols, 0.0f);
        for(int64_t i = 0; i < rows; ++i)
            for(int64_t j = 0; j < cols; ++j)
            {
                float acc = 0.0f;
                for(int64_t l = 0; l < kk; ++l)
                    acc += static_cast<float>(A[l * rows + i]) * static_cast<float>(B[j * kk + l]);
                out[j * rows + i] = refRndBf(acc);
            }
        return out;
    }

    // CPU reference for the residual-add + RMSNorm reduce path over a column-major [rows, cols]
    // GEMM output, mirroring the shim's BF16 rounding at each storage point. Fills `rstd` with
    // the FP32 per-row reciprocal RMS and writes the downstream value to `out`: (z * gamma)
    // scaled by rstd when apply_scale is set (full flow) or left unscaled (decomposed producer).
    void cpuResidualRmsnormRef(const std::vector<float>&        gemm,
                               const std::vector<hip_bfloat16>& residual,
                               const std::vector<hip_bfloat16>& gamma,
                               int64_t                          rows,
                               int64_t                          cols,
                               float                            eps,
                               bool                             apply_scale,
                               std::vector<float>&              out,
                               std::vector<float>&              rstd)
    {
        out.assign(static_cast<size_t>(rows) * cols, 0.0f);
        rstd.assign(rows, 0.0f);
        for(int64_t i = 0; i < rows; ++i)
        {
            float              sumSq = 0.0f;
            std::vector<float> zStored(cols);
            for(int64_t j = 0; j < cols; ++j)
            {
                const float z
                    = gemm[j * rows + i] + static_cast<float>(residual[j * rows + i]);
                zStored[j] = refRndBf(z);  // stored back to D as BF16
                sumSq += z * z;            // FP32 accumulation of the pre-store value
            }
            rstd[i] = 1.0f / std::sqrt(sumSq / static_cast<float>(cols) + eps);
            for(int64_t j = 0; j < cols; ++j)
            {
                float v = zStored[j] * static_cast<float>(gamma[j]);
                if(apply_scale)
                    v *= rstd[i];
                out[j * rows + i] = refRndBf(v);
            }
        }
    }

    class FusedEpilogueTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
        }
        void TearDown() override
        {
            if(fused)
                hipblasLtFusedEpilogueDestroy(fused);
            if(stats)
                hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
        }

        // Set gamma/eps so an RMSNorm handle passes attach-time validation.
        void completeRmsnorm()
        {
            int   dummy_gamma_storage = 0;
            void* gamma               = &dummy_gamma_storage;
            const float eps           = 1e-5f;
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        // Set the residual input pointer so a residual-add handle passes attach-time
        // validation. Without a residual output pointer, the API uses this pointer as the
        // in-place destination for the updated residual stream.
        void completeResidual()
        {
            int   dummy_residual_storage = 0;
            void* residual               = &dummy_residual_storage;
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &residual, sizeof(residual)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        // Create and set the opaque RMSNorm handoff descriptor so a decomposed producer or
        // consumer handle passes attach-time validation.
        void completeStats()
        {
            ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t attach()
        {
            // The attribute value is the handle (a pointer); pass its pointer-sized storage.
            return hipblasLtMatmulDescSetAttribute(desc,
                                                   HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                                   &fused,
                                                   sizeof(hipblasLtFusedEpilogueDescriptor_t));
        }

        hipblasLtMatmulDesc_t                     desc  = nullptr;
        hipblasLtFusedEpilogueDescriptor_t        fused = nullptr;
        hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    };
}

// ---- RMSNorm math reference ----

TEST(FusedEpilogueMath, rmsnormReferenceNormalizesRows)
{
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 4;
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> gamma(cols, 1.0f);
    std::vector<float>       output(rows * cols, 0.0f);

    cpuRmsNorm(output.data(), input.data(), gamma.data(), rows, cols, 0.0f);

    for(std::size_t row = 0; row < rows; ++row)
    {
        float mean_sq = 0.0f;
        for(std::size_t col = 0; col < cols; ++col)
        {
            const auto v = output[row * cols + col];
            mean_sq += v * v;
        }
        mean_sq /= static_cast<float>(cols);
        EXPECT_NEAR(mean_sq, 1.0f, 1e-6f);
    }
}

TEST(FusedEpilogueMath, rmsnormReferenceAppliesGamma)
{
    constexpr std::size_t rows = 1;
    constexpr std::size_t cols = 4;
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> gamma = {1.0f, 0.5f, 2.0f, -1.0f};
    std::vector<float>       output(rows * cols, 0.0f);

    cpuRmsNorm(output.data(), input.data(), gamma.data(), rows, cols, 0.0f);

    const float inv_rms = 1.0f / std::sqrt(7.5f);
    EXPECT_NEAR(output[0], 1.0f * inv_rms, 1e-6f);
    EXPECT_NEAR(output[1], 2.0f * inv_rms * 0.5f, 1e-6f);
    EXPECT_NEAR(output[2], 3.0f * inv_rms * 2.0f, 1e-6f);
    EXPECT_NEAR(output[3], 4.0f * inv_rms * -1.0f, 1e-6f);
}

// ---- Lifecycle ----

TEST(FusedEpilogueLifecycle, createAddDestroy)
{
    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    EXPECT_NE(fused, nullptr);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueDestroy(fused), HIPBLAS_STATUS_SUCCESS);
}

TEST(FusedEpilogueLifecycle, createNullRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(FusedEpilogueLifecycle, rmsnormStatsCreateDestroy)
{
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    EXPECT_NE(stats, nullptr);
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats), HIPBLAS_STATUS_SUCCESS);
}

TEST(FusedEpilogueLifecycle, rmsnormStatsCreateNullRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Add: ordering legality ----

TEST_F(FusedEpilogueTest, legalOrderAccepted)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_AMAX),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, illegalOrderRejected)
{
    // Requant then RMSNorm violates the supported RMSNorm order (requant must come last).
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, amaxAfterRequantRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_AMAX),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, swigluRejectedByRmsnormChainValidator)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_SWIGLU),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, rmsnormBeforeResidualRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, duplicateStageRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, unknownEpilogueRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, static_cast<hipblasLtFuseableEpilogue_t>(999)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Add: decomposed-flow ordering and family mixing ----

TEST_F(FusedEpilogueTest, decomposedProducerOrderAccepted)
{
    // Producer chain: residual add -> partial RMSNorm stats.
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, decomposedConsumerAccepted)
{
    // Consumer chain: RMSNorm scale-apply only.
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, partialStatsBeforeResidualRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, fullRmsnormThenPartialStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, partialStatsThenFullRmsnormRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, producerAndConsumerStagesInOneChainRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- SetAttribute validation ----

TEST_F(FusedEpilogueTest, setUnknownAttributeRejected)
{
    const float eps = 1e-5f;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, static_cast<hipblasLtFusedEpilogueAttribute_t>(999), &eps, sizeof(eps)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullResidualPointerRejected)
{
    void* residual = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &residual, sizeof(residual)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullRmsnormGammaRejected)
{
    void* gamma = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullRmsnormStatsRejected)
{
    hipblasLtFusedEpilogueRMSNormDescriptor_t null_stats = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &null_stats, sizeof(null_stats)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullResidualOutputAcceptedAsInPlace)
{
    void* residual_output = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, setInvalidRequantComputeModeRejected)
{
    auto mode = static_cast<hipblasLtRequantScaleComputeMode_t>(999);
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE, &mode, sizeof(mode)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setInvalidRequantGranularityRejected)
{
    auto granularity = static_cast<hipblasLtRequantScaleGranularity_t>(999);
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY,
                                                 &granularity,
                                                 sizeof(granularity)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Attach-time completeness validation ----

TEST_F(FusedEpilogueTest, attachResidualMissingPointerRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    // residual pointer never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachResidualInPlaceWritebackAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    // No residual-output pointer is required; unset means update the residual input in place.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualSeparateWritebackAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    int   dummy_residual_output_storage = 0;
    void* residual_output               = &dummy_residual_output_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualOutputCanBeClearedToInPlace)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    int   dummy_residual_output_storage = 0;
    void* residual_output               = &dummy_residual_output_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    residual_output = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachRmsnormMissingGammaRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    const float eps = 1e-5f;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    // gamma never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachRmsnormMissingEpsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    int   dummy = 0;
    void* gamma = &dummy;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
              HIPBLAS_STATUS_SUCCESS);
    // eps never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteRmsnormAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualRmsnormAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    completeRmsnorm();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

// ---- Attach-time completeness validation: decomposed flow ----

TEST_F(FusedEpilogueTest, attachPartialStatsMissingStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    // stats handoff descriptor never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachPartialStatsMissingGammaEpsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeStats();
    // gamma/eps never set -> attach must reject (the producer computes the partial stats).
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteProducerAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    completeRmsnorm();
    completeStats();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachScaleApplyMissingStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    // stats handoff descriptor never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteConsumerAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    completeStats();
    // scale-apply only needs the handoff descriptor; gamma/eps live on the producer.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachNullFusedEpilogueDetaches)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    ASSERT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t null_fused = nullptr;
    EXPECT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                                              HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                              &null_fused,
                                              sizeof(null_fused)),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachRequantMissingScaleRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    // scale pointer never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteRequantAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    int   dummy_scale_storage = 0;
    void* scale               = &dummy_scale_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER, &scale, sizeof(scale)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

// ---- Chains the CPU shim does not implement are rejected by matmul (NOT_SUPPORTED) ----
//
// The hipblasLtMatmul wrapper checks whether the attached chain is shim-supported before
// kernel selection, so an unsupported chain returns NOT_SUPPORTED without requiring a GPU or
// valid layouts. The full residual+RMSNorm flow IS shim-supported and is exercised for
// correctness by the end-to-end test below instead.

TEST_F(FusedEpilogueTest, attachedUnsupportedChainMatmulNotSupported)
{
    // RMSNorm + AMax: legal ordering, but AMax is not emulated by the CPU shim.
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_AMAX),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    ASSERT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(hipblasLtMatmul(nullptr,
                              desc,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              0,
                              nullptr),
              HIPBLAS_STATUS_NOT_SUPPORTED);
}

// ---- End-to-end correctness of the CPU shim (full residual + RMSNorm flow) ----
//
// Runs a real column-major BF16 GEMM with an attached residual+RMSNorm fused epilogue and
// compares the device output against a CPU reference. The reference mirrors the shim's BF16
// rounding at each storage point (GEMM output, residual sum, normalized output) so the compare
// stays tight. Skipped when no HIP device is present.

TEST(FusedEpilogueEndToEnd, residualRmsnormFullFlowMatchesReference)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "no HIP device available";

    // BF16 storage for A/B/C/D, residual, and gamma; FP32 compute/accumulation.
    using bf16 = hip_bfloat16;
    auto toF   = [](bf16 v) { return static_cast<float>(v); };
    auto toBf  = [](float f) { return bf16(f); };

    // D = [m, n], contraction dim k. Column-major throughout.
    const int64_t m = 64, n = 48, k = 32;
    const float   eps = 1e-5f;

    std::vector<bf16> hA(m * k), hB(k * n), hC(m * n, toBf(0.0f)), hResidual(m * n), hGamma(n);
    for(int64_t i = 0; i < m * k; ++i)
        hA[i] = toBf(static_cast<float>((i % 13) - 6) * 0.05f);
    for(int64_t i = 0; i < k * n; ++i)
        hB[i] = toBf(static_cast<float>((i % 11) - 5) * 0.04f);
    for(int64_t i = 0; i < m * n; ++i)
        hResidual[i] = toBf(static_cast<float>((i % 7) - 3) * 0.1f);
    for(int64_t j = 0; j < n; ++j)
        hGamma[j] = toBf(0.5f + static_cast<float>(j % 5) * 0.1f);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dResidual = nullptr,
         *dGamma = nullptr, *dWorkspace = nullptr;
    const size_t maxWorkspace = 32 * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dC, hC.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, hC.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWorkspace, maxWorkspace), hipSuccess);
    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dC, hC.data(), hC.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dResidual, hResidual.data(), hResidual.size() * sizeof(bf16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Full flow: residual add + RMSNorm on a single matmul call.
    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &dResidual, sizeof(dResidual)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);

    // Run the GEMM with the fused epilogue attached: the base GEMM writes D on device, then the
    // shim applies residual add + RMSNorm on the host. Skip if the shape has no GEMM solution.
    int                   algoCount = 0;
    const hipblasStatus_t status    = runBf16GemmWithFusedEpilogue(
        handle, m, n, k, dA, dB, dC, dD, fused, dWorkspace, maxWorkspace, &algoCount);
    if(algoCount == 0)
        GTEST_SKIP() << "no GEMM solution found for the test shape";
    ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    std::vector<bf16> hD(m * n, toBf(0.0f));
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(bf16), hipMemcpyDeviceToHost),
              hipSuccess);

    // CPU reference: y = RMSNorm(A@B + residual, gamma, eps), column-major, mirroring the BF16
    // rounding the device path applies at each storage point.
    const std::vector<float> gemm = cpuGemmBf16(hA, hB, m, n, k);
    std::vector<float>       ref, rstd;
    cpuResidualRmsnormRef(gemm, hResidual, hGamma, m, n, eps, /*apply_scale=*/true, ref, rstd);

    for(int64_t idx = 0; idx < m * n; ++idx)
        EXPECT_NEAR(toF(hD[idx]), ref[idx], 2e-2f) << "mismatch at flat index " << idx;

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dC));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dWorkspace));
}

// ---- End-to-end correctness of the CPU shim (decomposed GEMM -> residual -> RMSNorm -> GEMM) ----
//
// Exercises the decomposed flow across two matmul calls linked by the opaque RMSNorm handoff
// descriptor:
//   GEMM1 (producer):  h2 = (x @ W0 + residual) * gamma, and the per-row scale rstd is stashed
//   GEMM2 (consumer):  y  = rstd * (h2 @ W1)
// The combined result equals RMSNorm(x @ W0 + residual) @ W1. The device output is compared
// against a CPU reference that mirrors the shim's BF16 rounding. Skipped when no HIP device.

TEST(FusedEpilogueEndToEnd, decomposedResidualRmsnormFlowMatchesReference)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "no HIP device available";

    using bf16 = hip_bfloat16;
    auto toF   = [](bf16 v) { return static_cast<float>(v); };
    auto toBf  = [](float f) { return bf16(f); };

    // GEMM1 (producer): x[M0,K0] @ W0[K0,N0] -> [M0,N0].
    // GEMM2 (consumer): h2[M1,K1] @ W1[K1,N1] -> [M1,N1], with M1 == M0 (per-row scale carries
    // through) and K1 == N0 (GEMM2 contracts over GEMM1's feature dimension).
    const int64_t M0 = 32, N0 = 24, K0 = 16;
    const int64_t M1 = M0, K1 = N0, N1 = 20;
    const float   eps = 1e-5f;

    std::vector<bf16> hX(M0 * K0), hW0(K0 * N0), hResidual(M0 * N0), hGamma(N0), hW1(K1 * N1);
    std::vector<bf16> hC1(M0 * N0, toBf(0.0f)), hC2(M1 * N1, toBf(0.0f));
    for(int64_t i = 0; i < M0 * K0; ++i)
        hX[i] = toBf(static_cast<float>((i % 13) - 6) * 0.05f);
    for(int64_t i = 0; i < K0 * N0; ++i)
        hW0[i] = toBf(static_cast<float>((i % 11) - 5) * 0.04f);
    for(int64_t i = 0; i < M0 * N0; ++i)
        hResidual[i] = toBf(static_cast<float>((i % 7) - 3) * 0.3f);
    for(int64_t j = 0; j < N0; ++j)
        hGamma[j] = toBf(0.5f + static_cast<float>(j % 5) * 0.1f);
    for(int64_t i = 0; i < K1 * N1; ++i)
        hW1[i] = toBf(static_cast<float>((i % 9) - 4) * 0.05f);

    void *dX = nullptr, *dW0 = nullptr, *dC1 = nullptr, *dH2 = nullptr, *dResidual = nullptr,
         *dGamma = nullptr, *dW1 = nullptr, *dC2 = nullptr, *dD2 = nullptr, *dWorkspace = nullptr;
    const size_t maxWorkspace = 32 * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dX, hX.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW0, hW0.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dC1, hC1.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dH2, hC1.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW1, hW1.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dC2, hC2.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD2, hC2.size() * sizeof(bf16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWorkspace, maxWorkspace), hipSuccess);
    ASSERT_EQ(hipMemcpy(dX, hX.data(), hX.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW0, hW0.data(), hW0.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dC1, hC1.data(), hC1.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dResidual, hResidual.data(), hResidual.size() * sizeof(bf16), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW1, hW1.data(), hW1.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dC2, hC2.data(), hC2.size() * sizeof(bf16), hipMemcpyHostToDevice),
              hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Handoff descriptor shared by the producer and consumer calls.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);

    // Producer chain (GEMM1): residual add + partial RMSNorm stats.
    hipblasLtFusedEpilogueDescriptor_t prod = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&prod), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  prod, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &dResidual, sizeof(dResidual)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
              HIPBLAS_STATUS_SUCCESS);

    // Run GEMM1: writes h2 to dH2 and stashes the per-row scale into the handoff descriptor.
    int algoCount = 0;
    const hipblasStatus_t s1 = runBf16GemmWithFusedEpilogue(
        handle, M0, N0, K0, dX, dW0, dC1, dH2, prod, dWorkspace, maxWorkspace, &algoCount);
    if(algoCount == 0)
        GTEST_SKIP() << "no GEMM1 solution found for the test shape";
    ASSERT_EQ(s1, HIPBLAS_STATUS_SUCCESS);

    // Consumer chain (GEMM2): RMSNorm scale-apply reading the same handoff descriptor.
    hipblasLtFusedEpilogueDescriptor_t cons = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&cons), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(cons, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  cons, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
              HIPBLAS_STATUS_SUCCESS);

    // Run GEMM2: applies the deferred per-row scale from the handoff descriptor.
    const hipblasStatus_t s2 = runBf16GemmWithFusedEpilogue(
        handle, M1, N1, K1, dH2, dW1, dC2, dD2, cons, dWorkspace, maxWorkspace, &algoCount);
    if(algoCount == 0)
        GTEST_SKIP() << "no GEMM2 solution found for the test shape";
    ASSERT_EQ(s2, HIPBLAS_STATUS_SUCCESS);

    std::vector<bf16> hD2(M1 * N1, toBf(0.0f));
    ASSERT_EQ(hipMemcpy(hD2.data(), dD2, hD2.size() * sizeof(bf16), hipMemcpyDeviceToHost),
              hipSuccess);

    // CPU reference mirroring the shim's BF16 rounding at each storage point. The per-row scale
    // (rstd) is kept in FP32, matching the handoff descriptor.
    //  - GEMM1 producer: h2 = (rndBf(x@W0) + residual) * gamma, no scale applied; stash rstd.
    //  - GEMM2 consumer: y  = rstd * rndBf(h2 @ W1).
    const std::vector<float> gemm1 = cpuGemmBf16(hX, hW0, M0, N0, K0);
    std::vector<float>       h2ref, rstd;
    cpuResidualRmsnormRef(gemm1, hResidual, hGamma, M0, N0, eps, /*apply_scale=*/false, h2ref, rstd);

    const std::vector<float> gemm2 = cpuGemmBf16(h2ref, hW1, M1, N1, K1);
    std::vector<float>       yref(M1 * N1, 0.0f);
    for(int64_t i = 0; i < M1; ++i)
        for(int64_t jj = 0; jj < N1; ++jj)
            yref[jj * M1 + i] = refRndBf(gemm2[jj * M1 + i] * rstd[i]); // consumer scale-apply

    for(int64_t idx = 0; idx < M1 * N1; ++idx)
        EXPECT_NEAR(toF(hD2[idx]), yref[idx], 3e-2f) << "mismatch at flat index " << idx;

    hipblasLtFusedEpilogueDestroy(prod);
    hipblasLtFusedEpilogueDestroy(cons);
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dX));
    static_cast<void>(hipFree(dW0));
    static_cast<void>(hipFree(dC1));
    static_cast<void>(hipFree(dH2));
    static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dW1));
    static_cast<void>(hipFree(dC2));
    static_cast<void>(hipFree(dD2));
    static_cast<void>(hipFree(dWorkspace));
}
