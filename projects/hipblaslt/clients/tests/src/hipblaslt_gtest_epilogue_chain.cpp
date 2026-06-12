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
//  - Attaching an incomplete RMSNorm handle (null gamma or unset eps) is rejected at
//    descriptor-set time (INVALID_VALUE).
//  - A complete-but-unimplemented fused epilogue is rejected by hipblasLtMatmul with
//    NOT_SUPPORTED before kernel selection/launch.

#include <cmath>
#include <gtest/gtest.h>
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

        hipblasStatus_t attach()
        {
            // The attribute value is the handle (a pointer); pass its pointer-sized storage.
            return hipblasLtMatmulDescSetAttribute(desc,
                                                   HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                                   &fused,
                                                   sizeof(hipblasLtFusedEpilogueDescriptor_t));
        }

        hipblasLtMatmulDesc_t              desc  = nullptr;
        hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
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

// ---- Add: ordering legality ----

TEST_F(FusedEpilogueTest, legalOrderAccepted)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_AMAX),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_FP8_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, illegalOrderRejected)
{
    // FP8 requant then RMSNorm violates the supported RMSNorm order (requant must come last).
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_FP8_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, amaxAfterFp8Rejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_FP8_REQUANT),
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

// ---- SetAttribute validation ----

TEST_F(FusedEpilogueTest, setUnknownAttributeRejected)
{
    const float eps = 1e-5f;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, static_cast<hipblasLtFusedEpilogueAttribute_t>(999), &eps, sizeof(eps)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Attach-time completeness validation ----

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

// ---- Complete-but-unimplemented config rejected by matmul (NOT_SUPPORTED) ----
//
// The hipblasLtMatmul wrapper guards an attached fused epilogue before kernel selection,
// so this returns NOT_SUPPORTED without requiring a GPU or valid layouts.

TEST_F(FusedEpilogueTest, attachedFusedEpilogueMatmulNotSupported)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
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
