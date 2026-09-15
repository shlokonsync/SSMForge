#include <gtest/gtest.h>

#include "ssmforge/hippo.hpp"

#include <cmath>

TEST(HiPPOTTest, CorrectDimensions)
{
    const int N = 4;

    auto result = ssmforge::hippo_legs(N);

    // A should be N x N.
    EXPECT_EQ(result.A.rows(), N);
    EXPECT_EQ(result.A.cols(), N);

    // B should contain N elements.
    EXPECT_EQ(result.B.size(), N);
}


TEST(HiPPOTTest, CorrectInputVector)
{
    const int N = 4;

    auto result = ssmforge::hippo_legs(N);

    // HiPPO-LegS:
    //
    // B[n] = sqrt(2n + 1)
    //
    // n starts from 0.
    for (int n = 0; n < N; ++n)
    {
        double expected = std::sqrt(2.0 * n + 1.0);

        EXPECT_NEAR(
            result.B(n),
            expected,
            1e-10
        );
    }
}


TEST(HiPPOTTest, CorrectDiagonal)
{
    const int N = 4;

    auto result = ssmforge::hippo_legs(N);

    // Diagonal of HiPPO-LegS:
    //
    // A[n,n] = -(n + 1)
    //
    // Therefore:
    // A[0,0] = -1
    // A[1,1] = -2
    // A[2,2] = -3
    // A[3,3] = -4

    for (int n = 0; n < N; ++n)
    {
        double expected = -(n + 1.0);

        EXPECT_NEAR(
            result.A(n, n),
            expected,
            1e-10
        );
    }
}


TEST(HiPPOTTest, UpperTriangularPartIsZero)
{
    const int N = 4;

    auto result = ssmforge::hippo_legs(N);

    // HiPPO-LegS is lower triangular.
    //
    // Therefore, whenever n < k:
    //
    // A[n,k] = 0

    for (int n = 0; n < N; ++n)
    {
        for (int k = n + 1; k < N; ++k)
        {
            EXPECT_NEAR(
                result.A(n, k),
                0.0,
                1e-10
            );
        }
    }
}


TEST(HiPPOTTest, CorrectLowerTriangularValues)
{
    const int N = 4;

    auto result = ssmforge::hippo_legs(N);

    // For n > k:
    //
    // A[n,k] = -sqrt((2n+1)(2k+1))

    for (int n = 0; n < N; ++n)
    {
        for (int k = 0; k < n; ++k)
        {
            double expected =
                -std::sqrt(
                    (2.0 * n + 1.0) *
                    (2.0 * k + 1.0)
                );

            EXPECT_NEAR(
                result.A(n, k),
                expected,
                1e-10
            );
        }
    }
}