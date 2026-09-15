#include <gtest/gtest.h>

#include "ssmforge/hippo.hpp"
#include "ssmforge/nplr.hpp"

#include <cmath>

TEST(NPLRTest, CorrectDimensions)
{
    const int N = 8;

    // First construct the original HiPPO system.
    auto hippo = ssmforge::hippo_legs(N);
    Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

    // Convert it to NPLR form.
    auto result =
        ssmforge::nplr_decompose(
            hippo.A,
            hippo.B,
            C
        );

    // N should be an N x N diagonal matrix.
    EXPECT_EQ(result.N.rows(), N);
    EXPECT_EQ(result.N.cols(), N);

    // P and Q are rank-1 vectors stored as N x 1 matrices.
    EXPECT_EQ(result.P.rows(), N);
    EXPECT_EQ(result.P.cols(), 1);

    EXPECT_EQ(result.Q.rows(), N);
    EXPECT_EQ(result.Q.cols(), 1);

    // B should contain N elements.
    EXPECT_EQ(result.B.size(), N);

    // C should contain N elements.
    EXPECT_EQ(result.C.cols(), N);
}


TEST(NPLRTest, LowRankVectorsNonZero)
{
    const int N = 8;

    auto hippo = ssmforge::hippo_legs(N);
    Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

    auto result =
        ssmforge::nplr_decompose(
            hippo.A,
            hippo.B,
            C
        );

    // P and Q should be non-zero low-rank vectors.
    EXPECT_GT(result.P.norm(), 0.0);
    EXPECT_GT(result.Q.norm(), 0.0);
}


TEST(NPLRTest, NIsDiagonal)
{
    const int N = 8;

    auto hippo = ssmforge::hippo_legs(N);
    Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

    auto result =
        ssmforge::nplr_decompose(
            hippo.A,
            hippo.B,
            C
        );

    // N contains the eigenvalues of the normal matrix.
    //
    // Therefore N should be diagonal.
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                EXPECT_NEAR(
                    std::abs(result.N(i, j)),
                    0.0,
                    1e-10
                );
            }
        }
    }
}


TEST(NPLRTest, LowRankCorrectionHasRankOne)
{
    const int N = 8;

    auto hippo = ssmforge::hippo_legs(N);
    Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

    auto result =
        ssmforge::nplr_decompose(
            hippo.A,
            hippo.B,
            C
        );

    // P and Q are N x 1.
    //
    // Therefore:
    //
    //     P Q*
    //
    // can have rank at most 1.
    Eigen::MatrixXcd correction =
        result.P * result.Q.adjoint();

    EXPECT_EQ(correction.rows(), N);
    EXPECT_EQ(correction.cols(), N);
}


TEST(NPLRTest, InputVectorHasCorrectSize)
{
    const int N = 8;

    auto hippo = ssmforge::hippo_legs(N);
    Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

    auto result =
        ssmforge::nplr_decompose(
            hippo.A,
            hippo.B,
            C
        );

    EXPECT_EQ(result.B.rows(), N);
}