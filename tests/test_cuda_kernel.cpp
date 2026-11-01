#include <gtest/gtest.h>

#include "ssmforge/bilinear.hpp"
#include "ssmforge/cuda_kernel.hpp"
#include "ssmforge/discrete_kernel.hpp"
#include "ssmforge/discretize.hpp"
#include "ssmforge/hippo.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

TEST(CudaKernelTest, MatchesReferenceSmallConfigurations)
{
    const int state_sizes[] = {2, 4, 8, 16};
    const int sequence_lengths[] = {8, 16, 64, 256};
    const double step_sizes[] = {0.01, 0.1, 0.5};

    for (int N : state_sizes)
    {
        for (int L : sequence_lengths)
        {
            for (double dt : step_sizes)
            {
                const auto hippo = ssmforge::hippo_legs(N);
                const Eigen::VectorXd B = Eigen::VectorXd::Ones(N);
                const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

                const auto discrete =
                    ssmforge::discretize_bilinear(hippo.A, B, dt);

                const Eigen::VectorXd reference =
                    ssmforge::compute_discrete_kernel(
                        discrete.Ad,
                        discrete.Bd,
                        C,
                        L
                    );

                const Eigen::VectorXd cuda_kernel =
                    ssmforge::compute_cuda_kernel(
                        discrete.Ad,
                        discrete.Bd,
                        C,
                        L
                    );

                ASSERT_EQ(reference.size(), cuda_kernel.size());

                for (int t = 0; t < L; ++t)
                {
                    EXPECT_NEAR(reference(t), cuda_kernel(t), 1e-10)
                        << "Mismatch at N=" << N
                        << ", L=" << L
                        << ", dt=" << dt
                        << ", t=" << t;
                }
            }
        }
    }
}

TEST(CudaKernelTest, MatchesReferenceLargeLength)
{
    const int N = 64;
    const int L = 4096;
    const double dt = 0.1;

    const auto hippo = ssmforge::hippo_legs(N);
    const Eigen::VectorXd B = Eigen::VectorXd::Ones(N);
    const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

    const auto discrete =
        ssmforge::discretize_bilinear(hippo.A, B, dt);

    const Eigen::VectorXd reference =
        ssmforge::compute_discrete_kernel(
            discrete.Ad,
            discrete.Bd,
            C,
            L
        );

    const Eigen::VectorXd cuda_kernel =
        ssmforge::compute_cuda_kernel(
            discrete.Ad,
            discrete.Bd,
            C,
            L
        );

    ASSERT_EQ(reference.size(), cuda_kernel.size());

    const double max_error =
        (reference - cuda_kernel).cwiseAbs().maxCoeff();
    EXPECT_LT(max_error, 1e-10);
}

TEST(CudaKernelTest, RejectsInvalidAdNonSquare)
{
    Eigen::MatrixXd Ad(4, 3);
    Ad.setZero();
    Eigen::VectorXd Bd = Eigen::VectorXd::Ones(4);
    Eigen::RowVectorXd Cd = Eigen::RowVectorXd::Ones(4);

    EXPECT_THROW(
        ssmforge::compute_cuda_kernel(Ad, Bd, Cd, 16),
        std::invalid_argument
    );
}

TEST(CudaKernelTest, RejectsInvalidBdSize)
{
    Eigen::MatrixXd Ad = Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd Bd = Eigen::VectorXd::Ones(3);
    Eigen::RowVectorXd Cd = Eigen::RowVectorXd::Ones(4);

    EXPECT_THROW(
        ssmforge::compute_cuda_kernel(Ad, Bd, Cd, 16),
        std::invalid_argument
    );
}

TEST(CudaKernelTest, RejectsInvalidCdSize)
{
    Eigen::MatrixXd Ad = Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd Bd = Eigen::VectorXd::Ones(4);
    Eigen::RowVectorXd Cd = Eigen::RowVectorXd::Ones(3);

    EXPECT_THROW(
        ssmforge::compute_cuda_kernel(Ad, Bd, Cd, 16),
        std::invalid_argument
    );
}

TEST(CudaKernelTest, RejectsInvalidLength)
{
    Eigen::MatrixXd Ad = Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd Bd = Eigen::VectorXd::Ones(4);
    Eigen::RowVectorXd Cd = Eigen::RowVectorXd::Ones(4);

    EXPECT_THROW(
        ssmforge::compute_cuda_kernel(Ad, Bd, Cd, 0),
        std::invalid_argument
    );

    EXPECT_THROW(
        ssmforge::compute_cuda_kernel(Ad, Bd, Cd, -5),
        std::invalid_argument
    );
}
