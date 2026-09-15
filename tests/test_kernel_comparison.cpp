#include <gtest/gtest.h>

#include <cmath>

#include "ssmforge/bilinear.hpp"
#include "ssmforge/discrete_kernel.hpp"
#include "ssmforge/discretize.hpp"
#include "ssmforge/hippo.hpp"
#include "ssmforge/nplr.hpp"
#include "ssmforge/s4_kernel.hpp"

TEST(KernelComparisonTest, MultipleConfigurations)
{
    const int state_sizes[] = {2, 4, 8};
    const int sequence_lengths[] = {8, 16, 32};
    const double step_sizes[] = {0.01, 0.1, 0.5};

    for (int N : state_sizes)
    {
        for (int L : sequence_lengths)
        {
            for (double dt : step_sizes)
            {
                const auto hippo =
                    ssmforge::hippo_legs(N);

                Eigen::RowVectorXd C =
                    Eigen::RowVectorXd::Ones(N);

                const auto discrete =
                    ssmforge::discretize_bilinear(
                        hippo.A,
                        hippo.B,
                        dt
                    );

                const Eigen::VectorXd reference_kernel =
                    ssmforge::compute_discrete_kernel(
                        discrete.Ad,
                        discrete.Bd,
                        C,
                        L
                    );

                const auto nplr =
                    ssmforge::nplr_decompose(
                        hippo.A,
                        hippo.B,
                        C
                    );

                const Eigen::VectorXd fast_kernel =
                    ssmforge::compute_s4_kernel(
                        nplr,
                        L,
                        dt
                    );

                ASSERT_EQ(
                    reference_kernel.size(),
                    fast_kernel.size()
                );

                for (int t = 0; t < L; ++t)
                {
                    EXPECT_NEAR(
                        reference_kernel(t),
                        fast_kernel(t),
                        1e-5
                    ) << "N = " << N
                      << ", L = " << L
                      << ", dt = " << dt
                      << ", t = " << t;
                }
            }
        }
    }
}