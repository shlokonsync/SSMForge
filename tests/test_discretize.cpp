#include <gtest/gtest.h>

#include <Eigen/Dense>
#include "ssmforge/discretize.hpp"

TEST(DiscretizationTest, SimpleScalarSystem)
{
    // Simple system:
    // dx/dt = -x + u
    Eigen::MatrixXd A(1, 1);
    A << -1.0;

    Eigen::VectorXd B(1);
    B << 1.0;

    double dt = 0.1;

    // Convert continuous-time SSM to discrete-time SSM using ZOH.
    auto result = ssmforge::discretize_zoh(A, B, dt);

    // For this system:
    // Ad = exp(-0.1) ≈ 0.904837
    // Bd = 1 - exp(-0.1) ≈ 0.0951626
    EXPECT_NEAR(result.Ad(0, 0), 0.904837, 1e-5);
    EXPECT_NEAR(result.Bd(0), 0.0951626, 1e-5);
}

TEST(DiscretizationTest, BilinearScalarSystem)
{
    // dx/dt = -x + u
    Eigen::MatrixXd A(1, 1);
    A << -1.0;

    Eigen::VectorXd B(1);
    B << 1.0;

    double dt = 0.1;

    // Bilinear transform:
    // Ad = (1 + dt/2 * (-1)) / (1 - dt/2 * (-1)) = (1 - 0.05) / (1 + 0.05) = 0.95 / 1.05 ≈ 0.9047619
    // Bd = dt / (1 - dt/2 * (-1)) = 0.1 / 1.05 ≈ 0.0952381
    auto result = ssmforge::discretize_bilinear(A, B, dt);

    EXPECT_NEAR(result.Ad(0, 0), 0.9047619, 1e-5);
    EXPECT_NEAR(result.Bd(0), 0.0952381, 1e-5);
}