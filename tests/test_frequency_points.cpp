#include <gtest/gtest.h>

#include "ssmforge/frequency_points.hpp"

#include <cmath>
#include <complex>

TEST(FrequencyPointsTest, GeneratesCorrectPoints)
{
    const int L = 4;

    Eigen::VectorXcd points =
        ssmforge::frequency_points(L);

    ASSERT_EQ(points.size(), L);

    const double tolerance = 1e-12;

    // 1
    EXPECT_NEAR(points(0).real(), 1.0, tolerance);
    EXPECT_NEAR(points(0).imag(), 0.0, tolerance);

    // i
    EXPECT_NEAR(points(1).real(), 0.0, tolerance);
    EXPECT_NEAR(points(1).imag(), 1.0, tolerance);

    // -1
    EXPECT_NEAR(points(2).real(), -1.0, tolerance);
    EXPECT_NEAR(points(2).imag(), 0.0, tolerance);

    // -i
    EXPECT_NEAR(points(3).real(), 0.0, tolerance);
    EXPECT_NEAR(points(3).imag(), -1.0, tolerance);
}


TEST(FrequencyPointsTest, AllPointsLieOnUnitCircle)
{
    const int L = 16;

    Eigen::VectorXcd points =
        ssmforge::frequency_points(L);

    for (int k = 0; k < L; ++k)
    {
        // Every point should satisfy:
        //
        //     |w_k| = 1
        //
        EXPECT_NEAR(
            std::abs(points(k)),
            1.0,
            1e-12
        );
    }
}


TEST(FrequencyPointsTest, RejectsInvalidLength)
{
    EXPECT_THROW(
        ssmforge::frequency_points(0),
        std::invalid_argument
    );

    EXPECT_THROW(
        ssmforge::frequency_points(-5),
        std::invalid_argument
    );
}