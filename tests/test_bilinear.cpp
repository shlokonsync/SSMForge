#include <gtest/gtest.h>

#include "ssmforge/bilinear.hpp"

#include <complex>

TEST(BilinearTest, MapsStableEigenvalueInsideUnitCircle)
{
    using Complex = std::complex<double>;

    Eigen::VectorXcd lambda(1);

    // Stable continuous-time eigenvalue.
    lambda(0) = Complex(-1.0, 0.0);

    const double dt = 0.1;

    Eigen::VectorXcd result =
        ssmforge::bilinear_transform(lambda, dt);

    // The transformed eigenvalue should be
    // approximately 0.90476.
    EXPECT_NEAR(
        result(0).real(),
        0.9047619,
        1e-6
    );

    EXPECT_NEAR(
        result(0).imag(),
        0.0,
        1e-12
    );

    // Stable discrete-time eigenvalues must
    // lie inside the unit circle.
    EXPECT_LT(
        std::abs(result(0)),
        1.0
    );
}


TEST(BilinearTest, HandlesComplexEigenvalue)
{
    using Complex = std::complex<double>;

    Eigen::VectorXcd lambda(1);

    lambda(0) = Complex(-2.0, 3.0);

    const double dt = 0.1;

    Eigen::VectorXcd result =
        ssmforge::bilinear_transform(lambda, dt);

    // The result should be a valid finite
    // complex number.
    EXPECT_TRUE(std::isfinite(result(0).real()));
    EXPECT_TRUE(std::isfinite(result(0).imag()));

    EXPECT_LT(
        std::abs(result(0)),
        1.0
    );
}


TEST(BilinearTest, RejectsInvalidDt)
{
    Eigen::VectorXcd lambda(1);

    lambda(0) = {-1.0, 0.0};

    EXPECT_THROW(
        ssmforge::bilinear_transform(lambda, 0.0),
        std::invalid_argument
    );

    EXPECT_THROW(
        ssmforge::bilinear_transform(lambda, -0.1),
        std::invalid_argument
    );
}