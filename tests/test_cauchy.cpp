#include <gtest/gtest.h>

#include "ssmforge/cauchy.hpp"

#include <complex>

TEST(CauchyTest, ComputesCorrectValues)
{
    using Complex = std::complex<double>;

    // Two poles/eigenvalues.
    Eigen::VectorXcd lambda(2);
    lambda << Complex(1.0, 0.0),
              Complex(2.0, 0.0);

    // One evaluation point.
    Eigen::VectorXcd z(1);
    z << Complex(4.0, 0.0);

    // Weights.
    Eigen::VectorXcd v(2);
    v << Complex(2.0, 0.0),
         Complex(3.0, 0.0);

    Eigen::VectorXcd result =
        ssmforge::cauchy(lambda, z, v);

    /*
     * Manually:
     *
     * y = 2/(4-1) + 3/(4-2)
     *
     *   = 2/3 + 3/2
     *
     *   = 2.166666...
     */

    EXPECT_NEAR(
        result(0).real(),
        2.1666666667,
        1e-9
    );

    EXPECT_NEAR(
        result(0).imag(),
        0.0,
        1e-9
    );
}


TEST(CauchyTest, WorksWithMultipleEvaluationPoints)
{
    using Complex = std::complex<double>;

    Eigen::VectorXcd lambda(2);
    lambda << Complex(1.0, 0.0),
              Complex(2.0, 0.0);

    Eigen::VectorXcd z(2);
    z << Complex(4.0, 0.0),
         Complex(5.0, 0.0);

    Eigen::VectorXcd v(2);
    v << Complex(2.0, 0.0),
         Complex(3.0, 0.0);

    Eigen::VectorXcd result =
        ssmforge::cauchy(lambda, z, v);

    // For z = 4:
    //
    // 2/(4-1) + 3/(4-2)
    // = 2/3 + 3/2
    // = 2.166666...

    EXPECT_NEAR(
        result(0).real(),
        2.1666666667,
        1e-9
    );

    // For z = 5:
    //
    // 2/(5-1) + 3/(5-2)
    // = 1/2 + 1
    // = 1.5

    EXPECT_NEAR(
        result(1).real(),
        1.5,
        1e-9
    );
}


TEST(CauchyTest, WorksWithComplexNumbers)
{
    using Complex = std::complex<double>;

    Eigen::VectorXcd lambda(1);
    lambda << Complex(1.0, 1.0);

    Eigen::VectorXcd z(1);
    z << Complex(3.0, 2.0);

    Eigen::VectorXcd v(1);
    v << Complex(2.0, 0.0);

    Eigen::VectorXcd result =
        ssmforge::cauchy(lambda, z, v);

    // Direct calculation:
    //
    // 2 / ((3+2i) - (1+i))
    //
    // = 2 / (2+i)
    //
    // = 0.8 - 0.4i

    EXPECT_NEAR(
        result(0).real(),
        0.8,
        1e-9
    );

    EXPECT_NEAR(
        result(0).imag(),
        -0.4,
        1e-9
    );
}


TEST(CauchyTest, RejectsMismatchedSizes)
{
    using Complex = std::complex<double>;

    Eigen::VectorXcd lambda(2);
    lambda << Complex(1.0, 0.0),
              Complex(2.0, 0.0);

    Eigen::VectorXcd z(1);
    z << Complex(4.0, 0.0);

    // Wrong size: lambda has 2 elements,
    // but v has only 1.
    Eigen::VectorXcd v(1);
    v << Complex(2.0, 0.0);

    EXPECT_THROW(
        ssmforge::cauchy(lambda, z, v),
        std::invalid_argument
    );
}