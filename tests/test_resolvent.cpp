#include <gtest/gtest.h>

#include "ssmforge/resolvent.hpp"

#include <complex>

TEST(ResolventTest, ComputesSimpleResolvent)
{
    using Complex = std::complex<double>;

    /*
     * Simple 1-state system:
     *
     * A = [2]
     * B = [3]
     * C = [4]
     *
     * Therefore:
     *
     * H(z) = C (zI - A)^(-1) B
     *
     *       = 4 * 1/(z - 2) * 3
     *
     *       = 12/(z - 2)
     */

    Eigen::MatrixXcd A(1, 1);
    A(0, 0) = Complex(2.0, 0.0);

    Eigen::VectorXcd B(1);
    B(0) = Complex(3.0, 0.0);

    Eigen::RowVectorXcd C(1);
    C(0) = Complex(4.0, 0.0);

    /*
     * Evaluate at z = 5.
     *
     * H(5) = 12/(5-2)
     *      = 4
     */

    Eigen::VectorXcd z(1);
    z(0) = Complex(5.0, 0.0);

    Eigen::VectorXcd result =
        ssmforge::evaluate_resolvent(A, B, C, z);

    EXPECT_NEAR(result(0).real(), 4.0, 1e-9);
    EXPECT_NEAR(result(0).imag(), 0.0, 1e-9);
}


TEST(ResolventTest, ComputesMultipleZValues)
{
    using Complex = std::complex<double>;

    /*
     * Again:
     *
     * A = [2]
     * B = [3]
     * C = [4]
     *
     * H(z) = 12/(z-2)
     */

    Eigen::MatrixXcd A(1, 1);
    A(0, 0) = Complex(2.0, 0.0);

    Eigen::VectorXcd B(1);
    B(0) = Complex(3.0, 0.0);

    Eigen::RowVectorXcd C(1);
    C(0) = Complex(4.0, 0.0);

    /*
     * Evaluate at:
     *
     * z = 5
     * z = 6
     * z = 8
     */

    Eigen::VectorXcd z(3);

    z(0) = Complex(5.0, 0.0);
    z(1) = Complex(6.0, 0.0);
    z(2) = Complex(8.0, 0.0);

    Eigen::VectorXcd result =
        ssmforge::evaluate_resolvent(A, B, C, z);

    /*
     * H(5) = 12/3 = 4
     * H(6) = 12/4 = 3
     * H(8) = 12/6 = 2
     */

    EXPECT_NEAR(result(0).real(), 4.0, 1e-9);
    EXPECT_NEAR(result(1).real(), 3.0, 1e-9);
    EXPECT_NEAR(result(2).real(), 2.0, 1e-9);
}


TEST(ResolventTest, ComputesComplexZ)
{
    using Complex = std::complex<double>;

    /*
     * 1-state system:
     *
     * A = [1]
     * B = [1]
     * C = [1]
     *
     * Therefore:
     *
     * H(z) = 1/(z-1)
     */

    Eigen::MatrixXcd A(1, 1);
    A(0, 0) = Complex(1.0, 0.0);

    Eigen::VectorXcd B(1);
    B(0) = Complex(1.0, 0.0);

    Eigen::RowVectorXcd C(1);
    C(0) = Complex(1.0, 0.0);

    /*
     * z = 2 + i
     *
     * H(z) = 1/(1+i)
     *
     *       = (1-i)/2
     *
     *       = 0.5 - 0.5i
     */

    Eigen::VectorXcd z(1);
    z(0) = Complex(2.0, 1.0);

    Eigen::VectorXcd result =
        ssmforge::evaluate_resolvent(A, B, C, z);

    EXPECT_NEAR(result(0).real(), 0.5, 1e-9);
    EXPECT_NEAR(result(0).imag(), -0.5, 1e-9);
}


TEST(ResolventTest, RejectsNonSquareA)
{
    using Complex = std::complex<double>;

    /*
     * A must be N x N.
     */

    Eigen::MatrixXcd A(2, 3);
    A.setZero();

    Eigen::VectorXcd B(2);
    B.setZero();

    Eigen::RowVectorXcd C(2);
    C.setZero();

    Eigen::VectorXcd z(1);
    z(0) = Complex(1.0, 0.0);

    EXPECT_THROW(
        ssmforge::evaluate_resolvent(A, B, C, z),
        std::invalid_argument
    );
}


TEST(ResolventTest, RejectsEmptyZ)
{
    using Complex = std::complex<double>;

    Eigen::MatrixXcd A(1, 1);
    A(0, 0) = Complex(1.0, 0.0);

    Eigen::VectorXcd B(1);
    B(0) = Complex(1.0, 0.0);

    Eigen::RowVectorXcd C(1);
    C(0) = Complex(1.0, 0.0);

    Eigen::VectorXcd z(0);

    EXPECT_THROW(
        ssmforge::evaluate_resolvent(A, B, C, z),
        std::invalid_argument
    );
}