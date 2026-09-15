#include <gtest/gtest.h>

#include "ssmforge/kernel.hpp"
#include "ssmforge/discrete_kernel.hpp"

#include <complex>

TEST(KernelTest, ComputesBasicKernel)
{
    using Complex = std::complex<double>;

    /*
     * We use a very simple 2-state system:
     *
     *        [1 0]
     * A =    [0 2]
     *
     *        [1]
     * B =    [1]
     *
     * C = [1 1]
     *
     * We want:
     *
     * K[t] = C A^t B
     */

    Eigen::MatrixXcd A(2, 2);

    A << Complex(1.0, 0.0), Complex(0.0, 0.0),
         Complex(0.0, 0.0), Complex(2.0, 0.0);

    Eigen::VectorXcd B(2);

    B << Complex(1.0, 0.0),
         Complex(1.0, 0.0);

    Eigen::RowVectorXcd C(2);

    C << Complex(1.0, 0.0),
         Complex(1.0, 0.0);

    const int L = 4;

    Eigen::VectorXcd K =
        ssmforge::compute_kernel(A, B, C, L);

    /*
     * Let's calculate manually.
     *
     * K[0] = C B
     *
     *      = [1 1] [1]
     *              [1]
     *
     *      = 2
     */

    EXPECT_NEAR(K(0).real(), 2.0, 1e-9);

    /*
     * A B:
     *
     * [1 0] [1] = [1]
     * [0 2] [1]   [2]
     *
     * K[1] = C A B
     *
     *      = [1 1] [1]
     *              [2]
     *
     *      = 3
     */

    EXPECT_NEAR(K(1).real(), 3.0, 1e-9);

    /*
     * A^2 B:
     *
     * A [1] = [1]
     *   [2]   [4]
     *
     * K[2] = [1 1] [1]
     *               [4]
     *
     *      = 5
     */

    EXPECT_NEAR(K(2).real(), 5.0, 1e-9);

    /*
     * A^3 B:
     *
     * A [1] = [1]
     *   [4]   [8]
     *
     * K[3] = 1 + 8 = 9
     */

    EXPECT_NEAR(K(3).real(), 9.0, 1e-9);
}


TEST(KernelTest, ComputesComplexKernel)
{
    using Complex = std::complex<double>;

    /*
     * Simple 1-state complex system:
     *
     * A = [2 + i]
     * B = [1]
     * C = [1]
     *
     * Therefore:
     *
     * K[t] = (2+i)^t
     */

    Eigen::MatrixXcd A(1, 1);
    A(0, 0) = Complex(2.0, 1.0);

    Eigen::VectorXcd B(1);
    B(0) = Complex(1.0, 0.0);

    Eigen::RowVectorXcd C(1);
    C(0) = Complex(1.0, 0.0);

    Eigen::VectorXcd K =
        ssmforge::compute_kernel(A, B, C, 3);

    /*
     * K[0] = 1
     *
     * K[1] = 2 + i
     *
     * K[2] = (2+i)^2
     *      = 3 + 4i
     */

    EXPECT_NEAR(K(0).real(), 1.0, 1e-9);
    EXPECT_NEAR(K(0).imag(), 0.0, 1e-9);

    EXPECT_NEAR(K(1).real(), 2.0, 1e-9);
    EXPECT_NEAR(K(1).imag(), 1.0, 1e-9);

    EXPECT_NEAR(K(2).real(), 3.0, 1e-9);
    EXPECT_NEAR(K(2).imag(), 4.0, 1e-9);
}


TEST(KernelTest, RejectsInvalidMatrix)
{
    using Complex = std::complex<double>;

    /*
     * A is not square, so it cannot be a valid
     * state-transition matrix.
     */

    Eigen::MatrixXcd A(2, 3);
    A.setZero();

    Eigen::VectorXcd B(2);
    B.setZero();

    Eigen::RowVectorXcd C(2);
    C.setZero();

    EXPECT_THROW(
        ssmforge::compute_kernel(A, B, C, 4),
        std::invalid_argument
    );
}


TEST(KernelTest, RejectsInvalidKernelLength)
{
    using Complex = std::complex<double>;

    Eigen::MatrixXcd A(1, 1);
    A(0, 0) = Complex(1.0, 0.0);

    Eigen::VectorXcd B(1);
    B(0) = Complex(1.0, 0.0);

    Eigen::RowVectorXcd C(1);
    C(0) = Complex(1.0, 0.0);

    /*
     * Kernel length must be positive.
     */

    EXPECT_THROW(
        ssmforge::compute_kernel(A, B, C, 0),
        std::invalid_argument
    );
}

TEST(KernelTest, ComputesRealDiscreteKernel)
{
    // A = [0.5 0; 0 0.25], B = [1; 1], C = [1 1]
    // K[0] = C B = 2
    // K[1] = C A B = [1 1] [0.5; 0.25] = 0.75
    // K[2] = C A^2 B = [1 1] [0.25; 0.0625] = 0.3125
    Eigen::MatrixXd A(2, 2);
    A << 0.5, 0.0,
         0.0, 0.25;

    Eigen::VectorXd B(2);
    B << 1.0, 1.0;

    Eigen::RowVectorXd C(2);
    C << 1.0, 1.0;

    Eigen::VectorXd K = ssmforge::compute_discrete_kernel(A, B, C, 3);

    ASSERT_EQ(K.size(), 3);
    EXPECT_NEAR(K(0), 2.0, 1e-9);
    EXPECT_NEAR(K(1), 0.75, 1e-9);
    EXPECT_NEAR(K(2), 0.3125, 1e-9);
}