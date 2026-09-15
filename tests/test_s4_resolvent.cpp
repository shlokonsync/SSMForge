#include <gtest/gtest.h>

#include "ssmforge/hippo.hpp"
#include "ssmforge/nplr.hpp"
#include "ssmforge/resolvent.hpp"
#include "ssmforge/s4_resolvent.hpp"

#include <complex>

TEST(S4ResolventTest, MatchesReferenceResolvent)
{
    using Complex = std::complex<double>;

    // --------------------------------------------------
    // Create a small SSM.
    //
    // We use a 4-state system so that we can actually
    // test the NPLR transformation.
    // --------------------------------------------------

    const int N = 4;

    auto hippo = ssmforge::hippo_legs(N);
    Eigen::MatrixXd A = hippo.A;
    Eigen::VectorXd B = hippo.B;
    Eigen::RowVectorXd C(N);

    C << 1.0, -1.0, 0.5, 2.0;

    // --------------------------------------------------
    // Pick several complex evaluation points.
    // --------------------------------------------------

    Eigen::VectorXcd z(3);

    z(0) = Complex(1.0, 0.5);
    z(1) = Complex(2.0, 1.0);
    z(2) = Complex(3.0, -0.5);

    // --------------------------------------------------
    // Reference implementation.
    //
    // Directly computes:
    //
    //     H(z) = C(zI-A)^(-1)B
    // --------------------------------------------------

    Eigen::MatrixXcd A_complex =
        A.cast<Complex>();

    Eigen::VectorXcd B_complex =
        B.cast<Complex>();

    Eigen::RowVectorXcd C_complex =
        C.cast<Complex>();

    Eigen::VectorXcd reference =
        ssmforge::evaluate_resolvent(
            A_complex,
            B_complex,
            C_complex,
            z
        );

    // --------------------------------------------------
    // Build NPLR representation.
    //
    // This transforms the system into:
    //
    //     A' = Lambda - P Q*
    //
    //     B' = V*B
    //
    //     C' = CV
    // --------------------------------------------------

    ssmforge::NPLR nplr =
        ssmforge::nplr_decompose(A, B, C);

    // --------------------------------------------------
    // Optimized S4 resolvent.
    //
    // Uses:
    //
    //     Woodbury
    //
    // instead of directly solving the N x N system.
    // --------------------------------------------------

    Eigen::VectorXcd optimized =
        ssmforge::evaluate_s4_resolvent(
            nplr,
            z
        );

    // --------------------------------------------------
    // Compare both implementations.
    //
    // They should produce the same complex numbers
    // up to floating-point error.
    // --------------------------------------------------

    ASSERT_EQ(reference.size(), optimized.size());

    for (int i = 0; i < z.size(); ++i)
    {
        EXPECT_NEAR(
            reference(i).real(),
            optimized(i).real(),
            1e-8
        );

        EXPECT_NEAR(
            reference(i).imag(),
            optimized(i).imag(),
            1e-8
        );
    }
}


TEST(S4ResolventTest, HandlesSingleEvaluationPoint)
{
    using Complex = std::complex<double>;

    const int N = 2;
    auto hippo = ssmforge::hippo_legs(N);
    Eigen::MatrixXd A = hippo.A;
    Eigen::VectorXd B = hippo.B;
    Eigen::RowVectorXd C(2);

    C << 1.0, 3.0;

    // One complex z.
    Eigen::VectorXcd z(1);

    z(0) = Complex(2.0, 1.0);

    // Reference.
    Eigen::VectorXcd reference =
        ssmforge::evaluate_resolvent(
            A.cast<Complex>(),
            B.cast<Complex>(),
            C.cast<Complex>(),
            z
        );

    // NPLR.
    ssmforge::NPLR nplr =
        ssmforge::nplr_decompose(A, B, C);

    // S4.
    Eigen::VectorXcd optimized =
        ssmforge::evaluate_s4_resolvent(
            nplr,
            z
        );

    EXPECT_NEAR(
        reference(0).real(),
        optimized(0).real(),
        1e-8
    );

    EXPECT_NEAR(
        reference(0).imag(),
        optimized(0).imag(),
        1e-8
    );
}


TEST(S4ResolventTest, RejectsInvalidNPLR)
{
    using Complex = std::complex<double>;

    ssmforge::NPLR nplr;

    // Incorrectly sized matrices.
    nplr.N = Eigen::MatrixXcd::Zero(3, 3);

    nplr.P = Eigen::MatrixXcd::Zero(2, 1);
    nplr.Q = Eigen::MatrixXcd::Zero(3, 1);

    nplr.B = Eigen::VectorXcd::Zero(3);
    nplr.C = Eigen::RowVectorXcd::Zero(3);

    Eigen::VectorXcd z(1);

    z(0) = Complex(1.0, 0.0);

    EXPECT_THROW(
        ssmforge::evaluate_s4_resolvent(nplr, z),
        std::invalid_argument
    );
}