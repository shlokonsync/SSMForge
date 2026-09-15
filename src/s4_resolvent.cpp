#include "ssmforge/s4_resolvent.hpp"

#include <stdexcept>

namespace ssmforge {

Eigen::VectorXcd evaluate_s4_resolvent(
    const NPLR& nplr,
    const Eigen::VectorXcd& z
) {
    // --------------------------------------------------
    // Validate dimensions
    // --------------------------------------------------

    const int N = static_cast<int>(nplr.N.rows());

    if (nplr.N.rows() != nplr.N.cols()) {
        throw std::invalid_argument(
            "N must be square"
        );
    }

    if (nplr.P.rows() != N || nplr.P.cols() != 1) {
        throw std::invalid_argument(
            "P must be N x 1 for rank-1 NPLR"
        );
    }

    if (nplr.Q.rows() != N || nplr.Q.cols() != 1) {
        throw std::invalid_argument(
            "Q must be N x 1 for rank-1 NPLR"
        );
    }

    if (nplr.B.size() != N) {
        throw std::invalid_argument(
            "B must have the same size as N"
        );
    }

    if (nplr.C.size() != N) {
        throw std::invalid_argument(
            "C must have the same size as N"
        );
    }

    if (z.size() == 0) {
        throw std::invalid_argument(
            "z must not be empty"
        );
    }

    // --------------------------------------------------
    // Extract Lambda.
    //
    // nplr.N is stored as a diagonal matrix:
    //
    //     Lambda =
    //
    //     [ lambda_0   0        0 ]
    //     [ 0          lambda_1  0 ]
    //     [ 0          0        ...]
    //
    // We only need the diagonal.
    // --------------------------------------------------

    Eigen::VectorXcd lambda =
        nplr.N.diagonal();

    // One H(z) value for every z.
    Eigen::VectorXcd result(z.size());

    // --------------------------------------------------
    // Evaluate the resolvent for every z.
    // --------------------------------------------------

    for (int j = 0; j < z.size(); ++j)
    {
        const std::complex<double> zj = z(j);

        // --------------------------------------------------
        // D = zI - Lambda
        //
        // Since Lambda is diagonal:
        //
        //     D(i,i) = z - lambda_i
        //
        // Therefore:
        //
        //     D^(-1)(i,i) = 1/(z - lambda_i)
        //
        // We never actually construct D^(-1).
        // --------------------------------------------------

        Eigen::VectorXcd DinvB =
            nplr.B.array() /
            (zj - lambda.array());

        Eigen::VectorXcd DinvP =
            nplr.P.col(0).array() /
            (zj - lambda.array());

        // --------------------------------------------------
        // First term:
        //
        //     C D^(-1) B
        //
        // --------------------------------------------------

        std::complex<double> first_term =
            (nplr.C * DinvB)(0);

        // --------------------------------------------------
        // Compute:
        //
        //     Q* D^(-1) P
        //
        // This is a scalar because P and Q are N x 1.
        // --------------------------------------------------

        std::complex<double> q_dinv_p =
            (nplr.Q.col(0).adjoint() * DinvP)(0);

        // --------------------------------------------------
        // Compute:
        //
        //     C D^(-1) P
        // --------------------------------------------------

        std::complex<double> c_dinv_p =
            (nplr.C * DinvP)(0);

        // --------------------------------------------------
        // Compute:
        //
        //     Q* D^(-1) B
        // --------------------------------------------------

        std::complex<double> q_dinv_b =
            (nplr.Q.col(0).adjoint() * DinvB)(0);

        // --------------------------------------------------
        // Woodbury correction:
        //
        //     C D^(-1) P
        //     --------------------  * Q*D^(-1)B
        //     1 + Q*D^(-1)P
        //
        // Therefore:
        //
        // H(z) =
        //
        // C D^(-1) B
        //
        // -
        //
        // (C D^(-1) P)
        // -----------------------
        // (1 + Q*D^(-1)P)
        //
        // * (Q*D^(-1)B)
        // --------------------------------------------------

        std::complex<double> denominator =
            1.0 + q_dinv_p;

        if (std::abs(denominator) < 1e-12) {
            throw std::runtime_error(
                "Woodbury denominator is numerically singular"
            );
        }

        std::complex<double> correction =
            (c_dinv_p / denominator) *
            q_dinv_b;

        result(j) =
            first_term - correction;
    }

    return result;
}

} // namespace ssmforge