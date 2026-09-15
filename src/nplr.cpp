#include "ssmforge/nplr.hpp"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <stdexcept>

namespace ssmforge {

NPLR nplr_decompose(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& B,
    const Eigen::RowVectorXd& C
) {
    const int N = static_cast<int>(A.rows());

    // --------------------------------------------------
    // Validate inputs
    // --------------------------------------------------

    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            "A must be square"
        );
    }

    if (B.size() != N) {
        throw std::invalid_argument(
            "B must have the same size as A"
        );
    }

    if (C.size() != N) {
        throw std::invalid_argument(
            "C must have the same size as A"
        );
    }

    // --------------------------------------------------
    // Step 1: Construct the low-rank correction vectors.
    //
    // For HiPPO-LegS:
    //
    //     p[n] = 0.5 * sqrt(2n + 1)
    //
    //          = sqrt(n + 0.5)
    //
    //     q[n] = sqrt(2n + 1)
    //
    // We want:
    //
    //     S = A + p q*
    //
    // so that:
    //
    //     A = S - p q*
    // --------------------------------------------------

    Eigen::VectorXd p_real(N);
    Eigen::VectorXd q_real(N);

    for (int n = 0; n < N; ++n)
    {
        p_real(n) = std::sqrt(n + 0.5);
        q_real(n) = std::sqrt(n + 0.5);
    }

    // --------------------------------------------------
    // Step 2: Construct the normal component.
    //
    //     S = A + p q*
    //
    // Since p and q are real here:
    //
    //     q* = q^T
    // --------------------------------------------------

    Eigen::MatrixXd S =
        A + p_real * q_real.transpose();

    // --------------------------------------------------
    // Step 3: Diagonalize S.
    //
    //     S V = V Lambda
    //
    // Lambda = eigenvalues
    // V      = eigenvectors
    // --------------------------------------------------

    Eigen::ComplexEigenSolver<Eigen::MatrixXd> solver(S);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Failed to diagonalize NPLR normal matrix"
        );
    }

    Eigen::VectorXcd eigenvalues =
        solver.eigenvalues();

    Eigen::MatrixXcd V =
        solver.eigenvectors();

    // --------------------------------------------------
    // Step 4: Store Lambda.
    //
    // We store it as a diagonal matrix because this makes
    // the structure explicit:
    //
    //     Lambda =
    //
    //     [ λ0  0   0 ]
    //     [ 0   λ1  0 ]
    //     [ 0   0   λ2 ]
    // --------------------------------------------------

    NPLR result;

    result.N =
        eigenvalues.asDiagonal();

    // --------------------------------------------------
    // Step 5: Transform P and Q into the eigenbasis.
    //
    // Original:
    //
    //     A = S - p q*
    //
    // and:
    //
    //     S = V Lambda V*
    //
    // Therefore:
    //
    //     A = V Lambda V* - p q*
    //
    // Multiplying by V* on the left and V on the right:
    //
    //     V* A V
    //
    // gives:
    //
    //     A' = Lambda - P Q*
    //
    // where:
    //
    //     P = V* p
    //     Q = V* q
    // --------------------------------------------------

    result.P =
        V.adjoint() *
        p_real.cast<std::complex<double>>();

    result.Q =
        V.adjoint() *
        q_real.cast<std::complex<double>>();

    // --------------------------------------------------
    // Step 6: Transform B.
    //
    // Original:
    //
    //     dx/dt = A x + B u
    //
    // Let:
    //
    //     x = V z
    //
    // Then:
    //
    //     dz/dt = A' z + V* B u
    //
    // Therefore:
    //
    //     B' = V* B
    // --------------------------------------------------

    result.B =
        V.adjoint() *
        B.cast<std::complex<double>>();

    // --------------------------------------------------
    // Step 7: Transform C.
    //
    // Original:
    //
    //     y = C x
    //
    // Since:
    //
    //     x = V z
    //
    // we get:
    //
    //     y = C V z
    //
    // Therefore:
    //
    //     C' = C V
    // --------------------------------------------------

    result.C =
        C.cast<std::complex<double>>() *
        V;

    // --------------------------------------------------
    // Step 8: Store V.
    //
    // Keeping V is useful because it lets us move between
    // the original basis and transformed basis.
    // --------------------------------------------------

    result.V = V;

    return result;
}

} // namespace ssmforge