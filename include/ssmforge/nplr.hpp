#pragma once

#include <Eigen/Dense>

namespace ssmforge {

/*
 * Normal Plus Low-Rank (NPLR) representation.
 *
 * We represent the state matrix in a new basis as:
 *
 *     A' = Lambda - P Q*
 *
 * where:
 *
 *     Lambda = diagonal normal component
 *     P,Q    = low-rank correction factors
 *
 * The complete transformed SSM is:
 *
 *     dz/dt = A' z + B' u
 *     y     = C' z
 */
struct NPLR
{
    // Diagonal normal component.
    // This is Lambda.
    Eigen::MatrixXcd N;

    // Low-rank correction factors.
    Eigen::MatrixXcd P;
    Eigen::MatrixXcd Q;

    // Input vector in the transformed basis.
    //
    //     B' = V* B
    //
    Eigen::VectorXcd B;

    // Output vector in the transformed basis.
    //
    //     C' = C V
    //
    Eigen::RowVectorXcd C;

    // Change-of-basis matrix.
    //
    //     x = V z
    //
    Eigen::MatrixXcd V;
};


/*
 * Convert an SSM into its NPLR representation.
 *
 * Original system:
 *
 *     dx/dt = A x + B u
 *     y     = C x
 *
 * Transformed system:
 *
 *     dz/dt = (Lambda - P Q*) z + B' u
 *     y     = C' z
 */
NPLR nplr_decompose(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& B,
    const Eigen::RowVectorXd& C
);

} // namespace ssmforge