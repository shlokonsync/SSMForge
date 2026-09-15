#pragma once

#include <Eigen/Dense>

namespace ssmforge {

/*
 * Evaluate the SSM resolvent / transfer function:
 *
 *     H(z) = C (zI - A)^(-1) B
 *
 * at multiple complex points z.
 *
 * This is a reference implementation.
 * Later we will replace the expensive linear solves
 * with the NPLR + Woodbury + Cauchy formulation.
 */
Eigen::VectorXcd evaluate_resolvent(
    const Eigen::MatrixXcd& A,
    const Eigen::VectorXcd& B,
    const Eigen::RowVectorXcd& C,
    const Eigen::VectorXcd& z
);

} // namespace ssmforge