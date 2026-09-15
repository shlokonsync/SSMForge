#pragma once

#include <Eigen/Dense>

namespace ssmforge {

/*
 * Compute the SSM convolution kernel:
 *
 *     K[t] = C A^t B
 *
 * for t = 0, 1, ..., L-1.
 *
 * A : state transition matrix
 * B : input vector
 * C : output vector
 * L : number of kernel elements
 */
Eigen::VectorXcd compute_kernel(
    const Eigen::MatrixXcd& A,
    const Eigen::VectorXcd& B,
    const Eigen::RowVectorXcd& C,
    int L
);

} // namespace ssmforge