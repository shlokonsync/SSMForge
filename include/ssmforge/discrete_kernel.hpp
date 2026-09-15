#pragma once

#include <Eigen/Dense>
#include "ssmforge/discretize.hpp"

namespace ssmforge {

/*
 * Compute discrete-time SSM convolution kernel:
 *
 *     K[t] = C * (Ad)^t * Bd
 *
 * for t = 0, 1, ..., L-1.
 *
 * Ad : discrete state transition matrix (N x N)
 * Bd : discrete input vector (N x 1)
 * C  : output row vector (1 x N)
 * L  : number of kernel elements
 */
Eigen::VectorXd compute_discrete_kernel(
    const Eigen::MatrixXd& Ad,
    const Eigen::VectorXd& Bd,
    const Eigen::RowVectorXd& C,
    int L
);

} // namespace ssmforge
