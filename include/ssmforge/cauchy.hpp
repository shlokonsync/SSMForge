#pragma once

#include <Eigen/Dense>

namespace ssmforge {

// Computes the Cauchy matrix-vector operation:
//
//     y_j = sum_i v_i / (z_j - lambda_i)
//
// lambda : poles/eigenvalues
// z      : evaluation points
// v      : weights
//
// All quantities are complex because the NPLR representation
// naturally uses complex eigenvalues.
Eigen::VectorXcd cauchy(
    const Eigen::VectorXcd& lambda,
    const Eigen::VectorXcd& z,
    const Eigen::VectorXcd& v
);

} // namespace ssmforge
