#pragma once

#include <Eigen/Dense>

#include "ssmforge/nplr.hpp"

namespace ssmforge {

// Compute an S4 convolution kernel using the NPLR representation.
//
// The NPLR representation is:
//
//     A = Lambda - P Q*
// 
// where Lambda is diagonal.
//
// L = desired kernel length
// dt = discretization step
Eigen::VectorXd compute_s4_kernel(
    const NPLR& nplr,
    int L,
    double dt
);

} // namespace ssmforge
