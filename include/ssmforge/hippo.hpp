#pragma once

#include <Eigen/Dense>

namespace ssmforge {

// Result of constructing a HiPPO state-space representation.
//
// A = continuous-time state matrix
// B = continuous-time input vector
struct HiPPO {
    Eigen::MatrixXd A;
    Eigen::VectorXd B;
};

// Construct the HiPPO-LegS representation of order N.
//
// N = number of state variables / memory coefficients.
HiPPO hippo_legs(int N);

} // namespace ssmforge