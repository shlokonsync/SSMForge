#pragma once

#include <Eigen/Dense>

#include "ssmforge/nplr.hpp"

namespace ssmforge {

/*
 * Evaluate the S4 resolvent using the NPLR representation.
 *
 * We want:
 *
 *     H(z) = C (zI - A)^(-1) B
 *
 * where:
 *
 *     A = Lambda - P Q*
 *
 * Using the Woodbury identity, we avoid directly
 * solving an N x N system.
 *
 * This implementation currently assumes a rank-1
 * correction P Q*.
 */
Eigen::VectorXcd evaluate_s4_resolvent(
    const NPLR& nplr,
    const Eigen::VectorXcd& z
);

} // namespace ssmforge