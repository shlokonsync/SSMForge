#pragma once

#include <Eigen/Dense>

namespace ssmforge {

/*
 * Apply the bilinear (Tustin) transform to continuous-time eigenvalues/poles:
 *
 *     z = (1 + dt/2 * lambda) / (1 - dt/2 * lambda)
 *
 * lambda : vector of continuous-time eigenvalues
 * dt     : sampling time step (must be > 0)
 */
Eigen::VectorXcd bilinear_transform(
    const Eigen::VectorXcd& lambda,
    double dt
);

} // namespace ssmforge
