#pragma once

#include <Eigen/Dense>

namespace ssmforge {

/*
 * Generate L frequency evaluation points (roots of unity) on the unit circle:
 *
 *     w_k = exp(2 * pi * i * k / L)
 *
 * for k = 0, 1, ..., L-1.
 *
 * L : number of frequency points (must be > 0)
 */
Eigen::VectorXcd frequency_points(int L);

} // namespace ssmforge