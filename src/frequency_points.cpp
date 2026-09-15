#include "ssmforge/frequency_points.hpp"

#include <cmath>
#include <stdexcept>

namespace ssmforge {

Eigen::VectorXcd frequency_points(int L) {
    if (L <= 0) {
        throw std::invalid_argument("L must be positive");
    }

    constexpr double PI = 3.14159265358979323846;
    Eigen::VectorXcd points(L);

    for (int k = 0; k < L; ++k) {
        double theta = 2.0 * PI * static_cast<double>(k) / static_cast<double>(L);
        points(k) = std::complex<double>(std::cos(theta), std::sin(theta));
    }

    return points;
}

} // namespace ssmforge
