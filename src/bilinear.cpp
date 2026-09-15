#include "ssmforge/bilinear.hpp"

#include <stdexcept>

namespace ssmforge {

Eigen::VectorXcd bilinear_transform(
    const Eigen::VectorXcd& lambda,
    double dt
) {
    if (dt <= 0.0) {
        throw std::invalid_argument("dt must be positive");
    }

    const int N = static_cast<int>(lambda.size());
    Eigen::VectorXcd result(N);

    for (int i = 0; i < N; ++i) {
        std::complex<double> lam = lambda(i);
        std::complex<double> num = 1.0 + 0.5 * dt * lam;
        std::complex<double> den = 1.0 - 0.5 * dt * lam;

        if (std::abs(den) < 1e-15) {
            throw std::runtime_error("Denominator is singular in bilinear transform");
        }

        result(i) = num / den;
    }

    return result;
}

} // namespace ssmforge