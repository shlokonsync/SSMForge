#include "ssmforge/cauchy.hpp"

#include <stdexcept>

namespace ssmforge {

Eigen::VectorXcd cauchy(
    const Eigen::VectorXcd& lambda,
    const Eigen::VectorXcd& z,
    const Eigen::VectorXcd& v
) {
    const int N = static_cast<int>(lambda.size());
    const int M = static_cast<int>(z.size());

    // All three vectors must have the same number of poles/weights.
    if (v.size() != N) {
        throw std::invalid_argument(
            "lambda and v must have the same size"
        );
    }

    Eigen::VectorXcd result(M);

    /*
     * We want:
     *
     *     result[j] =
     *         sum_i v[i] / (z[j] - lambda[i])
     *
     * For every evaluation point z[j].
     *
     * This is the direct O(MN) implementation.
     *
     * We intentionally keep this version simple first.
     * Later cauchy_simd.cpp can optimize the same operation.
     */

    for (int j = 0; j < M; ++j)
    {
        std::complex<double> sum = 0.0;

        for (int i = 0; i < N; ++i)
        {
            sum += v(i) / (z(j) - lambda(i));
        }

        result(j) = sum;
    }

    return result;
}

} // namespace ssmforge