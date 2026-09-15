#include "ssmforge/discrete_kernel.hpp"
#include <stdexcept>

namespace ssmforge {

Eigen::VectorXd compute_discrete_kernel(
    const Eigen::MatrixXd& Ad,
    const Eigen::VectorXd& Bd,
    const Eigen::RowVectorXd& C,
    int L
) {
    if (L <= 0) {
        throw std::invalid_argument("Kernel length must be positive");
    }

    const int N = static_cast<int>(Ad.rows());
    if (Ad.cols() != N || Bd.size() != N || C.size() != N) {
        throw std::invalid_argument("Dimension mismatch in compute_discrete_kernel");
    }

    Eigen::VectorXd K(L);
    Eigen::VectorXd state = Bd; // for t = 0: state = Ad^0 * Bd = Bd, K[0] = C * Bd

    for (int t = 0; t < L; ++t) {
        K(t) = C * state;
        if (t + 1 < L) {
            state = Ad * state;
        }
    }

    return K;
}

} // namespace ssmforge
