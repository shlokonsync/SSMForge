#include "ssmforge/resolvent.hpp"

#include <stdexcept>

namespace ssmforge {

Eigen::VectorXcd evaluate_resolvent(
    const Eigen::MatrixXcd& A,
    const Eigen::VectorXcd& B,
    const Eigen::RowVectorXcd& C,
    const Eigen::VectorXcd& z
) {
    const int N = static_cast<int>(A.rows());

    if (A.rows() != A.cols()) {
        throw std::invalid_argument("A must be square");
    }

    if (B.size() != N) {
        throw std::invalid_argument("B dimension mismatch");
    }

    if (C.size() != N) {
        throw std::invalid_argument("C dimension mismatch");
    }

    if (z.size() == 0) {
        throw std::invalid_argument("z vector must not be empty");
    }

    const int num_z = static_cast<int>(z.size());
    Eigen::VectorXcd H(num_z);
    const Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(N, N);

    for (int i = 0; i < num_z; ++i) {
        Eigen::MatrixXcd M = z(i) * I - A;
        Eigen::VectorXcd x = M.partialPivLu().solve(B);
        H(i) = (C * x)(0);
    }

    return H;
}

} // namespace ssmforge
