#include "ssmforge/discretize.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <stdexcept>

namespace ssmforge {

DiscreteSSM discretize_zoh(const Eigen::MatrixXd& A, const Eigen::VectorXd& B, double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("dt must be positive");
    }
    const Eigen::Index n = A.rows();
    if (A.cols() != n || B.size() != n) {
        throw std::invalid_argument("Dimension mismatch in discretize_zoh");
    }
    const Eigen::Index m = 1; // B is n x 1 vector

    // Construct augmented matrix M = [A, B; 0, 0] * dt of size (n + m) x (n + m)
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n + m, n + m);
    M.block(0, 0, n, n) = A * dt;
    M.block(0, n, n, m) = B;
    M.block(0, n, n, m) *= dt;

    // Matrix exponential: e^M = [Ad, Bd; 0, I]
    Eigen::MatrixXd expM = M.exp();

    DiscreteSSM result;
    result.Ad = expM.block(0, 0, n, n);
    result.Bd = expM.block(0, n, n, m);

    return result;
}

DiscreteSSM discretize_bilinear(const Eigen::MatrixXd& A, const Eigen::VectorXd& B, double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("dt must be positive");
    }
    const Eigen::Index n = A.rows();
    if (A.cols() != n || B.size() != n) {
        throw std::invalid_argument("Dimension mismatch in discretize_bilinear");
    }

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd I_minus = I - 0.5 * dt * A;
    Eigen::MatrixXd I_plus = I + 0.5 * dt * A;

    auto solver = I_minus.colPivHouseholderQr();

    DiscreteSSM result;
    result.Ad = solver.solve(I_plus);
    result.Bd = solver.solve(dt * B);

    return result;
}

} // namespace ssmforge
