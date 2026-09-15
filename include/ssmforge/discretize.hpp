#pragma once

#include <Eigen/Dense>

namespace ssmforge {

struct DiscreteSSM {
    Eigen::MatrixXd Ad;
    Eigen::VectorXd Bd;
};

/**
 * @brief Convert continuous-time state-space system (A, B) to discrete-time (Ad, Bd) using Zero-Order Hold (ZOH).
 * 
 * @param A Continuous-time system matrix (n x n)
 * @param B Continuous-time input vector/matrix (n x 1)
 * @param dt Sampling time step (seconds)
 * @return DiscreteSSM Struct containing discrete-time matrices Ad (n x n) and Bd (n x 1)
 */
DiscreteSSM discretize_zoh(const Eigen::MatrixXd& A, const Eigen::VectorXd& B, double dt);

/**
 * @brief Convert continuous-time state-space system (A, B) to discrete-time (Ad, Bd) using Bilinear (Tustin) transform.
 * 
 * Ad = (I - dt/2 * A)^(-1) * (I + dt/2 * A)
 * Bd = (I - dt/2 * A)^(-1) * (dt * B)
 * 
 * @param A Continuous-time system matrix (n x n)
 * @param B Continuous-time input vector/matrix (n x 1)
 * @param dt Sampling time step (seconds)
 * @return DiscreteSSM Struct containing discrete-time matrices Ad (n x n) and Bd (n x 1)
 */
DiscreteSSM discretize_bilinear(const Eigen::MatrixXd& A, const Eigen::VectorXd& B, double dt);

} // namespace ssmforge
