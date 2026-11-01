#pragma once

#include <Eigen/Dense>

namespace ssmforge {

Eigen::VectorXd compute_cuda_kernel(
    const Eigen::MatrixXd& Ad,
    const Eigen::VectorXd& Bd,
    const Eigen::RowVectorXd& Cd,
    int L
);

}