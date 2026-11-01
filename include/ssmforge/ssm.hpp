#pragma once

#include <Eigen/Dense>

namespace ssmforge {

class SSM {
public:

    SSM(
        const Eigen::MatrixXd& Ad,
        const Eigen::VectorXd& Bd,
        const Eigen::RowVectorXd& C,
        double D
    );

    Eigen::VectorXd forward(
        const Eigen::VectorXd& u
    );

    double step(double u);

    void reset();

//why you adding these things in private cauz you dont want user to play with things like say state we want use only forward set and reset etc
private:

    Eigen::MatrixXd Ad_;
    Eigen::VectorXd Bd_;
    Eigen::RowVectorXd C_;

    double D_;

    Eigen::VectorXd state_;
};

} // namespace ssmforge