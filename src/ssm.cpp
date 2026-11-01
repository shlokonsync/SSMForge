#include "ssmforge/ssm.hpp" //header included cauz it contains those imp variables and functions

#include <stdexcept>

namespace ssmforge {

SSM::SSM(
    const Eigen::MatrixXd& Ad,
    const Eigen::VectorXd& Bd,
    const Eigen::RowVectorXd& C,
    double D
)
    : Ad_(Ad),
      Bd_(Bd),
      C_(C),
      D_(D),
      state_(Eigen::VectorXd::Zero(Ad.rows()))
{
    if (Ad_.rows() != Ad_.cols()) {
        throw std::invalid_argument(
            "Ad must be square"
        );
    }

    if (Bd_.size() != Ad_.rows()) {
        throw std::invalid_argument(
            "Bd has incompatible size"
        );
    }

    if (C_.size() != Ad_.rows()) {
        throw std::invalid_argument(
            "C has incompatible size"
        );
    }
}

Eigen::VectorXd SSM::forward(
    const Eigen::VectorXd& u
)
{
    reset();

    Eigen::VectorXd output(u.size());

    for (Eigen::Index k = 0; k < u.size(); ++k) {
        output(k) = step(u(k));
    }

    return output;
}

double SSM::step(double u)
{
    state_ = Ad_ * state_ + Bd_ * u;

    return (C_ * state_)(0) + D_ * u;
}

void SSM::reset()
{
    state_.setZero();
}

} // namespace ssmforge