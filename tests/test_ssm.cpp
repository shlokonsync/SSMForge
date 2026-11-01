#include <gtest/gtest.h>

#include "ssmforge/ssm.hpp"

TEST(SSMTest, ImpulseResponse)
{
    Eigen::MatrixXd Ad(2, 2);

    Ad << 0.5, 0.0,
          0.0, 0.25;

    Eigen::VectorXd Bd(2);

    Bd << 1.0,
          2.0;

    Eigen::RowVectorXd C(2);

    C << 1.0,
         1.0;

    constexpr double D = 0.0;

    ssmforge::SSM ssm(
        Ad,
        Bd,
        C,
        D
    );

    Eigen::VectorXd u(3);

    u << 1.0,
         0.0,
         0.0;

    Eigen::VectorXd y = ssm.forward(u);

    EXPECT_NEAR(y(0), 3.0, 1e-12);
    EXPECT_NEAR(y(1), 1.0, 1e-12);
    EXPECT_NEAR(y(2), 0.375, 1e-12);
}

//second teST
TEST(SSMTest, ForwardMatchesStep)
{
    Eigen::MatrixXd Ad(2, 2);

    Ad << 0.8, 0.1,
          0.0, 0.5;

    Eigen::VectorXd Bd(2);

    Bd << 1.0,
          0.5;

    Eigen::RowVectorXd C(2);

    C << 1.0,
         -1.0;

    ssmforge::SSM ssm(
        Ad,
        Bd,
        C,
        0.1
    );

    Eigen::VectorXd u(5);

    u << 1.0,
         2.0,
         -1.0,
         0.5,
         3.0;

    Eigen::VectorXd y_forward = ssm.forward(u);

    ssm.reset();

    Eigen::VectorXd y_step(u.size());

    for (Eigen::Index i = 0; i < u.size(); ++i) {
        y_step(i) = ssm.step(u(i));
    }

    for (Eigen::Index i = 0; i < u.size(); ++i) {
        EXPECT_NEAR(
            y_forward(i),
            y_step(i),
            1e-12
        );
    }
}