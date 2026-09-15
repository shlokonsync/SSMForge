#include "ssmforge/cuda_kernel.hpp"
#include "ssmforge/discretize.hpp"
#include "ssmforge/discrete_kernel.hpp"
#include "ssmforge/hippo.hpp"
#include "ssmforge/bilinear.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <iomanip>
#include <iostream>

int main()
{
    const int N = 64;
    const int L = 16384;
    const double dt = 0.1;

    std::cout << std::scientific;
    std::cout << std::setprecision(12);

    std::cout << "SSMForge CUDA Kernel Test\n";
    std::cout << "========================\n";
    std::cout << "State size N: " << N << "\n";
    std::cout << "Kernel length L: " << L << "\n";
     std::cout << "Using strided CUDA implementation, stride = "
          << stride << '\n';
    std::cout << "Time step dt: " << dt << "\n\n";

    auto hippo = ssmforge::hippo_legs(N);
    Eigen::MatrixXd A = hippo.A;

    Eigen::VectorXd B =
        Eigen::VectorXd::Ones(N);

    Eigen::RowVectorXd C =
        Eigen::RowVectorXd::Ones(N);

    auto discrete =
        ssmforge::discretize_bilinear(A, B, dt);

    Eigen::VectorXd reference =
        ssmforge::compute_discrete_kernel(
            discrete.Ad,
            discrete.Bd,
            C,
            L
        );

    auto start = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd cuda_kernel =
        ssmforge::compute_cuda_kernel(
            discrete.Ad,
            discrete.Bd,
            C,
            L
        );

    auto end = std::chrono::high_resolution_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    const double max_error =
        (reference - cuda_kernel)
            .cwiseAbs()
            .maxCoeff();

    std::cout << "CUDA execution time: "
              << elapsed_ms
              << " ms\n";

    std::cout << "Maximum absolute error: "
              << max_error
              << "\n";

    std::cout << "First reference value: "
              << reference(0)
              << "\n";

    std::cout << "First CUDA value: "
              << cuda_kernel(0)
              << "\n";

    return 0;
}