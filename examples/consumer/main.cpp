#include <ssmforge/ssmforge.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>

int main()
{
    std::cout << "====================================================\n";
    std::cout << "        SSMForge External Consumer Application      \n";
    std::cout << "====================================================\n\n";

    const int N = 8;
    const int L = 16;
    const double dt = 0.1;

    // 1. Generate HiPPO-LegS matrix representation
    std::cout << "1. Constructing HiPPO-LegS representation (N = " << N << ")...\n";
    const auto hippo = ssmforge::hippo_legs(N);
    std::cout << "   Continuous matrix A: " << hippo.A.rows() << "x" << hippo.A.cols()
              << ", input vector B: " << hippo.B.size() << "\n\n";

    // 2. Continuous-to-discrete conversion using bilinear (Tustin) transform
    std::cout << "2. Discretizing via Bilinear transform (dt = " << dt << ")...\n";
    const Eigen::VectorXd B = Eigen::VectorXd::Ones(N);
    const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);
    const auto discrete = ssmforge::discretize_bilinear(hippo.A, B, dt);
    std::cout << "   Discrete matrix Ad: " << discrete.Ad.rows() << "x" << discrete.Ad.cols()
              << ", Bd: " << discrete.Bd.size() << "\n\n";

    // 3. Compute direct discrete recurrence kernel
    std::cout << "3. Computing discrete recurrence reference kernel...\n";
    const auto k_ref = ssmforge::compute_discrete_kernel(discrete.Ad, discrete.Bd, C, L);
    std::cout << "   Reference kernel K[0..3]: ";
    for (int t = 0; t < 4; ++t)
    {
        std::cout << std::fixed << std::setprecision(5) << k_ref(t) << " ";
    }
    std::cout << "...\n\n";

    // 4. Decompose system into Normal Plus Low-Rank (NPLR) form
    std::cout << "4. Performing NPLR decomposition...\n";
    const auto nplr = ssmforge::nplr_decompose(hippo.A, B, C);
    std::cout << "   Normal Lambda diagonal size: " << nplr.N.rows()
              << ", rank-1 factors P: " << nplr.P.rows() << "x" << nplr.P.cols()
              << ", Q: " << nplr.Q.rows() << "x" << nplr.Q.cols() << "\n\n";

    // 5. Compute S4 convolution kernel on CPU via resolvent evaluation & FFT
    std::cout << "5. Computing S4 convolution kernel on CPU...\n";
    const auto k_cpu = ssmforge::compute_s4_kernel(nplr, L, dt);
    std::cout << "   CPU S4 kernel K[0..3]:     ";
    for (int t = 0; t < 4; ++t)
    {
        std::cout << std::fixed << std::setprecision(5) << k_cpu(t) << " ";
    }
    std::cout << "...\n\n";

    // 6. Compute structured S4 convolution kernel on GPU using CUDA
    std::cout << "6. Computing Structured S4 convolution kernel on GPU (CUDA)...\n";
    ssmforge::CudaS4Timings timings;
    const auto k_gpu = ssmforge::compute_cuda_s4_kernel(nplr, L, dt, &timings);
    std::cout << "   CUDA S4 kernel K[0..3]:    ";
    for (int t = 0; t < 4; ++t)
    {
        std::cout << std::fixed << std::setprecision(5) << k_gpu(t) << " ";
    }
    std::cout << "...\n";
    std::cout << "   GPU Timing: Cauchy=" << timings.cauchy_kernel_ms << "ms, cuFFT="
              << timings.ifft_ms << "ms, Total GPU=" << timings.total_gpu_ms << "ms\n\n";

    // 7. Numerical Verification
    const double max_diff_cpu_gpu = (k_cpu - k_gpu).cwiseAbs().maxCoeff();
    const double max_diff_gpu_ref = (k_ref - k_gpu).cwiseAbs().maxCoeff();

    std::cout << "7. Numerical Validation:\n";
    std::cout << "   Max absolute difference (CPU vs GPU S4): " << std::scientific << max_diff_cpu_gpu << "\n";
    std::cout << "   Max absolute difference (GPU vs Direct): " << std::scientific << max_diff_gpu_ref << "\n\n";

    if (max_diff_cpu_gpu < 1e-8 && max_diff_gpu_ref < 1e-5)
    {
        std::cout << ">>> VERIFICATION PASSED: SSMForge library successfully consumed and validated! <<<\n";
        return 0;
    }
    else
    {
        std::cerr << ">>> VERIFICATION FAILED: Numerical error exceeds tolerance! <<<\n";
        return 1;
    }
}

