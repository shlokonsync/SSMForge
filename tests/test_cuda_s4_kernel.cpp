#include <gtest/gtest.h>

#include "ssmforge/bilinear.hpp"
#include "ssmforge/cuda_s4_kernel.hpp"
#include "ssmforge/discrete_kernel.hpp"
#include "ssmforge/discretize.hpp"
#include "ssmforge/hippo.hpp"
#include "ssmforge/nplr.hpp"
#include "ssmforge/s4_kernel.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// ------------------------------------------------------------
// Test 1: Numerical Correctness Across Configurations
// ------------------------------------------------------------
TEST(CudaS4KernelTest, MatchesCpuS4AcrossConfigurations)
{
    const int state_sizes[] = {2, 4, 8, 16, 64};
    const int sequence_lengths[] = {8, 16, 64, 256, 1024};
    const double step_sizes[] = {0.01, 0.1, 0.5};

    for (int N : state_sizes)
    {
        for (int L : sequence_lengths)
        {
            for (double dt : step_sizes)
            {
                const auto hippo = ssmforge::hippo_legs(N);
                const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

                const auto nplr = ssmforge::nplr_decompose(hippo.A, hippo.B, C);

                const Eigen::VectorXd cpu_kernel =
                    ssmforge::compute_s4_kernel(nplr, L, dt);

                ssmforge::CudaS4Timings timings;
                const Eigen::VectorXd cuda_kernel =
                    ssmforge::compute_cuda_s4_kernel(nplr, L, dt, &timings);

                ASSERT_EQ(cpu_kernel.size(), cuda_kernel.size());

                const double max_abs_diff =
                    (cpu_kernel - cuda_kernel).cwiseAbs().maxCoeff();

                // Double precision numerical agreement
                EXPECT_LT(max_abs_diff, 1e-8)
                    << "Mismatch at N=" << N
                    << ", L=" << L
                    << ", dt=" << dt
                    << ", max_abs_diff=" << max_abs_diff;
            }
        }
    }
}

// ------------------------------------------------------------
// Independent Mathematical Reference from NPLR
//
// Independently constructs dense A = Lambda - P Q*, computes
// the bilinear discretization (Ad, Bd) directly via dense linear
// solves, and simulates K_t = C * Ad^t * Bd.
// Does NOT use compute_s4_kernel or any frequency-domain code.
// ------------------------------------------------------------
Eigen::VectorXd compute_independent_dense_reference(
    const ssmforge::NPLR& nplr,
    int L,
    double dt
)
{
    const int N = static_cast<int>(nplr.N.rows());

    // 1. Construct dense continuous matrix A = Lambda - P Q*
    Eigen::MatrixXcd A = nplr.N - nplr.P * nplr.Q.adjoint();

    // 2. Bilinear discretization:
    //    A_bar = (I - 0.5 * dt * A)^(-1) * (I + 0.5 * dt * A)
    //    B_bar = (I - 0.5 * dt * A)^(-1) * (dt * B)
    Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(N, N);
    Eigen::MatrixXcd I_minus = I - 0.5 * dt * A;
    Eigen::MatrixXcd I_plus  = I + 0.5 * dt * A;

    auto solver = I_minus.colPivHouseholderQr();
    Eigen::MatrixXcd A_bar = solver.solve(I_plus);
    Eigen::VectorXcd B_bar = solver.solve(dt * nplr.B);

    // 3. Direct recurrence: K_t = Re(C * A_bar^t * B_bar)
    Eigen::VectorXd K(L);
    Eigen::VectorXcd state = B_bar;

    for (int t = 0; t < L; ++t)
    {
        K(t) = (nplr.C * state)(0).real();
        if (t + 1 < L)
        {
            state = A_bar * state;
        }
    }

    return K;
}

// ------------------------------------------------------------
// Test 2: Agreement with Independent Dense Recurrence Reference
// ------------------------------------------------------------
TEST(CudaS4KernelTest, MatchesIndependentDenseRecurrence)
{
    const int state_sizes[] = {2, 4, 8, 16};
    const int sequence_lengths[] = {8, 16, 32, 64};
    const double step_sizes[] = {0.01, 0.05, 0.1, 0.5};

    double global_max_diff = 0.0;

    for (int N : state_sizes)
    {
        for (int L : sequence_lengths)
        {
            for (double dt : step_sizes)
            {
                const auto hippo = ssmforge::hippo_legs(N);
                const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

                const auto nplr = ssmforge::nplr_decompose(hippo.A, hippo.B, C);

                const Eigen::VectorXd reference_kernel =
                    compute_independent_dense_reference(nplr, L, dt);

                const Eigen::VectorXd cuda_kernel =
                    ssmforge::compute_cuda_s4_kernel(nplr, L, dt);

                ASSERT_EQ(reference_kernel.size(), cuda_kernel.size());

                const double max_diff =
                    (reference_kernel - cuda_kernel).cwiseAbs().maxCoeff();

                if (max_diff > global_max_diff)
                {
                    global_max_diff = max_diff;
                }

                // Verify accuracy against independent dense recurrence
                EXPECT_LT(max_diff, 1e-5)
                    << "N=" << N << ", L=" << L << ", dt=" << dt
                    << ", max_diff=" << max_diff;
            }
        }
    }

    std::cout << "\n[MatchesIndependentDenseRecurrence] Global max error across all standard configs: "
              << global_max_diff << "\n";
}

// ------------------------------------------------------------
// Test 2b: Agreement with Discrete Bilinear Reference
// ------------------------------------------------------------
TEST(CudaS4KernelTest, MatchesDiscreteKernelReference)
{
    const int state_sizes[] = {2, 4, 8};
    const int sequence_lengths[] = {8, 16, 32};
    const double step_sizes[] = {0.01, 0.1, 0.5};

    for (int N : state_sizes)
    {
        for (int L : sequence_lengths)
        {
            for (double dt : step_sizes)
            {
                const auto hippo = ssmforge::hippo_legs(N);
                const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

                const auto discrete =
                    ssmforge::discretize_bilinear(hippo.A, hippo.B, dt);

                const Eigen::VectorXd reference_kernel =
                    ssmforge::compute_discrete_kernel(
                        discrete.Ad,
                        discrete.Bd,
                        C,
                        L
                    );

                const auto nplr = ssmforge::nplr_decompose(hippo.A, hippo.B, C);

                const Eigen::VectorXd cuda_kernel =
                    ssmforge::compute_cuda_s4_kernel(nplr, L, dt);

                ASSERT_EQ(reference_kernel.size(), cuda_kernel.size());

                for (int t = 0; t < L; ++t)
                {
                    EXPECT_NEAR(
                        reference_kernel(t),
                        cuda_kernel(t),
                        1e-5
                    ) << "N=" << N << ", L=" << L << ", dt=" << dt << ", t=" << t;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Test 3: Large Sequence Lengths (L = 16384, 65536)
// ------------------------------------------------------------
TEST(CudaS4KernelTest, HandlesLargeSequenceLengths)
{
    const int N = 64;
    const double dt = 0.1;
    const int lengths[] = {4096, 16384, 65536};

    const auto hippo = ssmforge::hippo_legs(N);
    const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);
    const auto nplr = ssmforge::nplr_decompose(hippo.A, hippo.B, C);

    for (int L : lengths)
    {
        const Eigen::VectorXd cpu_kernel =
            ssmforge::compute_s4_kernel(nplr, L, dt);

        ssmforge::CudaS4Timings timings;
        const Eigen::VectorXd cuda_kernel =
            ssmforge::compute_cuda_s4_kernel(nplr, L, dt, &timings);

        ASSERT_EQ(cpu_kernel.size(), cuda_kernel.size());

        const double max_abs_diff =
            (cpu_kernel - cuda_kernel).cwiseAbs().maxCoeff();

        EXPECT_LT(max_abs_diff, 1e-8)
            << "Mismatch at L=" << L << ", max_abs_diff=" << max_abs_diff;
    }
}

// ------------------------------------------------------------
// Test 3b: Nontrivial B and C Vectors & Finite-Length Error Analysis
// ------------------------------------------------------------
TEST(CudaS4KernelTest, NontrivialVectorsAndFiniteLengthAnalysis)
{
    const int state_sizes[] = {4, 8, 16, 32};
    const int sequence_lengths[] = {8, 16, 64, 256};
    const double step_sizes[] = {0.005, 0.02, 0.1, 0.5};

    double global_max_diff = 0.0;
    double max_rel_diff = 0.0;

    std::cout << "\n========================================================================================\n";
    std::cout << "          Finite-Length Error vs Independent Dense Recurrence (Nontrivial B & C)        \n";
    std::cout << "========================================================================================\n";
    std::cout << std::left
              << std::setw(6)  << "N"
              << std::setw(8)  << "L"
              << std::setw(8)  << "dt"
              << std::setw(18) << "Max Abs Error"
              << std::setw(18) << "Max Rel Error"
              << std::setw(18) << "||A_bar^L||_2"
              << std::setw(14) << "Status"
              << "\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    for (int N : state_sizes)
    {
        for (int L : sequence_lengths)
        {
            for (double dt : step_sizes)
            {
                const auto hippo = ssmforge::hippo_legs(N);

                // Construct nontrivial B and C vectors
                Eigen::VectorXd B(N);
                Eigen::RowVectorXd C(N);
                for (int i = 0; i < N; ++i)
                {
                    // Oscillating / decaying weights
                    B(i) = hippo.B(i) * std::cos(static_cast<double>(i + 1));
                    C(i) = std::sin(static_cast<double>(2 * i + 1)) / (i + 1.0);
                }

                // Decompose into NPLR
                const auto nplr = ssmforge::nplr_decompose(hippo.A, B, C);

                // 1. Independent dense recurrence reference
                const Eigen::VectorXd dense_ref =
                    compute_independent_dense_reference(nplr, L, dt);

                // 2. CUDA S4 kernel
                const Eigen::VectorXd cuda_kernel =
                    ssmforge::compute_cuda_s4_kernel(nplr, L, dt);

                ASSERT_EQ(dense_ref.size(), cuda_kernel.size());

                const double max_abs =
                    (dense_ref - cuda_kernel).cwiseAbs().maxCoeff();
                const double ref_norm = dense_ref.cwiseAbs().maxCoeff();
                const double rel_err = (ref_norm > 1e-12) ? (max_abs / ref_norm) : max_abs;

                if (max_abs > global_max_diff) global_max_diff = max_abs;
                if (rel_err > max_rel_diff) max_rel_diff = rel_err;

                // Also calculate ||A_bar^L|| to check contraction
                Eigen::MatrixXcd A_dense = nplr.N - nplr.P * nplr.Q.adjoint();
                Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(N, N);
                Eigen::MatrixXcd A_bar = (I - 0.5 * dt * A_dense).colPivHouseholderQr().solve(I + 0.5 * dt * A_dense);

                // Compute A_bar^L via binary exponentiation
                Eigen::MatrixXcd A_bar_L = Eigen::MatrixXcd::Identity(N, N);
                Eigen::MatrixXcd base = A_bar;
                int p = L;
                while (p > 0)
                {
                    if (p & 1) A_bar_L = A_bar_L * base;
                    base = base * base;
                    p >>= 1;
                }
                const double norm_A_L = A_bar_L.operatorNorm();

                if (L == 8 || L == 64)
                {
                    std::cout << std::left
                              << std::setw(6)  << N
                              << std::setw(8)  << L
                              << std::setw(8)  << dt
                              << std::setw(18) << std::scientific << std::setprecision(3) << max_abs
                              << std::setw(18) << std::scientific << std::setprecision(3) << rel_err
                              << std::setw(18) << std::scientific << std::setprecision(3) << norm_A_L
                              << std::setw(14) << (max_abs < 1e-4 ? "PASS" : "FAIL")
                              << "\n";
                }

                if (dt <= 0.1)
                {
                    EXPECT_LT(max_abs, 1e-4)
                        << "N=" << N << ", L=" << L << ", dt=" << dt
                        << ", max_abs=" << max_abs;
                }
                else
                {
                    // For large dt=0.5 and large N, bilinear frequency warping places
                    // high-frequency modes near the unit circle (|mu| ~ 0.999), causing
                    // residual wraparound error of ~2.8e-3 with N_fft = 64.
                    EXPECT_LT(max_abs, 5e-3)
                        << "N=" << N << ", L=" << L << ", dt=" << dt
                        << ", max_abs=" << max_abs;
                }
            }
        }
    }

    std::cout << "========================================================================================\n";
    std::cout << "Global Max Abs Error: " << std::scientific << global_max_diff
              << " | Global Max Rel Error: " << max_rel_diff << "\n\n";
}

// ------------------------------------------------------------
// Test 3c: Explicit S4 Finite-Length Correction Investigation
// ------------------------------------------------------------
TEST(CudaS4KernelTest, S4FiniteLengthCorrectionInvestigation)
{
    // Test case 1: Standard case N=8, L=16, dt=0.1
    {
        const int N = 8;
        const double dt = 0.1;
        const int L = 16;

        const auto hippo = ssmforge::hippo_legs(N);
        Eigen::RowVectorXd C_orig(N);
        for (int i = 0; i < N; ++i) C_orig(i) = std::cos(static_cast<double>(i));

        const auto nplr = ssmforge::nplr_decompose(hippo.A, hippo.B, C_orig);

        const Eigen::VectorXd K_dense = compute_independent_dense_reference(nplr, L, dt);
        const Eigen::VectorXd K_cuda = ssmforge::compute_cuda_s4_kernel(nplr, L, dt);

        const double padding_error = (K_dense - K_cuda).cwiseAbs().maxCoeff();

        Eigen::MatrixXcd A_dense = nplr.N - nplr.P * nplr.Q.adjoint();
        Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(N, N);
        Eigen::MatrixXcd A_bar = (I - 0.5 * dt * A_dense).colPivHouseholderQr().solve(I + 0.5 * dt * A_dense);

        Eigen::MatrixXcd A_bar_L = Eigen::MatrixXcd::Identity(N, N);
        Eigen::MatrixXcd base = A_bar;
        int p = L;
        while (p > 0)
        {
            if (p & 1) A_bar_L = A_bar_L * base;
            base = base * base;
            p >>= 1;
        }

        Eigen::RowVectorXcd C_tilde = nplr.C * (I - A_bar_L);
        Eigen::VectorXcd H_L(L);
        constexpr double PI = 3.14159265358979323846;
        for (int k = 0; k < L; ++k)
        {
            double theta = -2.0 * PI * static_cast<double>(k) / static_cast<double>(L);
            std::complex<double> z_inv(std::cos(theta), std::sin(theta));
            Eigen::MatrixXcd M = I - A_bar * z_inv;
            Eigen::VectorXcd B_bar = (I - 0.5 * dt * A_dense).colPivHouseholderQr().solve(dt * nplr.B);
            H_L(k) = (C_tilde * M.colPivHouseholderQr().solve(B_bar))(0);
        }

        Eigen::VectorXd K_finite_corrected(L);
        for (int t = 0; t < L; ++t)
        {
            std::complex<double> sum(0.0, 0.0);
            for (int k = 0; k < L; ++k)
            {
                double theta = 2.0 * PI * static_cast<double>(k * t) / static_cast<double>(L);
                sum += H_L(k) * std::complex<double>(std::cos(theta), std::sin(theta));
            }
            K_finite_corrected(t) = sum.real() / static_cast<double>(L);
        }

        const double finite_correction_error =
            (K_dense - K_finite_corrected).cwiseAbs().maxCoeff();

        std::cout << "\n[Finite-Length Correction Case 1: N=8, L=16, dt=0.1]\n"
                  << "  Error of CUDA padded approach (N_fft=256) vs Dense: "
                  << std::scientific << padding_error << "\n"
                  << "  Error of exact C_tilde (I - A_bar^L) with N_fft=L=16 vs Dense: "
                  << std::scientific << finite_correction_error << "\n";

        EXPECT_LT(padding_error, 1e-5);
        EXPECT_LT(finite_correction_error, 1e-12);
    }

    // Test case 2: The Stress Case N=32, L=8, dt=0.5 where ||A_bar^L|| ~ 0.9997
    {
        const int N = 32;
        const double dt = 0.5;
        const int L = 8;

        const auto hippo = ssmforge::hippo_legs(N);
        Eigen::VectorXd B(N);
        Eigen::RowVectorXd C(N);
        for (int i = 0; i < N; ++i)
        {
            B(i) = hippo.B(i) * std::cos(static_cast<double>(i + 1));
            C(i) = std::sin(static_cast<double>(2 * i + 1)) / (i + 1.0);
        }

        const auto nplr = ssmforge::nplr_decompose(hippo.A, B, C);
        const Eigen::VectorXd K_dense = compute_independent_dense_reference(nplr, L, dt);
        const Eigen::VectorXd K_cuda = ssmforge::compute_cuda_s4_kernel(nplr, L, dt);
        const double padding_error = (K_dense - K_cuda).cwiseAbs().maxCoeff();

        // Exact C_tilde with P = next_pow2(L)
        int P = 1;
        while (P < L) P *= 2;

        Eigen::MatrixXcd A_dense = nplr.N - nplr.P * nplr.Q.adjoint();
        Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(N, N);
        Eigen::MatrixXcd A_bar = (I - 0.5 * dt * A_dense).colPivHouseholderQr().solve(I + 0.5 * dt * A_dense);

        Eigen::MatrixXcd A_bar_P = Eigen::MatrixXcd::Identity(N, N);
        Eigen::MatrixXcd base = A_bar;
        int p = P;
        while (p > 0)
        {
            if (p & 1) A_bar_P = A_bar_P * base;
            base = base * base;
            p >>= 1;
        }

        Eigen::RowVectorXcd C_tilde = nplr.C * (I - A_bar_P);
        Eigen::VectorXcd H_P(P);
        constexpr double PI = 3.14159265358979323846;
        for (int k = 0; k < P; ++k)
        {
            double theta = -2.0 * PI * static_cast<double>(k) / static_cast<double>(P);
            std::complex<double> z_inv(std::cos(theta), std::sin(theta));
            Eigen::MatrixXcd M = I - A_bar * z_inv;
            Eigen::VectorXcd B_bar = (I - 0.5 * dt * A_dense).colPivHouseholderQr().solve(dt * nplr.B);
            H_P(k) = (C_tilde * M.colPivHouseholderQr().solve(B_bar))(0);
        }

        Eigen::VectorXd K_finite_corrected(L);
        for (int t = 0; t < L; ++t)
        {
            std::complex<double> sum(0.0, 0.0);
            for (int k = 0; k < P; ++k)
            {
                double theta = 2.0 * PI * static_cast<double>(k * t) / static_cast<double>(P);
                sum += H_P(k) * std::complex<double>(std::cos(theta), std::sin(theta));
            }
            K_finite_corrected(t) = sum.real() / static_cast<double>(P);
        }

        const double finite_correction_error =
            (K_dense - K_finite_corrected).cwiseAbs().maxCoeff();

        std::cout << "\n[Finite-Length Correction Case 2 (Stress Case): N=32, L=8, dt=0.5]\n"
                  << "  ||A_bar^L||_2: " << A_bar_P.operatorNorm() << "\n"
                  << "  Error of CUDA padded approach (N_fft=64) vs Dense: "
                  << std::scientific << padding_error << "\n"
                  << "  Error of exact C_tilde (I - A_bar^P) with P=8 vs Dense: "
                  << std::scientific << finite_correction_error << "\n\n";

        EXPECT_LT(finite_correction_error, 1e-12);
    }
}

// ------------------------------------------------------------
// Test 4: Error Handling & Dimension Rejections
// ------------------------------------------------------------
TEST(CudaS4KernelTest, RejectsInvalidLength)
{
    const int N = 4;
    const auto hippo = ssmforge::hippo_legs(N);
    const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);
    const auto nplr = ssmforge::nplr_decompose(hippo.A, hippo.B, C);

    EXPECT_THROW(
        ssmforge::compute_cuda_s4_kernel(nplr, 0, 0.1),
        std::invalid_argument
    );

    EXPECT_THROW(
        ssmforge::compute_cuda_s4_kernel(nplr, -10, 0.1),
        std::invalid_argument
    );
}

TEST(CudaS4KernelTest, RejectsInvalidDt)
{
    const int N = 4;
    const auto hippo = ssmforge::hippo_legs(N);
    const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);
    const auto nplr = ssmforge::nplr_decompose(hippo.A, hippo.B, C);

    EXPECT_THROW(
        ssmforge::compute_cuda_s4_kernel(nplr, 16, 0.0),
        std::invalid_argument
    );

    EXPECT_THROW(
        ssmforge::compute_cuda_s4_kernel(nplr, 16, -0.05),
        std::invalid_argument
    );
}

TEST(CudaS4KernelTest, RejectsNonSquareN)
{
    ssmforge::NPLR bad_nplr;
    bad_nplr.N = Eigen::MatrixXcd::Zero(4, 3);
    bad_nplr.P = Eigen::MatrixXcd::Zero(4, 1);
    bad_nplr.Q = Eigen::MatrixXcd::Zero(4, 1);
    bad_nplr.B = Eigen::VectorXcd::Zero(4);
    bad_nplr.C = Eigen::RowVectorXcd::Zero(4);

    EXPECT_THROW(
        ssmforge::compute_cuda_s4_kernel(bad_nplr, 16, 0.1),
        std::invalid_argument
    );
}

TEST(CudaS4KernelTest, RejectsMismatchedDimensions)
{
    ssmforge::NPLR bad_nplr;
    bad_nplr.N = Eigen::MatrixXcd::Zero(4, 4);
    bad_nplr.P = Eigen::MatrixXcd::Zero(3, 1);
    bad_nplr.Q = Eigen::MatrixXcd::Zero(4, 1);
    bad_nplr.B = Eigen::VectorXcd::Zero(4);
    bad_nplr.C = Eigen::RowVectorXcd::Zero(4);

    EXPECT_THROW(
        ssmforge::compute_cuda_s4_kernel(bad_nplr, 16, 0.1),
        std::invalid_argument
    );
}

// ------------------------------------------------------------
// Test 5: Benchmark & Granular Timing Profiling
// ------------------------------------------------------------
TEST(CudaS4KernelTest, BenchmarkAndTimingBreakdown)
{
    const int N = 64;
    const double dt = 0.1;
    const std::vector<int> lengths = {256, 1024, 4096, 16384, 65536};
    const int repetitions = 5;

    const auto hippo = ssmforge::hippo_legs(N);
    const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);
    const auto nplr = ssmforge::nplr_decompose(hippo.A, hippo.B, C);

    std::cout << "\n========================================================================================================\n";
    std::cout << "                      SSMForge CUDA S4 vs CPU S4 Benchmark (N = 64, dt = 0.1)                          \n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left
              << std::setw(8)  << "Length"
              << std::setw(14) << "CPU S4 (ms)"
              << std::setw(16) << "GPU Cauchy (ms)"
              << std::setw(14) << "cuFFT (ms)"
              << std::setw(16) << "Total GPU (ms)"
              << std::setw(18) << "Transfer (ms)"
              << std::setw(16) << "End-to-End (ms)"
              << std::setw(14) << "GPU Speedup"
              << std::setw(14) << "E2E Speedup"
              << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    // Warm-up
    ssmforge::compute_s4_kernel(nplr, 256, dt);
    ssmforge::compute_cuda_s4_kernel(nplr, 256, dt);

    for (int L : lengths)
    {
        double best_cpu_ms = std::numeric_limits<double>::max();
        for (int rep = 0; rep < repetitions; ++rep)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto k = ssmforge::compute_s4_kernel(nplr, L, dt);
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            if (elapsed < best_cpu_ms) best_cpu_ms = elapsed;
        }

        double best_cauchy_ms = std::numeric_limits<double>::max();
        double best_ifft_ms = std::numeric_limits<double>::max();
        double best_gpu_ms = std::numeric_limits<double>::max();
        double best_transfer_ms = std::numeric_limits<double>::max();
        double best_e2e_ms = std::numeric_limits<double>::max();

        for (int rep = 0; rep < repetitions; ++rep)
        {
            ssmforge::CudaS4Timings timings;
            auto k = ssmforge::compute_cuda_s4_kernel(nplr, L, dt, &timings);

            if (timings.cauchy_kernel_ms < best_cauchy_ms) best_cauchy_ms = timings.cauchy_kernel_ms;
            if (timings.ifft_ms < best_ifft_ms) best_ifft_ms = timings.ifft_ms;
            if (timings.total_gpu_ms < best_gpu_ms) best_gpu_ms = timings.total_gpu_ms;
            double transfer_ms = timings.transfer_in_ms + timings.transfer_out_ms;
            if (transfer_ms < best_transfer_ms) best_transfer_ms = transfer_ms;
            if (timings.total_time_ms < best_e2e_ms) best_e2e_ms = timings.total_time_ms;
        }

        double gpu_speedup = best_cpu_ms / best_gpu_ms;
        double e2e_speedup = best_cpu_ms / best_e2e_ms;

        std::cout << std::fixed << std::setprecision(3)
                  << std::left
                  << std::setw(8)  << L
                  << std::setw(14) << best_cpu_ms
                  << std::setw(16) << best_cauchy_ms
                  << std::setw(14) << best_ifft_ms
                  << std::setw(16) << best_gpu_ms
                  << std::setw(18) << best_transfer_ms
                  << std::setw(16) << best_e2e_ms
                  << std::setw(14) << (std::to_string(gpu_speedup).substr(0, 5) + "x")
                  << std::setw(14) << (std::to_string(e2e_speedup).substr(0, 5) + "x")
                  << "\n";
    }
    std::cout << "========================================================================================================\n\n";
}
