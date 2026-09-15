#include "ssmforge/cuda_kernel.hpp"

#include <cuda_runtime.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ssmforge {

namespace {

// ------------------------------------------------------------
// Matrix multiplication on the host:
//
// C = A * B
// ------------------------------------------------------------
Eigen::MatrixXd matrix_power(
    const Eigen::MatrixXd& A,
    int power
)
{
    const int N = static_cast<int>(A.rows());

    Eigen::MatrixXd result =
        Eigen::MatrixXd::Identity(N, N);

    Eigen::MatrixXd base = A;

    int exponent = power;

    while (exponent > 0)
    {
        if (exponent & 1)
        {
            result = result * base;
        }

        base = base * base;
        exponent >>= 1;
    }

    return result;
}

// ------------------------------------------------------------
// CUDA kernel
//
// Each thread handles:
//
// thread_id,
// thread_id + stride,
// thread_id + 2*stride,
// ...
//
// The state advances using P = A^stride.
// ------------------------------------------------------------
__global__ void compute_kernel_cuda_strided(
    const double* P,
    const double* initial_states,
    const double* C,
    double* K,
    double* workspace,
    int N,
    int L,
    int stride
)
{
    const int thread_id =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= stride)
    {
        return;
    }

    // Each thread owns a section of the workspace.
    double* x =
        workspace + thread_id * N;

    double* next_x =
        workspace + stride * N + thread_id * N;

    // Initial state:
    //
    // x = A^thread_id B
    //
    for (int i = 0; i < N; ++i)
    {
        x[i] = initial_states[thread_id * N + i];
    }

    // Compute:
    //
    // K[t] = C*x
    //
    // Then:
    //
    // x = P*x
    //
    // where P = A^stride.
    for (int t = thread_id; t < L; t += stride)
    {
        double result = 0.0;

        for (int i = 0; i < N; ++i)
        {
            result += C[i] * x[i];
        }

        K[t] = result;

        // Do not calculate another state after the
        // final output handled by this thread.
        if (t + stride >= L)
        {
            break;
        }

        // next_x = P*x
        for (int i = 0; i < N; ++i)
        {
            double value = 0.0;

            for (int j = 0; j < N; ++j)
            {
                // Column-major indexing.
                value += P[i + j * N] * x[j];
            }

            next_x[i] = value;
        }

        // Copy next_x back into x.
        for (int i = 0; i < N; ++i)
        {
            x[i] = next_x[i];
        }
    }
}

// ------------------------------------------------------------
// CUDA error helper
// ------------------------------------------------------------
void check_cuda_error(
    cudaError_t error,
    const char* message
)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(message) + ": " +
            cudaGetErrorString(error)
        );
    }
}

} // namespace

Eigen::VectorXd compute_cuda_kernel(
    const Eigen::MatrixXd& Ad,
    const Eigen::VectorXd& Bd,
    const Eigen::RowVectorXd& Cd,
    int L
)
{
    const int N =
        static_cast<int>(Ad.rows());

    if (Ad.rows() != Ad.cols())
    {
        throw std::invalid_argument(
            "Ad must be square"
        );
    }

    if (Bd.size() != N)
    {
        throw std::invalid_argument(
            "Bd must have the same size as Ad"
        );
    }

    if (Cd.size() != N)
    {
        throw std::invalid_argument(
            "Cd must have the same size as Ad"
        );
    }

    if (L <= 0)
    {
        throw std::invalid_argument(
            "Kernel length must be positive"
        );
    }

    Eigen::VectorXd K(L);

    // Number of independent CUDA threads.
    //
    // We do not need one thread per output.
    // Each thread handles multiple outputs.
    const int stride =
        std::min(L, 256);
    
        std::cout << "Using strided CUDA implementation, stride = "
          << stride << '\n';
    // --------------------------------------------------------
    // Compute P = A^stride on the CPU.
    // --------------------------------------------------------
    const Eigen::MatrixXd P =
        matrix_power(Ad, stride);

    // --------------------------------------------------------
    // Compute initial states:
    //
    // B, AB, A^2B, ..., A^(stride-1)B
    // --------------------------------------------------------
    Eigen::MatrixXd initial_states(N, stride);

    Eigen::VectorXd state = Bd;

    for (int t = 0; t < stride; ++t)
    {
        initial_states.col(t) = state;
        state = Ad * state;
    }

    double* d_P = nullptr;
    double* d_initial_states = nullptr;
    double* d_C = nullptr;
    double* d_K = nullptr;
    double* d_workspace = nullptr;

    const std::size_t matrix_bytes =
        static_cast<std::size_t>(N) *
        static_cast<std::size_t>(N) *
        sizeof(double);

    const std::size_t initial_states_bytes =
        static_cast<std::size_t>(N) *
        static_cast<std::size_t>(stride) *
        sizeof(double);

    const std::size_t vector_bytes =
        static_cast<std::size_t>(N) *
        sizeof(double);

    const std::size_t kernel_bytes =
        static_cast<std::size_t>(L) *
        sizeof(double);

    const std::size_t workspace_bytes =
        static_cast<std::size_t>(2) *
        static_cast<std::size_t>(stride) *
        static_cast<std::size_t>(N) *
        sizeof(double);

    try
    {
        check_cuda_error(
            cudaMalloc(
                reinterpret_cast<void**>(&d_P),
                matrix_bytes
            ),
            "Failed to allocate GPU memory for P"
        );

        check_cuda_error(
            cudaMalloc(
                reinterpret_cast<void**>(&d_initial_states),
                initial_states_bytes
            ),
            "Failed to allocate GPU memory for initial states"
        );

        check_cuda_error(
            cudaMalloc(
                reinterpret_cast<void**>(&d_C),
                vector_bytes
            ),
            "Failed to allocate GPU memory for C"
        );

        check_cuda_error(
            cudaMalloc(
                reinterpret_cast<void**>(&d_K),
                kernel_bytes
            ),
            "Failed to allocate GPU memory for K"
        );

        check_cuda_error(
            cudaMalloc(
                reinterpret_cast<void**>(&d_workspace),
                workspace_bytes
            ),
            "Failed to allocate GPU workspace"
        );

        check_cuda_error(
            cudaMemcpy(
                d_P,
                P.data(),
                matrix_bytes,
                cudaMemcpyHostToDevice
            ),
            "Failed to copy P to GPU"
        );

        check_cuda_error(
            cudaMemcpy(
                d_initial_states,
                initial_states.data(),
                initial_states_bytes,
                cudaMemcpyHostToDevice
            ),
            "Failed to copy initial states to GPU"
        );

        check_cuda_error(
            cudaMemcpy(
                d_C,
                Cd.data(),
                vector_bytes,
                cudaMemcpyHostToDevice
            ),
            "Failed to copy C to GPU"
        );

        const int threads_per_block = 128;

        const int blocks =
            (stride + threads_per_block - 1) /
            threads_per_block;

        compute_kernel_cuda_strided<<<blocks, threads_per_block>>>(
            d_P,
            d_initial_states,
            d_C,
            d_K,
            d_workspace,
            N,
            L,
            stride
        );

        check_cuda_error(
            cudaGetLastError(),
            "CUDA kernel launch failed"
        );

        check_cuda_error(
            cudaDeviceSynchronize(),
            "CUDA kernel execution failed"
        );

        check_cuda_error(
            cudaMemcpy(
                K.data(),
                d_K,
                kernel_bytes,
                cudaMemcpyDeviceToHost
            ),
            "Failed to copy K from GPU"
        );
    }
    catch (...)
    {
        cudaFree(d_P);
        cudaFree(d_initial_states);
        cudaFree(d_C);
        cudaFree(d_K);
        cudaFree(d_workspace);

        throw;
    }

    cudaFree(d_P);
    cudaFree(d_initial_states);
    cudaFree(d_C);
    cudaFree(d_K);
    cudaFree(d_workspace);

    return K;
}

} // namespace ssmforge
