#include "ssmforge/cuda_kernel.hpp"

#include <cuda_runtime.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ssmforge {

namespace {

// ------------------------------------------------------------
// Matrix power on the host:
//
// result = A^power
//
// Binary exponentiation is used so that the number of matrix
// multiplications grows logarithmically with power.
// ------------------------------------------------------------
Eigen::MatrixXd matrix_power(
    const Eigen::MatrixXd& A,
    int power
)
{
    const int N =
        static_cast<int>(A.rows());

    Eigen::MatrixXd result =
        Eigen::MatrixXd::Identity(N, N);

    Eigen::MatrixXd base = A;

    int exponent = power;
 // to cal say A^8, instead of doing a*a*a..., we do A2,A4 , A8 then store this A8
 // this takes log(n) time than O(N)
    while (exponent > 0)
    {
        if (exponent & 1)
        {
            result = result * base;
        }

        exponent >>= 1;

        if (exponent > 0)
        {
            base = base * base;
        }
    }

    return result;
}

// ------------------------------------------------------------
// CUDA kernel
//
// Each thread handles output positions:
//
// thread_id,
// thread_id + stride,
// thread_id + 2 * stride,
// ...
//
// The state advances using:
//
// P = A^stride
//
// Therefore:
//
// x(t + stride) = P*x(t)
// ------------------------------------------------------------
//This function executes on the GPU and is launched from the CPU.
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

    // First half of the workspace belongs to the current state.
    double* current =
        workspace + thread_id * N;

    // Second half belongs to the next state.
    double* next =
        workspace + stride * N + thread_id * N;

    // Initial state:
    //
    // current = A^thread_id * B
    //
    // The initial states were prepared on the CPU.
    for (int i = 0; i < N; ++i)
    {
        current[i] =
            initial_states[thread_id * N + i];
    }

    // Each thread generates several kernel values.
    for (
        int t = thread_id;
        t < L;
        t += stride
    )
    {
        double result = 0.0;

        // K[t] = C * current
        for (int i = 0; i < N; ++i)
        {
            result += C[i] * current[i];
        }

        K[t] = result;

        // No need to calculate a state after the final output.
        if (t + stride >= L)
        {
            break;
        }

        // next = P * current
        for (int i = 0; i < N; ++i)
        {
            double value = 0.0;

            for (int j = 0; j < N; ++j)
            {
                // Eigen stores matrices in column-major order.
                value +=
                    P[i + j * N] * current[j];
            }

            next[i] = value;
        }

        // Swap the two state buffers.
        //
        // This avoids copying next[i] back into current[i].
        double* temporary = current;
        current = next;
        next = temporary;
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
            std::string(message) +
            ": " +
            cudaGetErrorString(error)
        );
    }
}

// ------------------------------------------------------------
// Free a CUDA pointer safely
// ------------------------------------------------------------
void free_cuda_pointer(double* pointer)
{
    if (pointer != nullptr)
    {
        cudaFree(pointer);
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

    // We use at most 256 independent threads.
    //
    // Each thread handles multiple output positions.
    const int stride =
        std::min(L, 256);

    // --------------------------------------------------------
    // CPU preparation
    // --------------------------------------------------------

    const Eigen::MatrixXd P =
        matrix_power(Ad, stride);

    // Eigen stores this matrix in column-major order.
    //
    // Column t contains A^t * B.
    Eigen::MatrixXd initial_states(N, stride);

    Eigen::VectorXd state = Bd;

    for (int t = 0; t < stride; ++t)
    {
        initial_states.col(t) = state;
        state = Ad * state;
    }

    // --------------------------------------------------------
    // Device pointers
    // --------------------------------------------------------

    double* d_P = nullptr;
    double* d_initial_states = nullptr;
    double* d_C = nullptr;
    double* d_K = nullptr;
    double* d_workspace = nullptr;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

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

    const auto total_start =
        std::chrono::high_resolution_clock::now();

    try
    {
        // ----------------------------------------------------
        // Allocate device memory
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // Copy data from host to device
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // Kernel configuration
        // ----------------------------------------------------

        const int threads_per_block = 128;

        const int blocks =
            (stride + threads_per_block - 1) /
            threads_per_block;

        // ----------------------------------------------------
        // Create CUDA events
        // ----------------------------------------------------

        check_cuda_error(
            cudaEventCreate(&start_event),
            "Failed to create CUDA start event"
        );

        check_cuda_error(
            cudaEventCreate(&stop_event),
            "Failed to create CUDA stop event"
        );

        // ----------------------------------------------------
        // Warm-up execution
        //
        // This helps remove first-launch overhead from the
        // measured kernel timing.
        // ----------------------------------------------------

        compute_kernel_cuda_strided<<<
            blocks,
            threads_per_block
        >>>(
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
            "CUDA warm-up kernel launch failed"
        );

        check_cuda_error(
            cudaDeviceSynchronize(),
            "CUDA warm-up execution failed"
        );

        // ----------------------------------------------------
        // Actual timed kernel execution
        // ----------------------------------------------------

        check_cuda_error(
            cudaEventRecord(start_event),
            "Failed to record CUDA start event"
        );

        compute_kernel_cuda_strided<<<
            blocks,
            threads_per_block
        >>>(
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
            cudaEventRecord(stop_event),
            "Failed to record CUDA stop event"
        );

        check_cuda_error(
            cudaEventSynchronize(stop_event),
            "Failed to synchronize CUDA stop event"
        );

        float kernel_time_ms = 0.0f;

        check_cuda_error(
            cudaEventElapsedTime(
                &kernel_time_ms,
                start_event,
                stop_event
            ),
            "Failed to measure CUDA kernel time"
        );

        // ----------------------------------------------------
        // Copy result from device to host
        // ----------------------------------------------------

        check_cuda_error(
            cudaMemcpy(
                K.data(),
                d_K,
                kernel_bytes,
                cudaMemcpyDeviceToHost
            ),
            "Failed to copy K from GPU"
        );

        const auto total_end =
            std::chrono::high_resolution_clock::now();

        const double total_time_ms =
            std::chrono::duration<double, std::milli>(
                total_end - total_start
            ).count();

        std::cout << std::scientific
                  << std::setprecision(12);

        std::cout << "Actual CUDA kernel time: "
                  << kernel_time_ms
                  << " ms\n";

        std::cout << "Total CUDA function time: "
                  << total_time_ms
                  << " ms\n";
    }
    catch (...)
    {
        if (start_event != nullptr)
        {
            cudaEventDestroy(start_event);
        }

        if (stop_event != nullptr)
        {
            cudaEventDestroy(stop_event);
        }

        free_cuda_pointer(d_P);
        free_cuda_pointer(d_initial_states);
        free_cuda_pointer(d_C);
        free_cuda_pointer(d_K);
        free_cuda_pointer(d_workspace);

        throw;
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    check_cuda_error(
        cudaEventDestroy(start_event),
        "Failed to destroy CUDA start event"
    );

    check_cuda_error(
        cudaEventDestroy(stop_event),
        "Failed to destroy CUDA stop event"
    );

    free_cuda_pointer(d_P);
    free_cuda_pointer(d_initial_states);
    free_cuda_pointer(d_C);
    free_cuda_pointer(d_K);
    free_cuda_pointer(d_workspace);

    return K;
}

} // namespace ssmforge