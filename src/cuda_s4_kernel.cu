#include "ssmforge/cuda_s4_kernel.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ssmforge {

namespace {

// ------------------------------------------------------------
// CUDA & cuFFT Error Checking Helpers
// ------------------------------------------------------------

void check_cuda(cudaError_t result, const char* message)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(message) + ": " + cudaGetErrorString(result)
        );
    }
}

void check_cufft(cufftResult result, const char* message)
{
    if (result != CUFFT_SUCCESS)
    {
        throw std::runtime_error(
            std::string(message) + " (cuFFT error code: " + std::to_string(result) + ")"
        );
    }
}

// ------------------------------------------------------------
// RAII Guards for CUDA Resources
// ------------------------------------------------------------

struct DeviceBufferGuard {
    void* ptr = nullptr;

    DeviceBufferGuard() = default;

    explicit DeviceBufferGuard(std::size_t bytes)
    {
        if (bytes > 0)
        {
            check_cuda(cudaMalloc(&ptr, bytes), "Failed to allocate device memory");
        }
    }

    ~DeviceBufferGuard()
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
    }

    DeviceBufferGuard(const DeviceBufferGuard&) = delete;
    DeviceBufferGuard& operator=(const DeviceBufferGuard&) = delete;
};

struct CufftPlanGuard {
    cufftHandle plan = 0;
    bool active = false;

    CufftPlanGuard() = default;

    ~CufftPlanGuard()
    {
        if (active)
        {
            cufftDestroy(plan);
        }
    }

    CufftPlanGuard(const CufftPlanGuard&) = delete;
    CufftPlanGuard& operator=(const CufftPlanGuard&) = delete;
};

struct CudaEventGuard {
    cudaEvent_t evt = nullptr;

    CudaEventGuard()
    {
        check_cuda(cudaEventCreate(&evt), "Failed to create CUDA event");
    }

    ~CudaEventGuard()
    {
        if (evt != nullptr)
        {
            cudaEventDestroy(evt);
        }
    }

    CudaEventGuard(const CudaEventGuard&) = delete;
    CudaEventGuard& operator=(const CudaEventGuard&) = delete;
};

// ------------------------------------------------------------
// Device Complex Arithmetic Inline Primitives
// ------------------------------------------------------------

__device__ __forceinline__ cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cuDoubleComplex csub(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ __forceinline__ cuDoubleComplex cinv(cuDoubleComplex a)
{
    double denom = a.x * a.x + a.y * a.y;
    return make_cuDoubleComplex(a.x / denom, -a.y / denom);
}

__device__ __forceinline__ cuDoubleComplex cdiv(cuDoubleComplex a, cuDoubleComplex b)
{
    double denom = b.x * b.x + b.y * b.y;
    return make_cuDoubleComplex(
        (a.x * b.x + a.y * b.y) / denom,
        (a.y * b.x - a.x * b.y) / denom
    );
}

// ------------------------------------------------------------
// CUDA S4 Cauchy & Woodbury Evaluation Kernel
//
// Each GPU thread evaluates one frequency index k in [0, M-1]:
// 1. Bilinear transformation of frequency point om = exp(-2*pi*i*k/N_fft) -> z_k
// 2. Parallel Cauchy summation for the 4 resolvent terms
// 3. Sherman-Morrison-Woodbury rank-1 correction
// 4. Bilinear factor scaling: 2 / (1 + om)
// 5. Normalization by 1/N_fft for cuFFT inverse FFT
// ------------------------------------------------------------

__global__ void evaluate_s4_cauchy_woodbury_kernel(
    const cuDoubleComplex* __restrict__ w,
    const cuDoubleComplex* __restrict__ v00,
    const cuDoubleComplex* __restrict__ v01,
    const cuDoubleComplex* __restrict__ v10,
    const cuDoubleComplex* __restrict__ v11,
    cuDoubleComplex* __restrict__ k_frequency,
    cuDoubleComplex nyquist_val,
    int N,
    int N_fft,
    int M,
    double inv_N_fft
)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= M)
    {
        return;
    }

    // Special case: Nyquist frequency (k = N_fft / 2 for even N_fft)
    // where 1 + om = 0 and z -> infinity.
    // Closed-form limit from bilinear discretization: H(-1) = 0.5 * dt * (C * B)
    if (k == N_fft / 2)
    {
        k_frequency[k] = make_cuDoubleComplex(nyquist_val.x * inv_N_fft, 0.0);
        return;
    }

    // Evaluation point on unit circle: om = exp(-2*pi*i*k / N_fft)
    constexpr double PI = 3.14159265358979323846;
    const double theta = -2.0 * PI * static_cast<double>(k) / static_cast<double>(N_fft);
    double sin_theta, cos_theta;
    sincos(theta, &sin_theta, &cos_theta);
    const cuDoubleComplex om = make_cuDoubleComplex(cos_theta, sin_theta);

    // Bilinear transform: z = 2 * (1 - om) / (1 + om)
    const cuDoubleComplex denom_z = make_cuDoubleComplex(1.0 + om.x, om.y);
    const cuDoubleComplex num_z = make_cuDoubleComplex(2.0 * (1.0 - om.x), -2.0 * om.y);
    const cuDoubleComplex z = cdiv(num_z, denom_z);

    // Accumulators for Cauchy evaluations:
    // r00 = sum_i v00[i] / (z - w[i])
    // r01 = sum_i v01[i] / (z - w[i])
    // r10 = sum_i v10[i] / (z - w[i])
    // r11 = sum_i v11[i] / (z - w[i])
    cuDoubleComplex r00 = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex r01 = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex r10 = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex r11 = make_cuDoubleComplex(0.0, 0.0);

    for (int i = 0; i < N; ++i)
    {
        const cuDoubleComplex d = csub(z, w[i]);
        // Compute 1 / (z - w[i]) once and reuse across all 4 numerator vectors
        const cuDoubleComplex inv_d = cinv(d);

        r00 = cadd(r00, cmul(v00[i], inv_d));
        r01 = cadd(r01, cmul(v01[i], inv_d));
        r10 = cadd(r10, cmul(v10[i], inv_d));
        r11 = cadd(r11, cmul(v11[i], inv_d));
    }

    // Woodbury rank-1 correction:
    // H(z) = r00 - (r01 * r10) / (1 + r11)
    const cuDoubleComplex woodbury_denom = make_cuDoubleComplex(1.0 + r11.x, r11.y);
    const cuDoubleComplex prod = cmul(r01, r10);
    const cuDoubleComplex correction = cdiv(prod, woodbury_denom);
    const cuDoubleComplex H = csub(r00, correction);

    // Final bilinear transform factor: 2 / (1 + om)
    const cuDoubleComplex factor = cdiv(make_cuDoubleComplex(2.0, 0.0), denom_z);
    cuDoubleComplex K_val = cmul(H, factor);

    // cuFFT normalization: multiply by 1 / N_fft
    K_val.x *= inv_N_fft;
    K_val.y *= inv_N_fft;

    // For k=0 (DC component), imaginary part is mathematically 0
    if (k == 0)
    {
        K_val.y = 0.0;
    }

    k_frequency[k] = K_val;
}

} // namespace

// ------------------------------------------------------------
// Public API Implementation
// ------------------------------------------------------------

Eigen::VectorXd compute_cuda_s4_kernel(
    const NPLR& nplr,
    int L,
    double dt,
    CudaS4Timings* timings
)
{
    const auto total_wall_start = std::chrono::high_resolution_clock::now();

    // --------------------------------------------------------
    // 1. Input Validation
    // --------------------------------------------------------
    if (L <= 0)
    {
        throw std::invalid_argument("Kernel length must be positive");
    }

    if (dt <= 0.0)
    {
        throw std::invalid_argument("dt must be positive");
    }

    const int N = static_cast<int>(nplr.N.rows());

    if (nplr.N.rows() != nplr.N.cols())
    {
        throw std::invalid_argument("N must be square");
    }

    if (nplr.P.rows() != N || nplr.P.cols() != 1)
    {
        throw std::invalid_argument("P must be N x 1 for rank-1 NPLR");
    }

    if (nplr.Q.rows() != N || nplr.Q.cols() != 1)
    {
        throw std::invalid_argument("Q must be N x 1 for rank-1 NPLR");
    }

    if (nplr.B.size() != N)
    {
        throw std::invalid_argument("B must have size N");
    }

    if (nplr.C.size() != N)
    {
        throw std::invalid_argument("C must have size N");
    }

    // --------------------------------------------------------
    // 2. Determine FFT Length (Preventing circular aliasing)
    // --------------------------------------------------------
    int min_points = std::max(2 * L, static_cast<int>(std::ceil(24.0 / dt)));
    int N_fft = 1;
    while (N_fft < min_points)
    {
        N_fft *= 2;
    }

    // Number of frequency points on the half-spectrum [0, N_fft/2]
    const int M = N_fft / 2 + 1;
    const double inv_N_fft = 1.0 / static_cast<double>(N_fft);

    // --------------------------------------------------------
    // 3. Host Preprocessing: Precompute Pole & Numerator Vectors
    // --------------------------------------------------------
    const auto prep_start = std::chrono::high_resolution_clock::now();

    // Packed buffer containing: [w, v00, v01, v10, v11] (5 * N elements)
    std::vector<cuDoubleComplex> h_packed(5 * N);
    cuDoubleComplex* h_w   = h_packed.data();
    cuDoubleComplex* h_v00 = h_packed.data() + N;
    cuDoubleComplex* h_v01 = h_packed.data() + 2 * N;
    cuDoubleComplex* h_v10 = h_packed.data() + 3 * N;
    cuDoubleComplex* h_v11 = h_packed.data() + 4 * N;

    const auto lambda = nplr.N.diagonal();
    const auto& P = nplr.P.col(0);
    const auto& Q = nplr.Q.col(0);
    const auto& B = nplr.B;
    const auto& C = nplr.C;

    for (int i = 0; i < N; ++i)
    {
        std::complex<double> wi    = dt * lambda(i);
        std::complex<double> v00_i = dt * C(i) * B(i);
        std::complex<double> v01_i = dt * C(i) * P(i);
        std::complex<double> v10_i = dt * std::conj(Q(i)) * B(i);
        std::complex<double> v11_i = dt * std::conj(Q(i)) * P(i);

        h_w[i]   = make_cuDoubleComplex(wi.real(), wi.imag());
        h_v00[i] = make_cuDoubleComplex(v00_i.real(), v00_i.imag());
        h_v01[i] = make_cuDoubleComplex(v01_i.real(), v01_i.imag());
        h_v10[i] = make_cuDoubleComplex(v10_i.real(), v10_i.imag());
        h_v11[i] = make_cuDoubleComplex(v11_i.real(), v11_i.imag());
    }

    const std::complex<double> nyq = 0.5 * dt * (C * B)(0);
    const cuDoubleComplex nyquist_val = make_cuDoubleComplex(nyq.real(), nyq.imag());

    const auto prep_end = std::chrono::high_resolution_clock::now();
    const double host_prep_ms = std::chrono::duration<double, std::milli>(
        prep_end - prep_start
    ).count();

    // --------------------------------------------------------
    // 4. Device Memory Allocation (RAII)
    // --------------------------------------------------------
    const std::size_t packed_bytes = 5 * N * sizeof(cuDoubleComplex);
    const std::size_t freq_bytes   = M * sizeof(cuDoubleComplex);
    const std::size_t time_bytes   = N_fft * sizeof(double);

    DeviceBufferGuard d_packed_guard(packed_bytes);
    DeviceBufferGuard d_freq_guard(freq_bytes);
    DeviceBufferGuard d_time_guard(time_bytes);

    cuDoubleComplex* d_packed = reinterpret_cast<cuDoubleComplex*>(d_packed_guard.ptr);
    cuDoubleComplex* d_freq   = reinterpret_cast<cuDoubleComplex*>(d_freq_guard.ptr);
    double* d_time            = reinterpret_cast<double*>(d_time_guard.ptr);

    const cuDoubleComplex* d_w   = d_packed;
    const cuDoubleComplex* d_v00 = d_packed + N;
    const cuDoubleComplex* d_v01 = d_packed + 2 * N;
    const cuDoubleComplex* d_v10 = d_packed + 3 * N;
    const cuDoubleComplex* d_v11 = d_packed + 4 * N;

    // --------------------------------------------------------
    // 5. cuFFT Plan Setup (CUFFT_Z2D: Complex-to-Real 1D)
    // --------------------------------------------------------
    CufftPlanGuard fft_guard;
    check_cufft(
        cufftPlan1d(&fft_guard.plan, N_fft, CUFFT_Z2D, 1),
        "Failed to create cuFFT 1D Z2D plan"
    );
    fft_guard.active = true;

    // --------------------------------------------------------
    // 6. Asynchronous CUDA Stream & Events for Profiling
    // --------------------------------------------------------
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "Failed to create CUDA stream");
    struct StreamGuard {
        cudaStream_t s;
        ~StreamGuard() { if (s) cudaStreamDestroy(s); }
    } stream_guard{stream};

    check_cufft(cufftSetStream(fft_guard.plan, stream), "Failed to associate stream with cuFFT plan");

    CudaEventGuard evt_h2d_start;
    CudaEventGuard evt_h2d_stop;
    CudaEventGuard evt_kernel_start;
    CudaEventGuard evt_kernel_stop;
    CudaEventGuard evt_ifft_start;
    CudaEventGuard evt_ifft_stop;
    CudaEventGuard evt_d2h_start;
    CudaEventGuard evt_d2h_stop;

    // --------------------------------------------------------
    // 7. Host -> Device Transfer
    // --------------------------------------------------------
    check_cuda(cudaEventRecord(evt_h2d_start.evt, stream), "Event record failed");
    check_cuda(
        cudaMemcpyAsync(
            d_packed,
            h_packed.data(),
            packed_bytes,
            cudaMemcpyHostToDevice,
            stream
        ),
        "Host-to-device memcpy failed"
    );
    check_cuda(cudaEventRecord(evt_h2d_stop.evt, stream), "Event record failed");

    // --------------------------------------------------------
    // 8. Launch Cauchy & Woodbury Evaluation Kernel
    // --------------------------------------------------------
    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;

    check_cuda(cudaEventRecord(evt_kernel_start.evt, stream), "Event record failed");
    evaluate_s4_cauchy_woodbury_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_w,
        d_v00,
        d_v01,
        d_v10,
        d_v11,
        d_freq,
        nyquist_val,
        N,
        N_fft,
        M,
        inv_N_fft
    );
    check_cuda(cudaGetLastError(), "Cauchy kernel launch failed");
    check_cuda(cudaEventRecord(evt_kernel_stop.evt, stream), "Event record failed");

    // --------------------------------------------------------
    // 9. Execute cuFFT Inverse FFT (Z2D: Complex to Real)
    // --------------------------------------------------------
    check_cuda(cudaEventRecord(evt_ifft_start.evt, stream), "Event record failed");
    check_cufft(
        cufftExecZ2D(
            fft_guard.plan,
            reinterpret_cast<cufftDoubleComplex*>(d_freq),
            reinterpret_cast<cufftDoubleReal*>(d_time)
        ),
        "cufftExecZ2D execution failed"
    );
    check_cuda(cudaEventRecord(evt_ifft_stop.evt, stream), "Event record failed");

    // --------------------------------------------------------
    // 10. Device -> Host Transfer (First L elements)
    // --------------------------------------------------------
    Eigen::VectorXd kernel(L);
    check_cuda(cudaEventRecord(evt_d2h_start.evt, stream), "Event record failed");
    check_cuda(
        cudaMemcpyAsync(
            kernel.data(),
            d_time,
            static_cast<std::size_t>(L) * sizeof(double),
            cudaMemcpyDeviceToHost,
            stream
        ),
        "Device-to-host memcpy failed"
    );
    check_cuda(cudaEventRecord(evt_d2h_stop.evt, stream), "Event record failed");

    // Synchronize stream
    check_cuda(cudaStreamSynchronize(stream), "Stream synchronization failed");

    // --------------------------------------------------------
    // 11. Profile Timings
    // --------------------------------------------------------
    float transfer_in_ms = 0.0f;
    float cauchy_kernel_ms = 0.0f;
    float ifft_ms = 0.0f;
    float transfer_out_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(&transfer_in_ms, evt_h2d_start.evt, evt_h2d_stop.evt),
        "Failed to measure transfer in elapsed time"
    );
    check_cuda(
        cudaEventElapsedTime(&cauchy_kernel_ms, evt_kernel_start.evt, evt_kernel_stop.evt),
        "Failed to measure Cauchy kernel elapsed time"
    );
    check_cuda(
        cudaEventElapsedTime(&ifft_ms, evt_ifft_start.evt, evt_ifft_stop.evt),
        "Failed to measure cuFFT elapsed time"
    );
    check_cuda(
        cudaEventElapsedTime(&transfer_out_ms, evt_d2h_start.evt, evt_d2h_stop.evt),
        "Failed to measure transfer out elapsed time"
    );

    const auto total_wall_end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = std::chrono::duration<double, std::milli>(
        total_wall_end - total_wall_start
    ).count();

    if (timings != nullptr)
    {
        timings->host_prep_ms     = host_prep_ms;
        timings->transfer_in_ms   = static_cast<double>(transfer_in_ms);
        timings->cauchy_kernel_ms = static_cast<double>(cauchy_kernel_ms);
        timings->ifft_ms          = static_cast<double>(ifft_ms);
        timings->transfer_out_ms  = static_cast<double>(transfer_out_ms);
        timings->total_gpu_ms     = static_cast<double>(cauchy_kernel_ms + ifft_ms);
        timings->total_time_ms    = total_time_ms;
    }

    return kernel;
}

} // namespace ssmforge
