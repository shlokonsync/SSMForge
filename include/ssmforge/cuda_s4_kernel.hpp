#pragma once

#include <Eigen/Dense>

#include "ssmforge/nplr.hpp"

namespace ssmforge {

// Timing metrics for CUDA S4 kernel execution
struct CudaS4Timings {
    double host_prep_ms = 0.0;     // Precomputing numerators on host
    double transfer_in_ms = 0.0;   // Host-to-device memory transfer
    double cauchy_kernel_ms = 0.0; // GPU parallel Cauchy & Woodbury evaluation
    double ifft_ms = 0.0;          // cuFFT execution
    double transfer_out_ms = 0.0;  // Device-to-host memory transfer
    double total_gpu_ms = 0.0;     // GPU kernel + cuFFT
    double total_time_ms = 0.0;    // Complete end-to-end wall-clock time
};

// Compute an S4 convolution kernel on GPU using the structured NPLR representation:
//
//     A = Lambda - P Q*
//
// Evaluates the continuous-time resolvent at roots of unity via the bilinear transform,
// applies the Woodbury identity in parallel across frequencies on CUDA,
// and computes the inverse FFT via cuFFT.
//
// nplr: NPLR representation (Lambda diagonal, P, Q, B, C)
// L: desired kernel length (L > 0)
// dt: discretization step (dt > 0.0)
// timings: optional pointer to receive benchmark timing breakdown
Eigen::VectorXd compute_cuda_s4_kernel(
    const NPLR& nplr,
    int L,
    double dt,
    CudaS4Timings* timings = nullptr
);

} // namespace ssmforge
