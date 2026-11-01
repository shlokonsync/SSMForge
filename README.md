# SSMForge

SSMForge is a high-performance C++17 and CUDA library for structured state-space models (SSMs), continuous memory representations (HiPPO), numerical discretization, and fast sequence convolution kernels (Structured S4).

SSMForge provides both CPU reference implementations (powered by Eigen) and GPU-accelerated implementations (powered by custom CUDA kernels and cuFFT), designed for numerical precision, mathematical correctness, and reusable integration into external C++ applications.

---

## Features Currently Implemented

- **Continuous & Discrete State-Space Models**: Representation and simulation of linear continuous and discrete state-space systems ($A, B, C, D$).
- **Discretization Algorithms**:
  - Bilinear (Tustin) transformation with closed-form Cayley transform: $\bar{A} = (I - \frac{\Delta t}{2}A)^{-1}(I + \frac{\Delta t}{2}A)$, $\bar{B} = (I - \frac{\Delta t}{2}A)^{-1}\Delta t B$.
  - Zero-Order Hold (ZOH) discretization via matrix exponentiation.
- **HiPPO Memory Representations**:
  - Continuous HiPPO-LegS (High-order Polynomial Projection Operators on Legendre measure) matrix generation.
- **Normal Plus Low-Rank (NPLR) Decomposition**:
  - Diagonalization of the skew-symmetric component: $A = \Lambda - P Q^*$, where $\Lambda$ is diagonal and skew-symmetric/pure imaginary, and $P, Q$ are rank-1 factors.
- **Cauchy Kernel & Matrix Operations**:
  - Evaluation of Cauchy matrix elements $\frac{1}{z_k - \lambda_n}$ and Cauchy matrix-vector products.
- **Resolvent Evaluation**:
  - Direct matrix resolvent $(zI - A)^{-1}$.
  - Low-rank Woodbury S4 resolvent exploiting the NPLR structure: $(zI - A)^{-1} = (zI - \Lambda)^{-1} - \frac{(zI - \Lambda)^{-1} P Q^* (zI - \Lambda)^{-1}}{1 + Q^* (zI - \Lambda)^{-1} P}$.
- **Frequency Analysis**:
  - Evaluation on the unit circle via roots of unity frequency grids $\Omega = \{ \exp(-2\pi i k / L) \}$.
- **CPU S4 Convolution Kernel**:
  - Full pipeline computing $K = (C \bar{B}, C \bar{A} \bar{B}, \dots, C \bar{A}^{L-1} \bar{B})$ via Cauchy products and inverse FFT in $O((N + L) \log L)$ time.
- **CUDA Direct Recurrence Kernel**:
  - GPU-accelerated dense recurrence kernel evaluation.
- **Structured CUDA S4 Kernel**:
  - High-performance GPU pipeline evaluating Cauchy resolvents in parallel CUDA thread blocks followed by cuFFT inverse transform.
- **Reusable Library & CMake Package**:
  - Configurable CMake target `SSMForge::ssmforge` with CMake package config export (`SSMForgeConfig.cmake`), relocatable interfaces, and umbrella header `#include <ssmforge/ssmforge.hpp>`.

---

## Prerequisites

- **C++ Compiler**: Modern C++17 compliant compiler (GCC 9+, Clang 10+, or MSVC 2022 v19.3+).
- **CMake**: Version 3.20 or newer.
- **NVIDIA CUDA Toolkit**: CUDA 11.x, 12.x, or 13.x with `cudart` and `cufft`.
- **Eigen3**: Version 3.3 or newer (header-only linear algebra library).
- *(Optional)* **GoogleTest**: Automatically retrieved via CMake FetchContent when building tests.

---

## CUDA Requirements

- **GPU Compute Capability**: Compatible NVIDIA GPU (e.g., Ampere `sm_86`, Ada Lovelace `sm_89`, Hopper `sm_90`, etc.).
- **Architecture Configuration**: Configurable via CMake option `-DCMAKE_CUDA_ARCHITECTURES=<arch>` (defaults to `86`). Set this to match your GPU architecture (e.g., `80` for A100, `86` for RTX 30-series, `89` for RTX 40-series).
- **Toolchain**: Supported host C++ compiler compatible with the installed CUDA version.

---

## Build Instructions

To build the SSMForge static library:

```bash
# 1. Clone repository
git clone https://github.com/your-username/SSMForge.git
cd SSMForge

# 2. Configure build (specifying CUDA architecture)
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DSSMFORGE_BUILD_TESTS=ON \
    -DSSMFORGE_BUILD_BENCHMARKS=ON

# 3. Compile library and targets
cmake --build build --config Release
```

### Running Tests

Run the complete CTest test suite:

```bash
ctest --test-dir build -C Release --output-on-failure
```

---

## Installation Instructions

To install SSMForge to a designated directory prefix:

```bash
cmake --install build --prefix /path/to/install --config Release
```

This generates the following relocatable installation structure:

```text
/path/to/install/
├── include/
│   └── ssmforge/
│       ├── ssmforge.hpp          # Primary umbrella header
│       ├── ssm.hpp
│       ├── hippo.hpp
│       ├── nplr.hpp
│       ├── discretize.hpp
│       ├── bilinear.hpp
│       ├── cauchy.hpp
│       ├── resolvent.hpp
│       ├── s4_resolvent.hpp
│       ├── s4_kernel.hpp
│       ├── cuda_kernel.hpp
│       ├── cuda_s4_kernel.hpp
│       └── ...
├── lib/
│   └── ssmforge.lib (or libssmforge.a)
└── lib/cmake/SSMForge/
    ├── SSMForgeConfig.cmake
    ├── SSMForgeConfigVersion.cmake
    ├── SSMForgeTargets.cmake
    └── SSMForgeTargets-release.cmake
```

---

## Consuming SSMForge in Another C++ Project

SSMForge provides CMake exported targets via `SSMForgeConfig.cmake`.

### 1. `CMakeLists.txt` for Consumer Application

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed SSMForge package
find_package(SSMForge CONFIG REQUIRED)

# Link your application against SSMForge
add_executable(my_app main.cpp)
target_link_libraries(my_app
    PRIVATE
        SSMForge::ssmforge
)
```

### 2. Configuring the Consumer

Ensure that CMake knows where to find both SSMForge and Eigen3:

```bash
cmake -B build -S . \
    -DCMAKE_PREFIX_PATH="/path/to/ssmforge/install;/path/to/eigen/install" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

*(Note: If Eigen3 is installed in standard system paths, specifying its path in `CMAKE_PREFIX_PATH` is not required. You can also specify `-DEigen3_DIR=<path>` or `-DEIGEN3_ROOT=<path>`.)*

---

## Working C++ Example

Here is a self-contained example demonstrating how to consume SSMForge through the umbrella header `ssmforge/ssmforge.hpp`:

```cpp
#include <iostream>
#include <iomanip>
#include <ssmforge/ssmforge.hpp>

int main() {
    const int N = 8;        // State dimension
    const int L = 16;       // Sequence length
    const double dt = 0.1;  // Discretization step size

    // 1. Construct continuous HiPPO-LegS representation
    const auto hippo = ssmforge::hippo_legs(N);
    const Eigen::VectorXd B = Eigen::VectorXd::Ones(N);
    const Eigen::RowVectorXd C = Eigen::RowVectorXd::Ones(N);

    // 2. Discretize via Bilinear (Tustin) transform
    const auto discrete = ssmforge::discretize_bilinear(hippo.A, B, dt);

    // 3. Compute discrete recurrence reference kernel K_t = C * Ad^t * Bd
    const Eigen::VectorXd k_ref = ssmforge::compute_discrete_kernel(discrete.Ad, discrete.Bd, C, L);

    // 4. Perform Normal Plus Low-Rank (NPLR) decomposition: A = Lambda - P Q*
    const auto nplr = ssmforge::nplr_decompose(hippo.A, B, C);

    // 5. Compute S4 convolution kernel on CPU via resolvents & FFT
    const Eigen::VectorXd k_cpu = ssmforge::compute_s4_kernel(nplr, L, dt);

    // 6. Compute Structured S4 convolution kernel on GPU using CUDA
    ssmforge::CudaS4Timings timings;
    const Eigen::VectorXd k_gpu = ssmforge::compute_cuda_s4_kernel(nplr, L, dt, &timings);

    std::cout << "First 4 elements of GPU S4 kernel: ";
    for (int t = 0; t < 4; ++t) {
        std::cout << std::fixed << std::setprecision(5) << k_gpu(t) << " ";
    }
    std::cout << "\nGPU Total Time: " << timings.total_gpu_ms << " ms\n";

    return 0;
}
```

---

## Repository Structure

```text
SSMForge/
├── cmake/
│   └── SSMForgeConfig.cmake.in   # CMake package export template
├── examples/
│   └── consumer/                 # Standalone external consumer project
│       ├── CMakeLists.txt
│       └── main.cpp
├── include/
│   └── ssmforge/                 # Public library headers
│       ├── ssmforge.hpp          # Primary umbrella header
│       ├── ssm.hpp
│       ├── hippo.hpp
│       ├── nplr.hpp
│       ├── discretize.hpp
│       ├── bilinear.hpp
│       ├── cauchy.hpp
│       ├── resolvent.hpp
│       ├── s4_resolvent.hpp
│       ├── frequency_points.hpp
│       ├── s4_kernel.hpp
│       ├── cuda_kernel.hpp
│       ├── cuda_s4_kernel.hpp
│       └── discrete_kernel.hpp
├── src/                          # Library implementation files (.cpp & .cu)
│   ├── ssm.cpp
│   ├── discretize.cpp
│   ├── hippo.cpp
│   ├── nplr.cpp
│   ├── cauchy.cpp
│   ├── resolvent.cpp
│   ├── s4_resolvent.cpp
│   ├── bilinear.cpp
│   ├── frequency_points.cpp
│   ├── s4_kernel.cpp
│   ├── cuda_kernel.cu
│   └── cuda_s4_kernel.cu
├── tests/                        # Comprehensive CTest / GoogleTest suite
├── benchmarks/                   # Performance benchmarks
└── CMakeLists.txt                # Root CMake project configuration
```

---

## License

MIT License.
