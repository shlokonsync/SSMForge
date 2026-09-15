# SSMForge

SSMForge is a C++17 library for experimenting with structured state-space models (SSMs), numerical discretization, sequence kernels, and GPU acceleration using CUDA.

The project is designed to compare mathematically equivalent CPU and GPU implementations while studying correctness, numerical stability, and performance.

## Features

- Continuous-to-discrete state-space conversion
- Zero-order hold discretization
- Bilinear/Tustin discretization
- Direct state-space kernel generation
- HiPPO matrix construction
- NPLR-related state-space components
- S4-style kernel experimentation
- CPU reference implementations using Eigen
- CUDA implementation of the direct recurrence kernel
- GoogleTest-based correctness tests
- CPU and GPU benchmarks using CMake

## Mathematical Background

A continuous-time state-space model is written as

\[
\dot{x}(t) = Ax(t) + Bu(t)
\]

\[
y(t) = Cx(t) + Du(t)
\]

After discretization, the model becomes

\[
x_{t+1} = A_d x_t + B_d u_t
\]

\[
y_t = C_d x_t + Du_t
\]

For an impulse input, the sequence kernel can be generated using

\[
K_t = C_d A_d^t B_d
\]

SSMForge includes a direct recurrence implementation for generating this kernel.

## Project Structure

```text
SSMForge/
├── CMakeLists.txt
├── README.md
├── include/
│   └── ssmforge/
├── src/
├── tests/
├── benchmarks/
└── build/              # Generated locally; ignored by Git