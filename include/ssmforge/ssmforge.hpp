#pragma once

// ==============================================================================
// SSMForge: High-Performance Structured State Space Models in C++ and CUDA
// ==============================================================================
// This umbrella header provides access to the complete public API of SSMForge.
// Individual modular headers remain available for fine-grained inclusion.
// ==============================================================================

// Basic State-Space Model (SSM)
#include "ssmforge/ssm.hpp"

// HiPPO Matrix Formulation (HiPPO-LegS continuous matrices)
#include "ssmforge/hippo.hpp"

// Normal Plus Low-Rank (NPLR) Decomposition
#include "ssmforge/nplr.hpp"

// Continuous-to-Discrete Discretization (ZOH & Bilinear / Tustin)
#include "ssmforge/discretize.hpp"
#include "ssmforge/bilinear.hpp"

// Kernel Computations (Direct and Discrete Recurrence)
#include "ssmforge/kernel.hpp"
#include "ssmforge/discrete_kernel.hpp"

// Resolvent & Cauchy Kernel Operations
#include "ssmforge/cauchy.hpp"
#include "ssmforge/resolvent.hpp"
#include "ssmforge/s4_resolvent.hpp"
#include "ssmforge/frequency_points.hpp"

// S4 Convolution Kernel (CPU reference via FFT & Woodbury)
#include "ssmforge/s4_kernel.hpp"

// CUDA Accelerated Kernels
#include "ssmforge/cuda_kernel.hpp"
#include "ssmforge/cuda_s4_kernel.hpp"

