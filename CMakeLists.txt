cmake_minimum_required(VERSION 3.20)

project(ssmforge
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

# --------------------------------------------------
# C++ and CUDA Settings
# --------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Make CUDA architectures configurable instead of permanently hardcoded.
# Default to architecture 86 if not specified by user or toolchain.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to compile for")
endif()

# --------------------------------------------------
# Options
# --------------------------------------------------

option(SSMFORGE_BUILD_TESTS "Build SSMForge unit and regression tests" ON)
option(SSMFORGE_BUILD_BENCHMARKS "Build SSMForge performance benchmarks" ON)

# --------------------------------------------------
# Dependencies
# --------------------------------------------------

# CUDA Toolkit is required for CUDA Runtime (cudart) and cuFFT (cufft)
find_package(CUDAToolkit REQUIRED)

# Eigen3: Search for installed Eigen3 first; fall back to FetchContent if unavailable
find_package(Eigen3 3.3 QUIET CONFIG)
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.3 QUIET)
endif()

if(NOT TARGET Eigen3::Eigen)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.0
    )
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()

    # Define Eigen3::Eigen as an IMPORTED interface target so it does not interfere
    # with SSMForge's export set or leak build-tree targets.
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    target_include_directories(Eigen3::Eigen INTERFACE
        $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
    )
endif()

# GoogleTest is only fetched when tests are enabled
if(SSMFORGE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# --------------------------------------------------
# SSMForge Library Target
# --------------------------------------------------

add_library(ssmforge

    # Basic State Space Model
    src/ssm.cpp

    # Continuous -> Discrete conversion
    src/discretize.cpp

    # HiPPO matrix generation
    src/hippo.cpp

    # NPLR decomposition
    src/nplr.cpp

    # Cauchy evaluation
    src/cauchy.cpp

    # Kernel evaluations
    src/kernel.cpp
    src/discrete_kernel.cpp

    # Resolvent operations
    src/resolvent.cpp
    src/s4_resolvent.cpp

    # Bilinear transformation
    src/bilinear.cpp 

    # Frequency points (roots of unity)
    src/frequency_points.cpp

    # S4 convolution kernel (CPU)
    src/s4_kernel.cpp

    # CUDA accelerated kernels (Direct Recurrence & Structured S4)
    src/cuda_kernel.cu
    src/cuda_s4_kernel.cu
)

# Create namespaced alias target for in-tree and consumer consistency
add_library(SSMForge::ssmforge ALIAS ssmforge)

# Include directories: separate build interface and install interface
include(GNUInstallDirs)

target_include_directories(ssmforge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Compile features for C++17 and CUDA C++17
target_compile_features(ssmforge
    PUBLIC
        cxx_std_17
    PRIVATE
        cuda_std_17
)

# Link dependencies (Eigen3 and CUDA Toolkit)
target_link_libraries(ssmforge
    PUBLIC
        Eigen3::Eigen
        CUDA::cufft
        CUDA::cudart
)

set_target_properties(ssmforge PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    EXPORT_NAME ssmforge
)

# --------------------------------------------------
# Compiler Warnings & Options
# --------------------------------------------------

if(MSVC)
    target_compile_options(ssmforge
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
else()
    target_compile_options(ssmforge
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
endif()

# --------------------------------------------------
# Installation & Package Export
# --------------------------------------------------

# 1. Install library binary
install(TARGETS ssmforge
    EXPORT SSMForgeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 2. Install public headers
install(DIRECTORY include/ssmforge
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# 3. Export CMake targets and packages
set(SSMFORGE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/SSMForge")

install(EXPORT SSMForgeTargets
    FILE SSMForgeTargets.cmake
    NAMESPACE SSMForge::
    DESTINATION ${SSMFORGE_INSTALL_CMAKEDIR}
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SSMForgeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SSMForgeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SSMForgeConfig.cmake"
    INSTALL_DESTINATION "${SSMFORGE_INSTALL_CMAKEDIR}"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SSMForgeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SSMForgeConfigVersion.cmake"
    DESTINATION "${SSMFORGE_INSTALL_CMAKEDIR}"
)

# --------------------------------------------------
# Tests
# --------------------------------------------------

if(SSMFORGE_BUILD_TESTS)

    enable_testing()

    # SSM tests
    add_executable(test_ssm tests/test_ssm.cpp)
    target_link_libraries(test_ssm PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME SSMTests COMMAND test_ssm)

    # Discretization tests
    add_executable(test_discretize tests/test_discretize.cpp)
    target_link_libraries(test_discretize PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME DiscretizationTests COMMAND test_discretize)

    # HiPPO tests
    add_executable(test_hippo tests/test_hippo.cpp)
    target_link_libraries(test_hippo PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME HiPPOTests COMMAND test_hippo)

    # NPLR tests
    add_executable(test_nplr tests/test_nplr.cpp)
    target_link_libraries(test_nplr PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME NPLRTests COMMAND test_nplr)

    # Cauchy tests
    add_executable(test_cauchy tests/test_cauchy.cpp)
    target_link_libraries(test_cauchy PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME CauchyTests COMMAND test_cauchy)

    # Kernel tests
    add_executable(test_kernel tests/test_kernel.cpp)
    target_link_libraries(test_kernel PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME KernelTests COMMAND test_kernel)

    # Resolvent tests
    add_executable(test_resolvent tests/test_resolvent.cpp)
    target_link_libraries(test_resolvent PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME ResolventTests COMMAND test_resolvent)

    # S4 Resolvent tests
    add_executable(test_s4_resolvent tests/test_s4_resolvent.cpp)
    target_link_libraries(test_s4_resolvent PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME S4ResolventTests COMMAND test_s4_resolvent)

    # Bilinear tests
    add_executable(test_bilinear tests/test_bilinear.cpp)
    target_link_libraries(test_bilinear PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME BilinearTests COMMAND test_bilinear)

    # Frequency Points tests
    add_executable(test_frequency_points tests/test_frequency_points.cpp)
    target_link_libraries(test_frequency_points PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME FrequencyPointsTests COMMAND test_frequency_points)

    # S4 Kernel tests
    add_executable(test_s4_kernel tests/test_s4_kernel.cpp)
    target_link_libraries(test_s4_kernel PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME S4KernelTests COMMAND test_s4_kernel)

    # Kernel Comparison tests
    add_executable(test_kernel_comparison tests/test_kernel_comparison.cpp)
    target_link_libraries(test_kernel_comparison PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME KernelComparisonTests COMMAND test_kernel_comparison)

    # CUDA Direct Kernel tests
    add_executable(test_cuda_kernel tests/test_cuda_kernel.cpp)
    target_link_libraries(test_cuda_kernel PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME CudaKernelTests COMMAND test_cuda_kernel)

    # CUDA Structured S4 Kernel tests
    add_executable(test_cuda_s4_kernel tests/test_cuda_s4_kernel.cpp)
    target_link_libraries(test_cuda_s4_kernel PRIVATE ssmforge GTest::gtest_main)
    add_test(NAME CudaS4KernelTests COMMAND test_cuda_s4_kernel)

endif()

# --------------------------------------------------
# Benchmarks
# --------------------------------------------------

if(SSMFORGE_BUILD_BENCHMARKS)

    add_executable(KernelBenchmark
        benchmarks/kernel_benchmark.cpp
    )
    target_link_libraries(KernelBenchmark
        PRIVATE
            ssmforge
    )

    add_executable(CudaBenchmark
        benchmarks/cuda_benchmark.cu
    )
    target_link_libraries(CudaBenchmark
        PRIVATE
            ssmforge
    )

endif()
