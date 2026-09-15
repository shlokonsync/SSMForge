cmake_minimum_required(VERSION 3.20)

project(ssmforge
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

# --------------------------------------------------
# C++ Settings
# --------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# --------------------------------------------------
# Options
# --------------------------------------------------

option(SSMFORGE_BUILD_TESTS "Build tests" ON)

# --------------------------------------------------
# Dependencies
# --------------------------------------------------

include(FetchContent)

# -------------------------
# Eigen
# -------------------------

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen)

# -------------------------
# GoogleTest
# -------------------------

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

FetchContent_MakeAvailable(googletest)

# --------------------------------------------------
# SSMForge Library
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

    # Cuachy 
    src/cauchy.cpp

    #kernel
    src/kernel.cpp

    #resolvent
    src/resolvent.cpp

    #s4_resolvent
    src/s4_resolvent.cpp

    #bilinear
    src/bilinear.cpp 

    #frequency_points
    src/frequency_points.cpp
    #s4_kernel
    src/s4_kernel.cpp

    # discrete_kernel
    src/discrete_kernel.cpp

    #cuda kernel
    src/cuda_kernel.cu
)

# Tell users of the library where our header files are.
target_include_directories(ssmforge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# SSMForge uses Eigen for matrix/vector operations.
target_link_libraries(ssmforge
    PUBLIC
        Eigen3::Eigen
)
target_include_directories(ssmforge
    PUBLIC
    include
)

set_target_properties(ssmforge PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)


# --------------------------------------------------
# Compiler Warnings
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
# Tests
# --------------------------------------------------

if(SSMFORGE_BUILD_TESTS)

    enable_testing()

    # -------------------------
    # SSM tests
    # -------------------------

    add_executable(test_ssm
        tests/test_ssm.cpp
    )

    target_link_libraries(test_ssm
        PRIVATE
            ssmforge
            GTest::gtest_main
    )

    add_test(
        NAME SSMTests
        COMMAND test_ssm
    )

    # -------------------------
    # Discretization tests
    # -------------------------

    add_executable(test_discretize
        tests/test_discretize.cpp
    )

    target_link_libraries(test_discretize
        PRIVATE
            ssmforge
            GTest::gtest_main
    )

    add_test(
        NAME DiscretizationTests
        COMMAND test_discretize
    )

    # -------------------------
    # HiPPO tests
    # -------------------------

    add_executable(test_hippo
        tests/test_hippo.cpp
    )

    target_link_libraries(test_hippo
        PRIVATE
            ssmforge
            GTest::gtest_main
    )

    add_test(
        NAME HiPPOTests
        COMMAND test_hippo
    )

    # -------------------------
    # NPLR tests
    # -------------------------

    add_executable(test_nplr
        tests/test_nplr.cpp
    )

    target_link_libraries(test_nplr
        PRIVATE
            ssmforge
            GTest::gtest_main
    )

    add_test(
        NAME NPLRTests
        COMMAND test_nplr
    )
    # -------------------------
    # cauchy tests
    # -------------------------
    add_executable(test_cauchy
    tests/test_cauchy.cpp
)

target_link_libraries(test_cauchy
    PRIVATE
        ssmforge
        GTest::gtest_main
)

add_test(
    NAME CauchyTests
    COMMAND test_cauchy
)

    # -------------------------
    # kernel tests
    # -------------------------

        add_executable(test_kernel
            tests/test_kernel.cpp
        )

        target_link_libraries(test_kernel
            PRIVATE
                ssmforge
                GTest::gtest_main
        )

        add_test(
            NAME KernelTests
            COMMAND test_kernel
        )

        add_executable(test_resolvent
    tests/test_resolvent.cpp
)

target_link_libraries(test_resolvent
    PRIVATE
        ssmforge
        GTest::gtest_main
)

add_test(
    NAME ResolventTests
    COMMAND test_resolvent
)

add_executable(test_s4_resolvent
    tests/test_s4_resolvent.cpp
)

target_link_libraries(test_s4_resolvent
    PRIVATE
        ssmforge
        GTest::gtest_main
)

add_test(
    NAME S4ResolventTests
    COMMAND test_s4_resolvent
)

add_executable(test_bilinear
    tests/test_bilinear.cpp
)

target_link_libraries(test_bilinear
    PRIVATE
        ssmforge
        GTest::gtest_main
)

add_test(
    NAME BilinearTests
    COMMAND test_bilinear
)
add_executable(test_frequency_points
    tests/test_frequency_points.cpp
)

target_link_libraries(test_frequency_points
    PRIVATE
        ssmforge
        GTest::gtest_main
)

add_test(
    NAME FrequencyPointsTests
    COMMAND test_frequency_points
)

    add_executable(test_s4_kernel
        tests/test_s4_kernel.cpp
    )

    target_link_libraries(test_s4_kernel
        PRIVATE
            ssmforge
            GTest::gtest_main
    )

    add_test(
        NAME S4KernelTests
        COMMAND test_s4_kernel
    )

endif()

# --------------------------------------------------
# Benchmarks
# --------------------------------------------------

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