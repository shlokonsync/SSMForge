#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "ssmforge/bilinear.hpp"
#include "ssmforge/discrete_kernel.hpp"
#include "ssmforge/hippo.hpp"
#include "ssmforge/nplr.hpp"
#include "ssmforge/s4_kernel.hpp"

int main()
{
    const int N = 64;
    const double dt = 0.1;

    const std::vector<int> lengths = {
        256,
        1024,
        4096,
        16384,
        65536
    };

    const int repetitions = 5;

    std::cout << "SSMForge Kernel Scaling Benchmark\n";
    std::cout << "=================================\n";
    std::cout << "State size N: " << N << "\n";
    std::cout << "Time step dt: " << dt << "\n";
    std::cout << "Repetitions: " << repetitions << "\n\n";

    // ------------------------------------------------------------
    // 1. Construct the continuous-time HiPPO system
    // ------------------------------------------------------------

    const auto hippo =
        ssmforge::hippo_legs(N);

    Eigen::RowVectorXd C =
        Eigen::RowVectorXd::Ones(N);

    // ------------------------------------------------------------
    // 2. Construct the reference discrete-time system
    // ------------------------------------------------------------

    const auto discrete =
        ssmforge::discretize_bilinear(
            hippo.A,
            hippo.B,
            dt
        );

    // ------------------------------------------------------------
    // 3. Construct the NPLR representation
    // ------------------------------------------------------------

    const auto nplr =
        ssmforge::nplr_decompose(
            hippo.A,
            hippo.B,
            C
        );

    std::cout << std::scientific << std::setprecision(12);

    // ------------------------------------------------------------
    // 4. Test different kernel lengths
    // ------------------------------------------------------------

    for (int L : lengths)
    {
        double total_reference_time_ms = 0.0;
        double total_fast_time_ms = 0.0;

        double best_reference_time_ms =
            std::numeric_limits<double>::max();

        double best_fast_time_ms =
            std::numeric_limits<double>::max();

        Eigen::VectorXd reference_kernel;
        Eigen::VectorXd fast_kernel;

        // --------------------------------------------------------
        // 5. Warm up the reference implementation
        // --------------------------------------------------------

        reference_kernel =
            ssmforge::compute_discrete_kernel(
                discrete.Ad,
                discrete.Bd,
                C,
                L
            );

        // --------------------------------------------------------
        // 6. Measure the reference implementation
        // --------------------------------------------------------

        for (int run = 0; run < repetitions; ++run)
        {
            const auto start =
                std::chrono::high_resolution_clock::now();

            reference_kernel =
                ssmforge::compute_discrete_kernel(
                    discrete.Ad,
                    discrete.Bd,
                    C,
                    L
                );

            const auto end =
                std::chrono::high_resolution_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();

            total_reference_time_ms += elapsed_ms;

            if (elapsed_ms < best_reference_time_ms)
            {
                best_reference_time_ms = elapsed_ms;
            }
        }

        // --------------------------------------------------------
        // 7. Warm up the fast S4 implementation
        // --------------------------------------------------------

        fast_kernel =
            ssmforge::compute_s4_kernel(
                nplr,
                L,
                dt
            );

        // --------------------------------------------------------
        // 8. Measure the fast S4 implementation
        // --------------------------------------------------------

        for (int run = 0; run < repetitions; ++run)
        {
            const auto start =
                std::chrono::high_resolution_clock::now();

            fast_kernel =
                ssmforge::compute_s4_kernel(
                    nplr,
                    L,
                    dt
                );

            const auto end =
                std::chrono::high_resolution_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();

            total_fast_time_ms += elapsed_ms;

            if (elapsed_ms < best_fast_time_ms)
            {
                best_fast_time_ms = elapsed_ms;
            }
        }

        // --------------------------------------------------------
        // 9. Calculate average times
        // --------------------------------------------------------

        const double average_reference_time_ms =
            total_reference_time_ms / repetitions;

        const double average_fast_time_ms =
            total_fast_time_ms / repetitions;

        // --------------------------------------------------------
        // 10. Calculate speedups
        // --------------------------------------------------------

        const double average_speedup =
            average_reference_time_ms /
            average_fast_time_ms;

        const double best_speedup =
            best_reference_time_ms /
            best_fast_time_ms;

        // --------------------------------------------------------
        // 11. Calculate numerical error
        // --------------------------------------------------------

        const double maximum_error =
            (reference_kernel - fast_kernel)
                .cwiseAbs()
                .maxCoeff();

        // --------------------------------------------------------
        // 12. Print results
        // --------------------------------------------------------

        std::cout << "Kernel length L: "
                  << L
                  << "\n";

        std::cout << "Average reference time: "
                  << average_reference_time_ms
                  << " ms\n";

        std::cout << "Best reference time:    "
                  << best_reference_time_ms
                  << " ms\n";

        std::cout << "Average fast S4 time:   "
                  << average_fast_time_ms
                  << " ms\n";

        std::cout << "Best fast S4 time:      "
                  << best_fast_time_ms
                  << " ms\n";

        std::cout << "Average speedup:        "
                  << average_speedup
                  << "x\n";

        std::cout << "Best speedup:           "
                  << best_speedup
                  << "x\n";

        std::cout << "Maximum absolute error: "
                  << maximum_error
                  << "\n";

        std::cout << "First reference value:  "
                  << reference_kernel(0)
                  << "\n";

        std::cout << "First fast value:       "
                  << fast_kernel(0)
                  << "\n";

        std::cout << "---------------------------------\n";
    }

    return 0;
}