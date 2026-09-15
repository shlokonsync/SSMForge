#include "ssmforge/s4_kernel.hpp"

#include "ssmforge/frequency_points.hpp"

#include <unsupported/Eigen/FFT>

#include <cmath>
#include <complex>
#include <stdexcept>

namespace ssmforge {

Eigen::VectorXd compute_s4_kernel(
    const NPLR& nplr,
    int L,
    double dt
)
{
    if (L <= 0) {
        throw std::invalid_argument(
            "Kernel length must be positive"
        );
    }

    if (dt <= 0.0) {
        throw std::invalid_argument(
            "dt must be positive"
        );
    }

    const int N = static_cast<int>(nplr.N.rows());

    // Check dimensions of the NPLR representation.
    if (nplr.N.rows() != nplr.N.cols()) {
        throw std::invalid_argument(
            "N must be square"
        );
    }

    if (nplr.P.size() != N ||
        nplr.Q.size() != N ||
        nplr.B.size() != N ||
        nplr.C.size() != N)
    {
        throw std::invalid_argument(
            "Invalid NPLR dimensions"
        );
    }

    // In S4, to avoid circular convolution / time-domain aliasing
    // from the inverse FFT, we pad the FFT length based on the decay rate:
    // For HiPPO, real(lambda) ~ -0.5, so decay is exp(-0.5 * t * dt).
    // To ensure aliased tail < 1e-5: 0.5 * N_fft * dt >= 12 => N_fft >= 24 / dt.
    int min_points = std::max(2 * L, static_cast<int>(std::ceil(24.0 / dt)));
    int N_fft = 1;
    while (N_fft < min_points) {
        N_fft *= 2;
    }

    // FFT points:
    //
    //     omega_k = exp(-2*pi*i*k/N_fft)
    //
    // Only N_fft/2 + 1 points are needed.
    Eigen::VectorXcd omega =
        frequency_points(N_fft);

    // Number of frequency points to compute (first half)
    const int M = N_fft / 2 + 1;

    // Continuous eigenvalues.
    //
    // NPLR stores:
    //
    //     A = Lambda - P Q*
    //
    // The S4 implementation incorporates dt into Lambda.
    Eigen::VectorXcd w = dt * nplr.N.diagonal();

    Eigen::VectorXcd k_frequency(M);

    for (int k = 0; k < M; ++k)
    {
        const std::complex<double> om = std::conj(omega(k));

        // Bilinear transform:
        //
        //     z = 2(1 - omega)/(1 + omega)
        //
        // This is how S4 evaluates the continuous-time
        // resolvent at the FFT nodes.
        const std::complex<double> denominator =
            1.0 + om;

        if (std::abs(denominator) < 1e-12)
        {
            // At the Nyquist frequency (omega = -1), z -> infinity.
            // Using the bilinear discrete transfer function:
            // H(-1) = C (I + Ad)^-1 Bd = dt / 2 * C B
            k_frequency(k) = 0.5 * dt * (nplr.C * nplr.B)(0);
            continue;
        }

        const std::complex<double> z =
            2.0 * (1.0 - om) / denominator;

        // D = zI - Lambda.
        //
        // Because Lambda is diagonal, D^{-1} is just
        // element-wise division.
        Eigen::VectorXcd D_inv(N);

        for (int i = 0; i < N; ++i)
        {
            const std::complex<double> d =
                z - w(i);

            if (std::abs(d) < 1e-12)
            {
                throw std::runtime_error(
                    "Resolvent encountered singular point"
                );
            }

            D_inv(i) = 1.0 / d;
        }

        // -------------------------------------------------
        // Cauchy / resolvent terms
        // -------------------------------------------------
        //
        // r00 = dt * C D^-1 B
        // r01 = dt * C D^-1 P
        // r10 = dt * Q* D^-1 B
        // r11 = dt * Q* D^-1 P
        //
        // Since the rank is one, these are scalars.
        //

        Eigen::VectorXcd DinvB = nplr.B.array() * D_inv.array();
        Eigen::VectorXcd DinvP = nplr.P.col(0).array() * D_inv.array();

        const std::complex<double> r00 =
            dt * (nplr.C * DinvB)(0);

        const std::complex<double> r01 =
            dt * (nplr.C * DinvP)(0);

        // Q* means conjugate-transpose.
        const std::complex<double> r10 =
            dt * (nplr.Q.col(0).adjoint() * DinvB)(0);

        const std::complex<double> r11 =
            dt * (nplr.Q.col(0).adjoint() * DinvP)(0);

        // -------------------------------------------------
        // Woodbury correction
        // -------------------------------------------------
        //
        // A = Lambda - P Q*
        //
        // Therefore:
        //
        // H(z) =
        // C D^-1 B
        // -
        // (C D^-1 P)(Q* D^-1 B)
        // --------------------------------
        //       1 + Q* D^-1 P
        //

        const std::complex<double> woodbury_denominator =
            1.0 + r11;

        if (std::abs(woodbury_denominator) < 1e-12)
        {
            throw std::runtime_error(
                "Woodbury correction encountered singular denominator"
            );
        }

        k_frequency(k) =
            r00
            -
            (r01 * r10)
            / woodbury_denominator;

        // Final correction associated with the
        // bilinear transform.
        k_frequency(k) *=
            2.0 / (1.0 + om);
    }

    // -----------------------------------------------------
    // Reconstruct the full conjugate-symmetric spectrum.
    // -----------------------------------------------------
    //
    // We currently have:
    //
    //     k_frequency[0 ... N_fft/2]
    //
    // The remaining FFT values are obtained by
    // complex conjugate symmetry.
    //

    Eigen::VectorXcd full_spectrum(N_fft);

    full_spectrum.setZero();

    full_spectrum(0) = k_frequency(0);

    for (int k = 1; k < M; ++k)
    {
        full_spectrum(k) = k_frequency(k);
    }

    // For even N_fft, the Nyquist point N_fft/2 should not
    // be duplicated.
    const int last = (N_fft % 2 == 0)
        ? M - 2
        : M - 1;

    for (int k = 1; k <= last; ++k)
    {
        full_spectrum(N_fft - k) =
            std::conj(k_frequency(k));
    }

    // -----------------------------------------------------
    // Inverse FFT
    // -----------------------------------------------------

    Eigen::FFT<double> fft;

    Eigen::VectorXcd time_domain;

    fft.inv(time_domain, full_spectrum);

    // The theoretical kernel is real-valued.
    // Small imaginary parts can appear because of
    // floating-point error.
    Eigen::VectorXd kernel(L);

    for (int k = 0; k < L; ++k)
    {
        kernel(k) = time_domain(k).real();
    }

    return kernel;
}

} // namespace ssmforge