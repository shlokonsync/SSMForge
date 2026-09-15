#include "ssmforge/hippo.hpp"

#include <cmath>

namespace ssmforge {

HiPPO hippo_legs(int N)
{
    HiPPO result;

    // A is an N x N state-transition matrix.
    // B is an N-dimensional input vector.
    result.A = Eigen::MatrixXd::Zero(N, N);
    result.B = Eigen::VectorXd::Zero(N);

    // Construct the HiPPO-LegS matrix.
    //
    // For n > k:
    // A[n,k] = -sqrt((2n + 1)(2k + 1))
    //
    // For n = k:
    // A[n,n] = -(n + 1)
    //
    // For n < k:
    // A[n,k] = 0
    for (int n = 0; n < N; ++n) {

        // Input vector:
        // B[n] = sqrt(2n + 1)
        result.B(n) = std::sqrt(2.0 * n + 1.0);

        for (int k = 0; k < N; ++k) {

            if (n > k) {

                // Lower-triangular part of A.
                result.A(n, k) =
                    -std::sqrt(
                        (2.0 * n + 1.0) *
                        (2.0 * k + 1.0)
                    );

            }
            else if (n == k) {

                // Diagonal of A.
                result.A(n, k) = -(n + 1.0);
            }

            // If n < k, A[n,k] remains zero.
        }
    }

    return result;
}

} // namespace ssmforge
