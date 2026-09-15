#include "ssmforge/kernel.hpp"

#include <stdexcept>

namespace ssmforge {

Eigen::VectorXcd compute_kernel(
    const Eigen::MatrixXcd& A,
    const Eigen::VectorXcd& B,
    const Eigen::RowVectorXcd& C,
    int L
) {
    const int N = static_cast<int>(A.rows());

    // --------------------------------------------------
    // Validate dimensions.
    // --------------------------------------------------

    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            "A must be square"
        );
    }

    if (B.size() != N) {
        throw std::invalid_argument(
            "B must have the same size as A"
        );
    }

    if (C.size() != N) {
        throw std::invalid_argument(
            "C must have the same size as A"
        );
    }

    if (L <= 0) {
        throw std::invalid_argument(
            "Kernel length must be positive"
        );
    }

    // --------------------------------------------------
    // Allocate the kernel.
    //
    // K[t] = C A^t B
    // --------------------------------------------------

    Eigen::VectorXcd K(L);

    // x represents A^t B.
    //
    // At t = 0:
    //
    //     A^0 B = B
    //
    Eigen::VectorXcd x = B;

    // --------------------------------------------------
    // Compute:
    //
    //     K[t] = C A^t B
    //
    // without explicitly calculating A^t.
    // --------------------------------------------------

    for (int t = 0; t < L; ++t)
    {
        // Current state contribution:
        //
        //     K[t] = C x
        //
        // where x = A^t B.
        K(t) = (C * x)(0); // it is 1×1 Eigen matrix not a scalar so (0) says give me only one number

        // Move to the next power:
        //
        //     x = A^(t+1) B
        //
        x = A * x;
    }

    return K;
}

} // namespace ssmforge  