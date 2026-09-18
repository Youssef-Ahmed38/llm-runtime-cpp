#include "aurora/infrastructure/cpu_linear_algebra.hpp"

#include <numeric>
#include <stdexcept>

namespace aurora {

void CpuLinearAlgebra::matvec(const DenseMatrix& matrix, const std::span<const float> input,
                              const std::span<float> output) const {
  if (matrix.cols() != input.size() || matrix.rows() != output.size()) {
    throw std::invalid_argument("matvec dimensions do not match");
  }
#if defined(AURORA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(matrix.rows() >= 256)
#endif
  for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(matrix.rows()); ++row) {
    const auto row_index = static_cast<std::size_t>(row);
    const auto weights = matrix.data().subspan(row_index * matrix.cols(), matrix.cols());
    output[row_index] = std::inner_product(weights.begin(), weights.end(), input.begin(), 0.0F);
  }
}

}  // namespace aurora
