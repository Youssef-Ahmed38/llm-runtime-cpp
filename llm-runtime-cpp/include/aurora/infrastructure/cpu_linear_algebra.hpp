#pragma once

#include "aurora/ports/linear_algebra.hpp"

namespace aurora {

class CpuLinearAlgebra final : public ILinearAlgebra {
 public:
  void matvec(const DenseMatrix& matrix, std::span<const float> input,
              std::span<float> output) const override;
};

}  // namespace aurora
