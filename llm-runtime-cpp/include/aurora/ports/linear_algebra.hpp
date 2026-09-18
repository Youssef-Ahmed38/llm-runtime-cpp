#pragma once

#include "aurora/tensor.hpp"

#include <span>

namespace aurora {

// Port: the decoder owns inference semantics, while hardware-specific kernels
// (scalar, SIMD, CUDA, or vendor BLAS) are injected from infrastructure.
class ILinearAlgebra {
 public:
  virtual ~ILinearAlgebra() = default;
  virtual void matvec(const DenseMatrix& matrix, std::span<const float> input,
                      std::span<float> output) const = 0;
};

}  // namespace aurora
