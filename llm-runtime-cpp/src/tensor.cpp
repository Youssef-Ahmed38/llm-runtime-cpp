#include "aurora/tensor.hpp"

#include <cmath>
#include <stdexcept>

namespace aurora {

DenseMatrix::DenseMatrix(const std::size_t rows, const std::size_t cols)
    : rows_(rows), cols_(cols), values_(rows * cols) {}

float& DenseMatrix::operator()(const std::size_t row, const std::size_t col) noexcept {
  return values_[row * cols_ + col];
}

const float& DenseMatrix::operator()(const std::size_t row, const std::size_t col) const noexcept {
  return values_[row * cols_ + col];
}

void rms_norm(const std::span<const float> input, const std::span<const float> weight,
              const std::span<float> output, const float epsilon) {
  if (input.size() != weight.size() || input.size() != output.size()) {
    throw std::invalid_argument("rms_norm dimensions do not match");
  }
  float squared_sum = 0.0F;
  for (const float value : input) squared_sum += value * value;
  const float scale = 1.0F / std::sqrt(squared_sum / static_cast<float>(input.size()) + epsilon);
  for (std::size_t index = 0; index < input.size(); ++index) output[index] = input[index] * scale * weight[index];
}

void add_inplace(const std::span<float> destination, const std::span<const float> source) {
  if (destination.size() != source.size()) throw std::invalid_argument("add dimensions do not match");
  for (std::size_t index = 0; index < destination.size(); ++index) destination[index] += source[index];
}

void silu_multiply_inplace(const std::span<float> gate, const std::span<const float> up) {
  if (gate.size() != up.size()) throw std::invalid_argument("SwiGLU dimensions do not match");
  for (std::size_t index = 0; index < gate.size(); ++index) {
    const float value = gate[index];
    gate[index] = (value / (1.0F + std::exp(-value))) * up[index];
  }
}

}  // namespace aurora
