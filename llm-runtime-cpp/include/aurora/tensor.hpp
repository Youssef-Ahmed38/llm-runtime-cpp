#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace aurora {

class DenseMatrix {
 public:
  DenseMatrix() = default;
  DenseMatrix(std::size_t rows, std::size_t cols);

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] float& operator()(std::size_t row, std::size_t col) noexcept;
  [[nodiscard]] const float& operator()(std::size_t row, std::size_t col) const noexcept;
  [[nodiscard]] std::span<float> data() noexcept { return values_; }
  [[nodiscard]] std::span<const float> data() const noexcept { return values_; }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<float> values_;
};

void rms_norm(std::span<const float> input, std::span<const float> weight, std::span<float> output,
              float epsilon = 1.0e-5F);
void add_inplace(std::span<float> destination, std::span<const float> source);
void silu_multiply_inplace(std::span<float> gate, std::span<const float> up);

}  // namespace aurora
