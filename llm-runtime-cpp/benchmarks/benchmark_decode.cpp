#include "aurora/model.hpp"
#include "aurora/infrastructure/cpu_linear_algebra.hpp"
#include "aurora/runtime.hpp"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
  const auto model = aurora::DecoderModel::make_demo(); aurora::CpuLinearAlgebra cpu; aurora::DecoderRuntime runtime(model, cpu);
  const std::vector<std::uint32_t> prompt{'b', 'e', 'n', 'c', 'h'};
  for (const auto token : prompt) runtime.forward(token);
  constexpr std::uint32_t kSteps = 100;
  const auto started = std::chrono::steady_clock::now();
  for (std::uint32_t index = 0; index < kSteps; ++index) runtime.forward(index % model.config.vocab_size);
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::cout << "decode_tokens=" << kSteps << " seconds=" << elapsed << " tokens_per_second=" << static_cast<double>(kSteps) / elapsed << '\n';
}
