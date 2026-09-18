#include "aurora/model.hpp"
#include "aurora/infrastructure/cpu_linear_algebra.hpp"
#include "aurora/runtime.hpp"
#include "aurora/tensor.hpp"
#include "aurora/tokenizer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
void require(const bool condition, const char* message) { if (!condition) { std::cerr << "FAILED: " << message << '\n'; std::exit(1); } }
}

int main() {
  aurora::DenseMatrix matrix(2, 3); matrix(0, 0) = 1; matrix(0, 1) = 2; matrix(0, 2) = 3; matrix(1, 0) = 4; matrix(1, 1) = 5; matrix(1, 2) = 6;
  const std::vector<float> input{1, 1, 1}; std::vector<float> output(2); aurora::CpuLinearAlgebra{}.matvec(matrix, input, output);
  require(std::fabs(output[0] - 6.0F) < 0.0001F && std::fabs(output[1] - 15.0F) < 0.0001F, "matvec");
  aurora::ByteTokenizer tokenizer; const auto encoded = tokenizer.encode("Hello, عالم"); require(tokenizer.decode(encoded) == "Hello, عالم", "UTF-8 byte tokenizer round trip");
  const auto model = aurora::DecoderModel::make_demo(); aurora::CpuLinearAlgebra cpu; aurora::DecoderRuntime runtime(model, cpu); aurora::Sampler sampler({.temperature = 0.7F, .top_p = 0.9F, .top_k = 16, .seed = 1});
  const std::vector<std::uint32_t> prompt{'h', 'i'}; const auto result = runtime.generate(prompt, 4, sampler);
  require(result.size() == 4, "generation length"); require(runtime.forward('!').size() == model.config.vocab_size, "decoder output size");
  std::cout << "All Aurora runtime tests passed.\n";
}
