#pragma once

#include "aurora/ports/linear_algebra.hpp"
#include "aurora/ports/model_repository.hpp"
#include "aurora/ports/tokenizer.hpp"
#include "aurora/runtime.hpp"

#include <cstdint>
#include <string>

namespace aurora {

struct GenerationRequest {
  std::string model_path;
  std::string prompt;
  std::uint32_t tokens_to_generate = 64;
  SamplingConfig sampling{};
};

class GenerationService {
 public:
  GenerationService(const IModelRepository& models, const ITokenizer& tokenizer,
                    const ILinearAlgebra& linear_algebra)
      : models_(models), tokenizer_(tokenizer), linear_algebra_(linear_algebra) {}

  [[nodiscard]] std::string generate(const GenerationRequest& request) const;

 private:
  const IModelRepository& models_;
  const ITokenizer& tokenizer_;
  const ILinearAlgebra& linear_algebra_;
};

}  // namespace aurora
