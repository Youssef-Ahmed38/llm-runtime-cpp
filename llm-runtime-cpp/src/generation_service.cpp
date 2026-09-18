#include "aurora/application/generation_service.hpp"

#include <stdexcept>

namespace aurora {

std::string GenerationService::generate(const GenerationRequest& request) const {
  if (request.prompt.empty()) throw std::invalid_argument("prompt must not be empty");
  auto model = models_.load(request.model_path);
  auto tokens = tokenizer_.encode(request.prompt);
  if (tokens.empty()) throw std::invalid_argument("prompt produced no tokens");
  for (auto& token : tokens) token %= model.config.vocab_size;
  DecoderRuntime runtime(model, linear_algebra_);
  Sampler sampler(request.sampling);
  return tokenizer_.decode(runtime.generate(tokens, request.tokens_to_generate, sampler));
}

}  // namespace aurora
