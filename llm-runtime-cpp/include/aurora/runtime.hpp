#pragma once

#include "aurora/model.hpp"
#include "aurora/ports/linear_algebra.hpp"

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace aurora {

class KvCache {
 public:
  explicit KvCache(const ModelConfig& config);
  void clear() noexcept { length_ = 0; }
  [[nodiscard]] std::uint32_t length() const noexcept { return length_; }
  void append(std::uint32_t layer, std::span<const float> key, std::span<const float> value);
  [[nodiscard]] std::span<const float> key(std::uint32_t layer, std::uint32_t position) const;
  [[nodiscard]] std::span<const float> value(std::uint32_t layer, std::uint32_t position) const;
  void advance();

 private:
  ModelConfig config_;
  std::uint32_t length_ = 0;
  std::vector<float> keys_;
  std::vector<float> values_;
};

struct SamplingConfig {
  float temperature = 0.8F;
  float top_p = 0.95F;
  std::uint32_t top_k = 40;
  std::uint32_t seed = 7;
};

class Sampler {
 public:
  explicit Sampler(SamplingConfig config);
  [[nodiscard]] std::uint32_t sample(std::span<const float> logits);

 private:
  SamplingConfig config_;
  std::mt19937 generator_;
  std::vector<std::uint32_t> candidates_;
  std::vector<float> probabilities_;
  std::vector<std::size_t> probability_order_;
  std::vector<double> sampling_weights_;
};

class DecoderRuntime {
 public:
  DecoderRuntime(const DecoderModel& model, const ILinearAlgebra& linear_algebra);
  void reset();
  [[nodiscard]] std::vector<float> forward(std::uint32_t token);
  [[nodiscard]] std::vector<std::uint32_t> generate(std::span<const std::uint32_t> prompt,
                                                     std::uint32_t tokens_to_generate,
                                                     Sampler& sampler);

 private:
  struct Workspace {
    explicit Workspace(const ModelConfig& config);
    std::vector<float> state;
    std::vector<float> normalized;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> attention;
    std::vector<float> projected;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> logits;
    std::vector<float> scores;
  };

  [[nodiscard]] std::span<const float> decode(std::uint32_t token);
  const DecoderModel& model_;
  const ILinearAlgebra& linear_algebra_;
  KvCache cache_;
  Workspace workspace_;
};

}  // namespace aurora
