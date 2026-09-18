#pragma once

#include "aurora/tensor.hpp"

#include <cstdint>
#include <vector>

namespace aurora {

struct ModelConfig {
  std::uint32_t vocab_size = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t num_layers = 0;
  std::uint32_t num_heads = 0;
  std::uint32_t max_sequence_length = 0;

  [[nodiscard]] std::uint32_t head_size() const;
  void validate() const;
};

struct DecoderLayerWeights {
  std::vector<float> attention_norm;
  std::vector<float> ffn_norm;
  DenseMatrix query;
  DenseMatrix key;
  DenseMatrix value;
  DenseMatrix attention_output;
  DenseMatrix gate;
  DenseMatrix up;
  DenseMatrix down;
};

struct DecoderModel {
  ModelConfig config;
  DenseMatrix token_embedding;
  std::vector<DecoderLayerWeights> layers;
  std::vector<float> final_norm;
  DenseMatrix language_model_head;

  void validate() const;
  [[nodiscard]] static DecoderModel make_demo();
};

}  // namespace aurora
