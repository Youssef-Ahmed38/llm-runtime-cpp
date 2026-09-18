#include "aurora/model.hpp"

#include <random>
#include <stdexcept>

namespace aurora {
namespace {
void fill_random(DenseMatrix& matrix, std::mt19937& generator, const float scale) {
  std::normal_distribution<float> distribution(0.0F, scale);
  for (float& value : matrix.data()) value = distribution(generator);
}
}  // namespace

std::uint32_t ModelConfig::head_size() const { validate(); return hidden_size / num_heads; }
void ModelConfig::validate() const {
  if (vocab_size == 0 || hidden_size == 0 || intermediate_size == 0 || num_layers == 0 || num_heads == 0 || max_sequence_length == 0 || hidden_size % num_heads != 0) throw std::invalid_argument("invalid decoder configuration");
}
void DecoderModel::validate() const {
  config.validate(); const auto hidden = config.hidden_size; const auto intermediate = config.intermediate_size;
  if (token_embedding.rows() != config.vocab_size || token_embedding.cols() != hidden || language_model_head.rows() != config.vocab_size || language_model_head.cols() != hidden || final_norm.size() != hidden || layers.size() != config.num_layers) throw std::invalid_argument("invalid decoder model dimensions");
  for (const auto& layer : layers) if (layer.attention_norm.size() != hidden || layer.ffn_norm.size() != hidden || layer.query.rows() != hidden || layer.query.cols() != hidden || layer.key.rows() != hidden || layer.key.cols() != hidden || layer.value.rows() != hidden || layer.value.cols() != hidden || layer.attention_output.rows() != hidden || layer.attention_output.cols() != hidden || layer.gate.rows() != intermediate || layer.gate.cols() != hidden || layer.up.rows() != intermediate || layer.up.cols() != hidden || layer.down.rows() != hidden || layer.down.cols() != intermediate) throw std::invalid_argument("invalid decoder layer dimensions");
}
DecoderModel DecoderModel::make_demo() {
  DecoderModel model; model.config = {.vocab_size = 256, .hidden_size = 32, .intermediate_size = 64, .num_layers = 2, .num_heads = 4, .max_sequence_length = 256}; model.token_embedding = DenseMatrix(model.config.vocab_size, model.config.hidden_size); model.language_model_head = DenseMatrix(model.config.vocab_size, model.config.hidden_size); model.final_norm.assign(model.config.hidden_size, 1.0F); std::mt19937 generator(42); fill_random(model.token_embedding, generator, 0.04F); fill_random(model.language_model_head, generator, 0.04F);
  for (std::uint32_t index = 0; index < model.config.num_layers; ++index) { DecoderLayerWeights layer; layer.attention_norm.assign(model.config.hidden_size, 1.0F); layer.ffn_norm.assign(model.config.hidden_size, 1.0F); layer.query = DenseMatrix(model.config.hidden_size, model.config.hidden_size); layer.key = DenseMatrix(model.config.hidden_size, model.config.hidden_size); layer.value = DenseMatrix(model.config.hidden_size, model.config.hidden_size); layer.attention_output = DenseMatrix(model.config.hidden_size, model.config.hidden_size); layer.gate = DenseMatrix(model.config.intermediate_size, model.config.hidden_size); layer.up = DenseMatrix(model.config.intermediate_size, model.config.hidden_size); layer.down = DenseMatrix(model.config.hidden_size, model.config.intermediate_size); fill_random(layer.query, generator, 0.03F); fill_random(layer.key, generator, 0.03F); fill_random(layer.value, generator, 0.03F); fill_random(layer.attention_output, generator, 0.03F); fill_random(layer.gate, generator, 0.03F); fill_random(layer.up, generator, 0.03F); fill_random(layer.down, generator, 0.03F); model.layers.push_back(std::move(layer)); }
  model.validate(); return model;
}
}  // namespace aurora
