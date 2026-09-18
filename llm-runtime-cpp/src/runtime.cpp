#include "aurora/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace aurora {
KvCache::KvCache(const ModelConfig& config) : config_(config) { config_.validate(); const auto total = static_cast<std::size_t>(config_.num_layers) * config_.max_sequence_length * config_.hidden_size; keys_.resize(total); values_.resize(total); }
void KvCache::append(const std::uint32_t layer, const std::span<const float> key, const std::span<const float> value) { if (layer >= config_.num_layers || key.size() != config_.hidden_size || value.size() != config_.hidden_size || length_ >= config_.max_sequence_length) throw std::out_of_range("invalid KV-cache append"); const auto offset = (static_cast<std::size_t>(layer) * config_.max_sequence_length + length_) * config_.hidden_size; std::copy(key.begin(), key.end(), keys_.begin() + static_cast<std::ptrdiff_t>(offset)); std::copy(value.begin(), value.end(), values_.begin() + static_cast<std::ptrdiff_t>(offset)); }
std::span<const float> KvCache::key(const std::uint32_t layer, const std::uint32_t position) const { if (layer >= config_.num_layers || position >= length_) throw std::out_of_range("KV-cache key out of range"); const auto offset = (static_cast<std::size_t>(layer) * config_.max_sequence_length + position) * config_.hidden_size; return {keys_.data() + offset, config_.hidden_size}; }
std::span<const float> KvCache::value(const std::uint32_t layer, const std::uint32_t position) const { if (layer >= config_.num_layers || position >= length_) throw std::out_of_range("KV-cache value out of range"); const auto offset = (static_cast<std::size_t>(layer) * config_.max_sequence_length + position) * config_.hidden_size; return {values_.data() + offset, config_.hidden_size}; }
void KvCache::advance() { if (length_ >= config_.max_sequence_length) throw std::out_of_range("maximum context reached"); ++length_; }
Sampler::Sampler(const SamplingConfig config) : config_(config), generator_(config.seed) { if (config_.temperature <= 0.0F || config_.top_p <= 0.0F || config_.top_p > 1.0F) throw std::invalid_argument("invalid sampling configuration"); }
std::uint32_t Sampler::sample(const std::span<const float> logits) {
  if (logits.empty()) throw std::invalid_argument("cannot sample empty logits");
  candidates_.resize(logits.size());
  std::iota(candidates_.begin(), candidates_.end(), 0U);
  const auto top_k = std::min<std::size_t>(config_.top_k == 0 ? logits.size() : config_.top_k, logits.size());
  std::partial_sort(candidates_.begin(), candidates_.begin() + static_cast<std::ptrdiff_t>(top_k), candidates_.end(),
                    [&logits](const auto left, const auto right) { return logits[left] > logits[right]; });
  candidates_.resize(top_k);
  float maximum = -std::numeric_limits<float>::infinity();
  for (const auto token : candidates_) maximum = std::max(maximum, logits[token] / config_.temperature);
  probabilities_.resize(top_k);
  float normalizer = 0.0F;
  for (std::size_t index = 0; index < top_k; ++index) { probabilities_[index] = std::exp(logits[candidates_[index]] / config_.temperature - maximum); normalizer += probabilities_[index]; }
  for (float& probability : probabilities_) probability /= normalizer;
  probability_order_.resize(top_k);
  std::iota(probability_order_.begin(), probability_order_.end(), 0U);
  std::sort(probability_order_.begin(), probability_order_.end(), [this](const auto left, const auto right) { return probabilities_[left] > probabilities_[right]; });
  float cumulative = 0.0F; std::size_t keep = 0;
  for (const auto index : probability_order_) { cumulative += probabilities_[index]; ++keep; if (cumulative >= config_.top_p) break; }
  sampling_weights_.resize(keep);
  for (std::size_t index = 0; index < keep; ++index) sampling_weights_[index] = probabilities_[probability_order_[index]];
  std::discrete_distribution<std::size_t> distribution(sampling_weights_.begin(), sampling_weights_.end());
  return candidates_[probability_order_[distribution(generator_)]];
}
DecoderRuntime::Workspace::Workspace(const ModelConfig& config) : state(config.hidden_size), normalized(config.hidden_size), query(config.hidden_size), key(config.hidden_size), value(config.hidden_size), attention(config.hidden_size), projected(config.hidden_size), gate(config.intermediate_size), up(config.intermediate_size), logits(config.vocab_size), scores(config.max_sequence_length) {}
DecoderRuntime::DecoderRuntime(const DecoderModel& model, const ILinearAlgebra& linear_algebra) : model_(model), linear_algebra_(linear_algebra), cache_(model.config), workspace_(model.config) { model_.validate(); }
void DecoderRuntime::reset() { cache_.clear(); }
std::span<const float> DecoderRuntime::decode(const std::uint32_t token) {
  const auto& config = model_.config; if (token >= config.vocab_size) throw std::out_of_range("token outside vocabulary"); if (cache_.length() >= config.max_sequence_length) throw std::out_of_range("maximum context reached"); const auto hidden = static_cast<std::size_t>(config.hidden_size); const auto embedding = model_.token_embedding.data().subspan(static_cast<std::size_t>(token) * hidden, hidden); std::copy(embedding.begin(), embedding.end(), workspace_.state.begin()); const auto head_size = static_cast<std::size_t>(config.head_size());
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers; ++layer_index) { const auto& layer = model_.layers[layer_index]; rms_norm(workspace_.state, layer.attention_norm, workspace_.normalized); linear_algebra_.matvec(layer.query, workspace_.normalized, workspace_.query); linear_algebra_.matvec(layer.key, workspace_.normalized, workspace_.key); linear_algebra_.matvec(layer.value, workspace_.normalized, workspace_.value); cache_.append(layer_index, workspace_.key, workspace_.value); std::fill(workspace_.attention.begin(), workspace_.attention.end(), 0.0F); const auto positions = cache_.length() + 1U;
    for (std::uint32_t head = 0; head < config.num_heads; ++head) { const auto begin = static_cast<std::size_t>(head) * head_size; float maximum = -std::numeric_limits<float>::infinity(); for (std::uint32_t position = 0; position < positions; ++position) { const auto cached_key = position == cache_.length() ? std::span<const float>(workspace_.key) : cache_.key(layer_index, position); float dot = 0.0F; for (std::size_t item = 0; item < head_size; ++item) dot += workspace_.query[begin + item] * cached_key[begin + item]; workspace_.scores[position] = dot / std::sqrt(static_cast<float>(head_size)); maximum = std::max(maximum, workspace_.scores[position]); } float normalizer = 0.0F; for (std::uint32_t position = 0; position < positions; ++position) { workspace_.scores[position] = std::exp(workspace_.scores[position] - maximum); normalizer += workspace_.scores[position]; } for (std::uint32_t position = 0; position < positions; ++position) { const auto cached_value = position == cache_.length() ? std::span<const float>(workspace_.value) : cache_.value(layer_index, position); const float probability = workspace_.scores[position] / normalizer; for (std::size_t item = 0; item < head_size; ++item) workspace_.attention[begin + item] += probability * cached_value[begin + item]; } }
    linear_algebra_.matvec(layer.attention_output, workspace_.attention, workspace_.projected); add_inplace(workspace_.state, workspace_.projected); rms_norm(workspace_.state, layer.ffn_norm, workspace_.normalized); linear_algebra_.matvec(layer.gate, workspace_.normalized, workspace_.gate); linear_algebra_.matvec(layer.up, workspace_.normalized, workspace_.up); silu_multiply_inplace(workspace_.gate, workspace_.up); linear_algebra_.matvec(layer.down, workspace_.gate, workspace_.projected); add_inplace(workspace_.state, workspace_.projected); }
  cache_.advance(); rms_norm(workspace_.state, model_.final_norm, workspace_.normalized); linear_algebra_.matvec(model_.language_model_head, workspace_.normalized, workspace_.logits); return workspace_.logits;
}
std::vector<float> DecoderRuntime::forward(const std::uint32_t token) { const auto logits = decode(token); return {logits.begin(), logits.end()}; }
std::vector<std::uint32_t> DecoderRuntime::generate(const std::span<const std::uint32_t> prompt, const std::uint32_t tokens_to_generate, Sampler& sampler) { if (prompt.empty()) throw std::invalid_argument("prompt must contain at least one token"); std::span<const float> logits; for (const auto token : prompt) logits = decode(token); std::vector<std::uint32_t> result; result.reserve(tokens_to_generate); for (std::uint32_t index = 0; index < tokens_to_generate; ++index) { const auto token = sampler.sample(logits); result.push_back(token); logits = decode(token); } return result; }
}  // namespace aurora
