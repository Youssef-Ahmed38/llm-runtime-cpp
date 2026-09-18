#include "aurora/infrastructure/aurora_model_repository.hpp"

#include <array>
#include <fstream>
#include <span>
#include <stdexcept>

namespace aurora {
namespace {
constexpr std::array<char, 8> kMagic{'A', 'U', 'R', 'O', 'R', 'A', '0', '1'};
template <typename T> void write_value(std::ofstream& stream, const T& value) { stream.write(reinterpret_cast<const char*>(&value), sizeof(T)); if (!stream) throw std::runtime_error("failed to write model"); }
template <typename T> T read_value(std::ifstream& stream) { T value{}; stream.read(reinterpret_cast<char*>(&value), sizeof(T)); if (!stream) throw std::runtime_error("failed to read model"); return value; }
void write_vector(std::ofstream& stream, const std::span<const float> values) { write_value(stream, static_cast<std::uint64_t>(values.size())); stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size_bytes())); if (!stream) throw std::runtime_error("failed to write vector"); }
std::vector<float> read_vector(std::ifstream& stream) { const auto size = read_value<std::uint64_t>(stream); if (size > 1'000'000'000ULL) throw std::runtime_error("model vector is implausibly large"); std::vector<float> values(static_cast<std::size_t>(size)); stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float))); if (!stream) throw std::runtime_error("failed to read vector"); return values; }
void write_matrix(std::ofstream& stream, const DenseMatrix& matrix) { write_value(stream, static_cast<std::uint64_t>(matrix.rows())); write_value(stream, static_cast<std::uint64_t>(matrix.cols())); stream.write(reinterpret_cast<const char*>(matrix.data().data()), static_cast<std::streamsize>(matrix.data().size_bytes())); if (!stream) throw std::runtime_error("failed to write matrix"); }
DenseMatrix read_matrix(std::ifstream& stream) { const auto rows = read_value<std::uint64_t>(stream); const auto cols = read_value<std::uint64_t>(stream); if (rows == 0 || cols == 0 || rows > 1'000'000'000ULL || cols > 1'000'000'000ULL || rows > 1'000'000'000ULL / cols) throw std::runtime_error("model matrix shape is invalid"); DenseMatrix matrix(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)); stream.read(reinterpret_cast<char*>(matrix.data().data()), static_cast<std::streamsize>(matrix.data().size_bytes())); if (!stream) throw std::runtime_error("failed to read matrix"); return matrix; }
}  // namespace

void AuroraModelRepository::save(const DecoderModel& model, const std::string& path) const {
  model.validate(); std::ofstream stream(path, std::ios::binary | std::ios::trunc); if (!stream) throw std::runtime_error("could not open model output: " + path);
  stream.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  for (const auto value : {model.config.vocab_size, model.config.hidden_size, model.config.intermediate_size, model.config.num_layers, model.config.num_heads, model.config.max_sequence_length}) write_value(stream, value);
  write_matrix(stream, model.token_embedding); write_vector(stream, model.final_norm); write_matrix(stream, model.language_model_head);
  for (const auto& layer : model.layers) { write_vector(stream, layer.attention_norm); write_vector(stream, layer.ffn_norm); write_matrix(stream, layer.query); write_matrix(stream, layer.key); write_matrix(stream, layer.value); write_matrix(stream, layer.attention_output); write_matrix(stream, layer.gate); write_matrix(stream, layer.up); write_matrix(stream, layer.down); }
}

DecoderModel AuroraModelRepository::load(const std::string& path) const {
  std::ifstream stream(path, std::ios::binary); if (!stream) throw std::runtime_error("could not open model: " + path);
  std::array<char, kMagic.size()> magic{}; stream.read(magic.data(), static_cast<std::streamsize>(magic.size())); if (magic != kMagic) throw std::runtime_error("not an Aurora V1 model");
  DecoderModel model; model.config = {.vocab_size = read_value<std::uint32_t>(stream), .hidden_size = read_value<std::uint32_t>(stream), .intermediate_size = read_value<std::uint32_t>(stream), .num_layers = read_value<std::uint32_t>(stream), .num_heads = read_value<std::uint32_t>(stream), .max_sequence_length = read_value<std::uint32_t>(stream)};
  model.token_embedding = read_matrix(stream); model.final_norm = read_vector(stream); model.language_model_head = read_matrix(stream); model.layers.reserve(model.config.num_layers);
  for (std::uint32_t index = 0; index < model.config.num_layers; ++index) { DecoderLayerWeights layer; layer.attention_norm = read_vector(stream); layer.ffn_norm = read_vector(stream); layer.query = read_matrix(stream); layer.key = read_matrix(stream); layer.value = read_matrix(stream); layer.attention_output = read_matrix(stream); layer.gate = read_matrix(stream); layer.up = read_matrix(stream); layer.down = read_matrix(stream); model.layers.push_back(std::move(layer)); }
  model.validate(); return model;
}

}  // namespace aurora
