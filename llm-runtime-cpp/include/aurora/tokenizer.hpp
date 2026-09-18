#pragma once

#include "aurora/ports/tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace aurora {

// A deterministic byte tokenizer. It is intentionally dependency-free and makes the
// runtime usable with any UTF-8 prompt; production model adapters can replace it.
class ByteTokenizer final : public ITokenizer {
 public:
  static constexpr std::uint32_t kByteVocabularySize = 256;
  [[nodiscard]] std::vector<std::uint32_t> encode(const std::string& text) const override;
  [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& tokens) const override;
  [[nodiscard]] std::string decode_token(std::uint32_t token) const;
};

}  // namespace aurora
