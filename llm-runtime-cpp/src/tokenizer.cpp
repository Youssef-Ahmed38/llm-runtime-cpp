#include "aurora/tokenizer.hpp"

namespace aurora {

std::vector<std::uint32_t> ByteTokenizer::encode(const std::string& text) const {
  std::vector<std::uint32_t> tokens;
  tokens.reserve(text.size());
  for (const unsigned char character : text) tokens.push_back(character);
  return tokens;
}

std::string ByteTokenizer::decode_token(const std::uint32_t token) const {
  if (token >= kByteVocabularySize) return "";
  return std::string(1, static_cast<char>(token));
}

std::string ByteTokenizer::decode(const std::vector<std::uint32_t>& tokens) const {
  std::string result;
  result.reserve(tokens.size());
  for (const auto token : tokens) result += decode_token(token);
  return result;
}

}  // namespace aurora
