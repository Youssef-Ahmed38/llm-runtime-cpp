#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aurora {

class ITokenizer {
 public:
  virtual ~ITokenizer() = default;
  [[nodiscard]] virtual std::vector<std::uint32_t> encode(const std::string& text) const = 0;
  [[nodiscard]] virtual std::string decode(const std::vector<std::uint32_t>& tokens) const = 0;
};

}  // namespace aurora
