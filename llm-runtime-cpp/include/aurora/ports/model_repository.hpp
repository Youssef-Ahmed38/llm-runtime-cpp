#pragma once

#include "aurora/model.hpp"

#include <string>

namespace aurora {

class IModelRepository {
 public:
  virtual ~IModelRepository() = default;
  [[nodiscard]] virtual DecoderModel load(const std::string& path) const = 0;
  virtual void save(const DecoderModel& model, const std::string& path) const = 0;
};

}  // namespace aurora
