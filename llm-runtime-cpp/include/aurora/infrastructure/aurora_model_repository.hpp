#pragma once

#include "aurora/ports/model_repository.hpp"

namespace aurora {

// Infrastructure adapter for Aurora V1. GGUF and SafeTensors readers will be
// separate adapters and cannot leak their file-format details into the domain.
class AuroraModelRepository final : public IModelRepository {
 public:
  [[nodiscard]] DecoderModel load(const std::string& path) const override;
  void save(const DecoderModel& model, const std::string& path) const override;
};

}  // namespace aurora
