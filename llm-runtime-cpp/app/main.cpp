#include "aurora/application/generation_service.hpp"
#include "aurora/infrastructure/aurora_model_repository.hpp"
#include "aurora/infrastructure/cpu_linear_algebra.hpp"
#include "aurora/model.hpp"
#include "aurora/tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(const int argc, char** argv) {
  try {
    std::string prompt = "Aurora"; std::uint32_t tokens = 48; std::string model_path;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--prompt" && index + 1 < argc) prompt = argv[++index];
      else if (argument == "--tokens" && index + 1 < argc) tokens = static_cast<std::uint32_t>(std::stoul(argv[++index]));
      else if (argument == "--model" && index + 1 < argc) model_path = argv[++index];
      else if (argument == "--write-demo" && index + 1 < argc) { aurora::AuroraModelRepository{}.save(aurora::DecoderModel::make_demo(), argv[++index]); std::cout << "Demo model written.\n"; return 0; }
      else if (argument == "--help") { std::cout << "Usage: aurora [--model file.aurora] [--prompt text] [--tokens n] [--write-demo file.aurora]\n"; return 0; }
      else throw std::invalid_argument("unknown or incomplete option: " + argument);
    }
    aurora::ByteTokenizer tokenizer;
    aurora::CpuLinearAlgebra linear_algebra;
    if (model_path.empty()) throw std::invalid_argument("--model is required; use --write-demo to create a validated test model");
    aurora::AuroraModelRepository repository;
    aurora::GenerationService service(repository, tokenizer, linear_algebra);
    std::cout << "[Aurora Runtime] " << service.generate({.model_path = model_path, .prompt = prompt, .tokens_to_generate = tokens}) << '\n';
  } catch (const std::exception& error) {
    std::cerr << "aurora: " << error.what() << '\n'; return 1;
  }
}
