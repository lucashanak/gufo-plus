#include "src/models/qwen36_35b_a3b/probe.hpp"

#include <filesystem>
#include <string>

namespace gufo::models::qwen36_35b_a3b {

bool ProbeQwen36Gguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  return architecture == "qwen35moe";
}

bool ProbeQwen36(std::string_view model_root) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::exists(p)) {
      return false;
    }
    if (std::filesystem::is_regular_file(p)) {
      std::string err;
      auto reader = core::GgufReader::OpenFile(p.string(), &err);
      if (!reader)
        return false;
      return ProbeQwen36Gguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::qwen36_35b_a3b
