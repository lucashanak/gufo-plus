#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "gufo/ext/text.hpp"

namespace gufo::models::qwen36_35b_a3b {
class Model;

#if defined(ENGINE_ENABLE_HIP)
/// The HTTP runner: one session per request state on the snapshot
/// continuation cache, MTP from the artifact's own draft block.
std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

/// A runner over an already loaded model (tests), without persistence.
std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config = {});
#endif

}  // namespace gufo::models::qwen36_35b_a3b
