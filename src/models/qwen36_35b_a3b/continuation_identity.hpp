#ifndef GUFO_MODELS_QWEN36_35B_A3B_CONTINUATION_IDENTITY_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_CONTINUATION_IDENTITY_HPP_

#include <cstdint>
#include <string_view>
#include <vector>

namespace gufo::models::qwen36_35b_a3b {

// Durable identity of a served model: artifact (which carries the MTP block,
// so it covers the draft weights too), tokenizer, chat template and the
// serving options that change the state a continuation holds.
std::vector<std::uint8_t> Qwen36CompatibilityIdentity(
    std::string_view artifact_fingerprint, bool has_mtp,
    std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency);

// Process-local identity when no disk cache needs artifact fingerprints:
// the loaded model instance is an exact identity for this process.
std::vector<std::uint8_t> Qwen36ProcessIdentity(
    const void* model, std::uint32_t max_context, bool has_mtp,
    std::uint32_t max_draft_tokens, std::uint32_t decode_concurrency);

}  // namespace gufo::models::qwen36_35b_a3b
#endif
