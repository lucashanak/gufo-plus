#ifndef GUFO_MODELS_QWEN36_35B_A3B_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_CONTINUATION_LAYOUT_HPP_

#include <cstdint>
#include <gufo/ext/adapter.hpp>
#include <span>
#include <vector>

#include "src/models/qwen36_35b_a3b/config.hpp"

namespace gufo::models::qwen36_35b_a3b {

// State inventory shared by the adapter and its model-local capacity checks.
// IDs refer to this representation, never to an allocation or serving slot.
// Attention rows hold the session's 8-bit K/V and their F16 scales (one per
// 16 values); the F16 staging the wide kernels read is executor scratch.
enum class ContinuationPart {
  kTokens,
  kLogits,
  kMetadata,
  kConvolution,
  kRecurrent,
  kKey,
  kValue,
  kKeyScale,
  kValueScale,
  kDraftKey,
  kDraftValue,
  kDraftKeyScale,
  kDraftValueScale,
  kDraftResidual,
  kKeptHidden,
};

struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
};

[[nodiscard]] constexpr bool IsDraftRows(ContinuationPart part) noexcept {
  return part == ContinuationPart::kDraftKey ||
         part == ContinuationPart::kDraftValue ||
         part == ContinuationPart::kDraftKeyScale ||
         part == ContinuationPart::kDraftValueScale;
}

// Host metadata has an explicit fixed-size representation: positions, the
// kept hidden frontier, the draft length policy, the history-draft cooldown
// and the input identity. The vision layout is rebuilt from the matching
// request attachment on restoration.
inline constexpr std::size_t kContinuationMetadataBytes = 160;

// Backing classes are the cache's neutral contract (RFC step 6); the layout
// owns the arithmetic that fills them.
using ContinuationBackingClass = cache::BackingClass;

class ContinuationLayout {
public:
  // `kept_rows` is the trunk hidden capacity the draft block reads from
  // (the executor's longest speculative batch); MTP layouts only.
  ContinuationLayout(const Config&, bool mtp, std::uint32_t context,
                     std::uint32_t kept_rows);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  // Trunk components hold `target` rows, draft-block rows `draft`. Private
  // state belongs to the target checkpoint, whatever the draft frontier.
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      std::uint32_t target, std::uint32_t draft) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  [[nodiscard]] std::size_t BytesPerToken() const noexcept;
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes) const;

private:
  std::uint32_t context_;
  bool mtp_;
  std::vector<ContinuationComponent> components_;
};

}  // namespace gufo::models::qwen36_35b_a3b
#endif
