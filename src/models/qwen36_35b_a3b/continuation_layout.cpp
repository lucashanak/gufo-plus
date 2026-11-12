#include "src/models/qwen36_35b_a3b/continuation_layout.hpp"

#include <limits>
#include <stdexcept>

namespace gufo::models::qwen36_35b_a3b {

ContinuationLayout::ContinuationLayout(const Config& c, bool mtp,
                                       std::uint32_t context,
                                       std::uint32_t kept_rows)
    : context_(context), mtp_(mtp) {
  if (!context || context > c.context_length || !c.num_layers ||
      !c.full_attention_interval || !c.vocab_size || !c.hidden_size ||
      !c.ssm_conv_kernel || !c.AttentionKvDim() ||
      c.AttentionKvDim() % 16 != 0 || (mtp && !kept_rows))
    throw std::invalid_argument("invalid Qwen3.6 continuation geometry");
  const auto add = [&](ContinuationPart part, std::size_t row,
                       std::size_t state, std::uint32_t layer = 0) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    components_.push_back({{{id},
                            1,
                            row ? cache::ComponentKind::kAppendRows
                                : cache::ComponentKind::kPrivateState,
                            row,
                            row ? cache::Rows{256} : 0,
                            state},
                           part,
                           layer});
  };
  const std::size_t kv_row = c.AttentionKvDim();
  const std::size_t q8_row = kv_row * sizeof(std::int8_t);
  const std::size_t scale_row = kv_row / 16 * sizeof(std::uint16_t);
  add(ContinuationPart::kTokens, sizeof(std::int32_t), 0);
  add(ContinuationPart::kLogits, 0, std::size_t{c.vocab_size} * sizeof(float));
  add(ContinuationPart::kMetadata, 0, kContinuationMetadataBytes);
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    if (c.IsLinearLayer(layer)) {
      add(ContinuationPart::kConvolution, 0,
          std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels() *
              sizeof(float),
          layer);
      add(ContinuationPart::kRecurrent, 0,
          std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim *
              sizeof(float),
          layer);
    } else {
      add(ContinuationPart::kKey, q8_row, 0, layer);
      add(ContinuationPart::kValue, q8_row, 0, layer);
      add(ContinuationPart::kKeyScale, scale_row, 0, layer);
      add(ContinuationPart::kValueScale, scale_row, 0, layer);
    }
  }
  if (mtp) {
    add(ContinuationPart::kDraftKey, q8_row, 0);
    add(ContinuationPart::kDraftValue, q8_row, 0);
    add(ContinuationPart::kDraftKeyScale, scale_row, 0);
    add(ContinuationPart::kDraftValueScale, scale_row, 0);
    add(ContinuationPart::kDraftResidual, 0,
        std::size_t{c.hidden_size} * sizeof(float));
    add(ContinuationPart::kKeptHidden, 0,
        std::size_t{kept_rows} * c.hidden_size * sizeof(float));
  }
  // Check the complete retained representation, not just individual tensors.
  const auto per_token = BytesPerToken();
  if (per_token > std::numeric_limits<std::size_t>::max() / context ||
      per_token * context >
          std::numeric_limits<std::size_t>::max() - PrivateBytes())
    throw std::overflow_error("Qwen3.6 continuation payload overflows");
}

std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t target, std::uint32_t draft) const {
  if (target > context_ || draft > target || (!mtp_ && draft))
    throw std::invalid_argument("invalid Qwen3.6 continuation frontier");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& component : components_)
    positions.push_back({component.descriptor.id,
                         IsDraftRows(component.part) ? draft : target});
  return positions;
}

std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.state_bytes;
  return bytes;
}

std::size_t ContinuationLayout::BytesPerToken() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.row_bytes;
  return bytes;
}

std::vector<ContinuationBackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram) const {
  // Shared pages serve rows and private frontiers without stranding one
  // component's reservation while another component is under pressure.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}

}  // namespace gufo::models::qwen36_35b_a3b
