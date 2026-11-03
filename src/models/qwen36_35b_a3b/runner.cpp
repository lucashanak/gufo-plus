#include "src/models/qwen36_35b_a3b/runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/continuation.hpp"
#include "gufo/ext/gguf_identity.hpp"
#include "gufo/ext/host_budget.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/families/qwen/vision/encoder.hpp"
#include "src/families/qwen/vision/image_cache.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "src/models/qwen36_35b_a3b/engine.hpp"

namespace gufo::models::qwen36_35b_a3b {
namespace {

using namespace gufo::ext;

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

// Qwen3.6-35B-A3B carries its MTP block in the main artifact, so the
// artifact identity covers the draft weights too.
std::vector<std::uint8_t> Qwen36CompatibilityIdentity(
    std::string_view artifact_fingerprint, bool has_mtp,
    std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.6-35B-A3B disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen36-35b-a3b\n"
           << "artifact_id=" << ext::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen36-compiled-v1\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::Qwen36OfficialTemplateSha256()
           << '\n'
           << "state_abi=qwen36-35b-a3b-rocm-session-v1\n"
           << "payload_layout=q36-rocm-session-snapshot-v"
           << Session::kSnapshotPayloadVersion << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=q36-native-mtp-v1\n"
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

struct QwenImageContext final : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

tokenization::ChatTemplateOptions QwenChatOptions(
    const ChatRequest& request, std::uint32_t max_context,
    tokenization::QwenTemplateFamily family) {
  auto options = tokenization::ResolveQwenChatOptions(
      request.reasoning, request.add_vision_id, family);
  // gufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

std::optional<std::size_t> CountQwenPromptTokens(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache,
    tokenization::QwenTemplateFamily family) {
  const auto options = QwenChatOptions(request, max_context, family);
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  if (std::ranges::none_of(request.messages,
                           [](const auto& m) { return !m.images.empty(); })) {
    const auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
        tokenizer, request.messages, tools, options);
    if (!tokens)
      return std::nullopt;
    return tokens->size();
  }
  return models::qwen::vision::Prepare(
             tokenizer, request.messages, tools, options,
             encoder ? encoder->identity() : std::string_view{},
             std::numeric_limits<std::int32_t>::max(),
             request.cache_prompt ? &image_cache : nullptr)
      .tokens.size();
}

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache,
    tokenization::QwenTemplateFamily family) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  const auto options = QwenChatOptions(request, max_context, family);
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages,
          request.tool_choice == ChatRequest::ToolChoice::kNone
              ? std::span<const tokenization::ChatTool>{}
              : std::span<const tokenization::ChatTool>{request.tools},
          options,
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context, request.cache_prompt ? &image_cache : nullptr));
  const auto cache_prefix = prompt->stable_prefix_tokens;
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  for (const auto& image : prompt->images) {
    const auto identity = prompt->IdentityForPrefix(image.grid.offset);
    context->cache_prefixes.push_back(
        {image.grid.offset, {identity.begin(), identity.end()}});
  }
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

// Qwen3.6-35B-A3B (NinjaPear engine): one runner per request state on the
// snapshot continuation cache, as the Qwen 27B runner.
struct Qwen36RunnerTraits {
  using Model = models::qwen36_35b_a3b::Model;
  using Session = models::qwen36_35b_a3b::Session;
  using Snapshot = models::qwen36_35b_a3b::SessionSnapshot;
  static constexpr std::string_view kName = "Qwen3.6-35B-A3B";
  static constexpr std::string_view kShortName = "Qwen3.6";
  static constexpr std::string_view kStateAbi =
      "qwen36-35b-a3b-rocm-session-v1";
  static constexpr auto kTemplateFamily =
      tokenization::QwenTemplateFamily::kQwen36;
};

template<class Traits>
std::vector<std::int32_t> QwenMoeEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(std::string(Traits::kName) +
                                  " token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

template<class Traits>
class QwenMoeTextRunnerState final : public TextRunnerState {
public:
  QwenMoeTextRunnerState(const std::shared_ptr<typename Traits::Model>& model,
                         std::uint32_t max_context, bool use_mtp) {
    std::string error;
    session_ =
        model->CreateSession(use_mtp ? gufo::core::SessionMode::kSpeculative
                                     : gufo::core::SessionMode::kAutoregressive,
                             max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create " +
                               std::string(Traits::kName) +
                               " session: " + error);
    }
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Reset();
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    // K/V rows grow with the tokens held and are bounded by the model's K/V
    // budget; the claim charges each state a share of it, not its rows.
    return {.per_request_state_bytes =
                session_->AllocatedBytes() - session_->KvBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] typename Traits::Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<typename Traits::Session> session_;
  std::size_t position_{0};
};

template<class Traits>
class QwenMoeTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  QwenMoeTextRunnerSnapshot(std::shared_ptr<typename Traits::Model> model,
                            std::unique_ptr<typename Traits::Snapshot> snapshot,
                            std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<typename Traits::Model> model;
  std::unique_ptr<typename Traits::Snapshot> snapshot;
  std::size_t position;
};

template<class Traits>
QwenMoeTextRunnerState<Traits>& RequireQwenMoeState(TextRunnerState& state) {
  auto* qfn = dynamic_cast<QwenMoeTextRunnerState<Traits>*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not " +
                           std::string(Traits::kName));
  }
  return *qfn;
}

template<class Traits>
const QwenMoeTextRunnerState<Traits>& RequireQwenMoeState(
    const TextRunnerState& state) {
  const auto* qfn = dynamic_cast<const QwenMoeTextRunnerState<Traits>*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not " +
                           std::string(Traits::kName));
  }
  return *qfn;
}

template<class Traits>
class QwenMoeTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  QwenMoeTextRunner(
      std::shared_ptr<typename Traits::Model> model, std::uint32_t max_context,
      bool use_mtp, std::uint32_t max_draft_tokens,
      std::optional<TextRunnerPersistenceDescriptor> persistence = std::nullopt)
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens),
        persistence_(std::move(persistence)) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(Traits::kStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width = use_mtp_ ? 8u : 0u,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
        .sampling_preset = sampling::TextModelPreset::kQwen36,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Snapshots live in host memory, not in the device state pool.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(
            use_mtp_ ? gufo::core::SessionMode::kSpeculative
                     : gufo::core::SessionMode::kAutoregressive,
            max_context_),
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->tokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->tokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_, Traits::kTemplateFamily));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->tokenizer(),
                             model_->VisionEncoder(), max_context_,
                             image_cache_, Traits::kTemplateFamily);
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->tokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_, Traits::kTemplateFamily);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenMoeState<Traits>(state).session().ConfigureVision(
        QwenPrompt(context));
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_, Traits::kTemplateFamily)
                   .enable_thinking
               ? ext::InitialOutputState::kReasoning
               : ext::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<QwenMoeTextRunnerState<Traits>>(
        model_, max_context_, use_mtp_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& qfn = RequireQwenMoeState<Traits>(state);
    if (qfn.position() != prefix.size()) {
      throw std::logic_error(std::string(Traits::kName) +
                             " reused prefix does not match checkpoint");
    }
    qfn.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qfn = RequireQwenMoeState<Traits>(state);
    if (offset != qfn.position()) {
      throw std::logic_error(std::string(Traits::kName) +
                             " prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error(std::string(Traits::kName) +
                             " prefill has no remaining input");
    }
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, model_->PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix =
        QwenMoeEngineTokens<Traits>(prompt.first(next_position));
    std::string error;
    if (!qfn.session().Sync(prefix, &error)) {
      qfn.set_position(0);
      throw std::runtime_error(std::string(Traits::kName) +
                               " prefill failed: " + error);
    }
    qfn.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qfn = RequireQwenMoeState<Traits>(state);
    if (qfn.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = qfn.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error(std::string(Traits::kName) +
                               " token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (qfn.stop_at_eos() && model_->IsStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(std::string(Traits::kName) +
                                  " token ID exceeds engine range");
    }
    auto& qfn = RequireQwenMoeState<Traits>(state);
    std::string error;
    if (!qfn.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error(std::string(Traits::kName) +
                               " decode failed: " + error);
    }
    qfn.set_position(qfn.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          std::string(Traits::kName) +
          " MTP decode budget must be at least one token");
    }
    auto& qfn = RequireQwenMoeState<Traits>(state);
    if (qfn.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = qfn.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    typename Traits::Session::DecodeResult decoded;
    std::string error;
    const auto budget =
        std::min<std::size_t>(max_tokens, std::uint64_t{max_draft_tokens_} + 1);
    if (!qfn.session().DecodeStep(budget, working_sampler, &decoded, &error,
                                  qfn.stop_at_eos())) {
      throw std::runtime_error(std::string(Traits::kName) +
                               " MTP decode failed: " + error);
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    step.stop = decoded.stop;
    step.selections.reserve(decoded.tokens.size());
    for (const std::int32_t token : decoded.tokens) {
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
    qfn.set_position(qfn.session().Position());
    const auto stats_after = qfn.session().Statistics();
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<typename Traits::Session::BatchOutcome> outcomes(
        advances.size());
    std::vector<typename Traits::Session::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(Traits::kShortName) +
                                    " token exceeds engine range");
      }
      requests.push_back(
          {&RequireQwenMoeState<Traits>(advance.state.get()).session(),
           static_cast<std::int32_t>(advance.token),
           &outcomes[requests.size()]});
    }
    std::string error;
    (void)Traits::Session::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(
            std::runtime_error(std::string(Traits::kShortName) +
                               " advance failed: " + outcomes[i].error));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& state = RequireQwenMoeState<Traits>(advance.state.get());
      state.set_position(state.session().Position());
    }
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<typename Traits::Session::DecodeResult> results(count);
    std::vector<typename Traits::Session::BatchOutcome> outcomes(count);
    std::vector<typename Traits::Session::SpeculativeStats> before;
    std::vector<typename Traits::Session::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& state = RequireQwenMoeState<Traits>(decodes[i].state.get());
      auto& session = state.session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back(
          {&session,
           std::min<std::size_t>(decodes[i].max_tokens,
                                 std::uint64_t{max_draft_tokens_} + 1),
           &sampler, &results[i], state.stop_at_eos(), &outcomes[i]});
    }
    std::string error;
    (void)Traits::Session::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(
            std::runtime_error(std::string(Traits::kShortName) +
                               " MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& state = RequireQwenMoeState<Traits>(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      step.stop = results[i].stop;
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                               .physical_width = active_count};
      }
      for (const auto token : results[i].tokens) {
        step.selections.push_back({.stop = false,
                                   .token = static_cast<TextRunnerToken>(token),
                                   .piece = model_->TokenText(token)});
      }
      const auto stats = state.session().Statistics();
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenMoeState<Traits>(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    const std::uint64_t bytes =
        RequireQwenMoeState<Traits>(state).session().SnapshotBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error(std::string(Traits::kName) +
                                " snapshot size is unavailable");
    }
    return static_cast<std::size_t>(bytes);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& qfn = RequireQwenMoeState<Traits>(state);
    std::string error;
    auto snapshot = qfn.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error(std::string(Traits::kName) +
                               " snapshot failed: " + error);
    }
    return std::make_unique<QwenMoeTextRunnerSnapshot<Traits>>(
        model_, std::move(snapshot), qfn.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenMoeTextRunnerSnapshot<Traits>*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(std::string(Traits::kName) +
                                  " snapshot does not belong to this model");
    }
    auto& restored = RequireQwenMoeState<Traits>(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*qfn_snapshot->snapshot, &error)) {
      restored.Invalidate();
      throw std::runtime_error(std::string(Traits::kName) +
                               " snapshot restore failed: " + error);
    }
    restored.set_position(qfn_snapshot->position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenMoeTextRunnerSnapshot<Traits>*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          std::string(Traits::kName) +
          " persistent snapshot does not belong to this "
          "model");
    }
    return qfn_snapshot->PayloadBytes();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenMoeTextRunnerSnapshot<Traits>*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr ||
        destination.size() != qfn_snapshot->PayloadBytes() ||
        !qfn_snapshot->snapshot->CopyTo(destination)) {
      throw std::invalid_argument(std::string(Traits::kName) +
                                  " persistent snapshot serialization failed");
    }
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    (void)PersistentSnapshotPayloadBytes(snapshot);
    sink(dynamic_cast<const QwenMoeTextRunnerSnapshot<Traits>&>(snapshot)
             .snapshot->bytes());
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireQwenMoeState<Traits>(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      restored.Invalidate();
      throw std::runtime_error(std::string(Traits::kName) +
                               " persistent snapshot restore failed: " + error);
    }
    const std::uint32_t position = restored.session().Position();
    if (position == 0 || position > max_context_) {
      restored.Invalidate();
      throw std::runtime_error(std::string(Traits::kName) +
                               " persistent snapshot restored an invalid "
                               "position");
    }
    restored.set_position(position);
  }

private:
  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<typename Traits::Model> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

using Qwen36TextRunner = QwenMoeTextRunner<Qwen36RunnerTraits>;
}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config) {
  const bool use_mtp =
      speculative_config.backend == ext::TextSpeculativeBackend::kMtp;
  if (use_mtp && !model->HasMtp())
    throw std::invalid_argument(
        "Qwen3.6-35B-A3B MTP requires a model loaded with its MTP block");
  return std::make_shared<Qwen36TextRunner>(
      std::move(model), max_context, use_mtp,
      speculative_config.max_draft_tokens);
}

std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error)
      *error = std::move(message);
    return std::shared_ptr<ext::TextModelRunner>{};
  };
  std::shared_ptr<const core::GgufReader> reader = options.reader;
  if (!reader) {
    std::string load_error;
    auto owner = core::GgufReader::OpenFile(options.model_path, &load_error);
    if (!owner)
      return fail("Failed to open GGUF: " + load_error);
    reader = std::shared_ptr<const core::GgufReader>(std::move(owner));
  }
  if (reader->GetMetadataString("general.architecture").value_or("") !=
      "qwen35moe")
    return fail("Qwen3.6-35B-A3B requires architecture qwen35moe");
  if (options.enable_yarn)
    return fail("--enable-yarn applies only to Qwen3.8-Flash-Next");
  const auto& speculative = options.speculative;
  const bool use_mtp = speculative.backend == ext::TextSpeculativeBackend::kMtp;
  if (speculative.backend != ext::TextSpeculativeBackend::kDisabled && !use_mtp)
    return fail(
        "Qwen3.6-35B-A3B HTTP models support only MTP speculative decoding "
        "(--speculative mtp)");
  if (use_mtp && !speculative.draft_model_path.empty())
    return fail(
        "Qwen3.6-35B-A3B MTP is native to the model artifact; omit "
        "--mtp-model");
  if (!options.vision_model_path.empty())
    return fail("Qwen3.6-35B-A3B serves text only; omit --mmproj");
  if (use_mtp &&
      (speculative.max_draft_tokens == 0 || speculative.min_draft_tokens != 1))
    return fail(
        "Qwen3.6-35B-A3B MTP drafts a confidence-gated chain; custom draft "
        "floors are unsupported");
  std::string load_error;
  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            &load_error))
    return fail("Unsupported Qwen3.6-35B-A3B chat template: " + load_error);
  std::uint32_t context = options.max_context;
  if (context == 0) {
    const std::uint64_t native =
        reader->GetMetadataUint64("qwen35moe.context_length").value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max())
      return fail("GGUF has no valid native context length; specify --context");
    context = static_cast<std::uint32_t>(native);
  }
  auto model =
      Model::Load(options.model_path,
                  ModelOptions{
                      .max_context = context,
                      .mtp = use_mtp,
                      .max_draft_tokens = speculative.max_draft_tokens,
                      .decode_concurrency = static_cast<std::uint32_t>(
                          std::clamp<std::size_t>(options.session_count, 1, 8)),
                      .kv_cache_bytes = options.kv_cache_bytes,
                  },
                  &load_error);
  if (!model)
    return fail("Failed to create Qwen3.6-35B-A3B model: " + load_error);
  const std::uint32_t max_context = context;
  if (max_context == 0 || max_context > model->MaxContext())
    return fail(
        "HTTP context exceeds the loaded Qwen3.6-35B-A3B model context");
  if (use_mtp && !model->HasMtp())
    return fail(
        "Qwen3.6-35B-A3B MTP requires a model loaded with its MTP block");
  std::optional<TextRunnerPersistenceDescriptor> persistence;
  if (!options.disk_cache.directory.empty()) {
    std::string fingerprint = options.disk_cache.model_artifact_fingerprint;
    if (fingerprint.empty()) {
      try {
        fingerprint = core::GgufIdentityHex(*reader);
      } catch (const std::exception& e) {
        return fail(
            std::string("Qwen3.6-35B-A3B disk cache fingerprinting failed: ") +
            e.what());
      }
    }
    try {
      persistence = TextRunnerPersistenceDescriptor{
          .compatibility_identity = Qwen36CompatibilityIdentity(
              fingerprint, use_mtp, max_context, speculative.max_draft_tokens,
              model->DecodeConcurrency()),
          .payload_version = Session::kSnapshotPayloadVersion,
      };
    } catch (const std::exception& e) {
      return fail(e.what());
    }
  }
  return std::make_shared<Qwen36TextRunner>(
      std::move(model), max_context, use_mtp, speculative.max_draft_tokens,
      std::move(persistence));
}

}  // namespace gufo::models::qwen36_35b_a3b

#endif  // ENGINE_ENABLE_HIP
