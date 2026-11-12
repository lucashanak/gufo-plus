#include "src/models/qwen36_35b_a3b/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <gufo/ext/snapshot_copy.hpp>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen36_35b_a3b/continuation_hooks.hpp"
#include "src/models/qwen36_35b_a3b/kernels/rocm/executor.hpp"

namespace gufo::models::qwen36_35b_a3b {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
// Every field is four bytes wide, so the host image has no padding and the
// serialized metadata is deterministic.
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t target{}, draft{}, hidden_base{}, residual_valid{};
  std::uint32_t suffix_cooldown{}, image_bytes{};
  MtpLengthState policy{};
  std::array<std::uint8_t, 32> image{};
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
}  // namespace

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    this->session->continuation_hooks_ = this;
    this->session->session_->continuation_hooks_ = this;
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    BeforeRelease();
    session->continuation_hooks_ = nullptr;
    session->session_->continuation_hooks_ = nullptr;
  }
  bool IsValid() const noexcept override {
    return !restoring && !failed && (guarding || session->IsValid());
  }
  void BeforeExecution() override {
    if (!IsValid() || pending)
      throw std::logic_error("Qwen3.6 slot is not ready for execution");
  }
  void BeforeWrite(bool draft, std::uint32_t first,
                   std::uint32_t end) override {
    // Execution temporarily clears the session's public validity flag. The
    // adapter's restore/failure status and outstanding transfers still gate
    // every native write, including graph replay and speculative rollback.
    if (restoring || failed || pending)
      throw std::logic_error("Qwen3.6 slot is not ready for mutation");
    if (first >= end)
      return;
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows &&
          IsDraftRows(component.part) == draft)
        guard.BeforeOverwrite(component.descriptor.id, first, end);
  }
  void BeforeReset() override {
    if (invalidating)
      return;
    if (restoring || pending)
      throw std::logic_error("cannot reset a busy Qwen3.6 slot");
    GuardAll();
  }
  void AfterReset() override {
    // The native reset zeroes the recurrent state on the null stream. Settle
    // it before exposing an empty slot to captures on independent streams.
    Check(hipStreamSynchronize(nullptr));
  }
  void BeforeRestore() override {
    BeforeExecution();
    GuardAll();
  }
  void BeforeRelease() noexcept {
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(component.descriptor.id, 0, adapter.context_);
  }
  void GuardAll() {
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(component.descriptor.id, 0, adapter.context_);
  }
  // A capture reads the rows of a stable frontier: no pending speculative
  // batch, and the host token history in step with the device position.
  void Stable() const {
    if (!IsValid() || session->session_->spec_tokens_ != 0 ||
        session->tokens_.size() != session->session_->position_)
      throw std::logic_error(
          "cannot capture an invalid or speculative Qwen3.6 frontier");
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  struct CaptureDuringGuard {
    explicit CaptureDuringGuard(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~CaptureDuringGuard() { state.guarding = previous; }
    State& state;
    bool previous;
  };
  bool restoring{false}, guarding{false}, invalidating{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

struct ContinuationAdapter::Region {
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  std::vector<Piece> pieces;
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         core::SessionMode mode,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      mode_(mode),
      context_(context),
      layout_(model_ ? model_->config() : Config{},
              mode == core::SessionMode::kSpeculative, context,
              model_ ? model_->executor_->max_speculative() : 0),
      identity_(std::move(identity)) {
  if ((mode != core::SessionMode::kAutoregressive &&
       mode != core::SessionMode::kSpeculative) ||
      identity_.empty() || context > model_->MaxContext() ||
      (mode == core::SessionMode::kSpeculative && !model_->HasMtp()))
    throw std::invalid_argument("invalid Qwen3.6 adapter configuration");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  // The component representation, and the inference arithmetic the native
  // payload version tracks: a change to either refuses older checkpoints.
  put(1);
  put(Session::kSnapshotPayloadVersion);
  put(context);
  put(mode == core::SessionMode::kSpeculative);
  put(model_->executor_->max_speculative());
  put(model_->PrefillCapacity());
  put(model_->DecodeConcurrency());
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Qwen3.6 adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.BeforeExecution();
  return *state.session;
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(mode_, context_, &error);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Qwen3.6 slot");
  const auto& native = *state.session->session_;
  return layout_.Positions(native.position_, native.mtp_.position);
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Qwen3.6 prefill plan");
  // The runner's Prefill passes: one Sync per prefill-capacity chunk.
  std::vector<Rows> result;
  while (first < end) {
    first += std::min<Rows>(end - first, model_->PrefillCapacity());
    result.push_back(first);
  }
  return result;
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid Qwen3.6 component ID");
  return layout_.Components()[id.value - 1];
}
void* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  auto& native = *state.session->session_;
  switch (component.part) {
    case ContinuationPart::kTokens:
      return state.session->tokens_.data();
    case ContinuationPart::kKey:
      return native.attention_[component.layer].k_q8;
    case ContinuationPart::kValue:
      return native.attention_[component.layer].v_q8;
    case ContinuationPart::kKeyScale:
      return native.attention_[component.layer].k_scale;
    case ContinuationPart::kValueScale:
      return native.attention_[component.layer].v_scale;
    case ContinuationPart::kDraftKey:
      return native.mtp_.k_q8;
    case ContinuationPart::kDraftValue:
      return native.mtp_.v_q8;
    case ContinuationPart::kDraftKeyScale:
      return native.mtp_.k_scale;
    case ContinuationPart::kDraftValueScale:
      return native.mtp_.v_scale;
    default:
      throw std::invalid_argument("component has no append rows");
  }
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Qwen3.6 restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate Qwen3.6 frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  const auto frontier = [&](ContinuationPart part) -> std::uint32_t {
    for (std::size_t i = 0; i < ordered.size(); ++i)
      if (layout_.Components()[i].part == part) {
        if (ordered[i].valid_rows > context_)
          throw std::invalid_argument("Qwen3.6 frontier exceeds context");
        return static_cast<std::uint32_t>(ordered[i].valid_rows);
      }
    return 0;
  };
  const auto target = frontier(ContinuationPart::kTokens);
  const auto draft = frontier(ContinuationPart::kDraftKey);
  if (ordered != layout_.Positions(target, draft))
    throw std::invalid_argument("inconsistent Qwen3.6 component frontiers");
  // Allocate host capacity first; any refusal leaves execution valid. Guard
  // every range while the original state is still readable.
  auto& session = *state.session;
  session.tokens_.reserve(target);
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  // Rows load straight into the K/V planes: back them before any transfer.
  // Another session's reservation leaves a restoring slot's rows alone (its
  // release guard refuses), so mark the restore first.
  state.restoring = true;
  try {
    std::string error;
    if (!model_->executor_->ReserveKv(*session.session_, target, draft, &error))
      throw std::runtime_error(error);
    // Newly backed chunks are zeroed on the executor stream; settle them so
    // the loads on independent streams land after the fill.
    Check(hipStreamSynchronize(model_->executor_->stream()));
  } catch (...) {
    state.restoring = false;
    throw;
  }
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  session.valid_ = false;
  session.tokens_.resize(target);
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component, bool loading) const {
  auto& session = *state.session;
  auto& native = *session.session_;
  Region result;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false) {
    if (bytes)
      result.pieces.push_back({pointer, 0, bytes, host});
  };
  const auto bytes = component.descriptor.state_bytes;
  switch (component.part) {
    case ContinuationPart::kLogits:
      add(session.logits_.data(), bytes, true);
      break;
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), bytes, true);
      break;
    case ContinuationPart::kConvolution:
      add(native.linear_[component.layer].conv_state, bytes);
      break;
    case ContinuationPart::kRecurrent:
      add(native.linear_[component.layer].state, bytes);
      break;
    case ContinuationPart::kDraftResidual:
      // An invalid residual is never read; it serializes as zeros.
      if (loading || native.mtp_.residual_valid)
        add(native.mtp_.h, bytes);
      break;
    case ContinuationPart::kKeptHidden:
      // Only the leading kept rows are live; the rest serialize as zeros.
      add(native.mtp_.target_hidden,
          loading ? bytes
                  : std::size_t{session.KeptHiddenRows()} *
                        model_->config().hidden_size * sizeof(float));
      break;
    default:
      throw std::invalid_argument("component is not private state");
  }
  return result;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Qwen3.6 private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  state.Stable();
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Qwen3.6 private piece");
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  auto region = PrivateRegion(state, component, false);
  if (component.part == ContinuationPart::kMetadata) {
    auto& session = *state.session;
    const auto& native = *session.session_;
    Metadata metadata{};
    metadata.target = native.position_;
    metadata.draft = native.mtp_.position;
    metadata.hidden_base = native.position_ - session.KeptHiddenRows();
    metadata.residual_valid = native.mtp_.residual_valid;
    metadata.suffix_cooldown =
        static_cast<std::uint32_t>(session.suffix_cooldown_until_);
    metadata.policy = session.draft_length_.State();
    const auto identity = session.ImageIdentity(native.position_);
    metadata.image_bytes = static_cast<std::uint32_t>(identity.size());
    if (!identity.empty()) {
      if (identity.size() != metadata.image.size())
        throw std::logic_error("invalid Qwen3.6 image identity");
      std::ranges::copy(identity, metadata.image.begin());
    }
    std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
    region.pieces = {{host_metadata.data(), 0, host_metadata.size(), true}};
  }
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges. Clear only uncovered bytes, which
        // serialize as zeros: clearing a whole recycled copy on the host
        // costs more than the copy.
        std::vector<std::pair<std::size_t, std::size_t>> covered;
        covered.reserve(region.pieces.size());
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first < end)
            covered.emplace_back(first - offset, end - offset);
        }
        std::ranges::sort(covered);
        std::size_t cleared = 0;
        for (const auto& [first, end] : covered) {
          if (first > cleared)
            std::fill(bytes.begin() + cleared, bytes.begin() + first,
                      std::byte{});
          cleared = std::max(cleared, end);
        }
        std::fill(bytes.begin() + std::min(cleared, bytes.size()), bytes.end(),
                  std::byte{});
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(gufo::hip::CopySnapshotToHost(
                destination, source, end - first, transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Qwen3.6 load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect Qwen3.6 private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Qwen3.6 load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Qwen3.6 private piece");
    const auto region = PrivateRegion(state, component, true);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(gufo::hip::CopySnapshotToDevice(
                  destination, source, end - first, transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Qwen3.6 slot");
  const auto& native = *state.session->session_;
  // A guard may read while execution is mid-step: the host history then
  // trails the device position until the step commits its tokens.
  const std::size_t rows = component.part == ContinuationPart::kTokens
                               ? state.session->tokens_.size()
                           : IsDraftRows(component.part) ? native.mtp_.position
                                                         : native.position_;
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > rows || bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Qwen3.6 row capture");
  auto* source =
      bytes.empty()
          ? nullptr
          : static_cast<std::byte*>(RowData(state, component)) + first * row;
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        if (bytes.empty())
          return;
        if (component.part == ContinuationPart::kTokens)
          std::memcpy(bytes.data(), source, bytes.size());
        else
          Check(gufo::hip::CopySnapshotToHost(bytes.data(), source,
                                              bytes.size(), transfer.Native()));
      });
}
std::span<const std::byte> ContinuationAdapter::BorrowRowsView(
    const cache::Slot& slot, cache::ComponentId id, Rows first,
    Rows end) const {
  const auto& state = As(slot);
  const auto& component = Component(id);
  // The host token vector can move when execution appends. Device rows keep
  // their reserved address range for the session and are guarded before any
  // mutation or release.
  if (component.part == ContinuationPart::kTokens)
    return {};
  if (!state.IsValid() ||
      component.descriptor.kind != cache::ComponentKind::kAppendRows)
    throw std::logic_error("cannot borrow invalid Qwen3.6 rows");
  const auto& native = *state.session->session_;
  const auto rows =
      IsDraftRows(component.part) ? native.mtp_.position : native.position_;
  const auto row = component.descriptor.row_bytes;
  if (first >= end || end > rows)
    throw std::invalid_argument("invalid Qwen3.6 borrowed range");
  return {static_cast<const std::byte*>(
              RowData(const_cast<State&>(state), component)) +
              first * row,
          (end - first) * row};
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Qwen3.6 load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Qwen3.6 row restore");
    auto* destination =
        bytes.empty()
            ? nullptr
            : static_cast<std::byte*>(RowData(state, component)) + first * row;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          if (bytes.empty())
            return;
          if (component.part == ContinuationPart::kTokens)
            std::memcpy(destination, bytes.data(), bytes.size());
          else
            Check(gufo::hip::CopySnapshotToDevice(
                destination, bytes.data(), bytes.size(), transfer.Native()));
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Qwen3.6 validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  auto& session = *state.session;
  auto& native = *session.session_;
  auto policy = session.draft_length_;
  const bool mtp = mode_ == core::SessionMode::kSpeculative;
  const auto kept = mtp ? model_->executor_->max_speculative() : 0;
  if (metadata.version != 1 || metadata.draft > metadata.target ||
      (!mtp && metadata.draft) || metadata.hidden_base > metadata.target ||
      metadata.target - metadata.hidden_base > kept ||
      metadata.residual_valid > 1 ||
      (metadata.residual_valid && !metadata.draft) ||
      (metadata.image_bytes != 0 && metadata.image_bytes != 32) ||
      !policy.Restore(metadata.policy))
    return refuse();
  try {
    if (layout_.Positions(metadata.target, metadata.draft) != state.expected)
      return refuse();
    const auto identity = session.ImageIdentity(metadata.target);
    if (metadata.image_bytes && (identity.size() != 32 ||
                                 !std::ranges::equal(identity, metadata.image)))
      return refuse();
    if (metadata.image_bytes || (session.image_prompt_ && identity.empty())) {
      // Preserve the attached request's future images as well as its matching
      // computed prefix, including checkpoints before the first image.
      native.RestoreVisionLayout(session.image_prompt_->rope,
                                 model_->executor_->stream());
    } else {
      native.ConfigureVision(nullptr, nullptr, model_->executor_->stream());
      session.image_prompt_.reset();
    }
    // The restore replaces the token history: a DFlash2 draft restarts its
    // context from the restored frontier.
    if (session.dflash_) {
      session.dflash_->Reset();
      session.ResetRing(metadata.target);
    }
    Check(hipStreamSynchronize(model_->executor_->stream()));
  } catch (...) {
    return refuse();
  }
  native.position_ = metadata.target;
  native.mtp_.position = metadata.draft;
  native.mtp_.residual_valid = metadata.residual_valid != 0;
  native.spec_base_ = metadata.target;
  native.spec_tokens_ = 0;
  session.hidden_base_ = metadata.hidden_base;
  session.draft_length_ = policy;
  // The history index is rebuilt from the restored tokens on first use.
  session.lookup_ = LookupIndex{};
  session.suffix_cooldown_until_ = metadata.suffix_cooldown;
  session.draft_token_ = 0;
  session.stats_ = {};
  session.valid_ = true;
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    resetting = &state;
    state.BeforeRelease();
    struct ResetScope {
      bool& active;
      ~ResetScope() { active = false; }
    } reset_scope{state.invalidating};
    state.invalidating = true;
    state.session->Reset();
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting)
      resetting->failed = true;
    return false;
  }
}
}  // namespace gufo::models::qwen36_35b_a3b
