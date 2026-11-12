// Component-cache adapter: checkpoints captured through the adapter restore
// into another slot and continue exactly like the session they came from, at
// a prompt boundary, mid-prompt and mid-decode, in AR and MTP modes. Borrowed
// rows are preserved before a reset overwrites them, and refused restores
// leave a slot that serves cold prompts again.
#include "src/models/qwen36_35b_a3b/continuation_adapter.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"

namespace q36 = gufo::models::qwen36_35b_a3b;
namespace cache = gufo::cache;
namespace sampling = gufo::sampling;

namespace {

constexpr std::uint32_t kContext = 8192;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

template<class F>
void Reject(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid adapter operation was accepted");
}

void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}

std::unique_ptr<gufo::hip::TransferStream> Acquire(
    gufo::hip::TransferPool& streams) {
  auto stream = streams.TryAcquire();
  Require(bool(stream), "transfer stream exhausted");
  return stream;
}

struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};

// Odd piece and row counts make transfers straddle tensor and chunk edges.
constexpr std::size_t kPiece = (1 << 20) + 13;
constexpr cache::Rows kRows = 97;

Checkpoint Capture(q36::ContinuationAdapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& streams) {
  Checkpoint result{adapter.Positions(slot), {}};
  for (const auto& component : adapter.Components()) {
    const auto rows = result.positions[component.id.value - 1].valid_rows;
    auto& bytes = result.bytes.emplace_back(component.row_bytes
                                                ? rows * component.row_bytes
                                                : component.state_bytes);
    if (component.row_bytes) {
      for (cache::Rows first = 0; first < rows;) {
        const auto end = std::min(rows, first + kRows);
        auto stream = Acquire(streams);
        Wait(adapter.CopyRowsOut(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *stream));
        first = end;
      }
    } else {
      for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count = std::min(kPiece, bytes.size() - offset);
        auto stream = Acquire(streams);
        Wait(adapter.CapturePrivatePiece(
            slot, component.id, offset, std::span(bytes).subspan(offset, count),
            *stream));
        offset += count;
      }
    }
  }
  return result;
}

// Loads in reverse component and piece order, all pieces in flight before
// any settles: metadata arrives after the device regions.
void Load(q36::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& streams,
          bool omit_metadata_byte = false) {
  adapter.BeginRestore(slot, checkpoint.positions);
  Require(!slot.IsValid(), "BeginRestore exposed executable state");
  Reject([&] { (void)adapter.GetSession(slot); });
  std::vector<cache::Completion> pending;
  std::vector<std::unique_ptr<gufo::hip::TransferStream>> leases;
  const auto settle = [&] {
    for (auto& completion : pending)
      Wait(std::move(completion));
    pending.clear();
    leases.clear();
  };
  const auto acquire = [&]() -> cache::Stream& {
    auto stream = streams.TryAcquire();
    if (!stream) {
      settle();
      stream = Acquire(streams);
    }
    return *leases.emplace_back(std::move(stream));
  };
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    const std::span<const std::byte> bytes(checkpoint.bytes[i]);
    if (component.row_bytes) {
      for (cache::Rows end = checkpoint.positions[i].valid_rows; end;) {
        const auto first = end > kRows ? end - kRows : 0;
        pending.push_back(adapter.CopyRowsIn(
            slot, component.id, first, end,
            bytes.subspan(first * component.row_bytes,
                          (end - first) * component.row_bytes),
            acquire()));
        end = first;
      }
    } else {
      for (std::size_t end = bytes.size(); end;) {
        const auto first = end > kPiece ? end - kPiece : 0;
        auto count = end - first;
        if (omit_metadata_byte && i == 2 && first == 0)
          --count;
        pending.push_back(adapter.LoadPrivatePiece(
            slot, component.id, first, bytes.subspan(first, count), acquire()));
        end = first;
      }
    }
  }
  settle();
}

// Records overwrites of the rows a checkpoint borrowed and checks that the
// slot still held them, unchanged, when the guard ran.
struct Guard final : cache::MutationGuard {
  q36::ContinuationAdapter* adapter{};
  cache::Slot* slot{};
  gufo::hip::TransferPool* streams{};
  const Checkpoint* borrowed{};
  std::vector<bool> preserved;
  std::size_t resets{};
  bool release_failed{false};
  void Borrow(const Checkpoint& checkpoint) {
    borrowed = &checkpoint;
    preserved.assign(checkpoint.bytes.size(), false);
  }
  void BeforeOverwrite(cache::ComponentId id, cache::Rows first,
                       cache::Rows end) override {
    const auto index = id.value - 1;
    if (!borrowed || preserved[index] || first >= end)
      return;
    const auto rows = borrowed->positions[index].valid_rows;
    const auto& component = adapter->Components()[index];
    if (!component.row_bytes || first >= rows)
      return;
    std::vector<std::byte> bytes(rows * component.row_bytes);
    auto stream = Acquire(*streams);
    Wait(adapter->CopyRowsOut(*slot, id, 0, rows, bytes, *stream));
    Require(bytes == borrowed->bytes[index],
            "borrowed rows overwritten before preservation");
    preserved[index] = true;
  }
  void BeforeRelease(cache::ComponentId id, cache::Rows first,
                     cache::Rows end) noexcept override {
    try {
      BeforeOverwrite(id, first, end);
    } catch (...) {
      release_failed = true;
    }
  }
  void AfterReset() noexcept override { ++resets; }
};

struct Decoded {
  std::vector<std::int32_t> tokens;
  std::uint64_t drafted{}, accepted{};
};

/// Decodes `count` tokens past EOS with a fresh seeded sampler.
Decoded Decode(q36::Session& session, std::size_t count, bool sampled) {
  const auto history = session.Tokens();
  sampling::SamplerState sampler(
      {.temperature = sampled ? 0.8F : 0.0F,
       .top_k = 40,
       .top_p = 0.9F,
       .seed = 7},
      std::vector<sampling::TokenId>(history.begin(), history.end()));
  const auto before = session.Statistics();
  Decoded out;
  std::string error;
  while (out.tokens.size() < count) {
    q36::Session::DecodeResult step;
    Require(session.DecodeStep(count - out.tokens.size(), sampler, &step,
                               &error, false),
            error);
    Require(!step.tokens.empty(), "empty decode step");
    out.tokens.insert(out.tokens.end(), step.tokens.begin(), step.tokens.end());
  }
  const auto after = session.Statistics();
  out.drafted = after.drafted - before.drafted;
  out.accepted = after.accepted - before.accepted;
  return out;
}

void RequireSame(const Decoded& expected, const Decoded& actual,
                 const std::string& what) {
  Require(expected.tokens == actual.tokens, what + ": tokens differ");
  Require(expected.drafted == actual.drafted &&
              expected.accepted == actual.accepted,
          what + ": draft acceptance differs");
}

void SameFrontier(const q36::Session& a, const q36::Session& b,
                  const std::string& what) {
  Require(a.Position() == b.Position() &&
              std::ranges::equal(a.Tokens(), b.Tokens()),
          what + ": token history differs");
  Require(a.Logits().size() == b.Logits().size() &&
              std::memcmp(a.Logits().data(), b.Logits().data(),
                          a.Logits().size_bytes()) == 0,
          what + ": next-token logits differ");
}

void CheckMode(const std::shared_ptr<q36::Model>& model, bool mtp,
               std::span<const std::int32_t> prompt) {
  const std::string mode = mtp ? "MTP" : "AR";
  gufo::hip::TransferPool streams(8);
  q36::ContinuationAdapter adapter(
      model,
      mtp ? gufo::core::SessionMode::kSpeculative
          : gufo::core::SessionMode::kAutoregressive,
      kContext, {'q', '3', '6', 't'});
  Guard source_guard, target_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto target = adapter.CreateSlot(target_guard);
  source_guard.adapter = target_guard.adapter = &adapter;
  source_guard.streams = target_guard.streams = &streams;
  source_guard.slot = source.get();
  target_guard.slot = target.get();
  auto& a = adapter.GetSession(*source);
  auto& b = adapter.GetSession(*target);
  std::string error;

  // Mid-decode, with a drafted MTP frontier: the restored slot continues
  // exactly like the uninterrupted one, greedy and sampled.
  Require(a.Sync(prompt, &error), error);
  (void)Decode(a, 21, false);
  for (const bool sampled : {false, true}) {
    const auto checkpoint = Capture(adapter, *source, streams);
    Load(adapter, *target, checkpoint, streams);
    Require(adapter.Validate(*target, checkpoint.positions),
            mode + ": restore rejected");
    SameFrontier(a, b, mode + " mid-decode restore");
    const auto again = Capture(adapter, *target, streams);
    Require(again.positions == checkpoint.positions &&
                again.bytes == checkpoint.bytes,
            mode + ": a restored slot captures different bytes");
    const auto expected = Decode(a, 48, sampled);
    RequireSame(expected, Decode(b, 48, sampled),
                mode + (sampled ? " sampled" : " greedy") +
                    " continuation after restore");
    if (mtp)
      Require(expected.drafted > 0, mode + ": decode ran no drafts");
    SameFrontier(a, b, mode + " continued restore");
  }

  // Mid-prompt, off the prefill chunk: both slots prefill the rest.
  const auto boundary = model->PrefillCapacity() + 101;
  Require(adapter.Invalidate(*source) && adapter.Invalidate(*target),
          mode + ": reset failed");
  Require(a.Sync(prompt.first(boundary), &error), error);
  const auto partial = Capture(adapter, *source, streams);
  Require(a.Sync(prompt, &error), error);
  Load(adapter, *target, partial, streams);
  Require(adapter.Validate(*target, partial.positions),
          mode + ": mid-prompt restore rejected");
  Require(b.Sync(prompt, &error), error);
  SameFrontier(a, b, mode + " mid-prompt restore");
  RequireSame(Decode(a, 32, false), Decode(b, 32, false),
              mode + " continuation after a mid-prompt restore");

  // Borrowed rows survive appends and are preserved before the reset that
  // overwrites them.
  const auto borrowed = Capture(adapter, *source, streams);
  source_guard.Borrow(borrowed);
  (void)Decode(a, 16, false);
  Require(std::ranges::none_of(source_guard.preserved,
                               [](bool done) { return done; }),
          mode + ": appends overwrote borrowed rows");
  a.Reset();
  for (std::size_t i = 0; i < borrowed.bytes.size(); ++i)
    Require(!adapter.Components()[i].row_bytes ||
                !borrowed.positions[i].valid_rows || source_guard.preserved[i],
            mode + ": reset did not guard borrowed rows of component " +
                std::to_string(i + 1));
  source_guard.borrowed = nullptr;

  // Refusals: inconsistent positions leave the slot executable; a torn
  // private load fails validation, and the reset slot serves cold prompts.
  auto skewed = borrowed.positions;
  skewed.front().valid_rows += 1;
  Reject([&] { adapter.BeginRestore(*target, skewed); });
  (void)adapter.GetSession(*target);
  Load(adapter, *target, borrowed, streams, true);
  Require(!adapter.Validate(*target, borrowed.positions),
          mode + ": an incomplete restore validated");
  Require(adapter.Invalidate(*target), mode + ": failed restore reset");
  Require(target_guard.resets > 0, mode + ": reset skipped AfterReset");
  auto fresh =
      model->CreateSession(mtp ? gufo::core::SessionMode::kSpeculative
                               : gufo::core::SessionMode::kAutoregressive,
                           kContext, &error);
  Require(fresh != nullptr, error);
  const auto other = prompt.first(777);
  Require(b.Sync(other, &error) && fresh->Sync(other, &error), error);
  SameFrontier(b, *fresh, mode + " reused slot");
  RequireSame(Decode(*fresh, 32, false), Decode(b, 32, false),
              mode + " reused slot continuation");
  Require(!source_guard.release_failed && !target_guard.release_failed,
          mode + ": release guard failed");
  Require(adapter.Invalidate(*source) && adapter.Invalidate(*target),
          mode + ": final reset failed");
  std::cout << mode << ",continuation_adapter,PASS\n" << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[1]) != "--model") {
    std::cerr << "Usage: continuation_adapter_test --model MODEL.gguf\n";
    return 77;
  }
  try {
    std::string error;
    auto model = q36::Model::Load(argv[2],
                                  {.max_context = kContext,
                                   .mtp = true,
                                   .max_draft_tokens = 7,
                                   .vision_model_path = {},
                                   .decode_concurrency = 1},
                                  &error);
    Require(model != nullptr, error);
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    // A partial final prefill chunk, as the snapshot test.
    std::vector<std::int32_t> prompt(4095);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    CheckMode(model, false, prompt);
    if (model->HasMtp())
      CheckMode(model, true, prompt);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
