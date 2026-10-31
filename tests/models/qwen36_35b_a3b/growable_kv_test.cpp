// K/V rows that grow with the tokens a session holds. A fresh session maps
// nothing until it writes; its results must equal a session whose rows were
// already backed by an earlier, longer request, across chunk boundaries
// (2 MiB: kv_row 8-bit rows, the F16 staging rows and the scale rows), MTP
// verification and rollback, snapshot restore, cancellation and batched
// decoding. A small budget then has to refuse growth cleanly, leave its peers
// intact, and reuse the rows an idle session no longer holds.
//
// growable_kv_test --model MODEL.gguf [--batch-only]
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen36_35b_a3b/engine.hpp"

namespace q36 = gufo::models::qwen36_35b_a3b;
namespace sampling = gufo::sampling;
using gufo::core::SessionMode;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual, const std::string& message) {
  Require(!expected.empty() && expected.size() == actual.size() &&
              std::memcmp(expected.data(), actual.data(),
                          expected.size_bytes()) == 0,
          message);
}

constexpr std::uint32_t kContext = 40000;
constexpr std::uint32_t kChunk = 2U << 20;

/// First rows of the second chunk of each growable buffer kind.
struct Boundaries {
  std::uint32_t staging;  ///< F16 staging rows (2 bytes per value)
  std::uint32_t values;   ///< 8-bit K/V rows
  std::uint32_t scales;   ///< F16 scale rows (one per 16 values)
};

Boundaries ChunkRows(const q36::Model& model) {
  const std::uint32_t kv_row = model.config().AttentionKvDim();
  return {kChunk / (2 * kv_row), kChunk / kv_row, kChunk / (kv_row / 8)};
}

std::vector<std::int32_t> Prompt(const q36::Model& model, std::size_t tokens,
                                 const std::string& salt) {
  const auto pattern = model.Tokenize(
      salt +
      ": the ledger lists each shipment, its weight in kilograms and the "
      "port that received it. Rotterdam 412, Gdansk 88, Valencia 1301. "
      "def total(rows):\n    return sum(r['kg'] for r in rows)\n");
  std::vector<std::int32_t> prompt(tokens);
  for (std::size_t i = 0; i < tokens; ++i)
    prompt[i] = pattern[i % pattern.size()];
  return prompt;
}

std::unique_ptr<q36::Session> Create(q36::Model& model,
                                     std::uint32_t context = kContext) {
  std::string error;
  auto session =
      model.CreateSession(SessionMode::kSpeculative, context, &error);
  Require(session != nullptr, error);
  return session;
}

/// A session whose rows an earlier request already backed to `tokens`.
std::unique_ptr<q36::Session> Pregrown(q36::Model& model,
                                       std::uint32_t tokens) {
  auto session = Create(model);
  std::string error;
  Require(session->Sync(Prompt(model, tokens, "warm"), &error), error);
  session->Reset();
  return session;
}

/// Syncs both sessions through the same prefixes (identical pass
/// boundaries) and compares the logits after each.
void SyncBoth(q36::Session& fresh, q36::Session& reference,
              std::span<const std::int32_t> prompt,
              std::span<const std::uint32_t> ends, const char* what) {
  std::string error;
  for (const auto end : ends) {
    Require(fresh.Sync(prompt.first(end), &error), error);
    Require(reference.Sync(prompt.first(end), &error), error);
    RequireExact(reference.Logits(), fresh.Logits(),
                 std::string(what) + ": prefill logits differ at " +
                     std::to_string(end));
  }
}

/// Greedy MTP decoding of both sessions, compared token by token.
void DecodeBoth(q36::Session& fresh, q36::Session& reference, std::size_t steps,
                const char* what) {
  std::string error;
  const auto history = [](const q36::Session& s) {
    return std::vector<sampling::TokenId>(s.Tokens().begin(), s.Tokens().end());
  };
  sampling::SamplerState a({.seed = 7}, history(fresh));
  sampling::SamplerState b({.seed = 7}, history(reference));
  const auto drafted = fresh.Statistics().drafted;
  const auto reference_drafted = reference.Statistics().drafted;
  for (std::size_t i = 0; i < steps; ++i) {
    q36::Session::DecodeResult x, y;
    Require(fresh.DecodeStep(8, a, &x, &error, false), error);
    Require(reference.DecodeStep(8, b, &y, &error, false), error);
    Require(x.tokens == y.tokens, std::string(what) + ": tokens differ");
    RequireExact(reference.Logits(), fresh.Logits(),
                 std::string(what) + ": decode logits differ");
  }
  Require(fresh.Statistics().drafted - drafted ==
              reference.Statistics().drafted - reference_drafted,
          std::string(what) + ": drafts differ");
}

std::vector<std::uint8_t> Bytes(const q36::Session& session) {
  std::string error;
  const auto snapshot = session.SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  return {snapshot->bytes().begin(), snapshot->bytes().end()};
}

void CheckGrowth(q36::Model& model) {
  const auto b = ChunkRows(model);
  const std::uint32_t last = b.scales + 1200;
  auto reference = Pregrown(model, last + 200);
  auto fresh = Create(model);
  const auto empty = fresh->AllocatedBytes();
  Require(fresh->KvBytes() == 0, "a fresh session already backs K/V rows");
  Require(reference->KvBytes() > (100ULL << 20),
          "a pregrown session does not back its K/V rows");
  const auto prompt = Prompt(model, last, "growth");
  // Passes ending just before, on and after the staging, value and scale
  // chunk boundaries.
  const std::uint32_t ends[] = {b.staging - 3,  b.staging + 1, b.values - 2,
                                b.values + 1,   9000,          b.scales - 20,
                                b.scales + 300, last};
  SyncBoth(*fresh, *reference, prompt, ends, "growth");
  DecodeBoth(*fresh, *reference, 12, "growth");
  Require(Bytes(*fresh) == Bytes(*reference), "grown state bytes differ");
  Require(fresh->AllocatedBytes() > empty + (50ULL << 20),
          "growth did not back the written rows");
  std::cout << "growth across staging, value and scale chunks: logits, tokens "
               "and state exact; backed "
            << fresh->KvBytes() / (1 << 20) << " MiB of K/V rows\n";
}

void CheckRollbackAtBoundary(q36::Model& model) {
  const auto b = ChunkRows(model);
  auto reference = Pregrown(model, 2 * b.values);
  auto fresh = Create(model);
  // Verification batches of up to eight rows straddle the value chunk;
  // rejected drafts are cut back across it.
  const auto prompt = Prompt(model, b.values - 5, "rollback");
  const std::uint32_t ends[] = {b.values - 5};
  SyncBoth(*fresh, *reference, prompt, ends, "rollback");
  DecodeBoth(*fresh, *reference, 16, "rollback");
  Require(fresh->Position() > b.values, "decoding did not cross the chunk");
  Require(Bytes(*fresh) == Bytes(*reference), "rollback state bytes differ");
  std::cout << "MTP verification and rollback across a chunk: exact, drafted "
            << fresh->Statistics().drafted << " accepted "
            << fresh->Statistics().accepted << '\n';
}

void CheckRestore(q36::Model& model) {
  std::string error;
  auto source = Create(model);
  Require(source->Sync(Prompt(model, 9000, "restore"), &error), error);
  const auto snapshot = source->SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  auto fresh = Create(model);
  Require(fresh->KvBytes() == 0, "a fresh session already backs K/V rows");
  Require(fresh->RestoreSnapshot(*snapshot, &error), error);
  Require(fresh->KvBytes() > 0, "restore did not back its rows");
  RequireExact(source->Logits(), fresh->Logits(), "restored logits differ");
  const auto restored = Bytes(*fresh);
  Require(restored == Bytes(*source), "restored state bytes differ");
  // A longer session restores the same snapshot over rows it already backs.
  auto longer = Pregrown(model, 15000);
  Require(longer->RestoreSnapshot(*snapshot, &error), error);
  Require(Bytes(*longer) == restored, "restore over backed rows differs");
  DecodeBoth(*fresh, *source, 8, "restore");
  // The session restored over backed rows continues like one restored into
  // an empty session.
  auto again = Create(model);
  Require(again->RestoreSnapshot(*snapshot, &error), error);
  DecodeBoth(*longer, *again, 8, "restore over backed rows");
  std::cout << "restore into a session with no backed rows: exact\n";
}

void CheckCancellation(q36::Model& model) {
  std::string error;
  const auto prompt = Prompt(model, 17500, "cancel");
  auto reference = Create(model);
  int checks = 0;
  reference->SetCancellationCheck([&] { return ++checks < 0; });
  Require(reference->Sync(prompt, &error), error);
  reference->SetCancellationCheck({});
  // Cancel late in the prompt, after several passes have grown the rows.
  const int total = checks;
  auto fresh = Create(model);
  checks = 0;
  fresh->SetCancellationCheck([&] { return ++checks > total * 3 / 4; });
  Require(!fresh->Sync(prompt, &error), "the long prefill was not cancelled");
  Require(fresh->KvBytes() > 0, "the cancelled prefill backed no rows");
  fresh->SetCancellationCheck({});
  fresh->Reset();
  Require(fresh->Sync(prompt, &error), error);
  RequireExact(reference->Logits(), fresh->Logits(),
               "prefill after a cancelled growth differs");
  DecodeBoth(*fresh, *reference, 4, "cancellation");
  std::cout << "cancellation mid-growth, then a full prefill: exact\n";
}

void CheckBatch(q36::Model& model) {
  std::string error;
  const auto b = ChunkRows(model);
  // Lengths around the chunk boundaries and short ones; batched decoding
  // grows each member independently. Both cohorts decode batched, so their
  // draft policies match; only the backing of their rows differs.
  const std::uint32_t lengths[] = {
      100, b.values - 4, b.staging - 3, 5000, b.values + 3000, 300, 4090, 9000};
  std::vector<std::unique_ptr<q36::Session>> pregrown, growing;
  std::vector<sampling::SamplerState> pregrown_samplers, growing_samplers;
  for (std::size_t i = 0; i < 8; ++i) {
    const auto prompt = Prompt(model, lengths[i], "batch " + std::to_string(i));
    pregrown.push_back(Pregrown(model, 12000));
    growing.push_back(Create(model));
    Require(pregrown.back()->Sync(prompt, &error), error);
    Require(growing.back()->Sync(prompt, &error), error);
    const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
    pregrown_samplers.emplace_back(sampling::SamplingConfig{.seed = 11},
                                   history);
    growing_samplers.emplace_back(sampling::SamplingConfig{.seed = 11},
                                  history);
  }
  for (unsigned cycle = 0; cycle < 8; ++cycle) {
    std::vector<q36::Session::DecodeResult> x(8), y(8);
    std::vector<q36::Session::DecodeRequest> a, c;
    for (std::size_t i = 0; i < 8; ++i) {
      a.push_back({pregrown[i].get(), 8, &pregrown_samplers[i], &x[i], false});
      c.push_back({growing[i].get(), 8, &growing_samplers[i], &y[i], false});
    }
    Require(q36::Session::DecodeBatch(a, &error), error);
    Require(q36::Session::DecodeBatch(c, &error), error);
    for (std::size_t i = 0; i < 8; ++i) {
      Require(std::ranges::equal(pregrown[i]->Tokens(), growing[i]->Tokens()),
              "growing batch changed tokens of member " + std::to_string(i) +
                  " in cycle " + std::to_string(cycle));
      RequireExact(
          pregrown[i]->Logits(), growing[i]->Logits(),
          "growing batch changed logits of member " + std::to_string(i));
    }
  }
  Require(growing[1]->Position() > b.values &&
              growing[2]->Position() > b.staging &&
              growing[6]->Position() > b.values,
          "batched members did not cross their chunk boundaries");
  std::cout << "eight concurrent sessions of mixed lengths: tokens and logits "
               "exact\n";
}

void CheckBudget(const char* path, std::span<const float> expected,
                 std::uint32_t tokens, std::size_t per_session,
                 std::size_t full_claim) {
  std::string error;
  // Room for two sessions of `tokens` tokens and not a third.
  const std::size_t budget = per_session * 5 / 2;
  auto model = q36::Model::Load(path,
                                {.max_context = kContext,
                                 .mtp = true,
                                 .decode_concurrency = 4,
                                 .kv_cache_bytes = budget},
                                &error);
  Require(model != nullptr, error);
  // The startup claim charges a share of the budget, not the full context.
  const auto claim = model->SessionBytes(SessionMode::kSpeculative, kContext);
  Require(claim < full_claim && claim + per_session > budget / 4,
          "a budgeted session claims " + std::to_string(claim) + " of " +
              std::to_string(full_claim) + " bytes");
  const auto prompt = Prompt(*model, tokens, "budget");
  auto a = Create(*model);
  auto b = Create(*model);
  auto c = Create(*model);
  Require(a->Sync(prompt, &error), error);
  RequireExact(expected, a->Logits(), "budgeted prefill differs");
  Require(b->Sync(prompt, &error), error);
  const auto b_snapshot = b->SaveSnapshot(&error);
  Require(b_snapshot != nullptr, error);
  Require(!c->Sync(prompt, &error), "growth past the budget succeeded");
  Require(error.find("K/V memory exhausted") != std::string::npos,
          "unexpected budget error: " + error);
  // The refusal touched nothing else: a keeps decoding as before.
  sampling::SamplerState sampler(
      {.seed = 3},
      std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  q36::Session::DecodeResult step;
  Require(a->DecodeStep(8, sampler, &step, &error, false), error);
  // b goes idle: c reuses the rows b no longer holds.
  b->Reset();
  Require(c->Sync(prompt, &error), error);
  RequireExact(expected, c->Logits(), "prefill in released rows differs");
  // b comes back from its snapshot once c has gone idle in turn.
  c->Reset();
  Require(b->RestoreSnapshot(*b_snapshot, &error), error);
  RequireExact(expected, b->Logits(), "restore into released rows differs");
  std::cout << "budget " << budget / (1 << 20)
            << " MiB: third session refused cleanly, peers intact, idle rows "
               "reused, snapshot restored into released rows\n";
}

}  // namespace

int main(int argc, char** argv) {
  const bool batch_only =
      argc == 4 && std::string_view(argv[3]) == "--batch-only";
  if ((argc != 3 && !batch_only) || std::string_view(argv[1]) != "--model") {
    std::cerr << "Usage: growable_kv_test --model MODEL.gguf [--batch-only]\n";
    return 77;
  }
  try {
    std::string error;
    std::vector<float> budget_logits;
    std::size_t budget_session = 0;
    std::size_t full_claim = 0;
    constexpr std::uint32_t kBudgetTokens = 6000;
    {
      auto model = q36::Model::Load(
          argv[2],
          {.max_context = kContext, .mtp = true, .decode_concurrency = 8},
          &error);
      Require(model != nullptr, error);
      const auto b = ChunkRows(*model);
      std::cout << "chunk rows: staging " << b.staging << ", values "
                << b.values << ", scales " << b.scales << '\n';
      if (!batch_only) {
        CheckGrowth(*model);
        CheckRollbackAtBoundary(*model);
        CheckRestore(*model);
        CheckCancellation(*model);
      }
      CheckBatch(*model);
      auto reference = Create(*model);
      Require(reference->Sync(Prompt(*model, kBudgetTokens, "budget"), &error),
              error);
      budget_logits.assign(reference->Logits().begin(),
                           reference->Logits().end());
      budget_session = reference->KvBytes();
      full_claim = model->SessionBytes(SessionMode::kSpeculative, kContext);
    }
    CheckBudget(argv[2], budget_logits, kBudgetTokens, budget_session,
                full_claim);
    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
