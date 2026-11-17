// Demand-mapped K/V under churn. Eight sessions are grown deep into their
// context, then (a) destroyed and created again, so new regions reserve
// address ranges that released regions just left, and (b) trimmed to nothing
// through their state account and grown again with another prompt, so pages
// are mapped at addresses unmapped moments before. Every result must equal
// the first round's fresh sessions exactly.
//
// state_churn_test --model FIRST.gguf --mtp-model MTP.gguf ROUNDS
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/mem/account.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;
using gufo::core::SessionMode;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

constexpr std::uint32_t kContext = 20000;
constexpr std::uint32_t kLengths[] = {18000, 900,  7300,  12000,
                                      3641,  16500, 5200, 9800};

std::vector<std::int32_t> Prompt(const qfn::Model& model, std::size_t tokens,
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

struct Result {
  std::vector<float> logits;
  std::vector<std::int32_t> tokens;
};

/// Prefills slot `slot`'s prompt and takes one greedy MTP step.
Result Run(qfn::Model& model, qfn::Session& session, std::size_t slot) {
  std::string error;
  const auto prompt =
      Prompt(model, kLengths[slot], "churn " + std::to_string(slot));
  Require(session.Sync(prompt, &error), error);
  sampling::SamplerState sampler(
      {.seed = 5}, std::vector<sampling::TokenId>(prompt.begin(), prompt.end()));
  qfn::Session::DecodeResult step;
  Require(session.DecodeStep(1, sampler, &step, &error, false), error);
  return {{session.Logits().begin(), session.Logits().end()}, step.tokens};
}

void RequireSame(const Result& expected, const Result& actual,
                 const std::string& what) {
  Require(actual.tokens == expected.tokens, what + ": tokens differ");
  Require(expected.logits.size() == actual.logits.size() &&
              std::memcmp(expected.logits.data(), actual.logits.data(),
                          expected.logits.size() * sizeof(float)) == 0,
          what + ": logits differ");
}

std::unique_ptr<qfn::Session> Create(qfn::Model& model) {
  std::string error;
  auto session =
      model.CreateSession(SessionMode::kSpeculative, kContext, &error);
  Require(session != nullptr, error);
  Require(session->StateAccount() != nullptr, "session has no state account");
  return session;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr << "Usage: state_churn_test --model FIRST.gguf --mtp-model "
                 "MTP.gguf ROUNDS\n";
    return 77;
  }
  try {
    const auto rounds = static_cast<unsigned>(std::stoul(argv[5]));
    std::string error;
    auto model = qfn::Model::Load(argv[2],
                                  {.max_context = kContext,
                                   .mtp_model_path = argv[4],
                                   .decode_concurrency = 8},
                                  &error);
    Require(model != nullptr, error);
    std::vector<Result> expected(8);
    for (unsigned round = 0; round < rounds; ++round) {
      // (a) New sessions in address ranges the previous round released.
      std::vector<std::unique_ptr<qfn::Session>> sessions;
      for (std::size_t i = 0; i < 8; ++i) {
        sessions.push_back(Create(*model));
        const auto slot = (i + round) % 8;
        auto result = Run(*model, *sessions.back(), slot);
        if (round == 0)
          expected[slot] = std::move(result);
        else
          RequireSame(expected[slot], result,
                      "churn round " + std::to_string(round) + " slot " +
                          std::to_string(slot));
      }
      // (b) Each session drops every page, then grows again with another
      // prompt: its pages are mapped where they were just unmapped, and
      // recycled pages arrive from its peers.
      for (unsigned pass = 0; pass < 2; ++pass) {
        std::size_t released = 0;
        for (std::size_t i = 0; i < 8; ++i) {
          Require(hipDeviceSynchronize() == hipSuccess, "device sync");
          sessions[i]->Reset();
          released += sessions[i]->StateAccount()->TrimTo(0);
          const auto slot = (i + round + pass + 3) % 8;
          RequireSame(expected[slot], Run(*model, *sessions[i], slot),
                      "regrow round " + std::to_string(round) + " pass " +
                          std::to_string(pass) + " slot " +
                          std::to_string(slot));
        }
        Require(released > 0, "trims released nothing");
        std::cout << "round " << round << " pass " << pass << ": trimmed "
                  << (released >> 20) << " MiB\n";
      }
      std::cout << "round " << round
                << ": eight new sessions and sixteen regrowths exact\n"
                << std::flush;
    }
    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
