#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen36_35b_a3b/kernels/rocm/kernels.hpp"
#include "src/models/qwen36_35b_a3b/kernels/rocm/mmq/q36_mmq.h"
#include "tests/models/qwen36_35b_a3b/quant_fixtures.hpp"

namespace {

// The model's routing: 512 experts, top-10 without replacement.
constexpr int kExperts = 512;
constexpr int kUsed = 10;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

/// Top-k ids per token without replacement, skewed toward low experts so
/// the buckets are uneven; every 50th token drops one slot (-1).
std::vector<std::int32_t> MakeIds(int n_tokens, std::uint32_t seed) {
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n_tokens) * kUsed);
  for (int t = 0; t < n_tokens; ++t) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < kUsed) {
      const std::uint32_t r = NextRandom(&seed);
      const int e = static_cast<int>((r % 4 == 0 ? r / 4 : r / 8) % kExperts);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    if (t % 50 == 7) {
      chosen[3] = -1;
    }
    std::copy(chosen.begin(), chosen.end(),
              ids.begin() + static_cast<std::size_t>(t) * kUsed);
  }
  return ids;
}

struct Maps {
  std::vector<std::int32_t> src1;
  std::vector<std::int32_t> dst;
  std::vector<std::int32_t> bounds;
};

/// llama.cpp's mm_ids_helper contract on the host: rows sorted by expert
/// then assignment order; a negative id is counted below expert 0 and its
/// slot left zero; an id past the experts is dropped.
Maps Reference(const std::vector<std::int32_t>& ids, int n_tokens, int n_used,
               int si1, int sis1) {
  Maps m;
  const std::size_t rows = static_cast<std::size_t>(n_tokens) * n_used;
  m.src1.assign(rows, 0);
  m.dst.assign(rows, 0);
  m.bounds.assign(kExperts + 1, 0);
  std::vector<int> counts(kExperts, 0);
  int negatives = 0;
  for (int t = 0; t < n_tokens; ++t) {
    for (int i = 0; i < n_used; ++i) {
      const std::int32_t e = ids[static_cast<std::size_t>(t) * si1 + i];
      if (e < 0) {
        ++negatives;
      } else if (e < kExperts) {
        ++counts[e];
      }
    }
  }
  int running = negatives;
  for (int e = 0; e < kExperts; ++e) {
    m.bounds[e] = running;
    running += counts[e];
  }
  m.bounds[kExperts] = running;
  std::vector<int> cursor(m.bounds.begin(), m.bounds.end() - 1);
  for (int t = 0; t < n_tokens; ++t) {
    for (int i = 0; i < n_used; ++i) {
      const std::int32_t e = ids[static_cast<std::size_t>(t) * si1 + i];
      if (e < 0 || e >= kExperts) {
        continue;
      }
      const int pos = cursor[e]++;
      m.src1[pos] = t * sis1 + i % 1;
      m.dst[pos] = t * n_used + i;
    }
  }
  return m;
}

Maps Build(const std::vector<std::int32_t>& ids, int n_tokens, int n_used,
           int si1, int sis1) {
  const std::size_t rows = static_cast<std::size_t>(n_tokens) * n_used;
  std::int32_t* d_ids = nullptr;
  std::int32_t* d_src1 = nullptr;
  std::int32_t* d_dst = nullptr;
  std::int32_t* d_bounds = nullptr;
  CheckHip(hipMalloc(&d_ids, ids.size() * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_src1, rows * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_dst, rows * 4), "hipMalloc");
  CheckHip(hipMalloc(&d_bounds, (kExperts + 1) * 4), "hipMalloc");
  CheckHip(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice),
           "upload");
  CheckHip(hipMemset(d_src1, 0, rows * 4), "memset");
  CheckHip(hipMemset(d_dst, 0, rows * 4), "memset");
  if (q36_mmq_build_ids_maps(d_ids, d_src1, d_dst, d_bounds, kExperts, n_tokens,
                             n_used, 1, si1, sis1, nullptr) != 0) {
    throw std::runtime_error("id map build failed");
  }
  CheckHip(hipDeviceSynchronize(), "id maps");
  Maps m;
  m.src1.resize(rows);
  m.dst.resize(rows);
  m.bounds.resize(kExperts + 1);
  CheckHip(hipMemcpy(m.src1.data(), d_src1, rows * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(m.dst.data(), d_dst, rows * 4, hipMemcpyDeviceToHost),
           "download");
  CheckHip(hipMemcpy(m.bounds.data(), d_bounds, (kExperts + 1) * 4,
                     hipMemcpyDeviceToHost),
           "download");
  (void)hipFree(d_ids);
  (void)hipFree(d_src1);
  (void)hipFree(d_dst);
  (void)hipFree(d_bounds);
  return m;
}

bool Same(const Maps& a, const Maps& b, const char* what) {
  const bool ok = a.src1 == b.src1 && a.dst == b.dst && a.bounds == b.bounds;
  std::cout << what << ": " << (ok ? "identical" : "DIFFER") << " ("
            << a.bounds.back() << " routed rows)\n";
  if (!ok) {
    for (std::size_t i = 0; i < a.bounds.size(); ++i) {
      if (a.bounds[i] != b.bounds[i]) {
        std::cout << "  first bounds mismatch at " << i << ": " << a.bounds[i]
                  << " vs " << b.bounds[i] << '\n';
        break;
      }
    }
    for (std::size_t i = 0; i < a.src1.size(); ++i) {
      if (a.src1[i] != b.src1[i] || a.dst[i] != b.dst[i]) {
        std::cout << "  first map mismatch at " << i << ": src1 " << a.src1[i]
                  << " vs " << b.src1[i] << ", dst " << a.dst[i] << " vs "
                  << b.dst[i] << '\n';
        break;
      }
    }
  }
  return ok;
}

// Check the router independently of the assignment-map builder: softmax,
// selection without replacement, lowest-index ties, and renormalization.
void CheckRouter(int tokens, int experts, int used, int pattern) {
  namespace q = gufo::models::qwen36_35b_a3b::rocm;
  const int stride = experts + 1;
  const std::size_t count = static_cast<std::size_t>(tokens) * used;
  constexpr std::int32_t kIdGuard = -771;
  constexpr float kWeightGuard = -71.0F;
  std::vector<float> logits(static_cast<std::size_t>(tokens) * stride,
                            std::numeric_limits<float>::quiet_NaN());
  std::uint32_t random = 0xBADC0FFEU;
  for (int t = 0; t < tokens; ++t) {
    for (int e = 0; e < experts; ++e) {
      const auto bits = NextRandom(&random);
      float value =
          static_cast<float>(static_cast<int>(bits % 16384) - 8192) / 1024.0F;
      if (pattern == 1) {
        value = static_cast<float>(bits % 8);
      } else if (pattern == 2) {
        value = 0.0F;
      } else if (pattern == 3 && e % 3 == 0) {
        value = -std::numeric_limits<float>::infinity();
      }
      logits[static_cast<std::size_t>(t) * stride + e] = value;
    }
  }
  float* d_logits = nullptr;
  float* d_weights = nullptr;
  std::int32_t* d_ids = nullptr;
  CheckHip(hipMalloc(&d_logits, logits.size() * sizeof(float)),
           "router logits");
  CheckHip(hipMalloc(&d_weights, (count + 2) * sizeof(float)),
           "router weights");
  CheckHip(hipMalloc(&d_ids, (count + 2) * sizeof(std::int32_t)), "router ids");
  const std::vector<float> weight_guards(count + 2, kWeightGuard);
  const std::vector<std::int32_t> id_guards(count + 2, kIdGuard);
  CheckHip(hipMemcpy(d_logits, logits.data(), logits.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "upload logits");
  CheckHip(
      hipMemcpy(d_weights, weight_guards.data(),
                weight_guards.size() * sizeof(float), hipMemcpyHostToDevice),
      "upload weight guards");
  CheckHip(
      hipMemcpy(d_ids, id_guards.data(),
                id_guards.size() * sizeof(std::int32_t), hipMemcpyHostToDevice),
      "upload id guards");
  std::vector<float> first_weights;
  std::vector<std::int32_t> first_ids;
  for (int replay = 0; replay < 2; ++replay) {
    q::RouterTopK(d_logits, stride, d_ids + 1, d_weights + 1, tokens, experts,
                  used, nullptr);
    CheckHip(hipDeviceSynchronize(), "router");
    std::vector<float> weights(count + 2);
    std::vector<std::int32_t> ids(count + 2);
    CheckHip(hipMemcpy(weights.data(), d_weights,
                       weights.size() * sizeof(float), hipMemcpyDeviceToHost),
             "download router weights");
    CheckHip(hipMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t),
                       hipMemcpyDeviceToHost),
             "download router ids");
    if (weights.front() != kWeightGuard || weights.back() != kWeightGuard ||
        ids.front() != kIdGuard || ids.back() != kIdGuard) {
      throw std::runtime_error("router overwrote output guards");
    }
    if (replay != 0) {
      if (ids != first_ids ||
          !std::equal(weights.begin(), weights.end(), first_weights.begin(),
                      [](float a, float b) {
                        return std::bit_cast<std::uint32_t>(a) ==
                               std::bit_cast<std::uint32_t>(b);
                      })) {
        throw std::runtime_error("router replay differs");
      }
      continue;
    }
    first_weights = weights;
    first_ids = ids;
    for (int t = 0; t < tokens; ++t) {
      const float* row = logits.data() + static_cast<std::size_t>(t) * stride;
      const float maximum = *std::max_element(row, row + experts);
      std::vector<double> probabilities(experts);
      double denominator = 0.0;
      for (int e = 0; e < experts; ++e) {
        probabilities[e] = std::exp(static_cast<double>(row[e]) - maximum);
        denominator += probabilities[e];
      }
      for (double& probability : probabilities) {
        probability /= denominator;
      }
      std::vector<int> order(experts);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + used, order.end(),
                        [&](int a, int b) {
                          return probabilities[a] == probabilities[b]
                                     ? a < b
                                     : probabilities[a] > probabilities[b];
                        });
      double sum = 0.0;
      for (int i = 0; i < used; ++i) {
        sum += probabilities[order[i]];
      }
      sum = std::max(sum, 6.103515625e-5);
      for (int i = 0; i < used; ++i) {
        const auto at = 1 + static_cast<std::size_t>(t) * used + i;
        const double expected = probabilities[order[i]] / sum;
        if (ids[at] != order[i] || !std::isfinite(weights[at]) ||
            std::abs(weights[at] - expected) > 2e-6) {
          throw std::runtime_error("router differs from softmax reference");
        }
      }
    }
  }
  (void)hipFree(d_logits);
  (void)hipFree(d_weights);
  (void)hipFree(d_ids);
}

namespace fx = gufo::models::qwen36_35b_a3b::test;

template<typename T>
T* DeviceCopy(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

std::vector<std::uint8_t> Bytes(const void* device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  CheckHip(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost),
           "download");
  return host;
}

// The decode MoE prologue (top-k, routed Q8_1 input and shared gate/up in
// one launch) must write exactly what RouterTopK, the tier's
// quantize_q8_1 and DenseQ6KVec write: ids, weights, Q8_1 bytes and SwiGLU
// rows, byte for byte, at widths 1-8. The model's routing: 256 experts,
// top-8; hidden 2048, shared expert 512.
void CheckPrologue(int pattern) {
  namespace q = gufo::models::qwen36_35b_a3b::rocm;
  constexpr int kRows = 8, kRouted = 256, kTop = 8, kStride = kRouted + 1;
  constexpr int kHidden = 2048, kShared = 512;
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  std::uint32_t random = 0x9E3779B9U + static_cast<std::uint32_t>(pattern);
  std::vector<float> logits(static_cast<std::size_t>(kRows) * kStride);
  for (int t = 0; t < kRows; ++t) {
    for (int e = 0; e < kStride; ++e) {
      const auto bits = NextRandom(&random);
      float v =
          static_cast<float>(static_cast<int>(bits % 16384) - 8192) / 1024.0F;
      if (pattern == 1) {
        v = static_cast<float>(bits % 8);  // many exact ties
      } else if (pattern == 2) {
        v = 0.0F;
      } else if (pattern == 3 && e % 3 == 0) {
        v = -kInf;
      } else if (pattern == 4) {
        v = static_cast<float>(bits % 8) * 1e-41F;  // subnormal logits
      } else if (pattern == 5) {
        v *= 1e30F;
      } else if (pattern == 6 && e == 7 * t + 3) {
        v = t % 3 == 0 ? kNan : t % 3 == 1 ? kInf : -kInf;
      }
      logits[static_cast<std::size_t>(t) * kStride + e] = v;
    }
  }
  // Hidden rows: ordinary, zero, small, subnormal, huge, zero blocks,
  // nonfinite elements, and exact rounding ties (amax 127 gives d = 1).
  std::vector<float> x(static_cast<std::size_t>(kRows) * kHidden);
  for (int t = 0; t < kRows; ++t) {
    for (int i = 0; i < kHidden; ++i) {
      float v = fx::Uniform(&random, 2.0F);
      switch ((t + pattern) % kRows) {
        case 1:
          v = 0.0F;
          break;
        case 2:
          v /= 128.0F;
          break;
        case 3:
          v *= 1e-39F;
          break;
        case 4:
          v *= 1e30F;
          break;
        case 5:
          v = (i / 32) % 2 ? 0.0F : 2.0F * v;
          break;
        case 6:
          if (i == 100)
            v = kInf;
          if (i == 1000)
            v = kNan;
          if (i == 1500)
            v = -kInf;
          break;
        case 7: {
          constexpr float kTies[] = {127.0F, 2.5F, -3.5F, 0.5F, -0.5F, 63.5F};
          v = kTies[(i * 7 + t) % 6];
          break;
        }
        default:
          break;
      }
      x[static_cast<std::size_t>(t) * kHidden + i] = v;
    }
  }
  auto gate = fx::MakeQ6K(1, kShared, kHidden, 0x6A7E0001U);
  auto up = fx::MakeQ6K(1, kShared, kHidden, 0x6A7E0002U);
  auto* d_gate = DeviceCopy(gate.packed);
  auto* d_up = DeviceCopy(up.packed);
  if (!q::RepackQ6KRows(d_gate, kShared, kHidden) ||
      !q::RepackQ6KRows(d_up, kShared, kHidden))
    throw std::runtime_error("Q6_K repack rejected the shared expert");
  auto* d_logits = DeviceCopy(logits);
  auto* d_x = DeviceCopy(x);
  constexpr std::size_t kGuard = 256;
  const std::size_t id_bytes = kRows * kTop * sizeof(std::int32_t) + kGuard;
  const std::size_t q8_bytes = q36_mmq_q8_1_bytes(kRows, kHidden) + kGuard;
  const std::size_t y_bytes = kRows * kShared * sizeof(float) + kGuard;
  void* ref[4] = {};
  void* got[4] = {};
  const std::size_t sizes[4] = {id_bytes, id_bytes, q8_bytes, y_bytes};
  for (int i = 0; i < 4; ++i) {
    CheckHip(hipMalloc(&ref[i], sizes[i]), "reference output");
    CheckHip(hipMalloc(&got[i], sizes[i]), "prologue output");
  }
  const auto run = [&](int n, int first) {
    for (int i = 0; i < 4; ++i) {
      CheckHip(hipMemset(ref[i], 0xA5, sizes[i]), "guard");
      CheckHip(hipMemset(got[i], 0xA5, sizes[i]), "guard");
    }
    const float* lg = d_logits + static_cast<std::size_t>(first) * kStride;
    const float* xr = d_x + static_cast<std::size_t>(first) * kHidden;
    q::RouterTopK(lg, kStride, static_cast<std::int32_t*>(ref[0]),
                  static_cast<float*>(ref[1]), n, kRouted, kTop, nullptr);
    if (q36_mmq_quantize_q8_1(xr, ref[2], n, kHidden, nullptr) != 0 ||
        !q::DenseQ6KVec(d_gate, xr, static_cast<float*>(ref[3]), n, kShared,
                        kHidden, nullptr, d_up) ||
        !q::MoeSharedPrologue(lg, kStride, static_cast<std::int32_t*>(got[0]),
                              static_cast<float*>(got[1]), kRouted, kTop, xr,
                              got[2], d_gate, d_up, static_cast<float*>(got[3]),
                              n, kShared, kHidden, nullptr))
      throw std::runtime_error("prologue reference launch failed");
    CheckHip(hipDeviceSynchronize(), "prologue");
    const char* names[4] = {"ids", "weights", "Q8_1 bytes", "shared SwiGLU"};
    const std::size_t used_bytes[4] = {
        static_cast<std::size_t>(n) * kTop * sizeof(std::int32_t),
        static_cast<std::size_t>(n) * kTop * sizeof(float),
        q36_mmq_q8_1_bytes(n, kHidden),
        static_cast<std::size_t>(n) * kShared * sizeof(float)};
    for (int i = 0; i < 4; ++i) {
      const auto a = Bytes(ref[i], sizes[i]);
      const auto b = Bytes(got[i], sizes[i]);
      const auto diff = std::mismatch(a.begin(), a.end(), b.begin());
      if (diff.first != a.end()) {
        throw std::runtime_error(
            std::string("prologue ") + names[i] + " differ at byte " +
            std::to_string(diff.first - a.begin()) + ", pattern " +
            std::to_string(pattern) + ", rows " + std::to_string(first) + ".." +
            std::to_string(first + n - 1));
      }
      for (std::size_t j = used_bytes[i]; j < sizes[i]; ++j)
        if (b[j] != 0xA5)
          throw std::runtime_error(std::string("prologue ") + names[i] +
                                   " overwrote its guard");
    }
  };
  for (int first = 0; first < kRows; ++first)
    run(1, first);
  for (int n = 2; n <= kRows; ++n) {
    run(n, 0);
    run(n, kRows - n);
  }
  for (int i = 0; i < 4; ++i) {
    CheckHip(hipFree(ref[i]), "free");
    CheckHip(hipFree(got[i]), "free");
  }
  for (void* p : {static_cast<void*>(d_gate), static_cast<void*>(d_up),
                  static_cast<void*>(d_logits), static_cast<void*>(d_x)})
    CheckHip(hipFree(p), "free");
}

/// Host copies of group_moe_slots / group_moe_rows: only the entries the
/// routed kernels read are defined.
bool SlotGroupsMatch(const std::vector<std::int32_t>& ids,
                     const std::vector<std::int32_t>& g, int tokens, int used) {
  for (int anchor = 0; anchor < tokens * used; ++anchor) {
    const std::int32_t* group = g.data() + anchor * (tokens + 1);
    const int expert = ids[anchor];
    if (expert < 0) {
      if (group[0] != -2)
        return false;
      continue;
    }
    if (std::find(ids.begin(), ids.begin() + anchor, expert) !=
        ids.begin() + anchor) {
      if (group[0] != -1)
        return false;
      continue;
    }
    if (group[0] != expert)
      return false;
    for (int t = 0; t < tokens; ++t) {
      std::uint32_t slots = 0;
      for (int j = 0; j < used; ++j)
        if (ids[t * used + j] == expert)
          slots |= std::uint32_t{1} << j;
      if (group[t + 1] != static_cast<std::int32_t>(slots))
        return false;
    }
  }
  return true;
}

bool RowGroupsMatch(const std::vector<std::int32_t>& ids,
                    const std::vector<std::int32_t>& g, int n_rows) {
  for (int row = 0; row < n_rows; ++row) {
    const std::int32_t* group = g.data() + 3 * row;
    const int expert = ids[row];
    if (expert < 0) {
      if (group[0] != -2)
        return false;
      continue;
    }
    if (std::find(ids.begin(), ids.begin() + row, expert) !=
        ids.begin() + row) {
      if (group[0] != -1)
        return false;
      continue;
    }
    std::uint64_t mask = 0;
    for (int i = row; i < n_rows; ++i)
      if (ids[i] == expert)
        mask |= std::uint64_t{1} << i;
    const auto lo = static_cast<std::int32_t>(static_cast<std::uint32_t>(mask));
    const auto hi =
        static_cast<std::int32_t>(static_cast<std::uint32_t>(mask >> 32));
    if (group[0] != expert || group[1] != lo || group[2] != hi)
      return false;
  }
  return true;
}

// The grouping block of the shared down launch against the tier's grouping
// rules: router-like rows, shared experts, disjoint experts, duplicate
// slots and inactive (-1) slots; untouched memory past the tables.
void CheckGroupTables() {
  namespace q = gufo::models::qwen36_35b_a3b::rocm;
  constexpr int kTop = 8, kDownRows = 8, kDownCols = 512;
  auto w = fx::MakeQ6K(1, kDownRows, kDownCols, 0x6D0A0001U);
  auto* d_w = DeviceCopy(w.packed);
  if (!q::RepackQ6KRows(d_w, kDownRows, kDownCols))
    throw std::runtime_error("Q6_K repack rejected the down rows");
  std::vector<float> x(8 * kDownCols);
  std::uint32_t seed = 0x6D0A0002U;
  for (float& v : x)
    v = fx::Uniform(&seed, 1.0F);
  auto* d_x = DeviceCopy(x);
  float *d_y = nullptr, *d_ref = nullptr;
  CheckHip(hipMalloc(&d_y, 8 * kDownRows * sizeof(float)), "down output");
  CheckHip(hipMalloc(&d_ref, 8 * kDownRows * sizeof(float)), "down output");
  const std::size_t slot_count = 8 * kTop * 9, row_count = 64 * 3;
  std::int32_t *d_slots = nullptr, *d_rows = nullptr;
  CheckHip(hipMalloc(&d_slots, (slot_count + 64) * 4), "slot groups");
  CheckHip(hipMalloc(&d_rows, (row_count + 64) * 4), "row groups");
  constexpr std::int32_t kPoison = 0x7B7B7B7B;
  for (int pattern = 0; pattern < 4; ++pattern) {
    for (int n = 1; n <= 8; ++n) {
      std::vector<std::int32_t> ids(static_cast<std::size_t>(n) * kTop);
      std::uint32_t r = 0xC0DE0000U + pattern * 16 + n;
      for (int t = 0; t < n; ++t) {
        for (int j = 0; j < kTop; ++j) {
          int e = 0;
          if (pattern == 0)
            e = (t + 3 * j) % 16;  // router-like
          else if (pattern == 1)
            e = (5 * j) % 8;  // all shared
          else if (pattern == 2)
            e = 8 * t + j;  // disjoint
          else
            e = NextRandom(&r) % 9 == 0 ? -1 : int(NextRandom(&r) % 6);
          ids[t * kTop + j] = e;
        }
      }
      auto* d_ids = DeviceCopy(ids);
      CheckHip(hipMemset(d_slots, 0x7B, (slot_count + 64) * 4), "poison");
      CheckHip(hipMemset(d_rows, 0x7B, (row_count + 64) * 4), "poison");
      if (!q::MoeSharedDownGroups(d_w, d_x, d_y, n, kDownRows, kDownCols, d_ids,
                                  kTop, n > 1 ? d_slots : nullptr, d_rows,
                                  nullptr) ||
          !q::DenseQ6KVec(d_w, d_x, d_ref, n, kDownRows, kDownCols, nullptr))
        throw std::runtime_error("grouping launch rejected the shape");
      CheckHip(hipDeviceSynchronize(), "grouping");
      std::vector<std::int32_t> slots(slot_count + 64), rows(row_count + 64);
      CheckHip(hipMemcpy(slots.data(), d_slots, slots.size() * 4,
                         hipMemcpyDeviceToHost),
               "slot groups download");
      CheckHip(hipMemcpy(rows.data(), d_rows, rows.size() * 4,
                         hipMemcpyDeviceToHost),
               "row groups download");
      const std::size_t slot_used =
          n > 1 ? static_cast<std::size_t>(n) * kTop * (n + 1) : 0;
      const std::size_t row_used = static_cast<std::size_t>(n) * kTop * 3;
      bool ok = (n == 1 || SlotGroupsMatch(ids, slots, n, kTop)) &&
                RowGroupsMatch(ids, rows, n * kTop);
      for (std::size_t i = slot_used; i < slots.size(); ++i)
        ok = ok && slots[i] == kPoison;
      for (std::size_t i = row_used; i < rows.size(); ++i)
        ok = ok && rows[i] == kPoison;
      if (!ok)
        throw std::runtime_error("grouping tables differ, pattern " +
                                 std::to_string(pattern) + ", " +
                                 std::to_string(n) + " rows");
      if (Bytes(d_y, n * kDownRows * sizeof(float)) !=
          Bytes(d_ref, n * kDownRows * sizeof(float)))
        throw std::runtime_error("grouping launch changed the down rows");
      CheckHip(hipFree(d_ids), "free");
    }
  }
  // Shapes the single grouping block cannot cover are rejected unlaunched:
  // a slot table for one token, more than 64 down rows, nine tokens.
  auto* d_zero = DeviceCopy(std::vector<std::int32_t>(128, 0));
  if (q::MoeSharedDownGroups(d_w, d_x, d_y, 1, kDownRows, kDownCols, d_zero,
                             kTop, d_slots, nullptr, nullptr) ||
      q::MoeSharedDownGroups(d_w, d_x, d_y, 8, kDownRows, kDownCols, d_zero, 9,
                             nullptr, d_rows, nullptr) ||
      q::MoeSharedDownGroups(d_w, d_x, d_y, 9, kDownRows, kDownCols, d_zero,
                             kTop, nullptr, nullptr, nullptr))
    throw std::runtime_error("grouping launch accepted an unsupported shape");
  CheckHip(hipFree(d_zero), "free");
  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x),
                  static_cast<void*>(d_y), static_cast<void*>(d_ref),
                  static_cast<void*>(d_slots), static_cast<void*>(d_rows)})
    CheckHip(hipFree(p), "free");
}

}  // namespace

int main() {
  try {
    for (int pattern : {0, 1, 2, 3}) {
      CheckRouter(1, 512, 10, pattern);
      CheckRouter(8, 512, 10, pattern);
      CheckRouter(2048, 512, 10, pattern);
    }
    CheckRouter(17, 8, 8, 1);
    CheckRouter(17, 513, 32, 0);
    CheckRouter(17, 1024, 32, 1);
    CheckRouter(17, 1024, 1, 3);
    std::cout << "Router softmax, ties, guards and replay: passed\n";
    for (int pattern = 0; pattern < 7; ++pattern)
      CheckPrologue(pattern);
    std::cout << "MoE prologue ids, weights, Q8_1 bytes and shared rows "
                 "identical at widths 1-8: passed\n";
    CheckGroupTables();
    std::cout << "MoE grouping tables: passed\n";
    bool ok = true;
    for (int n_tokens : {37, 512, 2048}) {
      const auto ids = MakeIds(n_tokens, 0x1234ABCDU + n_tokens);
      // Gate/up: [n_tokens][kUsed] ids, one activation row per token.
      const Maps pair_ref = Reference(ids, n_tokens, kUsed, kUsed, 1);
      const Maps pair_scan = Build(ids, n_tokens, kUsed, kUsed, 1);
      ok = Same(pair_ref, pair_scan,
                (std::string("gate/up maps, ") + std::to_string(n_tokens) +
                 " tokens")
                    .c_str()) &&
           ok;
      // Down: every (token, slot) row is its own single-expert token.
      const Maps down_ref = Reference(ids, n_tokens * kUsed, 1, 1, 1);
      const Maps down_scan = Build(ids, n_tokens * kUsed, 1, 1, 1);
      ok = Same(down_ref, down_scan,
                (std::string("down maps, ") + std::to_string(n_tokens) +
                 " tokens")
                    .c_str()) &&
           ok;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
