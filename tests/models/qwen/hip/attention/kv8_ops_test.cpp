// 8-bit attention K/V rows of Qwen3.8-27B
// (src/models/qwen/hip/kernels/kv8.hpp). The store kernel must write the bytes
// of the host reference, and every 8-bit reader must produce exactly what its
// FP16 counterpart produces on an FP16 plane holding the dequantized values
// f16(scale * code).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/kernels/kv8.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

namespace kv8 = gufo::hip::kv8;

constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kWidth = kKvHeads * kHeadDim;
constexpr std::uint32_t kLayers = 2;
constexpr std::uint32_t kLayer = 1;
constexpr std::uint32_t kContext = 4096;
constexpr std::size_t kRowBytes = kv8::RowBytes(kWidth);

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

float ToHalf(float x) {
  return __half2float(__float2half(x));
}

template<typename T>
struct Device {
  explicit Device(std::size_t n) : count(n) {
    HIP_CHECK(hipMalloc(&data, std::max<std::size_t>(n, 1) * sizeof(T)));
    HIP_CHECK(hipMemset(data, 0, std::max<std::size_t>(n, 1) * sizeof(T)));
  }
  ~Device() { (void)hipFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void Upload(const std::vector<T>& host) {
    HIP_CHECK(hipMemcpy(data, host.data(), host.size() * sizeof(T),
                        hipMemcpyHostToDevice));
  }
  std::vector<T> Download() const {
    std::vector<T> host(count);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(
        hipMemcpy(host.data(), data, count * sizeof(T), hipMemcpyDeviceToHost));
    return host;
  }
  T* data{nullptr};
  std::size_t count;
};

std::vector<float> Random(std::size_t n, std::mt19937& rng, float scale) {
  std::normal_distribution<float> normal(0.0F, scale);
  std::vector<float> values(n);
  for (auto& value : values)
    value = normal(rng);
  return values;
}

// Both planes of one model: 8-bit (bytes) and the FP16 plane of the values
// its readers see, filled through LaunchStoreKv8 for `rows` positions.
struct Planes {
  Device<std::uint8_t> q8_k{std::size_t{kLayers} * kContext * kRowBytes};
  Device<std::uint8_t> q8_v{std::size_t{kLayers} * kContext * kRowBytes};
  Device<__half> f16_k{std::size_t{kLayers} * kContext * kWidth};
  Device<__half> f16_v{std::size_t{kLayers} * kContext * kWidth};
};

std::vector<__half> Dequantize(const std::vector<std::uint8_t>& plane) {
  std::vector<__half> values(std::size_t{kLayers} * kContext * kWidth);
  for (std::size_t row = 0; row < std::size_t{kLayers} * kContext; ++row) {
    const auto* bytes = plane.data() + row * kRowBytes;
    for (std::uint32_t i = 0; i < kWidth; ++i) {
      __half scale_bits;
      std::memcpy(&scale_bits, bytes + kWidth + (i / kv8::kBlock) * 2, 2);
      const auto code = static_cast<std::int8_t>(bytes[i]);
      values[row * kWidth + i] =
          __float2half(__half2float(scale_bits) * static_cast<float>(code));
    }
  }
  return values;
}

void FillPlanes(Planes& planes, std::uint32_t rows, std::mt19937& rng) {
  auto k = Random(std::size_t{rows} * kWidth, rng, 1.5F);
  auto v = Random(std::size_t{rows} * kWidth, rng, 0.7F);
  // Edge blocks: all zero, one huge outlier, values that round to F16 ties.
  std::fill_n(k.begin(), kv8::kBlock, 0.0F);
  k[kv8::kBlock + 3] = 300.0F;
  for (std::uint32_t i = 0; i < kv8::kBlock; ++i)
    v[2 * kv8::kBlock + i] = static_cast<float>(i) - 7.5F;
  Device<float> d_k(k.size()), d_v(v.size());
  d_k.Upload(k);
  d_v.Upload(v);
  gufo::hip::LaunchStoreKv8(d_k.data, d_v.data, planes.q8_k.data,
                            planes.q8_v.data, kLayer, 0, nullptr, rows,
                            kContext, kWidth, nullptr);
  const auto q8_k = planes.q8_k.Download();
  const auto q8_v = planes.q8_v.Download();
  // Byte-exact against the host reference.
  for (int plane = 0; plane < 2; ++plane) {
    const auto& source = plane == 0 ? k : v;
    const auto& bytes = plane == 0 ? q8_k : q8_v;
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto* stored =
          bytes.data() + (std::size_t{kLayer} * kContext + row) * kRowBytes;
      for (std::uint32_t block = 0; block < kWidth / kv8::kBlock; ++block) {
        std::int8_t codes[kv8::kBlock];
        float scale = 0.0F;
        kv8::QuantizeBlock(
            source.data() + std::size_t{row} * kWidth + block * kv8::kBlock,
            codes, &scale, ToHalf);
        Expect(
            std::memcmp(codes, stored + block * kv8::kBlock, kv8::kBlock) == 0,
            "8-bit codes differ from the host reference at row " +
                std::to_string(row) + " block " + std::to_string(block));
        const __half reference = __float2half(scale);
        Expect(std::memcmp(&reference, stored + kWidth + block * 2, 2) == 0,
               "8-bit scale differs from the host reference");
      }
    }
  }
  planes.f16_k.Upload(Dequantize(q8_k));
  planes.f16_v.Upload(Dequantize(q8_v));
}

void ExpectSame(const std::vector<float>& lhs, const std::vector<float>& rhs,
                const std::string& what) {
  Expect(lhs.size() == rhs.size(), what + ": size");
  for (std::size_t i = 0; i < lhs.size(); ++i)
    Expect(std::memcmp(&lhs[i], &rhs[i], sizeof(float)) == 0,
           what + ": element " + std::to_string(i) + " " +
               std::to_string(lhs[i]) + " vs " + std::to_string(rhs[i]));
}

void TestDecode(Planes& planes, std::mt19937& rng) {
  const std::size_t scratch_elements =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  for (const std::uint32_t pos : {0U, 50U, 127U, 128U, 300U, 511U, 2000U}) {
    const auto q = Random(std::size_t{kHeads} * kHeadDim, rng, 1.0F);
    const auto gate = Random(q.size(), rng, 1.0F);
    Device<float> d_q(q.size()), d_gate(gate.size());
    d_q.Upload(q);
    d_gate.Upload(gate);
    Device<float> f16_out(q.size()), q8_out(q.size());
    Device<float> scratch(scratch_elements);
    Device<std::uint32_t> d_pos(1);
    d_pos.Upload({pos});
    gufo::hip::LaunchAttention(d_q.data, nullptr, nullptr, d_gate.data, nullptr,
                               nullptr, planes.f16_k.data, planes.f16_v.data,
                               f16_out.data, kLayer, pos, kContext, kHeads,
                               kKvHeads, kHeadDim, nullptr, scratch.data, true);
    gufo::hip::LaunchAttention(
        d_q.data, nullptr, nullptr, d_gate.data, nullptr, nullptr,
        planes.q8_k.data, planes.q8_v.data, q8_out.data, kLayer, pos, kContext,
        kHeads, kKvHeads, kHeadDim, nullptr, scratch.data, true, true);
    ExpectSame(f16_out.Download(), q8_out.Download(),
               "decode attention at " + std::to_string(pos));
    if (pos < gufo::hip::detail::kSplitKDecodeAttentionMinContext) {
      gufo::hip::LaunchAttention(
          d_q.data, nullptr, nullptr, d_gate.data, nullptr, nullptr,
          planes.f16_k.data, planes.f16_v.data, f16_out.data, kLayer,
          d_pos.data, kContext, kHeads, kKvHeads, kHeadDim, nullptr, true);
      gufo::hip::LaunchAttention(
          d_q.data, nullptr, nullptr, d_gate.data, nullptr, nullptr,
          planes.q8_k.data, planes.q8_v.data, q8_out.data, kLayer, d_pos.data,
          kContext, kHeads, kKvHeads, kHeadDim, nullptr, true, true);
      ExpectSame(f16_out.Download(), q8_out.Download(),
                 "graph decode attention at " + std::to_string(pos));
    }
  }
}

void TestVerification(Planes& planes, std::mt19937& rng) {
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  for (const std::uint32_t start : {20U, 200U, 700U, 3000U}) {
    for (const std::size_t rows : {std::size_t{2}, std::size_t{8}}) {
      const auto q = Random(rows * kHeads * kHeadDim, rng, 1.0F);
      const auto gate = Random(q.size(), rng, 1.0F);
      Device<float> d_q(q.size()), d_gate(gate.size());
      d_q.Upload(q);
      d_gate.Upload(gate);
      Device<float> f16_out(q.size()), q8_out(q.size());
      Device<float> scratch(rows * row_scratch);
      gufo::hip::LaunchCausalDecodeAttention(
          d_q.data, d_gate.data, nullptr, nullptr, planes.f16_k.data,
          planes.f16_v.data, f16_out.data, kLayer, start, rows, kContext,
          kHeads, kKvHeads, kHeadDim, nullptr,
          std::span(scratch.data, rows * row_scratch));
      gufo::hip::LaunchCausalDecodeAttention(
          d_q.data, d_gate.data, nullptr, nullptr, planes.q8_k.data,
          planes.q8_v.data, q8_out.data, kLayer, start, rows, kContext, kHeads,
          kKvHeads, kHeadDim, nullptr,
          std::span(scratch.data, rows * row_scratch), true);
      ExpectSame(f16_out.Download(), q8_out.Download(),
                 "verification attention at " + std::to_string(start) + " x" +
                     std::to_string(rows));
    }
  }
}

void TestPrefill(Planes& planes, std::mt19937& rng) {
  // Workspaces as large as production's idle FFN scratch.
  const std::size_t workspace = std::size_t{2048} * 17408;
  Device<float> packed_k(workspace), packed_v(workspace);
  for (const auto [start, batch] :
       {std::pair<std::uint32_t, std::size_t>{0, 1024},
        std::pair<std::uint32_t, std::size_t>{1500, 1024},
        std::pair<std::uint32_t, std::size_t>{2000, 96}}) {
    const auto q = Random(batch * kHeads * kHeadDim, rng, 1.0F);
    const auto gate = Random(q.size(), rng, 1.0F);
    Device<float> d_q(q.size()), d_gate(gate.size());
    d_q.Upload(q);
    d_gate.Upload(gate);
    Device<float> f16_out(q.size()), q8_out(q.size());
    const std::span<float> k_space(packed_k.data, workspace);
    const std::span<float> v_space(packed_v.data, workspace);
    Expect(gufo::hip::LaunchQwenWmmaAttention(
               d_q.data, nullptr, nullptr, d_gate.data, nullptr, nullptr,
               planes.f16_k.data, planes.f16_v.data, f16_out.data, kLayer,
               start, batch, kContext, kHeads, kKvHeads, kHeadDim, nullptr,
               nullptr, 0, true, k_space, v_space),
           "FP16 WMMA prefill unsupported");
    Expect(gufo::hip::LaunchQwenWmmaAttention(
               d_q.data, nullptr, nullptr, d_gate.data, nullptr, nullptr,
               planes.q8_k.data, planes.q8_v.data, q8_out.data, kLayer, start,
               batch, kContext, kHeads, kKvHeads, kHeadDim, nullptr, nullptr, 0,
               true, k_space, v_space, 0, true),
           "8-bit WMMA prefill unsupported");
    const auto f16 = f16_out.Download();
    const auto q8 = q8_out.Download();
    if (batch >= 1024) {
      // Both pack the same F16 heads: identical arithmetic.
      ExpectSame(f16, q8, "WMMA prefill at " + std::to_string(start));
    } else {
      // FP16 attends the canonical plane directly; 8-bit packs heads first.
      float worst = 0.0F;
      for (std::size_t i = 0; i < f16.size(); ++i)
        worst = std::max(worst, std::fabs(f16[i] - q8[i]));
      Expect(worst < 2e-2F,
             "small-batch WMMA prefill differs by " + std::to_string(worst));
    }
  }
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen 8-bit KV ops test");
  if (device_status != 0)
    return device_status;
  try {
    std::mt19937 rng(20261031);
    Planes planes;
    FillPlanes(planes, 3200, rng);
    TestDecode(planes, rng);
    TestVerification(planes, rng);
    TestPrefill(planes, rng);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Qwen 8-bit KV ops tests passed.\n";
#endif
  return 0;
}
