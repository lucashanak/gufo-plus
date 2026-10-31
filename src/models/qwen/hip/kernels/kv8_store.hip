#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen/hip/kernels/kv8.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace gufo::hip {
namespace {

// One block per position; one thread per 16-value block of K, then of V.
__global__ void StoreKv8Kernel(const float* __restrict__ k,
                               const float* __restrict__ v,
                               std::uint8_t* __restrict__ k_plane,
                               std::uint8_t* __restrict__ v_plane,
                               std::uint32_t layer_idx, std::uint32_t start_pos,
                               const std::uint32_t* __restrict__ d_pos,
                               std::uint32_t max_context, std::uint32_t width) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t blocks = width / kv8::kBlock;
  const std::uint32_t thread = threadIdx.x;
  if (thread >= 2 * blocks)
    return;
  const bool value_plane = thread >= blocks;
  const std::uint32_t block = value_plane ? thread - blocks : thread;
  const std::uint32_t position = (d_pos != nullptr ? *d_pos : start_pos) + row;
  const float* source = (value_plane ? v : k) +
                        static_cast<std::size_t>(row) * width +
                        block * kv8::kBlock;
  std::uint8_t* destination =
      (value_plane ? v_plane : k_plane) +
      (static_cast<std::size_t>(layer_idx) * max_context + position) *
          kv8::RowBytes(width);
  float rounded[kv8::kBlock];
  float maximum = 0.0F;
#pragma unroll
  for (std::uint32_t i = 0; i < kv8::kBlock; ++i) {
    rounded[i] = __half2float(__float2half_rn(source[i]));
    maximum = fmaxf(maximum, fabsf(rounded[i]));
  }
  const __half scale_bits = __float2half_rn(__fmul_rn(maximum, 1.0F / 127.0F));
  const float scale = __half2float(scale_bits);
  union {
    uint4 packed;
    std::int8_t codes[kv8::kBlock];
  } out;
#pragma unroll
  for (std::uint32_t i = 0; i < kv8::kBlock; ++i) {
    float code = scale == 0.0F ? 0.0F : rintf(__fdiv_rn(rounded[i], scale));
    code = fminf(127.0F, fmaxf(-127.0F, code));
    out.codes[i] = static_cast<std::int8_t>(code);
  }
  *reinterpret_cast<uint4*>(destination + block * kv8::kBlock) = out.packed;
  reinterpret_cast<__half*>(destination + width)[block] = scale_bits;
}

}  // namespace

void LaunchStoreKv8(const float* k, const float* v, void* k_plane,
                    void* v_plane, std::uint32_t layer_idx,
                    std::uint32_t start_pos, const std::uint32_t* d_pos,
                    std::size_t rows, std::uint32_t max_context,
                    std::uint32_t width, hipStream_t stream) {
  if (rows == 0)
    return;
  if (k == nullptr || v == nullptr || k_plane == nullptr ||
      v_plane == nullptr || width == 0 || width % kv8::kBlock != 0 ||
      2 * width / kv8::kBlock > 1024)
    throw std::invalid_argument("invalid 8-bit K/V store");
  StoreKv8Kernel<<<static_cast<unsigned>(rows), 2 * width / kv8::kBlock, 0,
                   stream>>>(k, v, static_cast<std::uint8_t*>(k_plane),
                             static_cast<std::uint8_t*>(v_plane), layer_idx,
                             start_pos, d_pos, max_context, width);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
