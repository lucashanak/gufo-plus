#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_KV8_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_KV8_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

// The 8-bit attention K/V rows of Qwen3.8-27B. A row (one position of one
// attention layer) of `width` = kv_heads * head_dim values holds `width` int8
// codes followed by width / kKv8Block F16 scales, one per 16 consecutive
// values. Writers round each value x to F16 first; a block's scale is
// f16(max |x| * (1 / 127.0f)), and its codes are rint(x / scale) with an
// IEEE division, clamped to +-127 (all zero when the scale is zero). Readers
// see f16(scale * code): every attention kernel consumes exactly those F16
// values, so the FP16 kernels' arithmetic is unchanged.
namespace gufo::hip::kv8 {

inline constexpr std::uint32_t kBlock = 16;

[[nodiscard]] constexpr std::size_t RowBytes(std::size_t width) noexcept {
  return width + width / kBlock * 2;
}

/// The exact host reference of one block: 16 values to 16 codes and the F16
/// scale bits. `to_half` rounds a float to F16 and returns it as a float;
/// `half_bits` returns the F16 bits of a float that is already an F16 value.
template<typename ToHalf>
void QuantizeBlock(const float* values, std::int8_t* codes, float* scale,
                   ToHalf to_half) {
  float rounded[kBlock];
  float maximum = 0.0F;
  for (std::uint32_t i = 0; i < kBlock; ++i) {
    rounded[i] = to_half(values[i]);
    maximum = std::fmax(maximum, std::fabs(rounded[i]));
  }
  const float s = to_half(maximum * (1.0F / 127.0F));
  *scale = s;
  for (std::uint32_t i = 0; i < kBlock; ++i) {
    float code = s == 0.0F ? 0.0F : std::nearbyint(rounded[i] / s);
    code = std::fmin(127.0F, std::fmax(-127.0F, code));
    codes[i] = static_cast<std::int8_t>(code);
  }
}

#if defined(ENGINE_ENABLE_HIP)
/// f16(scale * code) as a float.
__device__ __forceinline__ float Value(std::int32_t code, float scale) {
  return __half2float(__float2half_rn(scale * static_cast<float>(code)));
}

/// Four consecutive values starting at `element` (a multiple of four) of a
/// row of `width` values.
__device__ __forceinline__ float4 LoadFloat4(const std::uint8_t* row,
                                             std::uint32_t width,
                                             std::uint32_t element) {
  const std::uint32_t word =
      *reinterpret_cast<const std::uint32_t*>(row + element);
  const float scale = __half2float(
      reinterpret_cast<const __half*>(row + width)[element / kBlock]);
  const auto code = [&](unsigned shift) {
    return static_cast<std::int32_t>(static_cast<std::int8_t>(word >> shift));
  };
  return make_float4(Value(code(0), scale), Value(code(8), scale),
                     Value(code(16), scale), Value(code(24), scale));
}

/// Eight consecutive values starting at `element` (a multiple of eight) as
/// eight packed F16 values, the bytes an FP16 row holds there.
__device__ __forceinline__ uint4 LoadHalf8(const std::uint8_t* row,
                                           std::uint32_t width,
                                           std::uint32_t element) {
  const uint2 codes = *reinterpret_cast<const uint2*>(row + element);
  const float scale = __half2float(
      reinterpret_cast<const __half*>(row + width)[element / kBlock]);
  union {
    uint4 packed;
    __half halves[8];
  } out;
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    const std::uint32_t word = i < 4 ? codes.x : codes.y;
    const auto code = static_cast<std::int32_t>(
        static_cast<std::int8_t>(word >> ((i % 4) * 8)));
    out.halves[i] = __float2half_rn(scale * static_cast<float>(code));
  }
  return out.packed;
}
#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::hip::kv8

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_KV8_HPP_
