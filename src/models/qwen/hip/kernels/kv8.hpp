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

#if defined(ENGINE_ENABLE_HIP) && defined(__HIP__)
/// Four codes (one 32-bit word) as two packed F16 pairs f16(scale * code).
/// The F16 word 0x64XX is 1024 + XX, so a code with its sign bit flipped
/// becomes exactly 1152 + code; one packed subtraction recovers the code and
/// one packed multiply rounds its exact product with the scale once.
__device__ __forceinline__ uint2 DequantWord(std::uint32_t word, __half scale) {
  using Half2 = _Float16 __attribute__((ext_vector_type(2)));
  const auto s = __builtin_bit_cast(_Float16, scale);
  const Half2 scales = {s, s};
  const Half2 bias = {static_cast<_Float16>(1152.0F),
                      static_cast<_Float16>(1152.0F)};
  const std::uint32_t flipped = word ^ 0x80808080U;
  std::uint32_t out[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    // Bytes 2i and 2i + 1 of the word, each under an 0x64 byte.
    const std::uint32_t bits = __builtin_amdgcn_perm(
        0x64646464U, flipped, i != 0 ? 0x04030402U : 0x04010400U);
    Half2 value = __builtin_bit_cast(Half2, bits) - bias;
    // Keep the exact subtraction apart from the rounded product.
    asm volatile("" : "+v"(value));
    value = value * scales;
    out[i] = __builtin_bit_cast(std::uint32_t, value);
  }
  return make_uint2(out[0], out[1]);
}

/// Four consecutive values starting at `element` (a multiple of four) of a
/// row of `width` values.
__device__ __forceinline__ float4 LoadFloat4(const std::uint8_t* row,
                                             std::uint32_t width,
                                             std::uint32_t element) {
  const std::uint32_t word =
      *reinterpret_cast<const std::uint32_t*>(row + element);
  const __half scale =
      reinterpret_cast<const __half*>(row + width)[element / kBlock];
  const uint2 packed = DequantWord(word, scale);
  const auto lo = __builtin_bit_cast(__half2, packed.x);
  const auto hi = __builtin_bit_cast(__half2, packed.y);
  return make_float4(__low2float(lo), __high2float(lo), __low2float(hi),
                     __high2float(hi));
}

/// Eight consecutive values starting at `element` (a multiple of eight) as
/// eight packed F16 values, the bytes an FP16 row holds there.
__device__ __forceinline__ uint4 LoadHalf8(const std::uint8_t* row,
                                           std::uint32_t width,
                                           std::uint32_t element) {
  const uint2 codes = *reinterpret_cast<const uint2*>(row + element);
  const __half scale =
      reinterpret_cast<const __half*>(row + width)[element / kBlock];
  const uint2 lo = DequantWord(codes.x, scale);
  const uint2 hi = DequantWord(codes.y, scale);
  return make_uint4(lo.x, lo.y, hi.x, hi.y);
}
#endif  // defined(ENGINE_ENABLE_HIP) && defined(__HIP__)

}  // namespace gufo::hip::kv8

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_KV8_HPP_
