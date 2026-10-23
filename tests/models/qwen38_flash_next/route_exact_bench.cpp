//   wave64  the 2560 x 6144 output projection's W8A8 GEMM at 1024-4096 rows
//           (the wave64 kernel; compare builds to time its pinned order).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {


void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

void Require(int status, const char* operation) {
  if (status != 0)
    throw std::runtime_error(operation);
}

std::uint32_t NextRandom(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

void* RandomDevice(std::size_t bytes, std::size_t block, std::size_t d_offset,
                   std::uint32_t seed) {
  std::vector<std::uint8_t> host(bytes);
  for (auto& b : host)
    b = static_cast<std::uint8_t>(NextRandom(&seed));
  for (std::size_t b = 0; b + block <= bytes; b += block) {
    const __half d = __float2half(1e-3F);
    std::memcpy(host.data() + b + d_offset, &d, 2);
  }
  void* device = nullptr;
  CheckHip(hipMalloc(&device, bytes + 4096), "weights");
  CheckHip(hipMemcpy(device, host.data(), bytes, hipMemcpyHostToDevice),
           "weights upload");
  return device;
}

template<typename F>
double TimeUs(F&& run, int reps = 50) {
  for (int i = 0; i < 5; ++i)
    run();
  hipEvent_t start, stop;
  CheckHip(hipEventCreate(&start), "event");
  CheckHip(hipEventCreate(&stop), "event");
  CheckHip(hipEventRecord(start, nullptr), "record");
  for (int i = 0; i < reps; ++i)
    run();
  CheckHip(hipEventRecord(stop, nullptr), "record");
  CheckHip(hipEventSynchronize(stop), "sync");
  float ms = 0.0F;
  CheckHip(hipEventElapsedTime(&ms, start, stop), "elapsed");
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return 1000.0 * ms / reps;
}

void KQuant() {
  struct Shape {
    const char* name;
    int m, k;
    std::vector<Format> formats;
  };
    void* q8 =
        RandomDevice(std::size_t(s.m) * s.k / 32 * 34, 34, 0, 0x4B2U + s.m);
    constexpr int kMaxRows = 64;
    float* x = nullptr;
    float* y = nullptr;
    void* xq = nullptr;
    CheckHip(hipMalloc(&x, std::size_t(kMaxRows) * s.k * 4), "x");
    CheckHip(hipMalloc(&y, std::size_t(kMaxRows) * s.m * 4), "y");
    CheckHip(hipMalloc(&xq, qfn_mmq_q8_1_bytes(kMaxRows, s.k)), "xq");
    CheckHip(hipMemset(x, 0, std::size_t(kMaxRows) * s.k * 4), "x");
      (void)hipFree(p);
  }
}

void Wave64() {
  constexpr int m = 2560, k = 6144;
  void* w = RandomDevice(std::size_t(m) * k / 32 * 34, 34, 0, 0x64U);
  for (const int batch : {1024, 2048, 4096}) {
    float* x = nullptr;
    float* y = nullptr;
    void* tiled = nullptr;
    CheckHip(hipMalloc(&x, std::size_t(batch) * k * 4), "x");
    CheckHip(hipMalloc(&y, std::size_t(batch) * m * 4), "y");
    CheckHip(hipMalloc(&tiled, q::Q8TiledBytes(batch, k)), "tiles");
    CheckHip(hipMemset(x, 0, std::size_t(batch) * k * 4), "x");
    q::QuantizeQ8Tiled(x, tiled, batch, k, nullptr);
    const double us = TimeUs(
        [&] {
          if (!q::W8A8Gemm(w, tiled, y, batch, m, k, nullptr))
            throw std::runtime_error("W8A8");
        },
        200);
    std::cout << std::fixed << std::setprecision(1) << "wave64 W8A8 " << m
              << "x" << k << " batch=" << batch << ": " << us << " us\n";
    for (void* p : {static_cast<void*>(x), static_cast<void*>(y), tiled})
      (void)hipFree(p);
  }
  (void)hipFree(w);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CheckHip(q::PrepareW8A8GemmWave64(), "prepare wave64");
    Require(qfn_mmq_init(0), "mmq init");
    const std::string mode = argc > 1 ? argv[1] : "all";
    if (mode == "all" || mode == "kquant")
      KQuant();
    if (mode == "all" || mode == "wave64")
      Wave64();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
