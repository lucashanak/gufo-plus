#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <initializer_list>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/weight_source.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace flash_next = ext::qwen::flash_next;

namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

/// Block formats a dense projection keeps as compact decode rows beside a
/// Q8_0 copy: the decode GEMVs (qfn_mmq_dense_kq_vec) read them.
bool CompactDecodeFormat(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ5_1:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kQ5_0:
    case core::GgmlType::kQ4_1:
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ2_K:
    case core::GgmlType::kIQ2_XXS:
    case core::GgmlType::kIQ2_XS:
    case core::GgmlType::kIQ2_S:
    case core::GgmlType::kIQ3_XXS:
    case core::GgmlType::kIQ3_S:
    case core::GgmlType::kIQ1_S:
    case core::GgmlType::kIQ1_M:
    case core::GgmlType::kIQ4_XS:
      return true;
    default:
      return false;
  }
}

/// Formats a dense projection (and the vocabulary head) can be uploaded in.
bool ProjectionFormat(core::GgmlType type) {
  return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kF32 ||
         type == core::GgmlType::kF16 || type == core::GgmlType::kBF16 ||
         CompactDecodeFormat(type);
}

/// Formats the routed expert kernels project (decode and prefill).
bool ExpertFormat(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ5_1:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ5_0:
    case core::GgmlType::kQ4_1:
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ2_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kIQ2_XXS:
    case core::GgmlType::kIQ2_XS:
    case core::GgmlType::kIQ2_S:
    case core::GgmlType::kIQ3_XXS:
    case core::GgmlType::kIQ3_S:
    case core::GgmlType::kIQ1_S:
    case core::GgmlType::kIQ4_XS:
      return true;
    default:
      return false;
  }
}

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
  /// Expands Q8_0 weights to F32 instead of narrowing F32 to F16.
  bool expand_q8{false};
  std::size_t source_bytes{0};
};

/// The trunk and the optional MTP block as one traversal's output.
struct Uploaded {
  DeviceTensor token_embd;
  DeviceTensor output;
  DeviceMixer hc_head;
  std::vector<DeviceLayer> layers;
  DeviceLayer mtp;
};

/// Walks the model's tensors in upload order. With `weights` null the walk
/// only measures: it sums the bytes every allocation would request and does
/// no device work, so the weight placement can be decided from the tensors.
struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator* weights;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  std::uint64_t requested{0};
  /// Uploads the tensors with format_state; null for a GGUF upload.
  flash_next::WeightSource* source{nullptr};
  /// Bytes the measuring walk requested: the whole upload.
  std::uint64_t measured{0};

  [[nodiscard]] bool Measuring() const noexcept { return weights == nullptr; }

  /// What a weight source's Upload sees of this walk.
  struct Context final : flash_next::UploadContext {
    Uploader& up;
    explicit Context(Uploader& uploader) : up(uploader) {}
    [[nodiscard]] bool Measuring() const noexcept override {
      return up.Measuring();
    }
    [[nodiscard]] bool Allocate(void** pointer, std::size_t size) override {
      *pointer = nullptr;
      if (up.Allocate(pointer, size) != hipSuccess)
        return false;
      if (!up.Measuring()) {
        up.allocations.push_back(*pointer);
        up.bytes += size;
      }
      return true;
    }
    // The measuring walk copies nothing.
    [[nodiscard]] bool CopyFromRegion(std::uint32_t region,
                                      std::uint64_t offset, std::size_t size,
                                      void* device) override {
      return up.Measuring() || up.stager.Copy(up.shard_base + region, offset,
                                              size, device, up.error);
    }
    [[nodiscard]] bool Finish() override {
      return up.Measuring() || up.stager.Finish(up.error);
    }
    void Fail(std::string_view message) override {
      up.Fail(std::string(message));
    }
  };

  /// Queues one payload copy from a mapped region, or from host memory the
  /// weight source prepared (kHostShard).
  [[nodiscard]] bool Source(const TensorRef& t, std::size_t size,
                            void* device) {
    if (t.shard == kHostShard) {
      if (hipMemcpy(device, t.data, size, hipMemcpyHostToDevice) == hipSuccess)
        return true;
      if (error != nullptr)
        *error = "host tensor copy failed";
      return false;
    }
    return stager.Copy(shard_base + t.shard, t.file_offset, size, device,
                       error);
  }

  DeviceTensor CopyFormat(const TensorRef& t,
                          const flash_next::UploadHint& hint) {
    if (source == nullptr) {
      Fail("no weight source uploads " + std::string(t.name));
      return {};
    }
    Context context(*this);
    const flash_next::Tensor d = source->Upload(t, hint, context);
    if (!ok || Measuring())
      return {};
    if (d.empty() || d.format_state() == nullptr) {
      Fail("weight source upload failed for " + std::string(t.name));
      return {};
    }
    // No GGUF route reads an encoded tensor: every type-gated kernel choice
    // (Q8_0 GEMVs, WMMA experts, tiled projections) declines kI32.
    DeviceTensor device = FromTensor(d);
    device.type = core::GgmlType::kI32;
    return device;
  }

  /// A tensor with format_state outside the projections the stages run:
  /// its bytes would reach a raw GGUF kernel.
  bool RejectFormat(const TensorRef& t) {
    if (t.format_state == nullptr)
      return false;
    Fail("encoded tensor " + std::string(t.name) +
         " is only supported on a stage-run projection");
    return true;
  }

  /// A failed allocation of `size` bytes for `name`. The placement and the
  /// share already allocated tell a full device heap (hipMalloc) from
  /// system-wide GPU memory that other processes hold (managed).
  void FailAllocation(std::string_view name, std::size_t size) {
    std::ostringstream message;
    message << "device allocation failed for " << name << " (" << size
            << " bytes";
    if (weights != nullptr) {
      constexpr double kGib = 1024.0 * 1024.0 * 1024.0;
      message << std::fixed << std::setprecision(1) << "; "
              << static_cast<double>(requested - size) / kGib << " of "
              << static_cast<double>(measured) / kGib
              << " GiB of weights allocated, as "
              << (weights->placement() == hip::Placement::kMalloc
                      ? "device memory"
                      : "managed memory");
    }
    message << ')';
    Fail(message.str());
  }

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t size) {
    requested += size;
    if (Measuring()) {
      return hipSuccess;
    }
    return weights->Allocate(pointer, size);
  }

  static bool DecodeKQuant(core::GgmlType type) {
    return CompactDecodeFormat(type);
  }

  /// Q8_0 blocks of `n` floats, as ggml's quantize_row_q8_0_ref.
  static void QuantizeQ8_0Row(const float* x, std::uint8_t* out,
                              std::size_t n) {
    for (std::size_t b = 0; b < n / 32; ++b) {
      float amax = 0.0F;
      for (int j = 0; j < 32; ++j)
        amax = std::max(amax, std::fabs(x[b * 32 + j]));
      const float d = amax / 127.0F;
      const float id = d != 0.0F ? 1.0F / d : 0.0F;
      const _Float16 dh = static_cast<_Float16>(d);
      std::memcpy(out + b * 34, &dh, 2);
      for (int j = 0; j < 32; ++j)
        reinterpret_cast<std::int8_t*>(out + b * 34 + 2)[j] =
            static_cast<std::int8_t>(std::round(x[b * 32 + j] * id));
    }
  }

  /// K-quant trunk projections stacked along rows: the file's rows go to
  /// `decode` unchanged and a Q8_0 re-encoding of them to `data`, so every
  /// Q8_0 path (prefill GEMMs, fused decode kernels) keeps its arithmetic
  /// while decode GEMVs read the compact rows. The 32-wide formats (mixer
  /// up projections, 320 columns) need only whole blocks.
  DeviceTensor KQuantProjection(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    const TensorRef& first = **parts.begin();
    std::size_t rows = 0;
    std::size_t raw = 0;
    for (const TensorRef* t : parts) {
      if (RejectFormat(*t))
        return d;
      if (t->empty() || t->type != first.type || t->cols != first.cols ||
          t->experts != 1 ||
          t->cols % gufo::quant::QuantizedBlockElements(t->type) != 0) {
        Fail("K-quant projection parts must share one type and shape");
        return d;
      }
      rows += t->rows;
      raw += t->SizeBytes();
    }
    const std::size_t q8_row = first.cols / 32 * 34;
    const std::size_t q8 = rows * q8_row;
    void* raw_ptr = nullptr;
    void* q8_ptr = nullptr;
    if (Allocate(&raw_ptr, raw + kTailMargin) != hipSuccess) {
      FailAllocation(first.name, raw + kTailMargin);
      return d;
    }
    if (Allocate(&q8_ptr, q8 + kTailMargin) != hipSuccess) {
      FailAllocation(first.name, q8 + kTailMargin);
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(raw_ptr);
    allocations.push_back(q8_ptr);
    bytes += raw + q8 + 2 * kTailMargin;
    std::vector<std::uint8_t> host(q8);
    std::size_t offset = 0;
    std::size_t row0 = 0;
    for (const TensorRef* t : parts) {
      if (!Source(*t, t->SizeBytes(),
                  static_cast<std::uint8_t*>(raw_ptr) + offset)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
      const auto* src = static_cast<const std::uint8_t*>(t->data);
      const std::size_t row_bytes = t->RowBytes();
      const unsigned workers =
          std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
      std::vector<std::thread> threads;
      for (unsigned w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
          std::vector<float> row(t->cols);
          for (std::size_t r = w; r < t->rows; r += workers) {
            const auto* in = src + r * row_bytes;
            (void)gufo::quant::DequantizeRow(t->type, in, row.data(), t->cols);
            QuantizeQ8_0Row(row.data(), host.data() + (row0 + r) * q8_row,
                            t->cols);
          }
        });
      }
      for (auto& th : threads)
        th.join();
      row0 += t->rows;
    }
    if (hipMemcpy(q8_ptr, host.data(), q8, hipMemcpyHostToDevice) !=
        hipSuccess) {
      Fail("Q8_0 copy upload failed for " + std::string(first.name));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(raw_ptr) + raw, 0,
                         kTailMargin, nullptr);
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(q8_ptr) + q8, 0,
                         kTailMargin, nullptr);
    d.data = q8_ptr;
    d.type = core::GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(first.cols);
    d.rows = static_cast<std::uint32_t>(rows);
    d.decode = raw_ptr;
    d.decode_type = first.type;
    max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
    return d;
  }

  /// A projection the executor runs through Dense / Experts, which hand
  /// encoded tensors to the stages (Linear, LinearGroup).
  DeviceTensor Projection(const TensorRef& t,
                          const flash_next::UploadHint& hint = {}) {
    if (t.empty() || !ok)
      return {};
    if (t.format_state != nullptr)
      return CopyFormat(t, hint);
    return Copy(t);
  }

  /// `rows_only`: the file's rows as they are, without the Q8_0 copy a
  /// K-quant projection carries (a table that is only gathered from).
  DeviceTensor Copy(const TensorRef& t, bool rows_only = false) {
    DeviceTensor d;
    if (t.empty() || !ok || RejectFormat(t)) {
      return d;
    }
    if (t.experts == 1 && !rows_only && DecodeKQuant(t.type)) {
      return KQuantProjection({&t});
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      FailAllocation(t.name, size + kTailMargin);
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!Source(t, size, ptr)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok || RejectFormat(t))
      return;
    const auto combined = Copy(t);
    if (!ok || (!Measuring() && !stager.Finish(error))) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (Allocate(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      if (Measuring()) {
        continue;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    if (Measuring()) {
      return;
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed; a BF16 router joins its F32 shared-expert gate row
  /// there, widened exactly on the host. A Q8_0 stack merges projections of
  /// one input into a single decode GEMV, unless `expand_q8` asks for the F32
  /// stack's F16 route over its dequantized values.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts,
                     bool expand_q8 = false) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    if (DecodeKQuant(type)) {
      return KQuantProjection(parts);
    }
    const auto wide = [](core::GgmlType t) {
      return t == core::GgmlType::kF32 || t == core::GgmlType::kBF16;
    };
    for (const TensorRef* t : parts) {
      if (RejectFormat(*t))
        return d;
      if (t->empty() || t->cols != (*parts.begin())->cols ||
          (type == core::GgmlType::kQ8_0 ? t->type != type
                                         : !wide(type) || !wide(t->type))) {
        Fail("stacked upload needs F32/BF16 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += type == core::GgmlType::kQ8_0 ? t->SizeBytes()
                                            : t->ElementCount() * sizeof(float);
    }
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (!Measuring()) {
      allocations.push_back(ptr);
      bytes += size + kTailMargin;
      std::size_t offset = 0;
      for (const TensorRef* t : parts) {
        auto* dst = static_cast<std::uint8_t*>(ptr) + offset;
        if (t->type == core::GgmlType::kBF16) {
          std::vector<float> wide_rows(t->ElementCount());
          const auto* src = static_cast<const std::uint16_t*>(t->data);
          for (std::size_t i = 0; i < wide_rows.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
            std::memcpy(&wide_rows[i], &bits, sizeof(float));
          }
          if (hipMemcpy(dst, wide_rows.data(), wide_rows.size() * sizeof(float),
                        hipMemcpyHostToDevice) != hipSuccess) {
            Fail("upload failed for " + std::string(t->name));
            return d;
          }
          offset += wide_rows.size() * sizeof(float);
          continue;
        }
        if (!Source(*t, t->SizeBytes(), dst)) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += t->SizeBytes();
      }
    }
    if (type == core::GgmlType::kQ8_0 && !expand_q8) {
      if (Measuring()) {
        return d;
      }
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    void* f32 = ptr;
    std::size_t f32_bytes = size;
    if (type == core::GgmlType::kQ8_0) {
      const std::size_t elements = rows * (*parts.begin())->cols;
      f32_bytes = elements * sizeof(float);
      if (Allocate(&f32, f32_bytes + kTailMargin) != hipSuccess) {
        Fail("stacked tensor allocation failed");
        return d;
      }
      if (!Measuring()) {
        allocations.push_back(f32);
        bytes += f32_bytes + kTailMargin;
        conversions.push_back({ptr, f32, elements, true, size});
      }
    }
    const std::size_t count = f32_bytes / sizeof(float);
    void* half = nullptr;
    if (Allocate(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({f32, half, count, false, f32_bytes});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  /// A tensor its kernels read as F32 (taps, inject rows), widened exactly
  /// from any other file format on the host.
  DeviceTensor CopyF32(const TensorRef& t) {
    if (t.empty() || t.type == core::GgmlType::kF32 || !ok) {
      return Copy(t);
    }
    DeviceTensor d;
    if (RejectFormat(t)) {
      return d;
    }
    const std::size_t count = t.ElementCount();
    void* ptr = nullptr;
    if (Allocate(&ptr, count * sizeof(float) + kTailMargin) != hipSuccess) {
      FailAllocation(t.name, count * sizeof(float) + kTailMargin);
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += count * sizeof(float) + kTailMargin;
    std::vector<float> host(count);
    const auto* src = static_cast<const std::uint8_t*>(t.data);
    for (std::size_t r = 0; r < t.rows * t.experts; ++r) {
      if (!gufo::quant::DequantizeRow(t.type, src + r * t.RowBytes(),
                                      host.data() + r * t.cols, t.cols)) {
        Fail("cannot widen " + std::string(t.name) + " to F32");
        return d;
      }
    }
    if (hipMemcpy(ptr, host.data(), count * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(ptr) + count * sizeof(float), 0,
                  kTailMargin) != hipSuccess) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = ptr;
    d.type = core::GgmlType::kF32;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    // Inject rows of every format, BF16 and Q8_0 included, are widened to
    // the F32 rows the fused mixer reads, so they run the F32 kernels on
    // their exact values. A separate BF16 projection would take the
    // library GEMM, whose sums depend on the pass width.
    return {Copy(m.norm), Copy(m.down), Copy(m.up), CopyF32(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      const core::GgmlType type = (*parts.begin())->type;
      for (const TensorRef* t : parts) {
        if (t->empty() || t->format_state != nullptr || t->type != type ||
            (type != core::GgmlType::kQ8_0 && !DecodeKQuant(type)) ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Projection(l.ssm_qkv);
      d.ssm_gate = Projection(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta}, true);
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Projection(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Projection(l.attn_q);
      d.attn_k = Projection(l.attn_k);
      d.attn_v = Projection(l.attn_v);
    }
    d.attn_out = Projection(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Projection(l.indexer_q);
    d.indexer_k = Projection(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = CopyF32(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Projection(l.ffn_gate_exps);
    d.ffn_up_exps = Projection(l.ffn_up_exps);
    d.ffn_down_exps = Projection(l.ffn_down_exps);
    constexpr flash_next::UploadHint kShared{.beside_routed_experts = true};
    d.shexp_gate = Projection(l.shexp_gate, kShared);
    d.shexp_up = Projection(l.shexp_up, kShared);
    d.shexp_down = Projection(l.shexp_down, kShared);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    if (!l.nextn_fc_embedding.empty()) {
      d.nextn_fc_embedding = Projection(l.nextn_fc_embedding);
      d.nextn_fc_hidden = Projection(l.nextn_fc_hidden);
    }
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  Uploaded Model(const ModelWeights& w, const MtpWeights* mtp,
                 std::uint32_t trunk_shards) {
    Uploaded u;
    // A tied head projects the embedding table, so it keeps the decode
    // rows and Q8_0 copy of any projection; an untied table is only
    // gathered from.
    const bool tied =
        w.output.format_state == nullptr && w.output.data == w.token_embd.data;
    u.token_embd = Copy(w.token_embd, !tied);
    u.output = tied ? u.token_embd : Projection(w.output);
    u.hc_head = Mixer(w.hc_head);
    u.layers.reserve(w.layers.size());
    for (const auto& l : w.layers) {
      u.layers.push_back(Layer(l));
      if (!ok) {
        return u;
      }
    }
    if (mtp != nullptr) {
      shard_base = trunk_shards;
      u.mtp = Layer(mtp->block);
    }
    return u;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

bool DeviceModel::SetDraftVocab(std::uint32_t prefix_rows,
                                std::vector<std::int32_t> specials,
                                std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  const DeviceTensor& head = draft_source();
  if (head.empty() || head.rows == 0 || head.format_state != nullptr ||
      (draft_source_.empty() && output_bytes_ % head.rows != 0))
    return fail("draft vocabulary needs a row-addressable output head");
  if (prefix_rows == 0 || prefix_rows >= head.rows)
    return fail("draft vocabulary prefix must leave rows out");
  for (const auto id : specials) {
    if (id < static_cast<std::int32_t>(prefix_rows) ||
        static_cast<std::uint32_t>(id) >= head.rows)
      return fail("draft special tokens must lie above the prefix");
  }
  // The prefix is the head's own leading rows: no copy. It keeps the Q6_K
  // rows: an IQ4_NL prefix lowered acceptance enough to lose on sampled
  // code and at 16K context.
  draft_output_ = head;
  draft_output_.rows = prefix_rows;
  draft_specials_ = std::move(specials);
  return true;
}

bool DeviceModel::PrepareDraftHead(std::string* error_msg) {
  if (!draft_full_output_.empty() || output_.decode == nullptr ||
      output_.decode_type != core::GgmlType::kQ6_K || output_.cols % 256 != 0)
    return true;
  const std::size_t blocks32 = std::size_t{output_.rows} * (output_.cols / 32);
  const std::size_t bytes = blocks32 * 18;  // block_iq4_nl
  void* rows = nullptr;
  if (hipMalloc(&rows, bytes) != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "draft head allocation failed";
    return false;
  }
  allocations_.push_back(rows);
  bytes_ += bytes;
  Q6KToIq4Nl(output_.decode, rows, blocks32, nullptr);
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "draft head conversion failed: " +
                   std::string(hipGetErrorString(status));
    return false;
  }
  draft_full_output_ = output_;
  draft_full_output_.decode = rows;
  draft_full_output_.decode_type = core::GgmlType::kIQ4_NL;
  return true;
}

bool DeviceModel::UseDraftSource(const DeviceTensor& head,
                                 std::string* error_msg) {
  TensorRef ref;
  ref.type = head.type;
  ref.cols = head.cols;
  const std::size_t row_bytes = ref.RowBytes();
  if (head.empty() || head.format_state != nullptr || row_bytes == 0 ||
      head.rows != output_.rows || head.cols != output_.cols ||
      head.experts != 1) {
    if (error_msg != nullptr)
      *error_msg = "draft source must be a GGUF matrix of the head's shape";
    return false;
  }
  draft_source_ = head;
  draft_source_bytes_ = row_bytes * head.rows;
  return true;
}

void WidenBF16(const std::uint16_t* src, float* dst,
               std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
    std::memcpy(dst + i, &bits, sizeof(float));
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const MtpWeights* mtp,
    flash_next::WeightSource& source, std::string* error_msg) {
  const auto trunk = source.Regions(false);
  std::vector<core::GgufMappedRegion> shards(trunk.begin(), trunk.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    const auto extra = source.Regions(true);
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  return UploadShards(w, mtp, std::move(shards), shard_count, &source,
                      error_msg);
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  return UploadShards(w, mtp, std::move(shards), shard_count, nullptr,
                      error_msg);
}

std::unique_ptr<DeviceModel> DeviceModel::UploadShards(
    const ModelWeights& w, const MtpWeights* mtp,
    std::vector<core::GgufMappedRegion> shards, std::uint32_t shard_count,
    flash_next::WeightSource* source, std::string* error_msg) {
  // The binder also accepts formats only the CPU reference reads. Reject the
  // routed experts and heads the HIP kernels cannot project before
  // allocating device weights.
  const auto reject = [&](const TensorRef& t) {
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format " +
                   std::string(core::ToString(t.type)) + ": " +
                   std::string(t.name);
    return false;
  };
  // A weight source's encoded tensors (format_state) run through the
  // stages, not these kernels.
  const auto expert_supported = [&](const TensorRef& t) {
    return t.format_state != nullptr || ExpertFormat(t.type) || reject(t);
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return expert_supported(l.ffn_gate_exps) &&
           expert_supported(l.ffn_up_exps) && expert_supported(l.ffn_down_exps);
  };
  if (!(w.output.format_state != nullptr || ProjectionFormat(w.output.type) ||
        reject(w.output)) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  // Measure first: the placement is decided once, from the bytes this walk
  // will request of the device heap.
  std::uint64_t measured = 0;
  {
    std::vector<Conversion> conversions;
    std::vector<void*> allocations;
    std::size_t bytes = 0;
    std::size_t max_half_cols = 1;
    std::size_t max_q8_cols = 32;
    Uploader measure{*stager, nullptr,       conversions, allocations,
                     bytes,   max_half_cols, max_q8_cols, error_msg};
    measure.source = source;
    (void)measure.Model(w, mtp, shard_count);
    if (source != nullptr)
      source->UploadDone(true);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  std::vector<Conversion> conversions;
  Uploader up{*stager,   &weights,          conversions,     m->allocations_,
              m->bytes_, m->max_half_cols_, m->max_q8_cols_, error_msg};
  up.measured = measured;
  up.source = source;
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  // The source hears of the walk's end only once its queued region reads
  // have landed, so it may free what they read into.
  bool finished = false;
  if (up.ok)
    finished = stager->Finish(error_msg);
  else if (source != nullptr)
    (void)stager->Finish(nullptr);
  if (source != nullptr)
    source->UploadDone(false);
  if (!up.ok || !finished) {
    return nullptr;
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  // Row-addressed users (draft tables) read the head's Q8_0 rows when it
  // also keeps K-quant decode rows.
  m->output_bytes_ =
      m->output_.decode != nullptr
          ? std::size_t{m->output_.rows} * (m->output_.cols / 32 * 34)
          : w.output.SizeBytes();
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
  for (const auto& c : conversions) {
    if (c.expand_q8) {
      ExpandQ8_0(c.source, static_cast<float*>(c.destination), c.count,
                 nullptr);
    } else {
      NarrowActivations(static_cast<const float*>(c.source), c.destination,
                        false, c.count, nullptr);
    }
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.source_bytes + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
