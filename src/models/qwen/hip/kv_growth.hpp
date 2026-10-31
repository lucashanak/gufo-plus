#ifndef GUFO_MODELS_QWEN_HIP_KV_GROWTH_HPP_
#define GUFO_MODELS_QWEN_HIP_KV_GROWTH_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/kv_memory.hpp"

namespace gufo::hip {

class QwenGpuExecutor;

/// The K and V planes of one session: `segments` (plane x attention layer)
/// regions of `max_rows` rows each, `stride` bytes apart, in one virtual
/// reservation. Rows [0, rows()) of every segment are backed by mapped chunks;
/// the layout, and every address kernels and captured graphs use, is the one
/// a preallocated plane has. New chunks read as zeros.
class QwenGrowableKv {
public:
  QwenGrowableKv() = default;
  QwenGrowableKv(const QwenGrowableKv&) = delete;
  QwenGrowableKv& operator=(const QwenGrowableKv&) = delete;
  ~QwenGrowableKv();

  [[nodiscard]] bool Reserve(KvPool* pool, std::size_t segments,
                             std::size_t row_bytes, std::size_t max_rows,
                             std::string* error);
  [[nodiscard]] void* data() const noexcept { return base_; }
  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t max_rows() const noexcept { return max_rows_; }
  [[nodiscard]] std::size_t mapped_bytes() const noexcept {
    return mapped_chunks_ * chunk_;
  }
  /// Bytes every segment occupies once all its rows are backed.
  [[nodiscard]] static std::size_t FullBytes(std::size_t segments,
                                             std::size_t row_bytes,
                                             std::size_t max_rows) noexcept;
  /// Bytes Grow(rows) would map.
  [[nodiscard]] std::size_t GrowthBytes(std::size_t rows) const noexcept;
  /// Backs rows [0, rows) of every segment, charging the pool. New chunks are
  /// zeroed on `stream` before later work on it. On failure nothing changes.
  [[nodiscard]] bool Grow(std::size_t rows, hipStream_t stream,
                          std::string* error);
  /// Releases chunks no segment needs for rows [0, rows). Every reader of the
  /// released rows has drained. Returns the bytes released.
  std::size_t Shrink(std::size_t rows) noexcept;

private:
  template<typename Visit>
  void ForEachChunk(std::size_t rows, Visit&& visit) const;

  KvPool* pool_{nullptr};
  std::uint8_t* base_{nullptr};
  std::size_t reserved_bytes_{0};
  std::size_t chunk_{0};
  std::size_t segments_{0};
  std::size_t row_bytes_{0};
  std::size_t max_rows_{0};
  std::size_t rows_{0};
  std::size_t mapped_chunks_{0};
  std::vector<hipMemGenericAllocationHandle_t> chunks_;
  std::vector<std::uint8_t> mapped_;
  /// When each chunk's address last lost a mapping (the epoch if never):
  /// remapping it waits out KvPool::kRemapQuarantine.
  std::vector<KvPool::Clock::time_point> unmapped_at_;
};

/// The sessions of one served model sharing a K/V byte budget. A growth that
/// does not fit first releases rows idle sessions no longer hold.
class QwenKvSessions {
public:
  /// `budget` zero means no limit beyond the device heap. `arena_bytes` of
  /// address space are reserved now, before sessions allocate and free their
  /// other buffers (KvPool::ReserveArena); zero skips the arena.
  explicit QwenKvSessions(std::size_t budget, std::size_t arena_bytes = 0)
      : pool_(budget) {
    if (arena_bytes != 0)
      (void)pool_.ReserveArena(arena_bytes, nullptr);
  }
  QwenKvSessions(const QwenKvSessions&) = delete;
  QwenKvSessions& operator=(const QwenKvSessions&) = delete;

  [[nodiscard]] KvPool& pool() noexcept { return pool_; }
  [[nodiscard]] std::size_t budget() const noexcept { return pool_.budget(); }

  void Register(QwenGpuExecutor* executor);
  void Unregister(QwenGpuExecutor* executor);
  /// Records that `executor` holds rows [0, end) and backs them (plus the
  /// arena's kernel margin). `busy` sessions (and `executor`) keep every
  /// mapped row. Throws "K/V memory exhausted" when the budget or the device
  /// heap cannot hold the rows; no mapping changes then.
  void Reserve(QwenGpuExecutor& executor, std::size_t end,
               std::span<QwenGpuExecutor* const> busy);

private:
  std::size_t ReleaseIdle(std::size_t want,
                          std::span<QwenGpuExecutor* const> busy);

  // Declared before the executors' planes are released: executors unregister
  // themselves, and the pool outlives every reservation.
  KvPool pool_;
  std::recursive_mutex mutex_;
  std::vector<QwenGpuExecutor*> executors_;
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // GUFO_MODELS_QWEN_HIP_KV_GROWTH_HPP_
