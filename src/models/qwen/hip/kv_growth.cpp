#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kv_growth.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "src/models/qwen/continuation_hooks.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
namespace {

bool Fail(std::string* error, const char* what, hipError_t status) {
  (void)hipGetLastError();  // HIP errors are sticky
  if (error)
    *error = std::string(what) + ": " + hipGetErrorString(status);
  return false;
}

}  // namespace

QwenGrowableKv::~QwenGrowableKv() {
  if (base_ == nullptr)
    return;
  (void)Shrink(0);
  pool_->ReleaseRange(
      base_, reserved_bytes_,
      *std::max_element(unmapped_at_.begin(), unmapped_at_.end()));
}

std::size_t QwenGrowableKv::FullBytes(std::size_t segments,
                                      std::size_t row_bytes,
                                      std::size_t max_rows) noexcept {
  const std::size_t total = segments * row_bytes * max_rows;
  const std::size_t chunk = KvPool::kChunkBytes;
  return (total + chunk - 1) / chunk * chunk;
}

bool QwenGrowableKv::Reserve(KvPool* pool, std::size_t segments,
                             std::size_t row_bytes, std::size_t max_rows,
                             std::string* error) {
  if (base_ != nullptr || pool == nullptr || segments == 0 || row_bytes == 0 ||
      max_rows == 0) {
    if (error)
      *error = "invalid growable K/V reservation";
    return false;
  }
  const std::size_t bytes = FullBytes(segments, row_bytes, max_rows);
  hipError_t status = hipSuccess;
  KvPool::Clock::time_point unmapped_at{};
  void* base = pool->ReserveRange(bytes, &status, &unmapped_at);
  if (base == nullptr)
    return Fail(error, "K/V address reservation", status);
  pool_ = pool;
  base_ = static_cast<std::uint8_t*>(base);
  reserved_bytes_ = bytes;
  chunk_ = KvPool::kChunkBytes;
  segments_ = segments;
  row_bytes_ = row_bytes;
  max_rows_ = max_rows;
  chunks_.assign(bytes / chunk_, hipMemGenericAllocationHandle_t{});
  mapped_.assign(bytes / chunk_, 0);
  unmapped_at_.assign(bytes / chunk_, unmapped_at);
  return true;
}

// Visits, in increasing order and once each, every chunk that holds a byte of
// rows [0, rows) of some segment.
template<typename Visit>
void QwenGrowableKv::ForEachChunk(std::size_t rows, Visit&& visit) const {
  rows = std::min(rows, max_rows_);
  if (rows == 0)
    return;
  const std::size_t stride = max_rows_ * row_bytes_;
  std::size_t next = 0;  // first chunk not yet visited
  for (std::size_t segment = 0; segment < segments_; ++segment) {
    const std::size_t first = segment * stride;
    const std::size_t end = first + rows * row_bytes_;
    for (std::size_t chunk = std::max(next, first / chunk_);
         chunk < (end + chunk_ - 1) / chunk_; ++chunk)
      visit(chunk);
    next = std::max(next, (end + chunk_ - 1) / chunk_);
  }
}

std::size_t QwenGrowableKv::GrowthBytes(std::size_t rows) const noexcept {
  std::size_t bytes = 0;
  ForEachChunk(rows, [&](std::size_t chunk) {
    if (!mapped_[chunk])
      bytes += chunk_;
  });
  return bytes;
}

bool QwenGrowableKv::Grow(std::size_t rows, hipStream_t stream,
                          std::string* error) {
  rows = std::min(rows, max_rows_);
  const std::size_t growth = GrowthBytes(rows);
  if (growth == 0) {
    rows_ = std::max(rows_, rows);
    return true;
  }
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(stream, &capture) != hipSuccess ||
      capture != hipStreamCaptureStatusNone) {
    (void)hipGetLastError();
    if (error)
      *error = "K/V rows cannot grow during graph capture";
    return false;
  }
  if (!pool_->Charge(growth)) {
    if (error)
      *error = "K/V memory budget exhausted";
    return false;
  }
  // Addresses that lost a mapping recently fault when remapped too soon.
  KvPool::Clock::time_point latest{};
  ForEachChunk(rows, [&](std::size_t chunk) {
    if (!mapped_[chunk])
      latest = std::max(latest, unmapped_at_[chunk]);
  });
  if (latest != KvPool::Clock::time_point{})
    KvPool::AwaitQuarantine(latest);
  const auto prop = KvChunkProp();
  hipMemAccessDesc access{};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  std::vector<std::size_t> added;
  const auto undo = [&](hipError_t status, const char* what) {
    for (const std::size_t chunk : added) {
      unmapped_at_[chunk] = KvPool::Clock::now();
      (void)hipMemUnmap(base_ + chunk * chunk_, chunk_);
      (void)hipMemRelease(chunks_[chunk]);
      chunks_[chunk] = {};
      mapped_[chunk] = 0;
    }
    mapped_chunks_ -= added.size();
    pool_->Refund(growth);
    return Fail(error, what, status);
  };
  hipError_t status = hipSuccess;
  const char* failure = nullptr;
  ForEachChunk(rows, [&](std::size_t chunk) {
    if (failure != nullptr || mapped_[chunk])
      return;
    hipMemGenericAllocationHandle_t handle{};
    status = hipMemCreate(&handle, chunk_, &prop, 0);
    if (status != hipSuccess) {
      failure = "K/V chunk allocation";
      return;
    }
    std::uint8_t* at = base_ + chunk * chunk_;
    status = hipMemMap(at, chunk_, 0, handle, 0);
    if (status != hipSuccess) {
      (void)hipMemRelease(handle);
      failure = "K/V chunk mapping";
      return;
    }
    chunks_[chunk] = handle;
    mapped_[chunk] = 1;
    ++mapped_chunks_;
    added.push_back(chunk);
    status = hipMemSetAccess(at, chunk_, &access, 1);
    if (status != hipSuccess)
      failure = "K/V chunk access";
  });
  if (failure != nullptr)
    return undo(status, failure);
  // New pages map beside rows queued work may be using; those stay mapped, so
  // the queue need not drain. Zero runs of consecutive new chunks.
  for (std::size_t i = 0; i < added.size();) {
    std::size_t j = i + 1;
    while (j < added.size() && added[j] == added[j - 1] + 1)
      ++j;
    status =
        hipMemsetAsync(base_ + added[i] * chunk_, 0, (j - i) * chunk_, stream);
    if (status != hipSuccess)
      return undo(status, "K/V chunk zero fill");
    i = j;
  }
  rows_ = std::max(rows_, rows);
  return true;
}

std::size_t QwenGrowableKv::Shrink(std::size_t rows) noexcept {
  rows = std::min(rows, max_rows_);
  std::vector<std::uint8_t> keep(mapped_.size(), 0);
  ForEachChunk(rows, [&](std::size_t chunk) { keep[chunk] = 1; });
  std::size_t released = 0;
  const auto now = KvPool::Clock::now();
  for (std::size_t chunk = 0; chunk < mapped_.size(); ++chunk) {
    if (!mapped_[chunk] || keep[chunk])
      continue;
    unmapped_at_[chunk] = now;
    (void)hipMemUnmap(base_ + chunk * chunk_, chunk_);
    (void)hipMemRelease(chunks_[chunk]);
    chunks_[chunk] = {};
    mapped_[chunk] = 0;
    --mapped_chunks_;
    released += chunk_;
  }
  rows_ = std::min(rows_, rows);
  if (released != 0)
    pool_->Refund(released);
  return released;
}

void QwenKvSessions::Register(QwenGpuExecutor* executor) {
  const std::lock_guard lock(mutex_);
  executors_.push_back(executor);
}

void QwenKvSessions::Unregister(QwenGpuExecutor* executor) {
  const std::lock_guard lock(mutex_);
  std::erase(executors_, executor);
}

void QwenKvSessions::Reserve(QwenGpuExecutor& executor, std::size_t end,
                             std::span<QwenGpuExecutor* const> busy) {
  const std::lock_guard lock(mutex_);
  auto& arena = executor.arena_;
  auto& kv = *arena.kv_rows_;
  arena.kv_live_rows_ = std::min(end, kv.max_rows());
  const std::size_t rows = arena.KvRowsToKeep();
  const std::size_t growth = kv.GrowthBytes(rows);
  if (growth == 0)
    return;
  std::vector<QwenGpuExecutor*> keep(busy.begin(), busy.end());
  keep.push_back(&executor);
  if (pool_.Available() < growth)
    (void)ReleaseIdle(growth - pool_.Available(), keep);
  if (pool_.Available() < growth)
    throw std::runtime_error(
        "K/V memory exhausted: " + std::to_string(growth) +
        " more bytes needed, " + std::to_string(pool_.mapped()) + " of " +
        std::to_string(pool_.budget()) + " budgeted bytes in use");
  std::string error;
  // The device heap can run out before the budget: release idle rows and
  // retry once.
  if (kv.Grow(rows, executor.arena_.stream, nullptr) ||
      (ReleaseIdle(growth, keep) > 0 &&
       kv.Grow(rows, executor.arena_.stream, nullptr)) ||
      kv.Grow(rows, executor.arena_.stream, &error))
    return;
  throw std::runtime_error("K/V memory exhausted: " + error);
}

std::size_t QwenKvSessions::ReleaseIdle(
    std::size_t want, std::span<QwenGpuExecutor* const> busy) {
  std::size_t released = 0;
  bool drained = false;
  for (auto* executor : executors_) {
    if (released >= want)
      break;
    if (std::ranges::find(busy, executor) != busy.end())
      continue;
    auto& arena = executor->arena_;
    auto& kv = *arena.kv_rows_;
    const std::size_t keep = arena.KvRowsToKeep();
    if (kv.rows() <= keep)
      continue;
    // Releasing rows is an overwrite for anything that borrowed them.
    try {
      if (executor->continuation_hooks_)
        executor->continuation_hooks_->BeforeWrite(
            static_cast<std::uint32_t>(arena.kv_live_rows_),
            static_cast<std::uint32_t>(kv.rows()));
    } catch (const std::exception&) {
      continue;  // a pending transfer or failed slot keeps its rows
    }
    if (!drained) {
      // Unmapping is not ordered after queued work or borrowed copies.
      if (hipDeviceSynchronize() != hipSuccess) {
        (void)hipGetLastError();
        return released;
      }
      drained = true;
    }
    released += kv.Shrink(keep);
  }
  return released;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
