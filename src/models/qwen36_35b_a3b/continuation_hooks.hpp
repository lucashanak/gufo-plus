#ifndef GUFO_MODELS_QWEN36_35B_A3B_CONTINUATION_HOOKS_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_CONTINUATION_HOOKS_HPP_

#include <cstdint>

namespace gufo::models::qwen36_35b_a3b {
// Installed only in sessions owned by the continuation adapter; standalone
// sessions leave the pointer null. The callbacks run on the host before
// state changes or device work is submitted (including a graph replay).
class ContinuationHooks {
public:
  virtual ~ContinuationHooks() = default;
  virtual void BeforeExecution() = 0;
  // Trunk (draft false) or draft-block rows [first, end) are about to be
  // written or released.
  virtual void BeforeWrite(bool draft, std::uint32_t first,
                           std::uint32_t end) = 0;
  virtual void BeforeReset() = 0;
  virtual void AfterReset() = 0;
  virtual void BeforeRestore() = 0;
};
}  // namespace gufo::models::qwen36_35b_a3b
#endif
