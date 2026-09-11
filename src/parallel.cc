#include "parallel.h"

#include <tbb/global_control.h>

#include <thread>

namespace gka {

unsigned HardwareThreads() {
  unsigned count = std::thread::hardware_concurrency();
  return count == 0 ? 1u : count;
}

struct ThreadLimit::Impl {
  tbb::global_control control;
  explicit Impl(std::size_t max_threads)
      : control(tbb::global_control::max_allowed_parallelism, max_threads) {}
};

ThreadLimit::ThreadLimit(int max_threads) {
  if (max_threads > 0) impl_ = std::make_unique<Impl>(static_cast<std::size_t>(max_threads));
}

ThreadLimit::~ThreadLimit() = default;

int ThreadLimit::ActiveThreads() const {
  return static_cast<int>(
      tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));
}

}  // namespace gka
