#ifndef GKA_SRC_PARALLEL_H_
#define GKA_SRC_PARALLEL_H_

#include <tbb/parallel_for.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>

// Thin wrapper over tbb::parallel_for.
//
// Two things the raw tbb::parallel_for does not give us and every call site
// here needs:
//   - an exception escaping a parallel body calls std::terminate, so bodies
//     must be wrapped and the first failure rethrown on the calling thread;
//   - the worker count is a process-wide TBB setting, not a policy argument,
//     so capping it needs an RAII handle rather than a parameter.
namespace gka {

// Number of hardware threads, or 1 when the platform will not say.
unsigned HardwareThreads();

// Caps how many threads TBB may use for as long as it lives. max_threads <= 0
// means "leave the default alone", i.e. use every core. Construct one in
// main() before any ParallelFor and keep it alive.
class ThreadLimit {
 public:
  explicit ThreadLimit(int max_threads);
  ~ThreadLimit();
  ThreadLimit(const ThreadLimit&) = delete;
  ThreadLimit& operator=(const ThreadLimit&) = delete;

  // Threads actually available after the cap, for reporting.
  int ActiveThreads() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Runs body(i) for every i in [0, count), in parallel and in no particular
// order. If any body throws, the remaining ones still run and the first
// exception raised is rethrown here once they have all finished.
template <typename Body>
void ParallelFor(std::size_t count, Body&& body) {
  if (count == 0) return;
  if (count == 1) {  // not worth a task graph, and keeps stack traces simple
    body(std::size_t{0});
    return;
  }

  std::mutex error_mutex;
  std::exception_ptr first_error;

  tbb::parallel_for(std::size_t{0}, count, [&](std::size_t i) {
    try {
      body(i);
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!first_error) first_error = std::current_exception();
    }
  });

  if (first_error) std::rethrow_exception(first_error);
}

}  // namespace gka

#endif  // GKA_SRC_PARALLEL_H_
