#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <execution>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

// Thin wrapper over the C++17 parallel STL (backed by TBB on libstdc++).
//
// Two things the raw std::for_each(std::execution::par, ...) does not give
// us and every call site here needs:
//   - an exception escaping a parallel body calls std::terminate, so bodies
//     must be wrapped and the first failure rethrown on the calling thread;
//   - the worker count is a process-wide TBB setting, not a policy argument,
//     so capping it needs an RAII handle rather than a parameter.
namespace gka {

// Number of hardware threads, or 1 when the platform will not say.
unsigned hardwareThreads();

// Caps how many threads the parallel STL may use for as long as it lives.
// maxThreads <= 0 means "leave the default alone", i.e. use every core.
// Construct one in main() before any parallelFor and keep it alive.
class ThreadLimit {
public:
    explicit ThreadLimit(int maxThreads);
    ~ThreadLimit();
    ThreadLimit(const ThreadLimit&) = delete;
    ThreadLimit& operator=(const ThreadLimit&) = delete;

    // Threads actually available after the cap, for reporting.
    int activeThreads() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Runs body(i) for every i in [0, count), in parallel and in no particular
// order. If any body throws, the remaining ones still run and the first
// exception raised is rethrown here once they have all finished.
template <typename Body>
void parallelFor(std::size_t count, Body&& body) {
    if (count == 0) return;
    if (count == 1) {  // not worth a task graph, and keeps stack traces simple
        body(std::size_t{0});
        return;
    }

    std::vector<std::size_t> indices(count);
    std::iota(indices.begin(), indices.end(), std::size_t{0});

    std::mutex errorMutex;
    std::exception_ptr firstError;

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i) {
        try {
            body(i);
        } catch (...) {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (!firstError) firstError = std::current_exception();
        }
    });

    if (firstError) std::rethrow_exception(firstError);
}

}  // namespace gka
