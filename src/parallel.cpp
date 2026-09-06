#include "parallel.h"

#include <thread>

#include <tbb/global_control.h>

namespace gka {

unsigned hardwareThreads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1u : n;
}

struct ThreadLimit::Impl {
    tbb::global_control control;
    explicit Impl(std::size_t maxThreads)
        : control(tbb::global_control::max_allowed_parallelism, maxThreads) {}
};

ThreadLimit::ThreadLimit(int maxThreads) {
    if (maxThreads > 0)
        impl_ = std::make_unique<Impl>(static_cast<std::size_t>(maxThreads));
}

ThreadLimit::~ThreadLimit() = default;

int ThreadLimit::activeThreads() const {
    return static_cast<int>(
        tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));
}

}  // namespace gka
