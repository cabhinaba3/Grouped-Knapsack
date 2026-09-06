#include "harness.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "dantzig.h"
#include "parallel.h"

namespace gka {
namespace {

using Clock = std::chrono::steady_clock;

double millisBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

double SolveOutcome::errorPercent() const { return (vStar - vGrouped) / vStar * 100.0; }

double resolveDelta(const Instance& inst, double relativeDelta, int numGroups) {
    return relativeDelta > 0.0 ? deltaFromRelativeTolerance(inst.v, inst.w, relativeDelta)
                                : deltaForGroupCount(inst.v, inst.w, numGroups);
}

SolveOutcome solveBoth(const Instance& inst, double delta) {
    SolveOutcome outcome;
    outcome.zDantzig = dantzig(inst);

    std::vector<Group> groups = groupByRatioBins(inst.v, inst.w, delta);
    GroupedResult grouped = groupedAllocation(inst, groups);

    outcome.numGroups = groups.size();
    outcome.zGrouped = std::move(grouped.z);
    outcome.vStar = value(inst.v, outcome.zDantzig);
    outcome.vGrouped = value(inst.v, outcome.zGrouped);
    return outcome;
}

Timings timeBoth(const Instance& inst, double delta, SolveOutcome& outcome) {
    auto t0 = Clock::now();
    std::vector<double> zDantzig = dantzig(inst);
    auto t1 = Clock::now();

    // Full grouped pipeline, timed together: build the groups (sort by
    // ratio + bin), then run Algorithm 1 (sort groups, prefix sums,
    // water-fill the boundary group) -- the one-shot cost of solving this
    // instance from scratch, the same basis on which Dantzig above is timed.
    auto t2 = Clock::now();
    std::vector<Group> groups = groupByRatioBins(inst.v, inst.w, delta);
    GroupedResult grouped = groupedAllocation(inst, groups);
    auto t3 = Clock::now();

    outcome.numGroups = groups.size();
    outcome.zDantzig = std::move(zDantzig);
    outcome.zGrouped = std::move(grouped.z);
    outcome.vStar = value(inst.v, outcome.zDantzig);
    outcome.vGrouped = value(inst.v, outcome.zGrouped);

    return {millisBetween(t0, t1), millisBetween(t2, t3)};
}

InstanceBatch generateBatch(const std::vector<InstanceSpec>& specs, const GenerateOptions& opts,
                            double relativeDelta, int numGroups) {
    InstanceBatch batch;
    batch.instances.resize(specs.size());
    batch.deltas.resize(specs.size());

    parallelFor(specs.size(), [&](std::size_t i) {
        GenerateOptions instOpts = opts;
        instOpts.seed = specs[i].seed;
        batch.instances[i] = generateInstance(specs[i].n, instOpts);
        batch.deltas[i] = resolveDelta(batch.instances[i], relativeDelta, numGroups);
    });

    return batch;
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("median: need at least one value");

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    const double upper = values[mid];
    if (values.size() % 2 != 0) return upper;

    // Even count: nth_element left everything below `mid` unsorted but all
    // of it <= upper, so the lower middle element is their maximum.
    const double lower = *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (lower + upper);
}

double mean(const std::vector<double>& values) {
    if (values.empty()) throw std::invalid_argument("mean: need at least one value");
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

}  // namespace gka
