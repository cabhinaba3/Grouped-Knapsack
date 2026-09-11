#include "harness.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "dantzig.h"
#include "data_generate.h"
#include "grouped.h"
#include "parallel.h"

namespace gka {
namespace {

using Clock = std::chrono::steady_clock;

double MillisBetween(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

double SolveOutcome::ErrorPercent() const { return (v_star - v_grouped) / v_star * 100.0; }

double ResolveDelta(const Instance& inst, double relative_delta, int num_groups) {
  return relative_delta > 0.0 ? DeltaFromRelativeTolerance(inst.v, inst.w, relative_delta)
                              : DeltaForGroupCount(inst.v, inst.w, num_groups);
}

SolveOutcome SolveBoth(const Instance& inst, double delta) {
  SolveOutcome outcome;
  outcome.z_dantzig = Dantzig(inst);

  std::vector<Group> groups = GroupByRatioBins(inst.v, inst.w, delta);
  GroupedResult grouped = GroupedAllocation(inst, groups);

  outcome.num_groups = groups.size();
  outcome.z_grouped = std::move(grouped.z);
  outcome.v_star = Value(inst.v, outcome.z_dantzig);
  outcome.v_grouped = Value(inst.v, outcome.z_grouped);
  return outcome;
}

Timings TimeBoth(const Instance& inst, double delta, SolveOutcome& outcome) {
  auto dantzig_start = Clock::now();
  std::vector<double> z_dantzig = Dantzig(inst);
  auto dantzig_end = Clock::now();

  // Full grouped pipeline, timed together: build the groups (sort by
  // ratio + bin), then run Algorithm 1 (sort groups, prefix sums,
  // water-fill the boundary group) -- the one-shot cost of solving this
  // instance from scratch, the same basis on which Dantzig above is timed.
  auto grouped_start = Clock::now();
  std::vector<Group> groups = GroupByRatioBins(inst.v, inst.w, delta);
  GroupedResult grouped = GroupedAllocation(inst, groups);
  auto grouped_end = Clock::now();

  outcome.num_groups = groups.size();
  outcome.z_dantzig = std::move(z_dantzig);
  outcome.z_grouped = std::move(grouped.z);
  outcome.v_star = Value(inst.v, outcome.z_dantzig);
  outcome.v_grouped = Value(inst.v, outcome.z_grouped);

  return {MillisBetween(dantzig_start, dantzig_end), MillisBetween(grouped_start, grouped_end)};
}

InstanceBatch GenerateBatch(const std::vector<InstanceSpec>& specs, const GenerateOptions& opts,
                            double relative_delta, int num_groups) {
  InstanceBatch batch;
  batch.instances.resize(specs.size());
  batch.deltas.resize(specs.size());

  ParallelFor(specs.size(), [&](std::size_t i) {
    GenerateOptions inst_opts = opts;
    inst_opts.seed = specs[i].seed;
    batch.instances[i] = GenerateInstance(specs[i].n, inst_opts);
    batch.deltas[i] = ResolveDelta(batch.instances[i], relative_delta, num_groups);
  });

  return batch;
}

double Median(std::vector<double> values) {
  if (values.empty()) throw std::invalid_argument("Median: need at least one value");

  const std::size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
  const double upper = values[mid];
  if (values.size() % 2 != 0) return upper;

  // Even count: nth_element left everything below `mid` unsorted but all
  // of it <= upper, so the lower middle element is their maximum.
  const double lower =
      *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
  return 0.5 * (lower + upper);
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) throw std::invalid_argument("Mean: need at least one value");
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

}  // namespace gka
