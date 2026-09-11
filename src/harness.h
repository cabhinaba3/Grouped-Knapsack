#ifndef GKA_SRC_HARNESS_H_
#define GKA_SRC_HARNESS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "data_generate.h"

// The experiment-side layer both binaries share: solve one instance with
// both methods, time those solves, build batches of instances in parallel,
// and aggregate repeated measurements.
//
// Nothing here changes what dantzig.cc or grouped.cc compute; it only
// decides what gets run, when it gets timed, and what runs concurrently.
namespace gka {

// One instance solved both ways.
struct SolveOutcome {
  std::vector<double> z_dantzig;  // exact fractional optimum
  std::vector<double> z_grouped;  // Algorithm 1's allocation
  double v_star = 0.0;            // V*  = sum_i v_i * z_dantzig_i
  double v_grouped = 0.0;         // V^gp = sum_i v_i * z_grouped_i
  std::size_t num_groups = 0;     // m, the number of groups actually formed

  // (V* - V^gp) / V* as a percentage.
  double ErrorPercent() const;
};

// The grouping tolerance for one instance: when `relative_delta` is positive
// (must be in (0,1)), that fraction of this instance's trimmed ratio spread;
// otherwise the value derived from the same spread that yields roughly
// num_groups groups. Deriving either involves a sort, so callers do it once
// per instance rather than once per repetition.
double ResolveDelta(const Instance& inst, double relative_delta, int num_groups);

// Dantzig plus the full grouped pipeline (build the groups, then run
// Algorithm 1). Pure and side-effect free, so it is safe to call
// concurrently on distinct instances.
SolveOutcome SolveBoth(const Instance& inst, double delta);

// Wall-clock milliseconds for the same two solves, measured separately and
// leaving the solutions in `outcome`. Timing only means something on an
// otherwise idle core, so this must be called from a single thread while no
// other solve is running.
struct Timings {
  double dantzig_ms = 0.0;
  double grouped_ms = 0.0;
};
Timings TimeBoth(const Instance& inst, double delta, SolveOutcome& outcome);

// A batch of instances to build up front, one per (n, seed) pair.
struct InstanceSpec {
  int n = 0;
  std::uint64_t seed = 0;
};

struct InstanceBatch {
  std::vector<Instance> instances;
  std::vector<double> deltas;  // deltas[i] goes with instances[i]
};

// Builds every spec's instance and grouping tolerance in parallel. This is
// the untimed setup work, so running it concurrently costs the measurements
// nothing. opts supplies the value/cost/limit ranges and budget ratio; each
// spec's own seed overrides opts.seed. relative_delta and num_groups are
// forwarded to ResolveDelta as-is.
InstanceBatch GenerateBatch(const std::vector<InstanceSpec>& specs, const GenerateOptions& opts,
                            double relative_delta, int num_groups);

double Median(std::vector<double> values);
double Mean(const std::vector<double>& values);

}  // namespace gka

#endif  // GKA_SRC_HARNESS_H_
