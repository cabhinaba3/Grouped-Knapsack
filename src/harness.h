#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dataGenerate.h"
#include "grouped.h"

// The experiment-side layer both binaries share: solve one instance with
// both methods, time those solves, build batches of instances in parallel,
// and aggregate repeated measurements.
//
// Nothing here changes what dantzig.cpp or grouped.cpp compute; it only
// decides what gets run, when it gets timed, and what runs concurrently.
namespace gka {

// One instance solved both ways.
struct SolveOutcome {
    std::vector<double> zDantzig;   // exact fractional optimum
    std::vector<double> zGrouped;   // Algorithm 1's allocation
    double vStar = 0.0;             // V*  = sum_i v_i * zDantzig_i
    double vGrouped = 0.0;          // V^gp = sum_i v_i * zGrouped_i
    std::size_t numGroups = 0;      // m, the number of groups actually formed

    // (V* - V^gp) / V* as a percentage.
    double errorPercent() const;
};

// The grouping tolerance for one instance: `delta` when positive, otherwise
// the value derived from this instance's own ratio spread that yields
// roughly numGroups groups. Deriving delta involves a sort, so callers do
// it once per instance rather than once per repetition.
double resolveDelta(const Instance& inst, double delta, int numGroups);

// Dantzig plus the full grouped pipeline (build the groups, then run
// Algorithm 1). Pure and side-effect free, so it is safe to call
// concurrently on distinct instances.
SolveOutcome solveBoth(const Instance& inst, double delta);

// Wall-clock milliseconds for the same two solves, measured separately and
// leaving the solutions in `outcome`. Timing only means something on an
// otherwise idle core, so this must be called from a single thread while no
// other solve is running.
struct Timings {
    double dantzigMs = 0.0;
    double groupedMs = 0.0;
};
Timings timeBoth(const Instance& inst, double delta, SolveOutcome& outcome);

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
// spec's own seed overrides opts.seed.
InstanceBatch generateBatch(const std::vector<InstanceSpec>& specs, const GenerateOptions& opts,
                            double delta, int numGroups);

double median(std::vector<double> values);
double mean(const std::vector<double>& values);

}  // namespace gka
