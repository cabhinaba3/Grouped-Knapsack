#ifndef GKA_SRC_DATA_GENERATE_H_
#define GKA_SRC_DATA_GENERATE_H_

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

// Random instance generation for the bounded fractional knapsack problem
//
//   I = (v, w, u, C),  items indexed 0..n-1
//     v_i > 0   value
//     w_i > 0   cost
//     u_i in [0,1]  capacity limit
//     C >= 0    budget, with C < F^w_{[n]} := sum_i w_i * u_i
//
// so that the budget can never saturate every item at once.
namespace gka {

struct Instance {
  int n = 0;
  std::vector<double> v;
  std::vector<double> w;
  std::vector<double> u;
  double C = 0.0;

  // F^w_{[n]} = sum_i w_i * u_i, the total cost if every item were filled
  // to its capacity limit.
  double TotalCapacityCost() const;
};

// How one field (v, w, or u) gets drawn.
enum class DistributionKind { kUniform, kNormal, kCustom };

// Parses "uniform"/"normal" (case-sensitive); throws std::invalid_argument
// on anything else, including "custom" -- kCustom requires a sampler set
// directly in C++ and has no CLI spelling.
DistributionKind ParseDistributionKind(const std::string& text);
const char* DistributionKindName(DistributionKind kind);

// One field's random draw.
//   kUniform: samples uniformly in [lo, hi].
//   kNormal:  samples Normal(mean, stddev), re-drawing (bounded retries) any
//             value outside [lo, hi] so the field's hard constraints still
//             hold (v, w > 0; u in [0,1]) -- a rejection-sampled truncated
//             normal rather than a clamp, so the accepted distribution stays
//             genuinely Gaussian-shaped within the bounds.
//   kCustom:  calls `sampler(rng)` directly, an escape hatch for any other
//             distribution (lognormal, exponential, ...) a caller wants;
//             only reachable by constructing a FieldSpec in C++, not via CLI.
struct FieldSpec {
  DistributionKind dist = DistributionKind::kUniform;
  double lo = 0.0, hi = 1.0;                        // kUniform's range; kNormal's rejection bounds
  double mean = 0.0, stddev = 1.0;                  // kNormal only
  std::function<double(std::mt19937_64&)> sampler;  // kCustom only
};

struct GenerateOptions {
  FieldSpec v_spec{DistributionKind::kUniform, 1.0, 100.0, 50.5, 16.5, nullptr};  // v_i
  FieldSpec w_spec{DistributionKind::kUniform, 1.0, 100.0, 50.5, 16.5, nullptr};  // w_i
  FieldSpec u_spec{DistributionKind::kUniform, 0.0, 1.0, 0.5, 1.0 / 6, nullptr};  // u_i, in [0,1]
  double budget_ratio = 0.5;  // C = budget_ratio * F^w_{[n]}, must be in (0,1)
  uint64_t seed = 0;
};

// Generates one random instance with n items. The same (n, opts.seed) pair
// always reproduces the same instance, so callers may generate instances
// concurrently without affecting each other's results.
Instance GenerateInstance(int n, const GenerateOptions& opts);

// Generates one instance per entry of item_counts, typically an increasing
// sequence of n, e.g. {10, 100, 1000, ...}.
std::vector<Instance> GenerateInstances(const std::vector<int>& item_counts,
                                        const GenerateOptions& opts);

// Shape of the n-sweep the experiments run over.
//   kDense  - fine near the small end (steps of 1, then 50, then 250) and
//             doubling past 10000; good for resolving the runtime curve.
//   kSparse - a 1-2-5 sequence per decade; ~13 points to max_n, for quick runs.
enum class SweepShape { kDense, kSparse };

// Parses "dense"/"sparse"; throws std::invalid_argument on anything else.
SweepShape ParseSweepShape(const std::string& text);
const char* SweepShapeName(SweepShape shape);

// Item counts from 1 (kDense) or 10 (kSparse) up to and including max_n.
// max_n is taken as long long and rejected above INT_MAX: the counts are
// used as `int` n throughout, and a sweep that walks past 2^31 would both
// overflow that and ask for more memory than any machine has.
std::vector<int> ItemCounts(SweepShape shape, long long max_n);

// The sweep used when no --sweep/--max-n is given: kDense up to 100000,
// matching the sizes the README and the checked-in figures describe.
inline constexpr long long kDefaultMaxN = 100000;
std::vector<int> DefaultItemCounts();

}  // namespace gka

#endif  // GKA_SRC_DATA_GENERATE_H_
