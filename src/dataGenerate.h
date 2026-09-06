#pragma once

#include <cstdint>
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
    double totalCapacityCost() const;
};

struct GenerateOptions {
    double vMin = 1.0, vMax = 100.0;   // range for v_i
    double wMin = 1.0, wMax = 100.0;   // range for w_i
    double uMin = 0.0, uMax = 1.0;     // range for u_i, must stay within [0,1]
    double budgetRatio = 0.5;          // C = budgetRatio * F^w_{[n]}, must be in (0,1)
    uint64_t seed = 0;
};

// Generates one random instance with n items. The same (n, opts.seed) pair
// always reproduces the same instance, so callers may generate instances
// concurrently without affecting each other's results.
Instance generateInstance(int n, const GenerateOptions& opts);

// Generates one instance per entry of itemCounts, typically an increasing
// sequence of n, e.g. {10, 100, 1000, ...}.
std::vector<Instance> generateInstances(const std::vector<int>& itemCounts,
                                         const GenerateOptions& opts);

// Shape of the n-sweep the experiments run over.
//   Dense  - fine near the small end (steps of 1, then 50, then 250) and
//            doubling past 10000; good for resolving the runtime curve.
//   Sparse - a 1-2-5 sequence per decade; ~13 points to maxN, for quick runs.
enum class SweepShape { Dense, Sparse };

// Parses "dense"/"sparse"; throws std::invalid_argument on anything else.
SweepShape parseSweepShape(const std::string& text);
const char* sweepShapeName(SweepShape shape);

// Item counts from 1 (Dense) or 10 (Sparse) up to and including maxN.
// maxN is taken as long long and rejected above INT_MAX: the counts are
// used as `int` n throughout, and a sweep that walks past 2^31 would both
// overflow that and ask for more memory than any machine has.
std::vector<int> itemCounts(SweepShape shape, long long maxN);

// The sweep used when no --sweep/--max-n is given: Dense up to 100000,
// matching the sizes the README and the checked-in figures describe.
inline constexpr long long kDefaultMaxN = 100000;
std::vector<int> defaultItemCounts();

}  // namespace gka
