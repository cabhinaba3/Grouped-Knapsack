#include "dataGenerate.h"

#include <algorithm>
#include <climits>
#include <random>
#include <stdexcept>

namespace gka {

double Instance::totalCapacityCost() const {
    double total = 0.0;
    for (int i = 0; i < n; ++i) { total += w[i] * u[i]; }
    return total;
}

namespace {

void validateOptions(const GenerateOptions& opts) {
    if (opts.vMin <= 0.0 || opts.vMax < opts.vMin)
        throw std::invalid_argument("dataGenerate: need 0 < vMin <= vMax");
    if (opts.wMin <= 0.0 || opts.wMax < opts.wMin)
        throw std::invalid_argument("dataGenerate: need 0 < wMin <= wMax");
    if (opts.uMin < 0.0 || opts.uMax > 1.0 || opts.uMax < opts.uMin)
        throw std::invalid_argument("dataGenerate: need 0 <= uMin <= uMax <= 1");
    if (opts.budgetRatio <= 0.0 || opts.budgetRatio >= 1.0)
        throw std::invalid_argument("dataGenerate: need 0 < budgetRatio < 1");
}

// Combines the batch seed with n so every item count in a sweep gets its
// own reproducible random stream instead of repeating the same one.
uint64_t seedFor(uint64_t seed, int n) {
    return seed * 1000003ULL + static_cast<uint64_t>(n) + 1ULL;
}

}  // namespace

Instance generateInstance(int n, const GenerateOptions& opts) {
    if (n <= 0) throw std::invalid_argument("dataGenerate: n must be positive");
    validateOptions(opts);

    std::mt19937_64 rng(seedFor(opts.seed, n));
    std::uniform_real_distribution<double> vd(opts.vMin, opts.vMax);
    std::uniform_real_distribution<double> wd(opts.wMin, opts.wMax);
    std::uniform_real_distribution<double> ud(opts.uMin, opts.uMax);

    Instance inst;
    inst.n = n;
    inst.v.resize(n);
    inst.w.resize(n);
    inst.u.resize(n);

    double fTotal = 0.0;
    // u_i can legitimately be 0, so with tiny probability every draw is 0
    // (F^w_{[n]} = 0), which leaves no room for a positive budget C < F.
    // Retry the u draws a bounded number of times in that unlikely case.
    constexpr int kMaxRetries = 100;
    for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
        fTotal = 0.0;
        for (int i = 0; i < n; ++i) {
            if (attempt == 0) {
                inst.v[i] = vd(rng);
                inst.w[i] = wd(rng);
            }
            inst.u[i] = ud(rng);
            fTotal += inst.w[i] * inst.u[i];
        }
        if (fTotal > 0.0) break;
        if (attempt == kMaxRetries)
            throw std::runtime_error("dataGenerate: could not draw a positive F^w_{[n]}; widen uMax");
    }

    inst.C = opts.budgetRatio * fTotal;
    return inst;
}

std::vector<Instance> generateInstances(const std::vector<int>& itemCounts,
                                         const GenerateOptions& opts) {
    std::vector<Instance> out;
    out.reserve(itemCounts.size());
    for (int n : itemCounts) out.push_back(generateInstance(n, opts));
    return out;
}

SweepShape parseSweepShape(const std::string& text) {
    if (text == "dense") return SweepShape::Dense;
    if (text == "sparse") return SweepShape::Sparse;
    throw std::invalid_argument("--sweep: expected 'dense' or 'sparse', got '" + text + "'");
}

const char* sweepShapeName(SweepShape shape) {
    return shape == SweepShape::Dense ? "dense" : "sparse";
}

std::vector<int> itemCounts(SweepShape shape, long long maxN) {
    if (maxN <= 0) throw std::invalid_argument("itemCounts: maxN must be positive");
    if (maxN > static_cast<long long>(INT_MAX))
        throw std::invalid_argument("itemCounts: maxN must fit in an int (<= 2147483647)");

    std::vector<int> counts;
    if (shape == SweepShape::Dense) {
        for (long long i = 1; i <= maxN;) {
            counts.push_back(static_cast<int>(i));
            if (i < 50) { i += 1; }
            else if (i < 1000) { i += 50; }
            else if (i < 10000) { i += 250; }
            else { i *= 2; }
        }
    } else {
        // 10, 20, 50, 100, 200, 500, ... : three points per decade.
        constexpr int kMantissas[] = {1, 2, 5};
        for (long long decade = 10; decade <= maxN; decade *= 10) {
            for (int m : kMantissas) {
                long long n = decade * m;
                if (n > maxN) break;
                counts.push_back(static_cast<int>(n));
            }
        }
    }

    // The step rules above overshoot maxN rather than landing on it, so a
    // sweep asked for "up to 100000" would otherwise stop at 80000.
    if (counts.empty() || counts.back() != static_cast<int>(maxN))
        counts.push_back(static_cast<int>(maxN));

    return counts;
}

std::vector<int> defaultItemCounts() { return itemCounts(SweepShape::Dense, kDefaultMaxN); }

}  // namespace gka
