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

void validateDistributionParams(const char* name, const FieldSpec& spec) {
    if (spec.dist == DistributionKind::Normal && spec.stddev <= 0.0)
        throw std::invalid_argument(std::string("dataGenerate: ") + name + "'s stddev must be positive");
    if (spec.dist == DistributionKind::Custom && !spec.sampler)
        throw std::invalid_argument(std::string("dataGenerate: ") + name +
                                    " distribution is Custom but no sampler was set");
}

void validateOptions(const GenerateOptions& opts) {
    if (opts.vSpec.lo <= 0.0 || opts.vSpec.hi < opts.vSpec.lo)
        throw std::invalid_argument("dataGenerate: need 0 < v's lo <= hi");
    if (opts.wSpec.lo <= 0.0 || opts.wSpec.hi < opts.wSpec.lo)
        throw std::invalid_argument("dataGenerate: need 0 < w's lo <= hi");
    if (opts.uSpec.lo < 0.0 || opts.uSpec.hi > 1.0 || opts.uSpec.hi < opts.uSpec.lo)
        throw std::invalid_argument("dataGenerate: need 0 <= u's lo <= hi <= 1");
    if (opts.budgetRatio <= 0.0 || opts.budgetRatio >= 1.0)
        throw std::invalid_argument("dataGenerate: need 0 < budgetRatio < 1");

    validateDistributionParams("v", opts.vSpec);
    validateDistributionParams("w", opts.wSpec);
    validateDistributionParams("u", opts.uSpec);
}

// Combines the batch seed with n so every item count in a sweep gets its
// own reproducible random stream instead of repeating the same one.
uint64_t seedFor(uint64_t seed, int n) {
    return seed * 1000003ULL + static_cast<uint64_t>(n) + 1ULL;
}

// Draws one value for a field. Normal is rejection-sampled (redraw any
// value outside [lo, hi]) rather than clamped, so the accepted values stay
// genuinely Normal-shaped within the bounds instead of piling up at an edge.
double sampleField(const char* name, const FieldSpec& spec, std::mt19937_64& rng) {
    switch (spec.dist) {
        case DistributionKind::Uniform: {
            std::uniform_real_distribution<double> d(spec.lo, spec.hi);
            return d(rng);
        }
        case DistributionKind::Normal: {
            std::normal_distribution<double> d(spec.mean, spec.stddev);
            constexpr int kMaxRetries = 1000;
            for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
                double x = d(rng);
                if (x >= spec.lo && x <= spec.hi) return x;
            }
            throw std::runtime_error(std::string("dataGenerate: ") + name +
                                    "'s Normal(mean, stddev) rarely lands in [lo, hi]; "
                                    "move mean/stddev closer to that range");
        }
        case DistributionKind::Custom:
            return spec.sampler(rng);
    }
    throw std::logic_error("dataGenerate: unknown DistributionKind");
}

}  // namespace

DistributionKind parseDistributionKind(const std::string& text) {
    if (text == "uniform") return DistributionKind::Uniform;
    if (text == "normal") return DistributionKind::Normal;
    throw std::invalid_argument("expected 'uniform' or 'normal', got '" + text + "'");
}

const char* distributionKindName(DistributionKind kind) {
    switch (kind) {
        case DistributionKind::Uniform: return "uniform";
        case DistributionKind::Normal: return "normal";
        case DistributionKind::Custom: return "custom";
    }
    return "unknown";
}

Instance generateInstance(int n, const GenerateOptions& opts) {
    if (n <= 0) throw std::invalid_argument("dataGenerate: n must be positive");
    validateOptions(opts);

    std::mt19937_64 rng(seedFor(opts.seed, n));

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
                inst.v[i] = sampleField("v", opts.vSpec, rng);
                inst.w[i] = sampleField("w", opts.wSpec, rng);
            }
            inst.u[i] = sampleField("u", opts.uSpec, rng);
            fTotal += inst.w[i] * inst.u[i];
        }
        if (fTotal > 0.0) break;
        if (attempt == kMaxRetries)
            throw std::runtime_error(
                "dataGenerate: could not draw a positive F^w_{[n]}; widen u's range/distribution");
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
