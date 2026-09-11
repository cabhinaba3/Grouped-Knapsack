#include "data_generate.h"

#include <climits>
#include <random>
#include <stdexcept>

namespace gka {

double Instance::TotalCapacityCost() const {
  double total = 0.0;
  for (int i = 0; i < n; ++i) total += w[i] * u[i];
  return total;
}

namespace {

void ValidateDistributionParams(const char* name, const FieldSpec& spec) {
  if (spec.dist == DistributionKind::kNormal && spec.stddev <= 0.0)
    throw std::invalid_argument(std::string("data_generate: ") + name +
                                "'s stddev must be positive");
  if (spec.dist == DistributionKind::kCustom && !spec.sampler)
    throw std::invalid_argument(std::string("data_generate: ") + name +
                                " distribution is kCustom but no sampler was set");
}

void ValidateOptions(const GenerateOptions& opts) {
  if (opts.v_spec.lo <= 0.0 || opts.v_spec.hi < opts.v_spec.lo)
    throw std::invalid_argument("data_generate: need 0 < v's lo <= hi");
  if (opts.w_spec.lo <= 0.0 || opts.w_spec.hi < opts.w_spec.lo)
    throw std::invalid_argument("data_generate: need 0 < w's lo <= hi");
  if (opts.u_spec.lo < 0.0 || opts.u_spec.hi > 1.0 || opts.u_spec.hi < opts.u_spec.lo)
    throw std::invalid_argument("data_generate: need 0 <= u's lo <= hi <= 1");
  if (opts.budget_ratio <= 0.0 || opts.budget_ratio >= 1.0)
    throw std::invalid_argument("data_generate: need 0 < budget_ratio < 1");

  ValidateDistributionParams("v", opts.v_spec);
  ValidateDistributionParams("w", opts.w_spec);
  ValidateDistributionParams("u", opts.u_spec);
}

// Combines the batch seed with n so every item count in a sweep gets its
// own reproducible random stream instead of repeating the same one.
uint64_t SeedFor(uint64_t seed, int n) {
  return seed * 1000003ULL + static_cast<uint64_t>(n) + 1ULL;
}

// Draws one value for a field. kNormal is rejection-sampled (redraw any
// value outside [lo, hi]) rather than clamped, so the accepted values stay
// genuinely Normal-shaped within the bounds instead of piling up at an edge.
double SampleField(const char* name, const FieldSpec& spec, std::mt19937_64& rng) {
  switch (spec.dist) {
    case DistributionKind::kUniform: {
      std::uniform_real_distribution<double> dist(spec.lo, spec.hi);
      return dist(rng);
    }
    case DistributionKind::kNormal: {
      std::normal_distribution<double> dist(spec.mean, spec.stddev);
      constexpr int kMaxRejectionSamples = 1000;
      for (int attempt = 0; attempt < kMaxRejectionSamples; ++attempt) {
        double x = dist(rng);
        if (x >= spec.lo && x <= spec.hi) return x;
      }
      throw std::runtime_error(std::string("data_generate: ") + name +
                               "'s Normal(mean, stddev) rarely lands in [lo, hi]; "
                               "move mean/stddev closer to that range");
    }
    case DistributionKind::kCustom:
      return spec.sampler(rng);
  }
  throw std::logic_error("data_generate: unknown DistributionKind");
}

}  // namespace

DistributionKind ParseDistributionKind(const std::string& text) {
  if (text == "uniform") return DistributionKind::kUniform;
  if (text == "normal") return DistributionKind::kNormal;
  throw std::invalid_argument("expected 'uniform' or 'normal', got '" + text + "'");
}

const char* DistributionKindName(DistributionKind kind) {
  switch (kind) {
    case DistributionKind::kUniform:
      return "uniform";
    case DistributionKind::kNormal:
      return "normal";
    case DistributionKind::kCustom:
      return "custom";
  }
  return "unknown";
}

Instance GenerateInstance(int n, const GenerateOptions& opts) {
  if (n <= 0) throw std::invalid_argument("data_generate: n must be positive");
  ValidateOptions(opts);

  std::mt19937_64 rng(SeedFor(opts.seed, n));

  Instance inst;
  inst.n = n;
  inst.v.resize(n);
  inst.w.resize(n);
  inst.u.resize(n);

  double total_capacity_cost = 0.0;
  // u_i can legitimately be 0, so with tiny probability every draw is 0
  // (F^w_{[n]} = 0), which leaves no room for a positive budget C < F.
  // Retry the u draws a bounded number of times in that unlikely case.
  constexpr int kMaxZeroCapacityRetries = 100;
  for (int attempt = 0; attempt <= kMaxZeroCapacityRetries; ++attempt) {
    total_capacity_cost = 0.0;
    for (int i = 0; i < n; ++i) {
      if (attempt == 0) {
        inst.v[i] = SampleField("v", opts.v_spec, rng);
        inst.w[i] = SampleField("w", opts.w_spec, rng);
      }
      inst.u[i] = SampleField("u", opts.u_spec, rng);
      total_capacity_cost += inst.w[i] * inst.u[i];
    }
    if (total_capacity_cost > 0.0) break;
    if (attempt == kMaxZeroCapacityRetries)
      throw std::runtime_error(
          "data_generate: could not draw a positive F^w_{[n]}; widen u's range/distribution");
  }

  inst.C = opts.budget_ratio * total_capacity_cost;
  return inst;
}

std::vector<Instance> GenerateInstances(const std::vector<int>& item_counts,
                                        const GenerateOptions& opts) {
  std::vector<Instance> out;
  out.reserve(item_counts.size());
  for (int n : item_counts) out.push_back(GenerateInstance(n, opts));
  return out;
}

SweepShape ParseSweepShape(const std::string& text) {
  if (text == "dense") return SweepShape::kDense;
  if (text == "sparse") return SweepShape::kSparse;
  throw std::invalid_argument("--sweep: expected 'dense' or 'sparse', got '" + text + "'");
}

const char* SweepShapeName(SweepShape shape) {
  return shape == SweepShape::kDense ? "dense" : "sparse";
}

std::vector<int> ItemCounts(SweepShape shape, long long max_n) {
  if (max_n <= 0) throw std::invalid_argument("ItemCounts: max_n must be positive");
  if (max_n > static_cast<long long>(INT_MAX))
    throw std::invalid_argument("ItemCounts: max_n must fit in an int (<= 2147483647)");

  std::vector<int> counts;
  if (shape == SweepShape::kDense) {
    for (long long i = 1; i <= max_n;) {
      counts.push_back(static_cast<int>(i));
      if (i < 50) {
        i += 1;
      } else if (i < 1000) {
        i += 50;
      } else if (i < 10000) {
        i += 250;
      } else {
        i *= 2;
      }
    }
  } else {
    // 10, 20, 50, 100, 200, 500, ... : three points per decade.
    constexpr int kMantissas[] = {1, 2, 5};
    for (long long decade = 10; decade <= max_n; decade *= 10) {
      for (int mantissa : kMantissas) {
        long long count = decade * mantissa;
        if (count > max_n) break;
        counts.push_back(static_cast<int>(count));
      }
    }
  }

  // The step rules above overshoot max_n rather than landing on it, so a
  // sweep asked for "up to 100000" would otherwise stop at 80000.
  if (counts.empty() || counts.back() != static_cast<int>(max_n))
    counts.push_back(static_cast<int>(max_n));

  return counts;
}

std::vector<int> DefaultItemCounts() { return ItemCounts(SweepShape::kDense, kDefaultMaxN); }

}  // namespace gka
