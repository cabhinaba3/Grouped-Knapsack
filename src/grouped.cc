#include "grouped.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace gka {

double WaterLevel(const std::vector<double>& w, const std::vector<double>& u, double c) {
  const std::size_t count = w.size();
  if (u.size() != count)
    throw std::invalid_argument("WaterLevel: w and u must have the same length");
  if (count == 0) {
    if (c <= 0.0) return 0.0;
    throw std::invalid_argument("WaterLevel: c must be 0 for an empty group");
  }

  if (c <= 0.0) return 0.0;

  double total_capacity_cost = 0.0;
  for (std::size_t i = 0; i < count; ++i) total_capacity_cost += w[i] * u[i];
  double max_capacity_limit = *std::max_element(u.begin(), u.end());
  if (c >= total_capacity_cost) return max_capacity_limit;

  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) { return u[a] < u[b]; });

  double prev_capacity_limit = 0.0;
  double cost_below_level = 0.0;
  double active_weight_sum = 0.0;
  for (std::size_t i = 0; i < count; ++i) active_weight_sum += w[i];

  for (std::size_t idx : order) {
    double step = u[idx] - prev_capacity_limit;
    if (cost_below_level + active_weight_sum * step >= c)
      return prev_capacity_limit + (c - cost_below_level) / active_weight_sum;
    cost_below_level += active_weight_sum * step;
    active_weight_sum -= w[idx];
    prev_capacity_limit = u[idx];
  }

  // Unreachable when c < total_capacity_cost and every w_i > 0: the loop
  // above accumulates exactly total_capacity_cost by its last iteration, so
  // the return inside the loop always fires first.
  throw std::logic_error("WaterLevel: failed to converge");
}

GroupedResult GroupedAllocation(const std::vector<double>& v, const std::vector<double>& w,
                                const std::vector<double>& u, const std::vector<Group>& groups,
                                double C) {
  const std::size_t n = v.size();
  if (w.size() != n || u.size() != n)
    throw std::invalid_argument("GroupedAllocation: v, w, u must have the same length");
  if (groups.empty()) throw std::invalid_argument("GroupedAllocation: need at least one group");

  std::vector<char> covered(n, 0);
  for (const Group& group : groups) {
    if (group.w_rep <= 0.0)
      throw std::invalid_argument("GroupedAllocation: representative w_rep must be positive");
    for (int item : group.items) {
      if (item < 0 || static_cast<std::size_t>(item) >= n)
        throw std::out_of_range("GroupedAllocation: item index out of range");
      const auto idx = static_cast<std::size_t>(item);
      if (covered[idx]) throw std::invalid_argument("GroupedAllocation: groups must not overlap");
      covered[idx] = 1;
    }
  }
  if (!std::all_of(covered.begin(), covered.end(), [](char c) { return c != 0; }))
    throw std::invalid_argument("GroupedAllocation: groups must cover every item exactly once");

  const std::size_t m = groups.size();
  std::vector<std::size_t> order(m);
  std::iota(order.begin(), order.end(), 0);
  // rho_rep(a) > rho_rep(b), i.e. v_rep_a/w_rep_a > v_rep_b/w_rep_b.
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return groups[a].v_rep / groups[a].w_rep > groups[b].v_rep / groups[b].w_rep;
  });

  std::vector<double> prefix_cost(m);
  double running = 0.0;
  for (std::size_t k = 0; k < m; ++k) {
    const Group& group = groups[order[k]];
    double group_cost = 0.0;
    for (int item : group.items) {
      const auto idx = static_cast<std::size_t>(item);
      group_cost += w[idx] * u[idx];
    }
    running += group_cost;
    prefix_cost[k] = running;
  }

  int k_star = -1;
  for (std::size_t k = 0; k < m; ++k) {
    if (prefix_cost[k] > C) {
      k_star = static_cast<int>(k);
      break;
    }
  }
  if (k_star == -1)
    throw std::invalid_argument("GroupedAllocation: C >= F^w_{[n]}, violates C < F^w_{[n]}");

  const double prefix_cost_before_boundary =
      (k_star == 0) ? 0.0 : prefix_cost[static_cast<std::size_t>(k_star) - 1];
  const double c_star = C - prefix_cost_before_boundary;

  std::vector<double> z(n, 0.0);
  for (int k = 0; k < k_star; ++k) {
    for (int item : groups[order[static_cast<std::size_t>(k)]].items) {
      const auto idx = static_cast<std::size_t>(item);
      z[idx] = u[idx];
    }
  }
  // Groups after k_star stay at 0, already the default.

  const Group& boundary_group = groups[order[static_cast<std::size_t>(k_star)]];
  std::vector<double> w_sub, u_sub;
  w_sub.reserve(boundary_group.items.size());
  u_sub.reserve(boundary_group.items.size());
  for (int item : boundary_group.items) {
    const auto idx = static_cast<std::size_t>(item);
    w_sub.push_back(w[idx]);
    u_sub.push_back(u[idx]);
  }
  double water_level = WaterLevel(w_sub, u_sub, c_star);
  for (int item : boundary_group.items) {
    const auto idx = static_cast<std::size_t>(item);
    z[idx] = std::min(u[idx], water_level);
  }

  return {std::move(z), k_star, c_star};
}

GroupedResult GroupedAllocation(const Instance& inst, const std::vector<Group>& groups) {
  return GroupedAllocation(inst.v, inst.w, inst.u, groups, inst.C);
}

std::vector<Group> GroupByRatioBins(const std::vector<double>& v, const std::vector<double>& w,
                                    double delta) {
  const std::size_t n = v.size();
  if (w.size() != n)
    throw std::invalid_argument("GroupByRatioBins: v and w must have the same length");
  if (delta <= 0.0) throw std::invalid_argument("GroupByRatioBins: delta must be positive");
  if (n == 0) return {};

  std::vector<double> ratio(n);
  for (std::size_t i = 0; i < n; ++i) ratio[i] = v[i] / w[i];

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) { return ratio[a] < ratio[b]; });

  auto finalize = [&](Group& group) {
    double v_sum = 0.0, w_sum = 0.0;
    for (int item : group.items) {
      const auto idx = static_cast<std::size_t>(item);
      v_sum += v[idx];
      w_sum += w[idx];
    }
    const double count = static_cast<double>(group.items.size());
    group.v_rep = v_sum / count;
    group.w_rep = w_sum / count;
  };

  std::vector<Group> groups;
  double bin_start = ratio[order.front()];
  Group current;
  for (std::size_t idx : order) {
    if (!current.items.empty() && ratio[idx] > bin_start + 2.0 * delta) {
      finalize(current);
      groups.push_back(std::move(current));
      current = Group{};
      bin_start = ratio[idx];
    }
    current.items.push_back(static_cast<int>(idx));
  }
  finalize(current);
  groups.push_back(std::move(current));

  return groups;
}

namespace {

// The trimmed (5%-95%) spread of the ratio v_i/w_i distribution.
//
// The raw min/max ratio is an outlier-sensitive statistic that widens as n
// grows (more extreme order statistics get sampled), which would otherwise
// make a delta derived from it drift with n even though the bulk of the
// ratio distribution hasn't changed. Trimming the extreme tails ties delta
// to that bulk instead.
double TrimmedRatioSpread(const std::vector<double>& v, const std::vector<double>& w,
                          const char* caller) {
  const std::size_t n = v.size();
  if (w.size() != n)
    throw std::invalid_argument(std::string(caller) + ": v and w must have the same length");
  if (n == 0) throw std::invalid_argument(std::string(caller) + ": need at least one item");

  std::vector<double> ratio(n);
  for (std::size_t i = 0; i < n; ++i) ratio[i] = v[i] / w[i];
  std::sort(ratio.begin(), ratio.end());

  constexpr double kTrimFraction = 0.05;
  const std::size_t trim_count = static_cast<std::size_t>(kTrimFraction * static_cast<double>(n));
  std::size_t lo = trim_count;
  std::size_t hi = n - 1 - trim_count;
  if (lo >= hi) {  // too few items to trim meaningfully
    lo = 0;
    hi = n - 1;
  }

  return ratio[hi] - ratio[lo];
}

}  // namespace

double DeltaForGroupCount(const std::vector<double>& v, const std::vector<double>& w,
                          int num_groups) {
  if (num_groups <= 0)
    throw std::invalid_argument("DeltaForGroupCount: num_groups must be positive");
  double range = TrimmedRatioSpread(v, w, "DeltaForGroupCount");
  if (range <= 0.0) return 1.0;  // degenerate: (trimmed) ratios identical, one group
  return range / (2.0 * num_groups);
}

double DeltaFromRelativeTolerance(const std::vector<double>& v, const std::vector<double>& w,
                                  double relative_delta) {
  if (relative_delta <= 0.0 || relative_delta >= 1.0)
    throw std::invalid_argument("DeltaFromRelativeTolerance: relative_delta must be in (0,1)");
  double range = TrimmedRatioSpread(v, w, "DeltaFromRelativeTolerance");
  if (range <= 0.0) return 1.0;  // degenerate: (trimmed) ratios identical, one group
  return relative_delta * range;
}

}  // namespace gka
