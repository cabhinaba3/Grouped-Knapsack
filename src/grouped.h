#ifndef GKA_SRC_GROUPED_H_
#define GKA_SRC_GROUPED_H_

#include <vector>

#include "data_generate.h"

// Grouped Knapsack Allocation (Algorithm 1) and its SolveWaterLevel
// subroutine (Algorithm 2), from my_algo.tex.
//
// A group is a metric-grouping cell G_k: a subset of item indices plus a
// representative (v_rep_k, w_rep_k). The representative's only role in the
// algorithm is to rank groups by their representative efficiency ratio
// rho_rep_k = v_rep_k / w_rep_k before the greedy budget pass; the
// water-filling step inside the boundary group works on the group
// members' own w_i, u_i, never on the representative.
namespace gka {

struct Group {
  std::vector<int> items;
  double v_rep = 0.0;
  double w_rep = 0.0;
};

// Algorithm 2 (SolveWaterLevel): the level zeta such that
// sum_i w_i * min(u_i, zeta) = c over the given group members, for
// c in [0, sum_i w_i*u_i]. w and u must be restricted to one group's
// members (same convention as the tex's {w_i}_{i in Gm}).
double WaterLevel(const std::vector<double>& w, const std::vector<double>& u, double c);

struct GroupedResult {
  std::vector<double> z;
  int k_star = -1;
  double c_star = 0.0;
};

// Algorithm 1 (Grouped Knapsack Allocation). Groups may be passed in any
// order; they are sorted internally by rho_rep descending. Every item in
// [0, v.size()) must appear in exactly one group.
GroupedResult GroupedAllocation(const std::vector<double>& v, const std::vector<double>& w,
                                const std::vector<double>& u, const std::vector<Group>& groups,
                                double C);

// Convenience overload operating directly on a generated Instance.
GroupedResult GroupedAllocation(const Instance& inst, const std::vector<Group>& groups);

// Practical grouping construction: sorts items by efficiency ratio v_i/w_i
// and cuts contiguous bins of width 2*delta on that ratio line (the
// metric-grouping tolerance from my_algo.tex, using the ratio itself as
// the 1-D attribute). Each bin's representative is the mean (v, w) of its
// members. Contiguous ratio-sorted bins are automatically order-compatible:
// every earlier group's ratios dominate every later group's.
std::vector<Group> GroupByRatioBins(const std::vector<double>& v, const std::vector<double>& w,
                                    double delta);

// Delta such that binning the middle 90% of the observed ratio v_i/w_i
// distribution (5% trimmed off each tail, to avoid outlier order statistics
// skewing the estimate) into contiguous width-2*delta bins (see
// GroupByRatioBins) yields roughly num_groups groups.
double DeltaForGroupCount(const std::vector<double>& v, const std::vector<double>& w,
                          int num_groups);

// Delta as a fraction (relative_delta, in (0,1)) of the same trimmed ratio
// spread DeltaForGroupCount uses, rather than a fixed target group count.
// This keeps the grouping tolerance comparable across instances whose v, w
// ranges differ, instead of being an absolute value tied to one scale.
double DeltaFromRelativeTolerance(const std::vector<double>& v, const std::vector<double>& w,
                                  double relative_delta);

}  // namespace gka

#endif  // GKA_SRC_GROUPED_H_
