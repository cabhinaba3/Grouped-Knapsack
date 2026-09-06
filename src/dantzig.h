#pragma once

#include <vector>

#include "dataGenerate.h"

// A1: Dantzig's greedy algorithm for the bounded fractional knapsack.
//
// Sorting items by efficiency v_i/w_i descending and greedily filling each
// to its capacity limit u_i is optimal for the LP relaxation (eq:P in the
// paper), since the problem decomposes into a chain of single-item
// trade-offs ordered by marginal value per unit cost.
namespace gka {

// Sorts items by v_i/w_i descending (stable), fills z_i = u_i while the
// item's cost w_i*u_i still fits the remaining budget, gives the first
// item that doesn't fit whatever fraction the remaining budget allows,
// and leaves the rest at 0.
std::vector<double> dantzig(const std::vector<double>& v,
                             const std::vector<double>& w,
                             const std::vector<double>& u,
                             double C);

// Convenience overload operating directly on a generated Instance.
std::vector<double> dantzig(const Instance& inst);

// A5: total value sum_i v_i * z_i.
double value(const std::vector<double>& v, const std::vector<double>& z);

// Mean allocation (1/n) * sum_i z_i, i.e. the average fraction of an
// item's capacity limit taken up across all items (z_i in [0, u_i] subset
// [0,1]).
double averageAllocation(const std::vector<double>& z);

}  // namespace gka
