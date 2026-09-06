#include "grouped.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace gka {

double waterLevel(const std::vector<double>& w, const std::vector<double>& u, double c) {
    const std::size_t q = w.size();
    if (u.size() != q) throw std::invalid_argument("waterLevel: w and u must have the same length");
    if (q == 0) {
        if (c <= 0.0) return 0.0;
        throw std::invalid_argument("waterLevel: c must be 0 for an empty group");
    }

    if (c <= 0.0) return 0.0;

    double fTotal = 0.0;
    for (std::size_t i = 0; i < q; ++i) fTotal += w[i] * u[i];
    double uMax = *std::max_element(u.begin(), u.end());
    if (c >= fTotal) return uMax;

    std::vector<std::size_t> order(q);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                      [&](std::size_t a, std::size_t b) { return u[a] < u[b]; });

    double prevU = 0.0;
    double W = 0.0;
    double Sw = 0.0;
    for (std::size_t i = 0; i < q; ++i) Sw += w[i];

    for (std::size_t idx : order) {
        double du = u[idx] - prevU;
        if (W + Sw * du >= c) return prevU + (c - W) / Sw;
        W += Sw * du;
        Sw -= w[idx];
        prevU = u[idx];
    }

    // Unreachable when c < fTotal and every w_i > 0: the loop above
    // accumulates exactly fTotal by its last iteration, so the return
    // inside the loop always fires first.
    throw std::logic_error("waterLevel: failed to converge");
}

GroupedResult groupedAllocation(const std::vector<double>& v,
                                 const std::vector<double>& w,
                                 const std::vector<double>& u,
                                 const std::vector<Group>& groups,
                                 double C) {
    const std::size_t n = v.size();
    if (w.size() != n || u.size() != n)
        throw std::invalid_argument("groupedAllocation: v, w, u must have the same length");
    if (groups.empty()) throw std::invalid_argument("groupedAllocation: need at least one group");

    std::vector<char> covered(n, 0);
    for (const Group& g : groups) {
        if (g.wRep <= 0.0)
            throw std::invalid_argument("groupedAllocation: representative wRep must be positive");
        for (int i : g.items) {
            if (i < 0 || static_cast<std::size_t>(i) >= n)
                throw std::out_of_range("groupedAllocation: item index out of range");
            if (covered[static_cast<std::size_t>(i)])
                throw std::invalid_argument("groupedAllocation: groups must not overlap");
            covered[static_cast<std::size_t>(i)] = 1;
        }
    }
    if (!std::all_of(covered.begin(), covered.end(), [](char c) { return c != 0; }))
        throw std::invalid_argument("groupedAllocation: groups must cover every item exactly once");

    const std::size_t m = groups.size();
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), 0);
    // rhoRep(a) > rhoRep(b), i.e. vRep_a/wRep_a > vRep_b/wRep_b.
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return groups[a].vRep / groups[a].wRep > groups[b].vRep / groups[b].wRep;
    });

    std::vector<double> fLe(m);
    double running = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        const Group& g = groups[order[k]];
        double fg = 0.0;
        for (int i : g.items) fg += w[static_cast<std::size_t>(i)] * u[static_cast<std::size_t>(i)];
        running += fg;
        fLe[k] = running;
    }

    int kStar = -1;
    for (std::size_t k = 0; k < m; ++k) {
        if (fLe[k] > C) {
            kStar = static_cast<int>(k);
            break;
        }
    }
    if (kStar == -1)
        throw std::invalid_argument("groupedAllocation: C >= F^w_{[n]}, violates C < F^w_{[n]}");

    double fLePrev = (kStar == 0) ? 0.0 : fLe[static_cast<std::size_t>(kStar) - 1];
    double cStar = C - fLePrev;

    std::vector<double> z(n, 0.0);
    for (int k = 0; k < kStar; ++k) {
        for (int i : groups[order[static_cast<std::size_t>(k)]].items) z[static_cast<std::size_t>(i)] = u[static_cast<std::size_t>(i)];
    }
    // Groups after k* stay at 0, already the default.

    const Group& gStar = groups[order[static_cast<std::size_t>(kStar)]];
    std::vector<double> wSub, uSub;
    wSub.reserve(gStar.items.size());
    uSub.reserve(gStar.items.size());
    for (int i : gStar.items) {
        wSub.push_back(w[static_cast<std::size_t>(i)]);
        uSub.push_back(u[static_cast<std::size_t>(i)]);
    }
    double zeta = waterLevel(wSub, uSub, cStar);
    for (int i : gStar.items) z[static_cast<std::size_t>(i)] = std::min(u[static_cast<std::size_t>(i)], zeta);

    return {std::move(z), kStar, cStar};
}

GroupedResult groupedAllocation(const Instance& inst, const std::vector<Group>& groups) {
    return groupedAllocation(inst.v, inst.w, inst.u, groups, inst.C);
}

std::vector<Group> groupByRatioBins(const std::vector<double>& v,
                                     const std::vector<double>& w,
                                     double delta) {
    const std::size_t n = v.size();
    if (w.size() != n) throw std::invalid_argument("groupByRatioBins: v and w must have the same length");
    if (delta <= 0.0) throw std::invalid_argument("groupByRatioBins: delta must be positive");
    if (n == 0) return {};

    std::vector<double> ratio(n);
    for (std::size_t i = 0; i < n; ++i) ratio[i] = v[i] / w[i];

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                      [&](std::size_t a, std::size_t b) { return ratio[a] < ratio[b]; });

    auto finalize = [&](Group& g) {
        double vSum = 0.0, wSum = 0.0;
        for (int i : g.items) {
            vSum += v[static_cast<std::size_t>(i)];
            wSum += w[static_cast<std::size_t>(i)];
        }
        double count = static_cast<double>(g.items.size());
        g.vRep = vSum / count;
        g.wRep = wSum / count;
    };

    std::vector<Group> groups;
    double binStart = ratio[order.front()];
    Group current;
    for (std::size_t idx : order) {
        if (!current.items.empty() && ratio[idx] > binStart + 2.0 * delta) {
            finalize(current);
            groups.push_back(std::move(current));
            current = Group{};
            binStart = ratio[idx];
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
double trimmedRatioSpread(const std::vector<double>& v, const std::vector<double>& w,
                          const char* caller) {
    const std::size_t n = v.size();
    if (w.size() != n)
        throw std::invalid_argument(std::string(caller) + ": v and w must have the same length");
    if (n == 0) throw std::invalid_argument(std::string(caller) + ": need at least one item");

    std::vector<double> ratio(n);
    for (std::size_t i = 0; i < n; ++i) ratio[i] = v[i] / w[i];
    std::sort(ratio.begin(), ratio.end());

    constexpr double kTrimFraction = 0.05;
    std::size_t trim = static_cast<std::size_t>(kTrimFraction * static_cast<double>(n));
    std::size_t lo = trim;
    std::size_t hi = n - 1 - trim;
    if (lo >= hi) {  // too few items to trim meaningfully
        lo = 0;
        hi = n - 1;
    }

    return ratio[hi] - ratio[lo];
}

}  // namespace

double deltaForGroupCount(const std::vector<double>& v, const std::vector<double>& w, int numGroups) {
    if (numGroups <= 0) throw std::invalid_argument("deltaForGroupCount: numGroups must be positive");
    double range = trimmedRatioSpread(v, w, "deltaForGroupCount");
    if (range <= 0.0) return 1.0;  // degenerate: (trimmed) ratios identical, one group
    return range / (2.0 * numGroups);
}

double deltaFromRelativeTolerance(const std::vector<double>& v, const std::vector<double>& w,
                                   double relativeDelta) {
    if (relativeDelta <= 0.0 || relativeDelta >= 1.0)
        throw std::invalid_argument("deltaFromRelativeTolerance: relativeDelta must be in (0,1)");
    double range = trimmedRatioSpread(v, w, "deltaFromRelativeTolerance");
    if (range <= 0.0) return 1.0;  // degenerate: (trimmed) ratios identical, one group
    return relativeDelta * range;
}

}  // namespace gka
