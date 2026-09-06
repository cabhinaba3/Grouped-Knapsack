#include "dantzig.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace gka {

std::vector<double> dantzig(const std::vector<double>& v,
                             const std::vector<double>& w,
                             const std::vector<double>& u,
                             double C) {
    const std::size_t n = v.size();
    if (w.size() != n || u.size() != n)
        throw std::invalid_argument("dantzig: v, w, u must have the same length");

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    // v[a]/w[a] > v[b]/w[b], written without division since w > 0.
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return v[a] * w[b] > v[b] * w[a];
    });

    std::vector<double> z(n, 0.0);
    double rem = C;
    for (std::size_t idx : order) {
        if (rem <= 0.0) break;
        double cost = w[idx] * u[idx];
        if (cost <= rem) {
            z[idx] = u[idx];
            rem -= cost;
        } else {
            z[idx] = rem / w[idx];
            break;
        }
    }
    return z;
}

std::vector<double> dantzig(const Instance& inst) {
    return dantzig(inst.v, inst.w, inst.u, inst.C);
}

double value(const std::vector<double>& v, const std::vector<double>& z) {
    if (v.size() != z.size())
        throw std::invalid_argument("value: v and z must have the same length");
    double total = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) total += v[i] * z[i];
    return total;
}

double averageAllocation(const std::vector<double>& z) {
    if (z.empty()) return 0.0;
    double total = 0.0;
    for (double zi : z) total += zi;
    return total / static_cast<double>(z.size());
}

}  // namespace gka
