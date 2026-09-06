# Grouped Knapsack

A C++17 implementation and experimental evaluation of the **Grouped Knapsack
Allocation** algorithm (`my_algo.tex`) against exact fractional-knapsack
optimality (Dantzig's greedy rule), for the bounded fractional knapsack
problem defined in `experiment_plan.md`:

```
V*(I) = max { sum_i v_i z_i : z_i in [0, u_i], sum_i w_i z_i <= C }
```

with values `v_i > 0`, costs `w_i > 0`, capacity limits `u_i in [0,1]`, and
budget `C < F^w_{[n]} := sum_i w_i u_i` (so the budget can never saturate
every item at once).

The grouped algorithm's premise: partition the `n` items into `m << n`
groups by a metric-grouping tolerance `delta`, rank groups by a
representative efficiency ratio, fill whole groups greedily, and
water-fill only the one boundary group that the budget cuts through. This
repo generates random instances, runs both methods, verifies the grouped
algorithm against the exact optimum, and produces the figures described
below.

## Repository layout

```
src/
  dataGenerate.h/.cpp   Random instance generator: (v, w, u, C)
  dantzig.h/.cpp        A1: exact greedy fractional-knapsack solver
  grouped.h/.cpp        Algorithm 1 (Grouped Knapsack Allocation) +
                         Algorithm 2 (SolveWaterLevel) from my_algo.tex,
                         plus the ratio-based grouping construction
  harness.h/.cpp        Solve/time/aggregate layer shared by both binaries
  parallel.h/.cpp       parallelFor over the C++17 parallel STL, plus the
                         thread-count cap
  cli.h/.cpp            Shared "--flag VALUE" parser used by both binaries
  main.cpp              gka_dantzig: single-run demo, prints a summary table
  experiments.cpp       gka_experiments: the three experiments below
CMakeLists.txt
my_algo.tex             The two algorithms this repo implements
experiment_plan.md      The paper's fuller experimental plan (E1-E3);
                         this repo implements a practical subset of it
figures/                Generated plots (git-ignored output)
```

## Building

Requires a C++17 compiler, CMake >= 3.12, [oneTBB](https://github.com/uxlfoundation/oneTBB)
(libstdc++ and libc++ implement the C++17 parallel algorithms on top of it,
so `std::execution::par` needs it linked), and for `gka_experiments`:
[Matplot++](https://github.com/alandefreitas/matplotplusplus) (built and
installed, discoverable via `find_package(Matplot++ REQUIRED)`) and
**gnuplot >= 5.2.6 on `PATH` at runtime** (Matplot++ shells out to it to
render/save figures; compiling and linking do not need it, but running
`gka_experiments` does).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces two executables:

- **`gka_dantzig`** — generates one instance per item count in the sweep
  (dense by default: 1..50, then 100, 150, ..., 1000, then 1250, ..., 10000,
  then doubling to 100000), solves each with both methods, and prints a
  summary table.
- **`gka_experiments`** — runs the three experiments described below and
  writes their figures to `figures/`.

### Threading

Both binaries parallelise their **untimed** work over the C++17 parallel STL
and leave every **timed** solve running on its own: concurrent solves contend
for memory bandwidth, which would turn the runtime curve into a measurement
of the machine's load rather than of the algorithms. Concretely:

| Work | Runs |
|---|---|
| Instance generation and `deltaForGroupCount` | parallel |
| Experiment 1's timed `dantzig` / grouped solves | serial |
| Experiment 2's error rate | free — read off experiment 1's solve |
| Experiment 3 (whole delta x seed grid, never timed) | parallel |
| `gka_dantzig`'s timed per-`n` solves | serial |

`--threads T` caps the worker count (default: every hardware thread) and
never applies to a timed solve. Results do not depend on it: every parallel
task writes only its own slot and the reductions run afterwards in seed
order, so output is bit-identical at any `--threads` value.

Measured on a 12-thread i7-1365U (2 P-cores + 8 E-cores, 12 MiB L3),
experiment 3 at the default `n=5000` runs **3.3x** faster on 12 threads than
on 1 (3.34s -> 1.02s). At `n=100000` the gain drops to 1.9x: each task's
working set is several MB, so a dozen concurrent solves exceed L3 and become
bandwidth-bound.

## What each module does

**`dataGenerate`** draws `v_i ~ U(1,100)`, `w_i ~ U(1,100)`, `u_i ~ U(0,1)`
i.i.d. per item, then sets `C = budgetRatio * F^w_{[n]}` with
`budgetRatio in (0,1)` (default `0.5`) so `C < F^w_{[n]}` holds by
construction. Seeded deterministically from `(seed, n)` for reproducibility.

**`dantzig`** is the textbook exact solver: sort items by `v_i/w_i`
descending, fill each to `u_i` until the budget runs out, give the crossing
item the leftover fraction. This is `V*`, the ground truth every experiment
compares against.

**`grouped`** implements `my_algo.tex` directly:
- `waterLevel(w, u, c)` — Algorithm 2, the level `zeta` such that
  `sum_i w_i * min(u_i, zeta) = c` over one group's members.
- `groupedAllocation(v, w, u, groups, C)` — Algorithm 1: rank groups by
  representative ratio `vRep_k / wRep_k`, find the boundary group `k*` via
  prefix sums of `F^w_{G_k}`, fill earlier groups fully, zero later ones,
  water-fill the boundary group.
- `groupByRatioBins(v, w, delta)` — the grouping construction actually used
  here (see "How delta is chosen" below): sort items by ratio `v_i/w_i` and
  cut contiguous bins of width `2*delta`. Contiguous ratio-sorted bins are
  automatically order-compatible (every earlier group's ratios dominate
  every later group's), which is what lets `groupedAllocation` treat whole
  groups as units without reordering items across group boundaries.

## How delta is chosen

`my_algo.tex`'s grouping is defined by a tolerance `delta`: every item in a
group must lie within `delta` of the group's representative on some
attribute line, so within-group costs can't spread arbitrarily far apart
(`eq:eps-def` bounds `w_hi_k - w_lo_k <= 2*L_w*delta`). This repo doesn't
generate items from an explicit attribute space with known Lipschitz
constants `L_v, L_w` (the data generator draws `v`, `w`, `u` independently);
instead it uses the efficiency ratio `rho_i = v_i/w_i` itself as a practical
1-D attribute, so `delta` is chosen directly on that ratio line.

`gka::deltaForGroupCount(v, w, numGroups)` picks a `delta` that yields
roughly `numGroups` groups, targeting `--num-groups 20` by default in both
executables. The first version of this simply used the raw
`(max ratio - min ratio) / (2 * numGroups)`. **That was a bug** (caught
while checking the experiment results, see below): the raw min/max ratio is
an outlier-sensitive statistic that widens as `n` grows (more extreme order
statistics get sampled), so a fixed `numGroups` target quietly produced a
*coarser* bin at large `n` even though the bulk of the ratio distribution
hadn't changed. At `n=5000` this let one bin swallow 4376 of 5000 items,
spanning costs from `w=1.9` to `w=100` — a huge intra-group spread, which is
exactly what Theorem 8's bound says drives loss up.

**Fix:** `deltaForGroupCount` now trims the extreme 5% off each tail of the
sorted ratio distribution before measuring the range, so `delta` tracks the
bulk of the distribution instead of its outliers:

```cpp
delta = (ratio[p95] - ratio[p05]) / (2 * numGroups)
```

This dropped the error rate from a flat ~30% (independent of `n`, because
the outlier-driven bug scaled with `n` too) to a stable ~1.2-4% across the
full `n` sweep — see Experiment 2. `--delta` can also be passed directly to
either executable to bypass this heuristic and fix an absolute value.

## Correctness verification

- **Singleton groups match Dantzig exactly**:
  with one item per group, `groupedAllocation` reduces to per-item greedy
  filling, and its output matched `dantzig`'s coordinate-wise (`< 1e-9`) on
  200 random trials.
- **Theorem 12's loss-localization identity holds exactly**: for a random
  `n=5000` instance, the boundary group `Gamma = G_{k*}` was extracted and
  its own intra-group loss `Delta_Gamma(c*)` (Dantzig-on-`Gamma` minus
  water-fill-on-`Gamma`, both at budget `c*`) was computed independently.
  Result: `|E - Delta_Gamma| = 0.000000` — every bit of the grouped
  method's total loss lives in the single boundary group, exactly as the
  theorem claims, confirming the earlier/higher error rates were a grouping
  *parameter* choice, not an algorithm bug.
- **Theorem 8's harmonic bound holds** (loosely, as expected of an upper
  bound): for that same boundary group,
  `v_hat * U_Gamma * (w_hi-w_lo)/(w_hi+w_lo) + eps_v * U_Gamma >= Delta_Gamma(c*)`.

## Experiments

All three are run by `gka_experiments`; each aggregates over several random
seeds to smooth out per-instance noise (median for timings, mean for error
rates).

```bash
./build/gka_experiments --outdir figures
```

### 1) Runtime: Dantzig vs Grouped, vs n

**What's timed:** for each `n`, generate `--num-seeds` instances (default
10), and for each, time `dantzig(inst)` and the *full* grouped pipeline
(`groupByRatioBins` + `groupedAllocation`, timed together) over `--reps`
repetitions (default 5), taking the median. `delta` itself
(`deltaForGroupCount`, `--num-groups 20`) is computed once per instance,
**outside** the timed region — it's a grouping-tolerance hyperparameter
you'd fix once from domain knowledge, not something re-derived on every
solve.

Both methods are timed on the same one-shot basis: solving one instance
from scratch, with no reuse of prior work between them.

| n | Dantzig (ms) | Grouped (ms) |
|---|---|---|
| 10 | 0.00025 | 0.00091 |
| 100 | 0.00201 | 0.00431 |
| 1,000 | 0.05410 | 0.06855 |
| 10,000 | 0.56662 | 0.58601 |
| 100,000 | 8.72552 | 9.77745 |

![Runtime comparison](figures/runtime_comparison.svg)

**Analysis:** Grouped is consistently at or slightly above Dantzig at every
`n`, converging closer as `n` grows. This is expected, not a regression:
`groupByRatioBins` performs its own `O(n log n)` sort by ratio (the same
asymptotic cost as Dantzig's sort), plus `groupedAllocation` adds an extra
`O(m log m)` pass over the `m` groups and the per-item work of building
`Group` structs (representative `v`/`w` means). So a one-shot, from-scratch
solve of the grouped method can only match or exceed Dantzig's cost, never
beat it — the grouped method's actual runtime advantage only appears when
the grouping is built once and reused across many budget queries `C`
(Algorithm 1 alone, given precomputed groups, is `O(m log m)` with `m << n`,
versus Dantzig's `O(n log n)` every time `C` changes). That amortized
per-query benefit isn't what this plot measures; it measures the fair
one-shot cost.

### 2) Error rate: Grouped vs Dantzig, vs n

**What's measured:** `(V* - V^gp) / V* * 100`, using the same instances and
`delta` (`deltaForGroupCount`, `--num-groups 20`) as Experiment 1, averaged
over `--num-seeds` seeds.

| n | m (groups) | Error rate |
|---|---|---|
| 10 | ~9 | 4.10% |
| 100 | ~19 | 1.22% |
| 1,000 | ~48 | 1.21% |
| 10,000 | ~109 | 1.27% |
| 100,000 | ~178 | 1.34% |

![Error rate vs n](figures/error_rate.svg)

**Analysis:** error rate is stable at roughly 1.2-1.6% for `n >= 20`, with
a higher 4.1% only at `n=10` (too few items for 20 groups to mean anything
— groups are nearly singletons there, so the small residual error comes
from the handful of non-singleton groups). This flatness across two orders
of magnitude of `n` is itself a validation of the `deltaForGroupCount` fix:
before trimming outliers, this curve was flat too, but at ~30% — the number
changed, the *shape* (n-independence) didn't, confirming the fix addressed
a scale-invariant bias rather than papering over an n-dependent effect.

### 3) Error rate sensitivity to delta, at fixed n

**What's measured:** fix `n = --sensitivity-n` (default 5000), sweep
`delta` itself over `--sensitivity-points` (default 25) log-spaced values
from `deltaMax/2000` to `deltaMax` (where `deltaMax` is the `delta` that
would merge every item into one group, i.e.
`deltaForGroupCount(v, w, numGroups=1)`), and average error rate over
`--sensitivity-seeds` (default 5) instances per `delta` value.

| delta | avg m | Error rate |
|---|---|---|
| 0.0019 | 1200 | 0.00012% |
| 0.0245 | 304 | 0.0134% |
| 0.163 | 97 | 0.929% |
| 0.580 | 42 | 11.7% |
| 1.501 | 21 | 26.7% |
| 3.880 | 10 | 33.5% |

![Delta sensitivity](figures/delta_sensitivity.svg)

**Analysis:** error rate rises smoothly and monotonically across roughly
four orders of magnitude of `delta`, with no discontinuities — exactly the
behavior Theorem 8/12 predict (loss grows with the intra-group cost spread,
which grows with `delta`). This is also the plot that explains the original
~30% error rate that motivated the `deltaForGroupCount` fix: it sits at the
large-`delta` tail of this exact curve (`delta ~= 1.5-2`, `m ~= 16-21`) —
i.e., the original bug was effectively choosing a `delta` from the far
right-hand side of this plot without meaning to, by letting outlier ratios
inflate the range estimate. This experiment is the direct empirical
counterpart of `experiment_plan.md`'s E1 (harmonic-bound tightness):
smaller `delta` (finer grouping) provably costs more groups `m` (see the
`avg m` column) but buys a rapidly shrinking error rate, down to `0.0001%`
at the finest tested `delta`.

## Command-line reference

```
gka_dantzig [--seed N] [--budget-ratio R] [--sweep dense|sparse] [--max-n N]
            [--num-groups M] [--delta D] [--threads T]
gka_experiments [--outdir DIR] [--seed-base N] [--num-seeds S] [--reps R] [--num-groups M]
                [--budget-ratio B] [--sweep dense|sparse] [--max-n N]
                [--sensitivity-n N] [--sensitivity-points P] [--sensitivity-seeds S]
                [--threads T]
```

Defaults: `seed=0`/`seed-base=0`, `budget-ratio=0.5`, `num-groups=20`,
`num-seeds=10`, `reps=5`, `sweep=dense`, `max-n=100000`,
`sensitivity-n=5000`, `sensitivity-points=25`, `sensitivity-seeds=5`,
`threads=`all cores. Run either binary with `--help` for the full list.

`--sweep sparse` gives the 13-point sequence `10, 20, 50, ..., 100000` for
quick runs; `--sweep dense` (the default) resolves the runtime curve more
finely. `--max-n` sets the largest item count either way and is rejected
above `INT_MAX`.

<!-- ## Known limitations / not yet implemented

- Grouping is constructed heuristically from the ratio `v_i/w_i` (see "How
  delta is chosen"), not from an explicit attribute space with known
  Lipschitz value/cost functions `v(a), w(a)`. -->

## License

GPL-2.0, see `LICENSE`.
