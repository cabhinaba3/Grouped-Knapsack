# Grouped Knapsack

A C++17 implementation and experimental evaluation of the **Grouped Knapsack
Allocation** algorithm against exact fractional-knapsack
optimality (Dantzig's greedy rule):

$V^*(I)=\max\left[\sum_i v_i z_i \mid z_i \in [0, u_i],\sum_i w_i z_i <= C\right]$

with values $v_i > 0$, costs $w_i > 0$, capacity limits $u_i \in [0,1]$, and
budget $C < F^w_{[n]}$.

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

## Experiments

All three are run by `gka_experiments`; each aggregates over several random
seeds to smooth out per-instance noise (median for timings, mean for error
rates).

```bash
./build/gka_experiments --outdir figures
```

## Command-line reference

```
gka_dantzig [--seed N] [--budget-ratio R] [--sweep dense|sparse] [--max-n N]
            [--num-groups M] [--delta D] [--threads T]
            [--v-dist uniform|normal] [--v-mean M] [--v-stddev S]
            [--w-dist uniform|normal] [--w-mean M] [--w-stddev S]
            [--u-dist uniform|normal] [--u-mean M] [--u-stddev S]
gka_experiments [--outdir DIR] [--seed-base N] [--num-seeds S] [--reps R] [--num-groups M]
                [--delta D] [--budget-ratio B] [--sweep dense|sparse] [--max-n N]
                [--v-dist uniform|normal] [--v-mean M] [--v-stddev S]
                [--w-dist uniform|normal] [--w-mean M] [--w-stddev S]
                [--u-dist uniform|normal] [--u-mean M] [--u-stddev S]
                [--sensitivity-n N] [--sensitivity-points P] [--sensitivity-seeds S]
                [--threads T]
```

Defaults: `seed=0`/`seed-base=0`, `budget-ratio=0.5`, `num-groups=20`,
`num-seeds=10`, `reps=5`, `sweep=dense`, `max-n=100000`,
`sensitivity-n=5000`, `sensitivity-points=25`, `sensitivity-seeds=5`,
`threads=`all cores, `v/w/u-dist=uniform`. Run either binary with `--help`
for the full list.

`--sweep sparse` gives the 13-point sequence `10, 20, 50, ..., 100000` for
quick runs; `--sweep dense` (the default) resolves the runtime curve more
finely. `--max-n` sets the largest item count either way and is rejected
above `INT_MAX`.

`--delta` is a fraction in `(0,1)` of the instance's trimmed ratio spread
(see "How delta is chosen"), not an absolute tolerance; it overrides
`--num-groups` when set.

Each of `v_i`, `w_i`, `u_i` is drawn independently and can be switched from
the default `uniform` (over `[lo,hi]`, currently `[1,100]` for `v`/`w` and
`[0,1]` for `u`) to `normal`, via `--{v,w,u}-mean`/`--{v,w,u}-stddev`.
A `normal` draw outside `[lo,hi]` is rejected and redrawn (bounded retries)
rather than clamped, so accepted values stay genuinely Gaussian-shaped
within the range instead of piling up at an edge; too tight a `[lo,hi]`
for the given mean/stddev raises a clear error instead of retrying forever.
Other distributions (lognormal, exponential, ...) can be attached in C++
via `FieldSpec::sampler` but have no CLI spelling.

<!-- ## Known limitations / not yet implemented

- Grouping is constructed heuristically from the ratio `v_i/w_i` (see "How
  delta is chosen"), not from an explicit attribute space with known
  Lipschitz value/cost functions `v(a), w(a)`. -->

## License

GPL-2.0, see `LICENSE`.

## Special Note

A big shoutout to [Matplot++](https://github.com/alandefreitas/matplotplusplus) for the great work.

