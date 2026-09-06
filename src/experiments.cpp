#include <matplot/matplot.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cli.h"
#include "dataGenerate.h"
#include "harness.h"
#include "parallel.h"

// Three experiments comparing Dantzig's algorithm to the Grouped Knapsack
// Allocation (my_algo.tex):
//   1) runtime vs n: how long each method takes to solve one instance.
//   2) error rate vs n: (V* - V^gp) / V*, i.e. how much value the grouped
//      allocation gives up relative to the exact fractional optimum, as n
//      grows (grouping width chosen via --num-groups).
//   3) error rate vs delta: at one fixed n, how the same error rate
//      responds to the grouping tolerance delta itself (the metric-grouping
//      half-width from my_algo.tex). Coarser bins (larger delta) let items
//      with very different costs share a group, which is exactly what
//      inflates the intra-group loss bounded by Theorem 8.
// All three are aggregated over several random seeds to smooth out
// per-instance noise (median for timings, mean for the error rate).
//
// Threading: experiments 1 and 2 share one pass, whose *timed* solves run
// one at a time -- concurrent solves would contend for memory bandwidth and
// turn the runtime curve into a measurement of the machine's load rather
// than of the algorithms. Everything untimed runs in parallel: building the
// instances and their grouping tolerances for both passes, and the whole of
// experiment 3, which measures value loss and never touches a clock.
namespace {

struct Config {
    gka::GenerateOptions opts;
    std::string outdir = "figures";
    std::string sweepName = "dense";
    long long maxN = gka::kDefaultMaxN;
    std::uint64_t seedBase = 0;
    int numSeeds = 10;
    int reps = 5;
    int numGroups = 20;
    int sensitivityN = 5000;
    int sensitivityPoints = 25;
    int sensitivitySeeds = 5;
    int threads = 0;  // 0 means "every hardware thread"
};

// --- Small helpers ---------------------------------------------------------

std::vector<double> logspace(double lo, double hi, int points) {
    std::vector<double> out(static_cast<std::size_t>(points));
    const double logLo = std::log(lo);
    const double logHi = std::log(hi);
    for (int i = 0; i < points; ++i) {
        double t = points == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(points - 1);
        out[static_cast<std::size_t>(i)] = std::exp(logLo + t * (logHi - logLo));
    }
    return out;
}

// save() returns before the gnuplot subprocess has necessarily finished
// flushing the PNG to disk, so poll briefly rather than checking once.
void reportSaved(const std::string& path) {
    std::error_code ec;
    std::uintmax_t size = 0;
    for (int attempt = 0; attempt < 40; ++attempt) {
        size = std::filesystem::file_size(path, ec);
        if (!ec && size > 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (ec || size == 0) {
        std::cerr << "Warning: " << path
                  << " was not written correctly. Matplot++ needs gnuplot (>=5.2.6) on PATH"
                     " to render figures -- install it and re-run.\n";
    } else {
        std::cerr << "Wrote " << path << " (" << size << " bytes)\n";
    }
}

std::vector<gka::InstanceSpec> seedSpecs(int n, std::uint64_t seedBase, int count) {
    std::vector<gka::InstanceSpec> specs;
    specs.reserve(static_cast<std::size_t>(count));
    for (int s = 0; s < count; ++s)
        specs.push_back({n, seedBase + static_cast<std::uint64_t>(s)});
    return specs;
}

// --- Experiments 1 & 2: runtime and error rate vs n ------------------------

struct SweepSeries {
    std::vector<double> ns;
    std::vector<double> dantzigMs;
    std::vector<double> groupedMs;
    std::vector<double> errorPct;
};

SweepSeries runRuntimeAndError(const Config& cfg, const std::vector<int>& counts) {
    SweepSeries series;
    series.ns.resize(counts.size());
    series.dantzigMs.resize(counts.size());
    series.groupedMs.resize(counts.size());
    series.errorPct.resize(counts.size());

    std::cout << "Experiment 1+2: runtime and error rate vs n\n";
    std::cout << std::left << std::setw(8) << "n" << std::setw(14) << "dantzig_ms" << std::setw(14)
              << "grouped_ms" << std::setw(10) << "error_%" << "\n";

    for (std::size_t idx = 0; idx < counts.size(); ++idx) {
        const int n = counts[idx];

        // Untimed setup, in parallel: the seeds' instances and the grouping
        // tolerance each will be solved at. delta is a fixed hyperparameter
        // chosen once up front (here derived from the data just to pick a
        // sensible value for the demo), not something re-solved per query,
        // so it belongs out here rather than in the timed region below.
        gka::InstanceBatch batch = gka::generateBatch(seedSpecs(n, cfg.seedBase, cfg.numSeeds),
                                                      cfg.opts, -1.0, cfg.numGroups);

        std::vector<double> dTimes, gTimes, errRates;
        dTimes.reserve(static_cast<std::size_t>(cfg.numSeeds));
        gTimes.reserve(static_cast<std::size_t>(cfg.numSeeds));
        errRates.reserve(static_cast<std::size_t>(cfg.numSeeds));

        // Timed region: strictly one solve at a time.
        for (std::size_t s = 0; s < batch.instances.size(); ++s) {
            std::vector<double> dRep(static_cast<std::size_t>(cfg.reps));
            std::vector<double> gRep(static_cast<std::size_t>(cfg.reps));
            gka::SolveOutcome outcome;

            for (int r = 0; r < cfg.reps; ++r) {
                gka::Timings t = gka::timeBoth(batch.instances[s], batch.deltas[s], outcome);
                dRep[static_cast<std::size_t>(r)] = t.dantzigMs;
                gRep[static_cast<std::size_t>(r)] = t.groupedMs;
            }

            dTimes.push_back(gka::median(dRep));
            gTimes.push_back(gka::median(gRep));
            // Experiment 2 reads its error rate off the solve experiment 1
            // just timed, so the two experiments cost one pass, not two.
            errRates.push_back(outcome.errorPercent());
        }

        series.ns[idx] = static_cast<double>(n);
        series.dantzigMs[idx] = gka::median(dTimes);
        series.groupedMs[idx] = gka::median(gTimes);
        series.errorPct[idx] = gka::mean(errRates);

        std::cout << std::left << std::setw(8) << n << std::setw(14) << series.dantzigMs[idx]
                  << std::setw(14) << series.groupedMs[idx] << std::setw(10) << series.errorPct[idx]
                  << "\n";
    }

    return series;
}

// --- Experiment 3: error rate vs delta, at a fixed n -----------------------

struct SensitivitySeries {
    std::vector<double> deltas;
    std::vector<double> errorPct;
    std::vector<double> groupCounts;
};

SensitivitySeries runDeltaSensitivity(const Config& cfg) {
    // The seed set is fixed across the whole delta sweep, so build it once
    // instead of regenerating identical instances at every delta. The delta
    // handed to generateBatch is a placeholder: this experiment supplies its
    // own delta per point below, so deriving one per instance here would only
    // pay for a sort nothing reads.
    gka::InstanceBatch batch =
        gka::generateBatch(seedSpecs(cfg.sensitivityN, cfg.seedBase, cfg.sensitivitySeeds),
                           cfg.opts, 1.0, cfg.numGroups);
    const std::vector<gka::Instance>& instances = batch.instances;

    // deltaForGroupCount with numGroups=1 returns half the (trimmed) ratio
    // range, i.e. the delta that would merge everything into one group -- a
    // natural upper bound for the sweep.
    const double deltaMax = gka::deltaForGroupCount(instances[0].v, instances[0].w, 1);
    const double deltaMin = deltaMax / 2000.0;

    SensitivitySeries series;
    series.deltas = logspace(deltaMin, deltaMax, cfg.sensitivityPoints);
    series.errorPct.resize(series.deltas.size());
    series.groupCounts.resize(series.deltas.size());

    // No clocks here, so the whole grid runs concurrently. It is flattened to
    // one task per (delta, seed) rather than one per delta because the work
    // per delta is badly skewed: at a large delta the boundary group holds
    // nearly every item, so SolveWaterLevel sorts ~n of them, while at a
    // small delta it sorts a handful. One task per delta leaves whichever
    // worker draws the coarse end running long after the others are idle.
    const std::size_t numSeeds = instances.size();
    std::vector<double> errFlat(series.deltas.size() * numSeeds);
    std::vector<double> groupFlat(series.deltas.size() * numSeeds);

    gka::parallelFor(errFlat.size(), [&](std::size_t task) {
        const std::size_t di = task / numSeeds;
        const std::size_t si = task % numSeeds;
        gka::SolveOutcome outcome = gka::solveBoth(instances[si], series.deltas[di]);
        errFlat[task] = outcome.errorPercent();
        groupFlat[task] = static_cast<double>(outcome.numGroups);
    });

    // Reduced afterwards, in seed order, so the averages do not depend on the
    // order the tasks happened to finish in.
    for (std::size_t di = 0; di < series.deltas.size(); ++di) {
        const auto first = static_cast<std::ptrdiff_t>(di * numSeeds);
        const auto last = static_cast<std::ptrdiff_t>((di + 1) * numSeeds);
        series.errorPct[di] = gka::mean(std::vector<double>(errFlat.begin() + first, errFlat.begin() + last));
        series.groupCounts[di] =
            gka::mean(std::vector<double>(groupFlat.begin() + first, groupFlat.begin() + last));
    }

    std::cout << "\nExperiment 3: error rate vs delta (n=" << cfg.sensitivityN << ")\n";
    std::cout << std::left << std::setw(14) << "delta" << std::setw(12) << "avg_m" << std::setw(10)
              << "error_%" << "\n";
    for (std::size_t di = 0; di < series.deltas.size(); ++di) {
        std::cout << std::left << std::setw(14) << series.deltas[di] << std::setw(12)
                  << series.groupCounts[di] << std::setw(10) << series.errorPct[di] << "\n";
    }

    return series;
}

// --- Figures ---------------------------------------------------------------

// One palette for every figure: a single-series plot is red, and a two-series
// comparison pairs blue (the baseline) with red (the grouped algorithm). Both
// are pulled off pure RGB towards print-safe shades that keep a visible
// lightness gap, so the two curves stay distinguishable in greyscale too.
// Matplot++ stores a colour as {alpha, r, g, b} with alpha 0 meaning opaque.
// These have to be given componentwise: its string overload only knows the
// eight named colours and silently turns anything else, "#C1272D" included,
// into black.
constexpr std::array<float, 4> kRed{0.0f, 0.757f, 0.153f, 0.176f};   // #C1272D
constexpr std::array<float, 4> kBlue{0.0f, 0.106f, 0.310f, 0.612f};  // #1B4F9C

// Matplot++ has no tight_layout(), and its default axes rectangle
// ({.13, .11, .775, .815}) leaves nearly a tenth of the canvas blank along the
// right and top edges -- in a paper that reads as a badly cropped figure. The
// inset below is only what the tick labels and the axis label need, so the
// plot box gets the rest of the canvas.
matplot::axes_handle journalAxes() {
    using namespace matplot;

    figure_handle fig = figure(true);
    fig->size(760, 520);  // ~3:2, the usual single-column figure aspect
    fig->font("Helvetica");
    fig->font_size(11.0f);

    axes_handle ax = gca();
    ax->position({0.125f, 0.135f, 0.855f, 0.835f});  // {x, y, width, height}
    ax->font("Helvetica");
    ax->font_size(11.0f);
    ax->line_width(1.0f);  // a heavier frame and ticks than the 0.5 default
    ax->box(true);
    ax->grid(true);
    ax->minor_grid(true);  // decade subdivisions, so log axes stay readable
    return ax;
}

// Line, marker edge and marker fill all take the same colour: when they agree
// Matplot++ emits the series as a single gnuplot "linespoints" command, and
// each series reads as one object rather than as outlines over a stroke.
matplot::line_handle styleSeries(matplot::line_handle line, const std::array<float, 4>& color) {
    line->line_width(1.6f);
    line->color(color);
    line->marker_size(5.0f);
    line->marker_color(color);
    line->marker_face_color(color);
    line->marker_face(true);
    return line;
}

// Padded limits for a log axis: clamping exactly to the data would leave the
// first and last markers half-clipped by the frame.
std::array<double, 2> logLimits(const std::vector<double>& xs) {
    constexpr double pad = 1.12;
    return {xs.front() / pad, xs.back() * pad};
}

void plotSweep(const SweepSeries& series, const std::string& outdir) {
    using namespace matplot;

    axes_handle runtimeAx = journalAxes();
    styleSeries(runtimeAx->loglog(series.ns, series.dantzigMs, "-o"), kBlue);
    runtimeAx->hold(on);
    styleSeries(runtimeAx->loglog(series.ns, series.groupedMs, "-s"), kRed);
    runtimeAx->hold(off);
    runtimeAx->xlabel("Number of items n");
    runtimeAx->ylabel("Runtime (ms, median over seeds and reps)");
    runtimeAx->xlim(logLimits(series.ns));
    runtimeAx->legend({"Dantzig", "Grouped"});
    // Both curves climb to the right, so the top-left corner is the one place
    // the key cannot sit on top of data.
    legend_handle runtimeKey = runtimeAx->legend();
    runtimeKey->location(legend::general_alignment::topleft);
    runtimeKey->box(true);
    runtimeKey->font_size(10.0f);
    const std::string runtimePath = outdir + "/runtime_comparison.svg";
    save(runtimePath);
    reportSaved(runtimePath);

    axes_handle errorAx = journalAxes();
    styleSeries(errorAx->semilogx(series.ns, series.errorPct, "-o"), kRed);
    errorAx->xlabel("Number of items n");
    errorAx->ylabel("Error rate (V* - V^{gp}) / V* [%]");
    errorAx->xlim(logLimits(series.ns));
    const std::string errorPath = outdir + "/error_rate.svg";
    save(errorPath);
    reportSaved(errorPath);
}

void plotSensitivity(const SensitivitySeries& series, int sensitivityN, const std::string& outdir) {
    using namespace matplot;

    // A log-scale y-axis can't render an exact 0; floor it so a perfect
    // (near-singleton-group) result still shows up at the bottom of the plot.
    std::vector<double> floored(series.errorPct.size());
    std::transform(series.errorPct.begin(), series.errorPct.end(), floored.begin(),
                   [](double e) { return std::max(e, 1e-6); });

    axes_handle ax = journalAxes();
    styleSeries(ax->loglog(series.deltas, floored, "-o"), kRed);
    ax->xlabel("Grouping tolerance δ (ratio-bin half-width)");
    ax->ylabel("Error rate (V* - V^{gp}) / V* [%]");
    ax->xlim(logLimits(series.deltas));
    // The instance size is the one experimental parameter a reader needs to
    // interpret this curve, and the error rate rises to the right, so it goes
    // in a key in the empty top-left corner rather than in a title.
    ax->legend({"n = " + std::to_string(sensitivityN)});
    legend_handle key = ax->legend();
    key->location(legend::general_alignment::topleft);
    key->box(true);
    key->font_size(10.0f);
    const std::string sensitivityPath = outdir + "/delta_sensitivity.svg";
    save(sensitivityPath);
    reportSaved(sensitivityPath);
}

// --- Command line ----------------------------------------------------------

gka::cli::Parser buildParser(const char* prog, Config& cfg) {
    gka::cli::Parser parser(
        prog,
        "  Runs three experiments comparing Dantzig's algorithm to the Grouped Knapsack\n"
        "  Allocation (my_algo.tex):\n"
        "    1) runtime vs n            -> DIR/runtime_comparison.png\n"
        "    2) error rate vs n         -> DIR/error_rate.png\n"
        "    3) error rate vs delta     -> DIR/delta_sensitivity.png\n");

    parser.add("--outdir", "DIR", "Directory for the generated figures (default figures)", cfg.outdir)
        .add("--seed-base", "N", "First RNG seed; seeds run N..N+num-seeds-1 (default 0)", cfg.seedBase)
        .add("--num-seeds", "S", "Instances aggregated per item count (default 10)", cfg.numSeeds)
        .add("--reps", "R", "Timed repetitions per instance (default 5)", cfg.reps)
        .add("--num-groups", "M", "Target number of groups (default 20)", cfg.numGroups)
        .add("--budget-ratio", "B", "C = B * F^w_{[n]}, in (0,1) (default 0.5)", cfg.opts.budgetRatio)
        .add("--sweep", "SHAPE", "Item-count sweep: dense or sparse (default dense)", cfg.sweepName)
        .add("--max-n", "N", "Largest item count in the sweep (default 100000)",
             [&cfg](const std::string& value) { cfg.maxN = std::stoll(value); })
        .add("--sensitivity-n", "N", "Fixed item count for experiment 3 (default 5000)",
             cfg.sensitivityN)
        .add("--sensitivity-points", "P", "Delta values swept in experiment 3 (default 25)",
             cfg.sensitivityPoints)
        .add("--sensitivity-seeds", "S", "Instances per delta in experiment 3 (default 5)",
             cfg.sensitivitySeeds)
        .add("--threads", "T", "Worker threads for untimed work (default: all cores)", cfg.threads)
        .epilog(
            "  Experiments 1-2 sweep n over an increasing sequence, aggregating num-seeds\n"
            "  random instances (each timed over reps repetitions: median runtime, mean\n"
            "  error rate). Experiment 3 fixes n=sensitivity-n and sweeps the grouping\n"
            "  tolerance delta itself over sensitivity-points log-spaced values, averaging\n"
            "  error rate over sensitivity-seeds instances per delta.\n"
            "  --threads never applies to a timed solve: those always run one at a time so\n"
            "  the runtime curve measures the algorithms rather than the machine's load.\n"
            "  Requires gnuplot >= 5.2.6 on PATH at runtime to render/save the figures.\n");

    return parser;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    gka::cli::Parser parser = buildParser(argv[0], cfg);

    switch (parser.parse(argc, argv)) {
        case gka::cli::Parser::Status::HelpRequested: return 0;
        case gka::cli::Parser::Status::Error: return 1;
        case gka::cli::Parser::Status::Ok: break;
    }

    std::vector<int> counts;
    gka::SweepShape sweep = gka::SweepShape::Dense;
    try {
        gka::cli::requirePositive("--num-seeds", cfg.numSeeds);
        gka::cli::requirePositive("--reps", cfg.reps);
        gka::cli::requirePositive("--num-groups", cfg.numGroups);
        gka::cli::requirePositive("--sensitivity-n", cfg.sensitivityN);
        gka::cli::requireGreaterThan("--sensitivity-points", cfg.sensitivityPoints, 1);
        gka::cli::requirePositive("--sensitivity-seeds", cfg.sensitivitySeeds);
        sweep = gka::parseSweepShape(cfg.sweepName);
        counts = gka::itemCounts(sweep, cfg.maxN);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        parser.usage(std::cerr);
        return 1;
    }

    gka::ThreadLimit threadLimit(cfg.threads);
    std::filesystem::create_directories(cfg.outdir);

    std::cerr << "Sweep: " << gka::sweepShapeName(sweep) << ", " << counts.size()
              << " item counts up to " << counts.back() << "; untimed work on "
              << threadLimit.activeThreads() << " thread(s), timed solves serial.\n";

    try {
        SweepSeries sweepSeries = runRuntimeAndError(cfg, counts);
        plotSweep(sweepSeries, cfg.outdir);

        SensitivitySeries sensitivity = runDeltaSensitivity(cfg);
        plotSensitivity(sensitivity, cfg.sensitivityN, cfg.outdir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
