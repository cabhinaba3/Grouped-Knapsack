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
        // < 0 means "derive per instance from numGroups"; otherwise a
        // fraction in (0,1) of the instance's trimmed ratio spread.
        double delta = -1.0;
        int sensitivityN = 5000;
        int sensitivityPoints = 25;
        int sensitivitySeeds = 5;
        int threads = 0;
        std::string vDistName = "uniform";
        std::string wDistName = "uniform";
        std::string uDistName = "uniform";
    };
    
    std::vector<double> logspace(double lo, double hi, int points) {
        std::vector<double> out(static_cast<std::size_t>(points));
        const double logLo = std::log(lo);
        const double logHi = std::log(hi);
        for (int i = 0; i < points; ++i) {
            const double t = points == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(points - 1);
            out[static_cast<std::size_t>(i)] = std::exp(logLo + t * (logHi - logLo));
        }
        return out;
    }
    
    void reportSaved(const std::string &path) {
        std::error_code ec;
        for (int attempt = 0; attempt < 40; ++attempt) {
            const auto size = std::filesystem::file_size(path, ec);
            if (!ec && size > 1024) {
                std::cerr << "Wrote " << path << " (" << size << " bytes)\n";
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cerr << "Warning: " << path << " was not written correctly.\n";
    }
    
    std::vector<gka::InstanceSpec> seedSpecs(int n, std::uint64_t seedBase, int count) {
        std::vector<gka::InstanceSpec> specs;
        specs.reserve(static_cast<std::size_t>(count));
        for (int s = 0; s < count; ++s) {
            specs.push_back({n, seedBase + static_cast<std::uint64_t>(s)});
        }
        return specs;
    }
    
    struct SweepSeries {
        std::vector<double> ns;
        std::vector<double> dantzigMs;
        std::vector<double> groupedMs;
        std::vector<double> errorPct;
        std::vector<double> groupCounts;
    };

    SweepSeries runRuntimeAndError(const Config &cfg, const std::vector<int> &counts) {
        SweepSeries series;
        series.ns.resize(counts.size());
        series.dantzigMs.resize(counts.size());
        series.groupedMs.resize(counts.size());
        series.errorPct.resize(counts.size());
        series.groupCounts.resize(counts.size());

        std::cout << "Experiment 1+2: runtime and error rate vs n\n";
        std::cout << std::left << std::setw(8) << "n" << std::setw(14) << "dantzig_ms"
        << std::setw(14) << "grouped_ms" << std::setw(10) << "error_%" << std::setw(8) << "avg_m\n";

        for (std::size_t idx = 0; idx < counts.size(); ++idx) {
            const int n = counts[idx];
            gka::InstanceBatch batch = gka::generateBatch(seedSpecs(n, cfg.seedBase, cfg.numSeeds), cfg.opts, cfg.delta, cfg.numGroups);
            std::vector<double> dTimes, gTimes, errRates, groupCounts;
            dTimes.reserve(static_cast<std::size_t>(cfg.numSeeds));
            gTimes.reserve(static_cast<std::size_t>(cfg.numSeeds));
            errRates.reserve(static_cast<std::size_t>(cfg.numSeeds));
            groupCounts.reserve(static_cast<std::size_t>(cfg.numSeeds));

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
                errRates.push_back(outcome.errorPercent());
                groupCounts.push_back(static_cast<double>(outcome.numGroups));
            }
            series.ns[idx] = static_cast<double>(n);
            series.dantzigMs[idx] = gka::median(dTimes);
            series.groupedMs[idx] = gka::median(gTimes);
            series.errorPct[idx] = gka::mean(errRates);
            series.groupCounts[idx] = gka::mean(groupCounts);
            std::cout << std::left << std::setw(8) << n << std::setw(14) << series.dantzigMs[idx]
            << std::setw(14) << series.groupedMs[idx] << std::setw(10) << series.errorPct[idx]
            << std::setw(8) << series.groupCounts[idx] << "\n";
        }
        return series;
    }
    
    struct SensitivitySeries {
        std::vector<double> deltas;
        std::vector<double> errorPct;
        std::vector<double> groupCounts;
    };
    
    SensitivitySeries runDeltaSensitivity(const Config &cfg)
    {
        // The per-instance deltas generateBatch would derive are unused below
        // (this function sweeps its own delta values instead), so -1.0 (the
        // "derive from numGroups" sentinel) is passed only to get valid
        // instances back; any placeholder that passes validation would do.
        gka::InstanceBatch batch = gka::generateBatch(seedSpecs(cfg.sensitivityN, cfg.seedBase, cfg.sensitivitySeeds),
        cfg.opts, -1.0, cfg.numGroups);
        const std::vector<gka::Instance> &instances = batch.instances;
        const double deltaMax = gka::deltaForGroupCount(instances[0].v, instances[0].w, 1);
        const double deltaMin = deltaMax / 2000.0;
        SensitivitySeries series;
        series.deltas = logspace(deltaMin, deltaMax, cfg.sensitivityPoints);
        series.errorPct.resize(series.deltas.size());
        series.groupCounts.resize(series.deltas.size());
        
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
        
        for (std::size_t di = 0; di < series.deltas.size(); ++di) {
            const auto first = static_cast<std::ptrdiff_t>(di * numSeeds);
            const auto last = static_cast<std::ptrdiff_t>((di + 1) * numSeeds);
            series.errorPct[di] =
            gka::mean(std::vector<double>(errFlat.begin() + first, errFlat.begin() + last));
            series.groupCounts[di] =
            gka::mean(std::vector<double>(groupFlat.begin() + first, groupFlat.begin() + last));
        }
        
        std::cout << "\nExperiment 3: error rate vs delta (n=" << cfg.sensitivityN << ")\n";
        std::cout << std::left << std::setw(14) << "delta" << std::setw(12) << "avg_m"
        << std::setw(10) << "error_%\n";
        
        for (std::size_t di = 0; di < series.deltas.size(); ++di) {
            std::cout << std::left << std::setw(14) << series.deltas[di] << std::setw(12)
            << series.groupCounts[di] << std::setw(10) << series.errorPct[di] << "\n";
        }
        
        return series;
    }
    
    constexpr std::array<float, 4> kRed{0.0f, 0.757f, 0.153f, 0.176f};
    constexpr std::array<float, 4> kBlue{0.0f, 0.106f, 0.310f, 0.612f};
    
    struct JournalFigure {
        matplot::figure_handle fig;
        matplot::axes_handle ax;
    };
    
    JournalFigure journalFigure() {
        using namespace matplot;
        JournalFigure jf;
        jf.fig = figure(true);
        jf.fig->size(805, 565);
        jf.fig->font("Helvetica");
        jf.fig->font_size(9.0f);
        jf.ax = jf.fig->current_axes();
        jf.ax->position({0.16f, 0.17f, 0.79f, 0.76f});
        jf.ax->font("Helvetica");
        jf.ax->font_size(8.5f);
        jf.ax->line_width(1.0f);
        jf.ax->box(true);
        jf.ax->grid(true);
        jf.ax->minor_grid(false);
        return jf;
    }
    
    matplot::line_handle styleSeries(matplot::line_handle line, const std::array<float, 4> &color) {
        line->line_width(1.2f);
        line->color(color);
        line->marker_size(4.0f);
        line->marker_color(color);
        line->marker_face(false);
        return line;
    }
    
    std::array<double, 2> logLimits(const std::vector<double> &xs) {
        constexpr double pad = 1.12;
        return {xs.front() / pad, xs.back() * pad};
    }
    
    void saveJournalFigure(const matplot::axes_handle &ax, const std::string &path) {
        if (!ax) throw std::runtime_error("Cannot save figure: axes handle is null.");
        auto parent = ax->parent();
        if (!parent) throw std::runtime_error("Cannot save figure: axes has no parent figure.");
        parent->draw();
        parent->save(path, "pdfcairo");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        reportSaved(path);
    }
    
    void plotSweep(const SweepSeries &series, const std::string &outdir) {
        using namespace matplot;
        
        {
            JournalFigure jf = journalFigure();
            auto ax = jf.ax;
            auto dantzig = ax->loglog(series.ns, series.dantzigMs, "-o");
            styleSeries(dantzig, kBlue);
            ax->hold(on);
            auto grouped = ax->loglog(series.ns, series.groupedMs, "-s");
            styleSeries(grouped, kRed);
            ax->hold(off);
            ax->xlabel("Number of items n");
            ax->ylabel("Runtime (ms)");
            ax->xlim(logLimits(series.ns));
            ax->legend({"Dantzig", "Grouped"});
            saveJournalFigure(ax, outdir + "/runtime_comparison.pdf");
            ax.reset();
            jf.fig.reset();
        }
        
        {
            JournalFigure jf = journalFigure();
            auto ax = jf.ax;
            auto error = ax->semilogx(series.ns, series.errorPct, "-o");
            styleSeries(error, kRed);
            ax->xlabel("Number of items n");
            ax->ylabel("Relative error (%)");
            ax->xlim(logLimits(series.ns));
            ax->legend({"Error"});
            saveJournalFigure(ax, outdir + "/error_rate.pdf");
            ax.reset();
            jf.fig.reset();
        }
    }
    
    void plotSensitivity(const SensitivitySeries &series, int sensitivityN, const std::string &outdir) {
        using namespace matplot;
        std::vector<double> floored(series.errorPct.size());
        std::transform(series.errorPct.begin(), series.errorPct.end(), floored.begin(),
        [](double e) { return std::max(e, 1e-6); });
        JournalFigure jf = journalFigure();
        auto ax = jf.ax;
        auto error = ax->loglog(series.deltas, floored, "-o");
        styleSeries(error, kRed);
        ax->xlabel("Grouping tolerance delta");
        ax->ylabel("Relative error (%)");
        ax->xlim(logLimits(series.deltas));
        ax->legend({"n = " + std::to_string(sensitivityN)});
        saveJournalFigure(ax, outdir + "/delta_sensitivity.pdf");
        ax.reset();
        jf.fig.reset();
    }
    
    gka::cli::Parser buildParser(const char *prog, Config &cfg) {
        gka::cli::Parser parser(prog,
            "  Runs three experiments comparing Dantzig's algorithm "
            "to the Grouped Knapsack Allocation:\n"
            "    1) runtime vs n            -> DIR/runtime_comparison\n"
            "    2) error rate vs n         -> DIR/error_rate\n"
            "    3) error rate vs delta     -> DIR/delta_sensitivity\n");
        parser.add("--outdir", "DIR", "Directory for generated figures (default figures)", cfg.outdir)
            .add("--seed-base", "N", "First RNG seed", cfg.seedBase)
            .add("--num-seeds", "S", "Instances aggregated per item count (default 10)", cfg.numSeeds)
            .add("--reps", "R", "Timed repetitions per instance (default 5)", cfg.reps)
            .add("--num-groups", "M", "Target number of groups for experiments 1-2 when --delta is unset (default 20)", cfg.numGroups)
            .add("--delta", "D",
                 "Grouping tolerance for experiments 1-2 as a fraction of the ratio spread, "
                 "in (0,1); overrides --num-groups (default: derived per n)",
                 cfg.delta)
            .add("--budget-ratio", "B", "C = B * F^w_[n], in (0,1) (default 0.5)", cfg.opts.budgetRatio)
            .add("--v-dist", "DIST", "Distribution for item values v_i: uniform or normal (default uniform)",
                 cfg.vDistName)
            .add("--v-mean", "M", "Mean for --v-dist normal (default 50.5)", cfg.opts.vSpec.mean)
            .add("--v-stddev", "S", "Stddev for --v-dist normal (default 16.5)", cfg.opts.vSpec.stddev)
            .add("--w-dist", "DIST", "Distribution for item costs w_i: uniform or normal (default uniform)",
                 cfg.wDistName)
            .add("--w-mean", "M", "Mean for --w-dist normal (default 50.5)", cfg.opts.wSpec.mean)
            .add("--w-stddev", "S", "Stddev for --w-dist normal (default 16.5)", cfg.opts.wSpec.stddev)
            .add("--u-dist", "DIST",
                 "Distribution for capacity limits u_i: uniform or normal (default uniform)", cfg.uDistName)
            .add("--u-mean", "M", "Mean for --u-dist normal, in [0,1] (default 0.5)", cfg.opts.uSpec.mean)
            .add("--u-stddev", "S", "Stddev for --u-dist normal (default 0.1667)", cfg.opts.uSpec.stddev)
            .add("--sweep", "SHAPE", "Item-count sweep: dense or sparse (default dense)", cfg.sweepName)
            .add("--max-n", "N", "Largest item count in the sweep (default 100000)",
                 [&cfg](const std::string &value) { cfg.maxN = std::stoll(value); })
            .add("--sensitivity-n", "N", "Fixed item count for experiment 3 (default 5000)", cfg.sensitivityN)
            .add("--sensitivity-points", "P", "Delta values swept in experiment 3 (default 25)", cfg.sensitivityPoints)
            .add("--sensitivity-seeds", "S", "Instances per delta (default 5)", cfg.sensitivitySeeds)
            .add("--threads", "T", "Worker threads for untimed work (default: all cores)", cfg.threads)
            .epilog(
                "  Experiments 1-2 sweep n over an increasing sequence.\n"
                "  With --delta unset, each n derives its own delta to target\n"
                "  --num-groups groups, so avg_m stays roughly constant across n.\n"
                "  Pass --delta (a fraction of the ratio spread, in (0,1)) to hold\n"
                "  the relative grouping tolerance fixed instead, so avg_m is free\n"
                "  to grow with n. Being relative keeps it comparable across\n"
                "  instances whose v, w ranges differ.\n"
                "  --v-dist/--w-dist/--u-dist normal rejection-samples outside [lo,hi]\n"
                "  (item values/costs stay positive; capacity limits stay in [0,1]).\n"
                "  Runtime uses median timing over repetitions.\n"
                "  Error rate uses the mean across random instances.\n"
                "  Experiment 3 fixes n and sweeps delta on a log scale.\n"
                "  Timed solves are serial to avoid measuring machine contention.\n"
                "  Figures are written as vector PDF using pdfcairo.\n"
                "  Requires gnuplot >= 5.2.6 with pdfcairo support.\n");
        return parser;
    }

}  // namespace

int main(int argc, char **argv) {
    Config cfg;
    gka::cli::Parser parser = buildParser(argv[0], cfg);
    switch (parser.parse(argc, argv)) {
        case gka::cli::Parser::Status::HelpRequested:
            return 0;
        case gka::cli::Parser::Status::Error:
            return 1;
        case gka::cli::Parser::Status::Ok:
            break;
    }

    auto resolveDist = [](const char *flag, const std::string &name, gka::FieldSpec &spec) {
        try {
            spec.dist = gka::parseDistributionKind(name);
        } catch (const std::invalid_argument &e) {
            throw std::invalid_argument(std::string(flag) + ": " + e.what());
        }
    };

    std::vector<int> counts;
    gka::SweepShape sweep = gka::SweepShape::Dense;
    try {
        gka::cli::requirePositive("--num-seeds", cfg.numSeeds);
        gka::cli::requirePositive("--reps", cfg.reps);
        gka::cli::requirePositive("--num-groups", cfg.numGroups);
        if (cfg.delta > 0.0) gka::cli::requireInOpenInterval("--delta", cfg.delta, 0.0, 1.0);
        resolveDist("--v-dist", cfg.vDistName, cfg.opts.vSpec);
        resolveDist("--w-dist", cfg.wDistName, cfg.opts.wSpec);
        resolveDist("--u-dist", cfg.uDistName, cfg.opts.uSpec);
        gka::cli::requirePositive("--sensitivity-n", cfg.sensitivityN);
        gka::cli::requireGreaterThan("--sensitivity-points", cfg.sensitivityPoints, 1);
        gka::cli::requirePositive("--sensitivity-seeds", cfg.sensitivitySeeds);
        sweep = gka::parseSweepShape(cfg.sweepName);
        counts = gka::itemCounts(sweep, cfg.maxN);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        parser.usage(std::cerr);
        return 1;
    }

    gka::ThreadLimit threadLimit(cfg.threads);
    std::filesystem::create_directories(cfg.outdir);
    std::cerr << "Sweep: " << gka::sweepShapeName(sweep) << ", " << counts.size()
              << " item counts up to " << counts.back() << "; untimed work on "
              << threadLimit.activeThreads() << " thread(s), timed solves serial.\n";
    if (cfg.delta > 0.0) {
        std::cerr << "Experiments 1-2: relative delta fixed at " << cfg.delta
                   << " of the ratio spread (--num-groups ignored).\n";
    } else {
        std::cerr << "Experiments 1-2: delta derived per n to target " << cfg.numGroups << " groups.\n";
    }

    try {
        SweepSeries sweepSeries = runRuntimeAndError(cfg, counts);
        plotSweep(sweepSeries, cfg.outdir);
        SensitivitySeries sensitivity = runDeltaSensitivity(cfg);
        plotSensitivity(sensitivity, cfg.sensitivityN, cfg.outdir);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}