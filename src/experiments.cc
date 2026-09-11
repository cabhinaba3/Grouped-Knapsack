#include <matplot/matplot.h>

#include <algorithm>
#include <array>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "cli.h"
#include "data_generate.h"
#include "distribution_flags.h"
#include "grouped.h"
#include "harness.h"
#include "parallel.h"

// gka_experiments: runs the three experiments comparing Dantzig's greedy rule
// to the Grouped Knapsack Allocation (runtime and error rate vs n, then error
// rate vs the grouping tolerance delta) and writes their figures to disk.
namespace {

struct Config {
  gka::GenerateOptions opts;
  std::string outdir = "figures";
  std::string sweep_name = "dense";
  long long max_n = gka::kDefaultMaxN;
  std::uint64_t seed_base = 0;
  int num_seeds = 10;
  int reps = 5;
  int num_groups = 20;
  // < 0 means "derive per instance from num_groups"; otherwise a fraction
  // in (0,1) of the instance's trimmed ratio spread.
  double delta = -1.0;
  int sensitivity_n = 5000;
  int sensitivity_points = 25;
  int sensitivity_seeds = 5;
  int threads = 0;
  std::string v_dist_name = "uniform";
  std::string w_dist_name = "uniform";
  std::string u_dist_name = "uniform";
};

std::vector<double> LogSpace(double lo, double hi, int points) {
  std::vector<double> out(static_cast<std::size_t>(points));
  const double log_lo = std::log(lo);
  const double log_hi = std::log(hi);
  for (int i = 0; i < points; ++i) {
    const double t = points == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(points - 1);
    out[static_cast<std::size_t>(i)] = std::exp(log_lo + t * (log_hi - log_lo));
  }
  return out;
}

void ReportSaved(const std::string& path) {
  constexpr int kPollAttempts = 40;
  constexpr std::chrono::milliseconds kPollInterval(50);
  constexpr std::uintmax_t kMinValidFileSizeBytes = 1024;

  std::error_code ec;
  for (int attempt = 0; attempt < kPollAttempts; ++attempt) {
    const auto size = std::filesystem::file_size(path, ec);
    if (!ec && size > kMinValidFileSizeBytes) {
      std::cerr << "Wrote " << path << " (" << size << " bytes)\n";
      return;
    }
    std::this_thread::sleep_for(kPollInterval);
  }
  std::cerr << "Warning: " << path << " was not written correctly.\n";
}

std::vector<gka::InstanceSpec> SeedSpecs(int n, std::uint64_t seed_base, int count) {
  std::vector<gka::InstanceSpec> specs;
  specs.reserve(static_cast<std::size_t>(count));
  for (int s = 0; s < count; ++s) specs.push_back({n, seed_base + static_cast<std::uint64_t>(s)});
  return specs;
}

struct SweepSeries {
  std::vector<double> ns;
  std::vector<double> dantzig_ms;
  std::vector<double> grouped_ms;
  std::vector<double> error_pct;
  std::vector<double> group_counts;
};

SweepSeries RunRuntimeAndError(const Config& cfg, const std::vector<int>& counts) {
  SweepSeries series;
  series.ns.resize(counts.size());
  series.dantzig_ms.resize(counts.size());
  series.grouped_ms.resize(counts.size());
  series.error_pct.resize(counts.size());
  series.group_counts.resize(counts.size());

  std::cout << "Experiment 1+2: runtime and error rate vs n\n";
  std::cout << std::left << std::setw(8) << "n" << std::setw(14) << "dantzig_ms" << std::setw(14)
            << "grouped_ms" << std::setw(10) << "error_%" << std::setw(8) << "avg_m\n";

  for (std::size_t idx = 0; idx < counts.size(); ++idx) {
    const int n = counts[idx];
    gka::InstanceBatch batch = gka::GenerateBatch(SeedSpecs(n, cfg.seed_base, cfg.num_seeds),
                                                  cfg.opts, cfg.delta, cfg.num_groups);
    std::vector<double> dantzig_ms_by_seed, grouped_ms_by_seed, error_pct_by_seed,
        group_counts_by_seed;
    dantzig_ms_by_seed.reserve(static_cast<std::size_t>(cfg.num_seeds));
    grouped_ms_by_seed.reserve(static_cast<std::size_t>(cfg.num_seeds));
    error_pct_by_seed.reserve(static_cast<std::size_t>(cfg.num_seeds));
    group_counts_by_seed.reserve(static_cast<std::size_t>(cfg.num_seeds));

    for (std::size_t s = 0; s < batch.instances.size(); ++s) {
      std::vector<double> dantzig_reps(static_cast<std::size_t>(cfg.reps));
      std::vector<double> grouped_reps(static_cast<std::size_t>(cfg.reps));
      gka::SolveOutcome outcome;
      for (int r = 0; r < cfg.reps; ++r) {
        gka::Timings timing = gka::TimeBoth(batch.instances[s], batch.deltas[s], outcome);
        dantzig_reps[static_cast<std::size_t>(r)] = timing.dantzig_ms;
        grouped_reps[static_cast<std::size_t>(r)] = timing.grouped_ms;
      }
      dantzig_ms_by_seed.push_back(gka::Median(dantzig_reps));
      grouped_ms_by_seed.push_back(gka::Median(grouped_reps));
      error_pct_by_seed.push_back(outcome.ErrorPercent());
      group_counts_by_seed.push_back(static_cast<double>(outcome.num_groups));
    }
    series.ns[idx] = static_cast<double>(n);
    series.dantzig_ms[idx] = gka::Median(dantzig_ms_by_seed);
    series.grouped_ms[idx] = gka::Median(grouped_ms_by_seed);
    series.error_pct[idx] = gka::Mean(error_pct_by_seed);
    series.group_counts[idx] = gka::Mean(group_counts_by_seed);
    std::cout << std::left << std::setw(8) << n << std::setw(14) << series.dantzig_ms[idx]
              << std::setw(14) << series.grouped_ms[idx] << std::setw(10) << series.error_pct[idx]
              << std::setw(8) << series.group_counts[idx] << "\n";
  }
  return series;
}

struct SensitivitySeries {
  std::vector<double> deltas;
  std::vector<double> error_pct;
  std::vector<double> group_counts;
};

SensitivitySeries RunDeltaSensitivity(const Config& cfg) {
  // The per-instance deltas GenerateBatch would derive are unused below
  // (this function sweeps its own delta values instead), so -1.0 (the
  // "derive from num_groups" sentinel) is passed only to get valid
  // instances back; any placeholder that passes validation would do.
  gka::InstanceBatch batch =
      gka::GenerateBatch(SeedSpecs(cfg.sensitivity_n, cfg.seed_base, cfg.sensitivity_seeds),
                         cfg.opts, -1.0, cfg.num_groups);
  const std::vector<gka::Instance>& instances = batch.instances;
  const double delta_max = gka::DeltaForGroupCount(instances[0].v, instances[0].w, 1);
  const double delta_min = delta_max / 2000.0;
  SensitivitySeries series;
  series.deltas = LogSpace(delta_min, delta_max, cfg.sensitivity_points);
  series.error_pct.resize(series.deltas.size());
  series.group_counts.resize(series.deltas.size());

  const std::size_t num_seeds = instances.size();
  std::vector<double> error_flat(series.deltas.size() * num_seeds);
  std::vector<double> group_flat(series.deltas.size() * num_seeds);

  gka::ParallelFor(error_flat.size(), [&](std::size_t task) {
    const std::size_t delta_idx = task / num_seeds;
    const std::size_t seed_idx = task % num_seeds;
    gka::SolveOutcome outcome = gka::SolveBoth(instances[seed_idx], series.deltas[delta_idx]);
    error_flat[task] = outcome.ErrorPercent();
    group_flat[task] = static_cast<double>(outcome.num_groups);
  });

  for (std::size_t delta_idx = 0; delta_idx < series.deltas.size(); ++delta_idx) {
    const auto first = static_cast<std::ptrdiff_t>(delta_idx * num_seeds);
    const auto last = static_cast<std::ptrdiff_t>((delta_idx + 1) * num_seeds);
    series.error_pct[delta_idx] =
        gka::Mean(std::vector<double>(error_flat.begin() + first, error_flat.begin() + last));
    series.group_counts[delta_idx] =
        gka::Mean(std::vector<double>(group_flat.begin() + first, group_flat.begin() + last));
  }

  std::cout << "\nExperiment 3: error rate vs delta (n=" << cfg.sensitivity_n << ")\n";
  std::cout << std::left << std::setw(14) << "delta" << std::setw(12) << "avg_m" << std::setw(10)
            << "error_%\n";

  for (std::size_t delta_idx = 0; delta_idx < series.deltas.size(); ++delta_idx) {
    std::cout << std::left << std::setw(14) << series.deltas[delta_idx] << std::setw(12)
              << series.group_counts[delta_idx] << std::setw(10) << series.error_pct[delta_idx]
              << "\n";
  }

  return series;
}

constexpr std::array<float, 4> kRed{0.0f, 0.757f, 0.153f, 0.176f};
constexpr std::array<float, 4> kBlue{0.0f, 0.106f, 0.310f, 0.612f};

struct JournalFigure {
  matplot::figure_handle fig;
  matplot::axes_handle ax;
};

JournalFigure MakeJournalFigure() {
  using namespace matplot;
  JournalFigure journal_figure;
  journal_figure.fig = figure(true);
  journal_figure.fig->size(805, 565);
  journal_figure.fig->font("Helvetica");
  journal_figure.fig->font_size(9.0f);
  journal_figure.ax = journal_figure.fig->current_axes();
  journal_figure.ax->position({0.16f, 0.17f, 0.79f, 0.76f});
  journal_figure.ax->font("Helvetica");
  journal_figure.ax->font_size(8.5f);
  journal_figure.ax->line_width(1.0f);
  journal_figure.ax->box(true);
  journal_figure.ax->grid(true);
  journal_figure.ax->minor_grid(false);
  return journal_figure;
}

matplot::line_handle StyleSeries(matplot::line_handle line, const std::array<float, 4>& color) {
  line->line_width(1.2f);
  line->color(color);
  line->marker_size(4.0f);
  line->marker_color(color);
  line->marker_face(false);
  return line;
}

std::array<double, 2> LogLimits(const std::vector<double>& xs) {
  constexpr double kPad = 1.12;
  return {xs.front() / kPad, xs.back() * kPad};
}

void SaveJournalFigure(const matplot::axes_handle& ax, const std::string& path) {
  if (!ax) throw std::runtime_error("Cannot save figure: axes handle is null.");
  auto parent = ax->parent();
  if (!parent) throw std::runtime_error("Cannot save figure: axes has no parent figure.");
  // matplot::figure_type::save() redirects the backend's output to `path`
  // before calling draw() itself, so a prior draw() here would only render
  // once more to gnuplot's default terminal for no benefit -- and, with
  // GNUTERM left at its default, either as an unwanted ASCII-art dump or a
  // "no output" warning depending on terminal.
  parent->save(path, "pdfcairo");
  constexpr std::chrono::milliseconds kPostSaveSettleDelay(100);
  std::this_thread::sleep_for(kPostSaveSettleDelay);
  ReportSaved(path);
}

void PlotSweep(const SweepSeries& series, const std::string& outdir) {
  using namespace matplot;

  {
    JournalFigure journal_figure = MakeJournalFigure();
    auto ax = journal_figure.ax;
    auto dantzig = ax->loglog(series.ns, series.dantzig_ms, "-o");
    StyleSeries(dantzig, kBlue);
    ax->hold(on);
    auto grouped = ax->loglog(series.ns, series.grouped_ms, "-s");
    StyleSeries(grouped, kRed);
    ax->hold(off);
    ax->xlabel("Number of items n");
    ax->ylabel("Runtime (ms)");
    ax->xlim(LogLimits(series.ns));
    ax->legend({"Dantzig", "Grouped"});
    SaveJournalFigure(ax, outdir + "/runtime_comparison.pdf");
    ax.reset();
    journal_figure.fig.reset();
  }

  {
    JournalFigure journal_figure = MakeJournalFigure();
    auto ax = journal_figure.ax;
    auto error = ax->semilogx(series.ns, series.error_pct, "-o");
    StyleSeries(error, kRed);
    ax->xlabel("Number of items n");
    ax->ylabel("Relative error (%)");
    ax->xlim(LogLimits(series.ns));
    ax->legend({"Error"});
    SaveJournalFigure(ax, outdir + "/error_rate.pdf");
    ax.reset();
    journal_figure.fig.reset();
  }
}

void PlotSensitivity(const SensitivitySeries& series, int sensitivity_n,
                     const std::string& outdir) {
  using namespace matplot;
  std::vector<double> floored(series.error_pct.size());
  std::transform(series.error_pct.begin(), series.error_pct.end(), floored.begin(),
                 [](double e) { return std::max(e, 1e-6); });
  JournalFigure journal_figure = MakeJournalFigure();
  auto ax = journal_figure.ax;
  auto error = ax->loglog(series.deltas, floored, "-o");
  StyleSeries(error, kRed);
  ax->xlabel("Grouping tolerance delta");
  ax->ylabel("Relative error (%)");
  ax->xlim(LogLimits(series.deltas));
  ax->legend({"n = " + std::to_string(sensitivity_n)});
  SaveJournalFigure(ax, outdir + "/delta_sensitivity.pdf");
  ax.reset();
  journal_figure.fig.reset();
}

namespace po = boost::program_options;

constexpr char kDescription[] =
    "  Runs three experiments comparing Dantzig's algorithm "
    "to the Grouped Knapsack Allocation:\n"
    "    1) runtime vs n            -> DIR/runtime_comparison\n"
    "    2) error rate vs n         -> DIR/error_rate\n"
    "    3) error rate vs delta     -> DIR/delta_sensitivity\n";

constexpr char kEpilog[] =
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
    "  Requires gnuplot >= 5.2.6 with pdfcairo support.\n";

po::options_description BuildParser(Config& cfg) {
  po::options_description desc;
  desc.add_options()("help,h", "Show this message and exit")(
      "outdir", po::value<std::string>(&cfg.outdir)->value_name("DIR"),
      "Directory for generated figures (default figures)")(
      "seed-base", po::value<std::uint64_t>(&cfg.seed_base)->value_name("N"), "First RNG seed")(
      "num-seeds", po::value<int>(&cfg.num_seeds)->value_name("S"),
      "Instances aggregated per item count (default 10)")(
      "reps", po::value<int>(&cfg.reps)->value_name("R"),
      "Timed repetitions per instance (default 5)")(
      "num-groups", po::value<int>(&cfg.num_groups)->value_name("M"),
      "Target number of groups for experiments 1-2 when --delta is unset (default 20)")(
      "delta", po::value<double>(&cfg.delta)->value_name("D"),
      "Grouping tolerance for experiments 1-2 as a fraction of the ratio spread, "
      "in (0,1); overrides --num-groups (default: derived per n)")(
      "budget-ratio", po::value<double>(&cfg.opts.budget_ratio)->value_name("B"),
      "C = B * F^w_{[n]}, in (0,1) (default 0.5)");

  gka::AddDistributionFlags(desc, cfg.v_dist_name, cfg.opts.v_spec, cfg.w_dist_name,
                            cfg.opts.w_spec, cfg.u_dist_name, cfg.opts.u_spec);

  desc.add_options()("sweep", po::value<std::string>(&cfg.sweep_name)->value_name("SHAPE"),
                     "Item-count sweep: dense or sparse (default dense)")(
      "max-n", po::value<long long>(&cfg.max_n)->value_name("N"),
      "Largest item count in the sweep (default 100000)")(
      "sensitivity-n", po::value<int>(&cfg.sensitivity_n)->value_name("N"),
      "Fixed item count for experiment 3 (default 5000)")(
      "sensitivity-points", po::value<int>(&cfg.sensitivity_points)->value_name("P"),
      "Delta values swept in experiment 3 (default 25)")(
      "sensitivity-seeds", po::value<int>(&cfg.sensitivity_seeds)->value_name("S"),
      "Instances per delta (default 5)")("threads", po::value<int>(&cfg.threads)->value_name("T"),
                                         "Worker threads for untimed work (default: all cores)");

  return desc;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
  // Default to gnuplot's no-op terminal so SaveJournalFigure's draw() call
  // never tries to open an interactive window (e.g. Qt, which can abort the
  // process with no display attached) before the figure's explicit pdfcairo
  // save. "dumb" would avoid the same crash but renders each draw() as ASCII
  // art to stdout instead; only takes effect if GNUTERM isn't already set.
  setenv("GNUTERM", "unknown", 0);
#endif
  Config cfg;
  po::options_description desc = BuildParser(cfg);
  po::variables_map vm;
  if (auto exit_code = gka::cli::ExitCodeFor(
          gka::cli::Parse(argc, argv, argv[0], kDescription, kEpilog, desc, vm)))
    return *exit_code;

  std::vector<int> counts;
  gka::SweepShape sweep = gka::SweepShape::kDense;
  try {
    gka::cli::RequirePositive("--num-seeds", cfg.num_seeds);
    gka::cli::RequirePositive("--reps", cfg.reps);
    gka::cli::RequirePositive("--num-groups", cfg.num_groups);
    if (cfg.delta > 0.0) gka::cli::RequireInOpenInterval("--delta", cfg.delta, 0.0, 1.0);
    gka::ResolveDistributionFlags(cfg.v_dist_name, cfg.opts.v_spec, cfg.w_dist_name,
                                  cfg.opts.w_spec, cfg.u_dist_name, cfg.opts.u_spec);
    gka::cli::RequirePositive("--sensitivity-n", cfg.sensitivity_n);
    gka::cli::RequireGreaterThan("--sensitivity-points", cfg.sensitivity_points, 1);
    gka::cli::RequirePositive("--sensitivity-seeds", cfg.sensitivity_seeds);
    sweep = gka::ParseSweepShape(cfg.sweep_name);
    counts = gka::ItemCounts(sweep, cfg.max_n);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    gka::cli::PrintUsage(std::cerr, argv[0], kDescription, desc, kEpilog);
    return 1;
  }

  gka::ThreadLimit thread_limit(cfg.threads);
  std::filesystem::create_directories(cfg.outdir);
  std::cerr << "Sweep: " << gka::SweepShapeName(sweep) << ", " << counts.size()
            << " item counts up to " << counts.back() << "; untimed work on "
            << thread_limit.ActiveThreads() << " thread(s), timed solves serial.\n";
  if (cfg.delta > 0.0) {
    std::cerr << "Experiments 1-2: relative delta fixed at " << cfg.delta
              << " of the ratio spread (--num-groups ignored).\n";
  } else {
    std::cerr << "Experiments 1-2: delta derived per n to target " << cfg.num_groups
              << " groups.\n";
  }

  try {
    SweepSeries sweep_series = RunRuntimeAndError(cfg, counts);
    PlotSweep(sweep_series, cfg.outdir);
    SensitivitySeries sensitivity = RunDeltaSensitivity(cfg);
    PlotSensitivity(sensitivity, cfg.sensitivity_n, cfg.outdir);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
