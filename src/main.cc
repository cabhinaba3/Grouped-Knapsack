#include <algorithm>
#include <boost/program_options.hpp>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cli.h"
#include "dantzig.h"
#include "data_generate.h"
#include "distribution_flags.h"
#include "harness.h"
#include "parallel.h"

// gka_dantzig: solves one random instance per item count in the sweep with
// both Dantzig's greedy rule and the Grouped Knapsack Allocation, and prints
// a comparison table. Instances are built concurrently, but each solve is
// timed on its own with nothing else running, so the time_ms column stays
// comparable with the curves gka_experiments plots.
namespace {

struct Config {
  gka::GenerateOptions opts;
  std::string sweep_name = "dense";
  long long max_n = gka::kDefaultMaxN;
  int num_groups = 20;
  double delta = -1.0;  // < 0 means "derive per instance from num_groups"
  int threads = 0;      // 0 means "every hardware thread"
  std::string v_dist_name = "uniform";
  std::string w_dist_name = "uniform";
  std::string u_dist_name = "uniform";
};

namespace po = boost::program_options;

constexpr char kDescription[] =
    "  Generates random bounded-fractional-knapsack instances in memory for an\n"
    "  increasing sequence of item counts n and solves each with Dantzig's greedy\n"
    "  algorithm and the Grouped Knapsack Allocation (my_algo.tex). Items are\n"
    "  grouped by binning the efficiency ratio v_i/w_i into contiguous bins of\n"
    "  width 2*delta.\n";

constexpr char kEpilog[] =
    "  Solves are timed one at a time regardless of --threads, which only\n"
    "  controls how instances are generated ahead of the timed loop.\n"
    "  --v-dist/--w-dist/--u-dist normal rejection-samples outside [lo,hi]\n"
    "  (item values/costs stay positive; capacity limits stay in [0,1]).\n";

po::options_description BuildParser(Config& cfg) {
  po::options_description desc;
  desc.add_options()("help,h", "Show this message and exit")(
      "seed", po::value<std::uint64_t>(&cfg.opts.seed)->value_name("N"),
      "Base RNG seed (default 0)")("budget-ratio",
                                   po::value<double>(&cfg.opts.budget_ratio)->value_name("R"),
                                   "C = R * F^w_{[n]}, in (0,1) (default 0.5)")(
      "sweep", po::value<std::string>(&cfg.sweep_name)->value_name("SHAPE"),
      "Item-count sweep: dense or sparse (default dense)")(
      "max-n", po::value<long long>(&cfg.max_n)->value_name("N"),
      "Largest item count in the sweep (default 100000)")(
      "num-groups", po::value<int>(&cfg.num_groups)->value_name("M"),
      "Target number of groups when --delta is unset (default 20)")(
      "delta", po::value<double>(&cfg.delta)->value_name("D"),
      "Grouping tolerance as a fraction of the ratio spread, in (0,1); overrides --num-groups");

  gka::AddDistributionFlags(desc, cfg.v_dist_name, cfg.opts.v_spec, cfg.w_dist_name,
                            cfg.opts.w_spec, cfg.u_dist_name, cfg.opts.u_spec);

  desc.add_options()("threads", po::value<int>(&cfg.threads)->value_name("T"),
                     "Worker threads for instance generation (default: all cores)");

  return desc;
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  po::options_description desc = BuildParser(cfg);
  po::variables_map vm;

  if (auto exit_code = gka::cli::ExitCodeFor(
          gka::cli::Parse(argc, argv, argv[0], kDescription, kEpilog, desc, vm)))
    return *exit_code;

  std::vector<int> counts;
  gka::SweepShape sweep = gka::SweepShape::kDense;
  try {
    gka::cli::RequirePositive("--num-groups", cfg.num_groups);
    if (cfg.delta > 0.0) gka::cli::RequireInOpenInterval("--delta", cfg.delta, 0.0, 1.0);
    gka::ResolveDistributionFlags(cfg.v_dist_name, cfg.opts.v_spec, cfg.w_dist_name,
                                  cfg.opts.w_spec, cfg.u_dist_name, cfg.opts.u_spec);
    sweep = gka::ParseSweepShape(cfg.sweep_name);
    counts = gka::ItemCounts(sweep, cfg.max_n);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    gka::cli::PrintUsage(std::cerr, argv[0], kDescription, desc, kEpilog);
    return 1;
  }

  gka::ThreadLimit thread_limit(cfg.threads);
  const std::size_t lookahead = static_cast<std::size_t>(std::max(1, thread_limit.ActiveThreads()));

  std::cerr << "Sweep: " << gka::SweepShapeName(sweep) << ", " << counts.size()
            << " item counts up to " << counts.back() << "; generating on "
            << thread_limit.ActiveThreads() << " thread(s), timing serially.\n";

  std::cout << std::left << std::setw(8) << "n" << std::setw(14) << "C" << std::setw(14)
            << "F_total" << std::setw(16) << "V*(dantzig)" << std::setw(16) << "V^gp(grouped)"
            << std::setw(12) << "E=V*-V^gp" << std::setw(6) << "m" << std::setw(12) << "avg_z"
            << std::setw(12) << "time_ms" << "\n";

  try {
    // Instances are built a chunk at a time rather than all up front: the
    // generation of a chunk parallelises cleanly, while peak memory stays
    // bounded by one chunk instead of the whole sweep.
    for (std::size_t start = 0; start < counts.size(); start += lookahead) {
      const std::size_t stop = std::min(start + lookahead, counts.size());

      std::vector<gka::InstanceSpec> specs;
      specs.reserve(stop - start);
      for (std::size_t i = start; i < stop; ++i) specs.push_back({counts[i], cfg.opts.seed});

      gka::InstanceBatch batch = gka::GenerateBatch(specs, cfg.opts, cfg.delta, cfg.num_groups);

      for (std::size_t j = 0; j < specs.size(); ++j) {
        const gka::Instance& inst = batch.instances[j];

        gka::SolveOutcome outcome;
        gka::Timings timings = gka::TimeBoth(inst, batch.deltas[j], outcome);

        std::cout << std::left << std::setw(8) << inst.n << std::setw(14) << inst.C << std::setw(14)
                  << inst.TotalCapacityCost() << std::setw(16) << outcome.v_star << std::setw(16)
                  << outcome.v_grouped << std::setw(12) << (outcome.v_star - outcome.v_grouped)
                  << std::setw(6) << outcome.num_groups << std::setw(12)
                  << gka::AverageAllocation(outcome.z_dantzig) << std::setw(12)
                  << (timings.dantzig_ms + timings.grouped_ms) << "\n";

        batch.instances[j] = gka::Instance{};  // release before the next chunk
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
