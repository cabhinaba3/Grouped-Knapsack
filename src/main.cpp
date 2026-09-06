#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.h"
#include "dantzig.h"
#include "dataGenerate.h"
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
    std::string sweepName = "dense";
    long long maxN = gka::kDefaultMaxN;
    int numGroups = 20;
    double delta = -1.0;  // < 0 means "derive per instance from numGroups"
    int threads = 0;      // 0 means "every hardware thread"
};

gka::cli::Parser buildParser(const char* prog, Config& cfg) {
    gka::cli::Parser parser(
        prog,
        "  Generates random bounded-fractional-knapsack instances in memory for an\n"
        "  increasing sequence of item counts n and solves each with Dantzig's greedy\n"
        "  algorithm and the Grouped Knapsack Allocation (my_algo.tex). Items are\n"
        "  grouped by binning the efficiency ratio v_i/w_i into contiguous bins of\n"
        "  width 2*delta.\n");

    parser.add("--seed", "N", "Base RNG seed (default 0)", cfg.opts.seed)
        .add("--budget-ratio", "R", "C = R * F^w_{[n]}, in (0,1) (default 0.5)", cfg.opts.budgetRatio)
        .add("--sweep", "SHAPE", "Item-count sweep: dense or sparse (default dense)", cfg.sweepName)
        .add("--max-n", "N", "Largest item count in the sweep (default 100000)",
             [&cfg](const std::string& value) { cfg.maxN = std::stoll(value); })
        .add("--num-groups", "M", "Target number of groups when --delta is unset (default 20)",
             cfg.numGroups)
        .add("--delta", "D", "Fixed grouping tolerance; overrides --num-groups", cfg.delta)
        .add("--threads", "T", "Worker threads for instance generation (default: all cores)",
             cfg.threads)
        .epilog(
            "  Solves are timed one at a time regardless of --threads, which only\n"
            "  controls how instances are generated ahead of the timed loop.\n");

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
        gka::cli::requirePositive("--num-groups", cfg.numGroups);
        sweep = gka::parseSweepShape(cfg.sweepName);
        counts = gka::itemCounts(sweep, cfg.maxN);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        parser.usage(std::cerr);
        return 1;
    }

    gka::ThreadLimit threadLimit(cfg.threads);
    const std::size_t lookahead = static_cast<std::size_t>(std::max(1, threadLimit.activeThreads()));

    std::cerr << "Sweep: " << gka::sweepShapeName(sweep) << ", " << counts.size()
              << " item counts up to " << counts.back() << "; generating on "
              << threadLimit.activeThreads() << " thread(s), timing serially.\n";

    std::cout << std::left << std::setw(8) << "n" << std::setw(14) << "C" << std::setw(14) << "F_total"
              << std::setw(16) << "V*(dantzig)" << std::setw(16) << "V^gp(grouped)" << std::setw(12)
              << "E=V*-V^gp" << std::setw(6) << "m" << std::setw(12) << "avg_z" << std::setw(12)
              << "time_ms" << "\n";

    try {
        // Instances are built a chunk at a time rather than all up front: the
        // generation of a chunk parallelises cleanly, while peak memory stays
        // bounded by one chunk instead of the whole sweep.
        for (std::size_t start = 0; start < counts.size(); start += lookahead) {
            const std::size_t stop = std::min(start + lookahead, counts.size());

            std::vector<gka::InstanceSpec> specs;
            specs.reserve(stop - start);
            for (std::size_t i = start; i < stop; ++i) specs.push_back({counts[i], cfg.opts.seed});

            gka::InstanceBatch batch =
                gka::generateBatch(specs, cfg.opts, cfg.delta, cfg.numGroups);

            for (std::size_t j = 0; j < specs.size(); ++j) {
                const gka::Instance& inst = batch.instances[j];

                gka::SolveOutcome outcome;
                gka::Timings timings = gka::timeBoth(inst, batch.deltas[j], outcome);

                std::cout << std::left << std::setw(8) << inst.n << std::setw(14) << inst.C
                          << std::setw(14) << inst.totalCapacityCost() << std::setw(16)
                          << outcome.vStar << std::setw(16) << outcome.vGrouped << std::setw(12)
                          << (outcome.vStar - outcome.vGrouped) << std::setw(6) << outcome.numGroups
                          << std::setw(12) << gka::averageAllocation(outcome.zDantzig)
                          << std::setw(12) << (timings.dantzigMs + timings.groupedMs) << "\n";

                batch.instances[j] = gka::Instance{};  // release before the next chunk
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
