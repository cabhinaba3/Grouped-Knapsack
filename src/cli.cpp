#include "cli.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace gka::cli {
namespace {

// std::sto* accept trailing garbage ("12abc" -> 12); reject it so a typo in
// a flag value fails loudly instead of silently changing the experiment.
void requireFullyConsumed(const std::string& flag, const std::string& text, std::size_t used) {
    if (used != text.size())
        throw std::invalid_argument(flag + ": '" + text + "' is not a valid value");
}

}  // namespace

Parser::Parser(std::string programName, std::string description)
    : program_(std::move(programName)), description_(std::move(description)) {}

Parser& Parser::add(std::string flag, std::string metavar, std::string help,
                    std::function<void(const std::string&)> apply) {
    options_.push_back({std::move(flag), std::move(metavar), std::move(help), std::move(apply)});
    return *this;
}

Parser& Parser::add(std::string flag, std::string metavar, std::string help, std::string& dest) {
    return add(std::move(flag), std::move(metavar), std::move(help),
               [&dest](const std::string& value) { dest = value; });
}

Parser& Parser::add(std::string flag, std::string metavar, std::string help, int& dest) {
    std::string name = flag;
    return add(std::move(flag), std::move(metavar), std::move(help),
               [&dest, name](const std::string& value) {
                   std::size_t used = 0;
                   int parsed = std::stoi(value, &used);
                   requireFullyConsumed(name, value, used);
                   dest = parsed;
               });
}

Parser& Parser::add(std::string flag, std::string metavar, std::string help, double& dest) {
    std::string name = flag;
    return add(std::move(flag), std::move(metavar), std::move(help),
               [&dest, name](const std::string& value) {
                   std::size_t used = 0;
                   double parsed = std::stod(value, &used);
                   requireFullyConsumed(name, value, used);
                   dest = parsed;
               });
}

Parser& Parser::add(std::string flag, std::string metavar, std::string help, std::uint64_t& dest) {
    std::string name = flag;
    return add(std::move(flag), std::move(metavar), std::move(help),
               [&dest, name](const std::string& value) {
                   std::size_t used = 0;
                   unsigned long long parsed = std::stoull(value, &used);
                   requireFullyConsumed(name, value, used);
                   dest = static_cast<std::uint64_t>(parsed);
               });
}

Parser& Parser::epilog(std::string text) {
    epilog_ = std::move(text);
    return *this;
}

void Parser::usage(std::ostream& os) const {
    os << "Usage: " << program_ << " [OPTIONS]\n\n" << description_ << "\nOptions:\n";

    std::size_t width = 0;
    for (const Option& o : options_) width = std::max(width, o.flag.size() + o.metavar.size() + 1);

    for (const Option& o : options_) {
        std::string left = o.flag + (o.metavar.empty() ? "" : " " + o.metavar);
        os << "  " << left << std::string(width - left.size() + 2, ' ') << o.help << "\n";
    }
    os << "  --help" << std::string(width - 6 + 2, ' ') << "Show this message and exit\n";

    if (!epilog_.empty()) os << "\n" << epilog_;
}

Parser::Status Parser::parse(int argc, char** argv) const {
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage(std::cout);
                return Status::HelpRequested;
            }

            auto it = std::find_if(options_.begin(), options_.end(),
                                   [&arg](const Option& o) { return o.flag == arg; });
            if (it == options_.end()) throw std::invalid_argument("unknown argument: " + arg);
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            it->apply(argv[++i]);
        }
        return Status::Ok;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        usage(std::cerr);
        return Status::Error;
    }
}

void requirePositive(const char* flag, int value) {
    if (value <= 0) throw std::invalid_argument(std::string(flag) + " must be positive");
}

void requirePositive(const char* flag, long long value) {
    if (value <= 0) throw std::invalid_argument(std::string(flag) + " must be positive");
}

void requireGreaterThan(const char* flag, int value, int bound) {
    if (value <= bound)
        throw std::invalid_argument(std::string(flag) + " must be greater than " +
                                    std::to_string(bound));
}

void requireInOpenInterval(const char* flag, double value, double lo, double hi) {
    if (value <= lo || value >= hi)
        throw std::invalid_argument(std::string(flag) + " must be in (" + std::to_string(lo) +
                                    ", " + std::to_string(hi) + ")");
}

}  // namespace gka::cli
