#include "cli.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace gka::cli {
namespace po = boost::program_options;

void PrintUsage(std::ostream& os, const std::string& program_name, const std::string& description,
                const po::options_description& desc, const std::string& epilog) {
  os << "Usage: " << program_name << " [OPTIONS]\n\n" << description << "\nOptions:\n" << desc;
  if (!epilog.empty()) os << "\n" << epilog;
}

Status Parse(int argc, char** argv, const std::string& program_name, const std::string& description,
             const std::string& epilog, const po::options_description& desc,
             po::variables_map& vm) {
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    PrintUsage(std::cerr, program_name, description, desc, epilog);
    return Status::kError;
  }

  if (vm.count("help")) {
    PrintUsage(std::cout, program_name, description, desc, epilog);
    return Status::kHelpRequested;
  }
  return Status::kOk;
}

std::optional<int> ExitCodeFor(Status status) {
  switch (status) {
    case Status::kHelpRequested:
      return 0;
    case Status::kError:
      return 1;
    case Status::kOk:
      return std::nullopt;
  }
  return std::nullopt;
}

void RequirePositive(const char* flag, int value) {
  if (value <= 0) throw std::invalid_argument(std::string(flag) + " must be positive");
}

void RequirePositive(const char* flag, long long value) {
  if (value <= 0) throw std::invalid_argument(std::string(flag) + " must be positive");
}

void RequireGreaterThan(const char* flag, int value, int bound) {
  if (value <= bound)
    throw std::invalid_argument(std::string(flag) + " must be greater than " +
                                std::to_string(bound));
}

void RequireInOpenInterval(const char* flag, double value, double lo, double hi) {
  if (value <= lo || value >= hi)
    throw std::invalid_argument(std::string(flag) + " must be in (" + std::to_string(lo) + ", " +
                                std::to_string(hi) + ")");
}

}  // namespace gka::cli
