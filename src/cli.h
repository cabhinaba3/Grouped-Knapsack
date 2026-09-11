#ifndef GKA_SRC_CLI_H_
#define GKA_SRC_CLI_H_

#include <boost/program_options.hpp>
#include <iosfwd>
#include <optional>
#include <string>

// Thin conveniences around Boost.Program_options for the "--flag VALUE"
// style both binaries use: a shared parse/--help/error-reporting flow, plus
// semantic validators Boost itself does not provide.
namespace gka::cli {

// Prints "Usage: {program_name} [OPTIONS]\n\n{description}\nOptions:\n{desc}",
// then epilog if non-empty.
void PrintUsage(std::ostream& os, const std::string& program_name, const std::string& description,
                const boost::program_options::options_description& desc, const std::string& epilog);

enum class Status { kOk, kHelpRequested, kError };

// Parses argv[1..argc) against desc (which must already register "help,h")
// and applies the results to whatever variables its options are bound to.
// On kError, the reason and desc's usage text have already gone to
// std::cerr; on kHelpRequested, desc's usage text (plus epilog) has gone to
// std::cout. Both mean "return from main now".
Status Parse(int argc, char** argv, const std::string& program_name, const std::string& description,
             const std::string& epilog, const boost::program_options::options_description& desc,
             boost::program_options::variables_map& vm);

// The exit code main() should return immediately for kHelpRequested (0) or
// kError (1); nullopt for kOk, meaning the caller should proceed.
std::optional<int> ExitCodeFor(Status status);

// Argument validators, throwing std::invalid_argument naming the flag.
void RequirePositive(const char* flag, int value);
void RequirePositive(const char* flag, long long value);
void RequireGreaterThan(const char* flag, int value, int bound);
// Throws unless lo < value < hi.
void RequireInOpenInterval(const char* flag, double value, double lo, double hi);

}  // namespace gka::cli

#endif  // GKA_SRC_CLI_H_
