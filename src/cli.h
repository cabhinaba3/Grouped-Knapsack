#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

// Minimal declarative parser for the "--flag VALUE" style both binaries use.
// Each binary registers its own flags against its own variables, so the
// parsing loop, the type conversions, the error messages and --help are
// written once instead of once per main().
namespace gka::cli {

class Parser {
public:
    Parser(std::string programName, std::string description);

    // Registers "--flag VALUE". metavar names the value in the usage text.
    // The typed overloads write straight into dest and report the flag name
    // if the value does not convert.
    Parser& add(std::string flag, std::string metavar, std::string help, std::string& dest);
    Parser& add(std::string flag, std::string metavar, std::string help, int& dest);
    Parser& add(std::string flag, std::string metavar, std::string help, double& dest);
    Parser& add(std::string flag, std::string metavar, std::string help, std::uint64_t& dest);
    // Escape hatch for values needing custom parsing (e.g. an enum).
    Parser& add(std::string flag, std::string metavar, std::string help,
                std::function<void(const std::string&)> apply);

    // Free-form text printed under the flag list by usage().
    Parser& epilog(std::string text);

    enum class Status { Ok, HelpRequested, Error };

    // Consumes argv[1..argc). On Error the reason and the usage text have
    // already been written to std::cerr; on HelpRequested the usage text has
    // gone to std::cout. Both mean "return from main now".
    Status parse(int argc, char** argv) const;

    void usage(std::ostream& os) const;

private:
    struct Option {
        std::string flag;
        std::string metavar;
        std::string help;
        std::function<void(const std::string&)> apply;
    };

    std::string program_;
    std::string description_;
    std::string epilog_;
    std::vector<Option> options_;
};

// Argument validators, throwing std::invalid_argument naming the flag.
void requirePositive(const char* flag, int value);
void requirePositive(const char* flag, long long value);
void requireGreaterThan(const char* flag, int value, int bound);

}  // namespace gka::cli
