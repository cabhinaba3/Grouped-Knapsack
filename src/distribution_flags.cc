#include "distribution_flags.h"

#include <stdexcept>
#include <string>

namespace gka {
namespace {

namespace po = boost::program_options;

void AddOneFieldFlags(po::options_description& desc, const char* flag, const char* description,
                      const char* mean_help, const char* stddev_help, std::string& dist_name,
                      FieldSpec& spec) {
  const std::string dist_flag = std::string(flag) + "-dist";
  const std::string mean_flag = std::string(flag) + "-mean";
  const std::string stddev_flag = std::string(flag) + "-stddev";
  const std::string dist_help =
      "Distribution for " + std::string(description) + ": uniform or normal (default uniform)";

  desc.add_options()(dist_flag.c_str(), po::value<std::string>(&dist_name)->value_name("DIST"),
                     dist_help.c_str());
  desc.add_options()(mean_flag.c_str(), po::value<double>(&spec.mean)->value_name("M"), mean_help);
  desc.add_options()(stddev_flag.c_str(), po::value<double>(&spec.stddev)->value_name("S"),
                     stddev_help);
}

void ResolveOneFieldFlag(const char* flag, const std::string& dist_name, FieldSpec& spec) {
  try {
    spec.dist = ParseDistributionKind(dist_name);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(flag) + ": " + e.what());
  }
}

}  // namespace

void AddDistributionFlags(po::options_description& desc, std::string& v_dist_name,
                          FieldSpec& v_spec, std::string& w_dist_name, FieldSpec& w_spec,
                          std::string& u_dist_name, FieldSpec& u_spec) {
  AddOneFieldFlags(desc, "v", "item values v_i", "Mean for --v-dist normal (default 50.5)",
                   "Stddev for --v-dist normal (default 16.5)", v_dist_name, v_spec);
  AddOneFieldFlags(desc, "w", "item costs w_i", "Mean for --w-dist normal (default 50.5)",
                   "Stddev for --w-dist normal (default 16.5)", w_dist_name, w_spec);
  AddOneFieldFlags(desc, "u", "capacity limits u_i",
                   "Mean for --u-dist normal, in [0,1] (default 0.5)",
                   "Stddev for --u-dist normal (default 0.1667)", u_dist_name, u_spec);
}

void ResolveDistributionFlags(const std::string& v_dist_name, FieldSpec& v_spec,
                              const std::string& w_dist_name, FieldSpec& w_spec,
                              const std::string& u_dist_name, FieldSpec& u_spec) {
  ResolveOneFieldFlag("--v-dist", v_dist_name, v_spec);
  ResolveOneFieldFlag("--w-dist", w_dist_name, w_spec);
  ResolveOneFieldFlag("--u-dist", u_dist_name, u_spec);
}

}  // namespace gka
