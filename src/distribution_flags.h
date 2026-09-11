#ifndef GKA_SRC_DISTRIBUTION_FLAGS_H_
#define GKA_SRC_DISTRIBUTION_FLAGS_H_

#include <boost/program_options.hpp>
#include <string>

#include "data_generate.h"

// Shared "--{v,w,u}-dist/-mean/-stddev" flag wiring for the three randomly
// drawn instance fields, used by both gka_dantzig and gka_experiments so the
// flag registration, help text and error-prefixing are written once instead
// of once per binary.
namespace gka {

// Registers all nine flags (three each for v, w, u) against the three
// dist-name strings and FieldSpecs.
void AddDistributionFlags(boost::program_options::options_description& desc,
                          std::string& v_dist_name, FieldSpec& v_spec, std::string& w_dist_name,
                          FieldSpec& w_spec, std::string& u_dist_name, FieldSpec& u_spec);

// Parses all three dist-name strings (collected by AddDistributionFlags)
// into their FieldSpec::dist once cli::Parse has succeeded, prefixing the
// owning flag on failure.
void ResolveDistributionFlags(const std::string& v_dist_name, FieldSpec& v_spec,
                              const std::string& w_dist_name, FieldSpec& w_spec,
                              const std::string& u_dist_name, FieldSpec& u_spec);

}  // namespace gka

#endif  // GKA_SRC_DISTRIBUTION_FLAGS_H_
