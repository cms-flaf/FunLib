/**
 * funlib_registry.h -- helpers to build and evaluate kernels by directory name.
 *
 * make_constants / make_kernel are header-only (used by RooFunLibPdf).
 * eval_kernel is defined in FunLib/cpp/funlib.cpp (the single TU that registers
 * all kernels); it caches kernels keyed by their full configuration.
 */
#ifndef FUNLIB_REGISTRY_H
#define FUNLIB_REGISTRY_H

#include "funlib_base.h"
#include <string>
#include <vector>

namespace funlib {

inline Constants make_constants(const std::vector<std::string>& cnames,
                                const std::vector<double>& cvals,
                                const std::vector<std::string>& snames,
                                const std::vector<std::string>& svals) {
    Constants c;
    for (size_t i = 0; i < cnames.size() && i < cvals.size(); i++)
        c.nums[cnames[i]] = cvals[i];
    for (size_t i = 0; i < snames.size() && i < svals.size(); i++)
        c.strs[snames[i]] = svals[i];
    return c;
}

inline IKernel* make_kernel(const std::string& dir, const Constants& c,
                            double xmin, double xmax, int npar) {
    auto it = registry().find(dir);
    if (it == registry().end()) return nullptr;
    return it->second(c, xmin, xmax, npar);
}

// All-in-one evaluation used by the Python validation bridge.  Kernels are
// cached internally so repeated point evaluations of one configuration are cheap.
double eval_kernel(const std::string& dir,
                   const std::vector<std::string>& cnames,
                   const std::vector<double>& cvals,
                   const std::vector<std::string>& snames,
                   const std::vector<std::string>& svals,
                   double xmin, double xmax, int npar, double x,
                   const std::vector<double>& params);

}  // namespace funlib

#endif  // FUNLIB_REGISTRY_H
