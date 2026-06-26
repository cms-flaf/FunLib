/**
 * funlib_base.h -- Shared infrastructure for the FunLib C++ kernels.
 *
 * Each FunLib function has a header-only C++ kernel in
 * FunLib/functions/<dir>/fn.h that defines a struct deriving from
 * funlib::IKernel and registers a factory in the global registry under its
 * directory name.  All kernels are compiled into a single shared library
 * (FunLib.{so,dylib}) together with the streamable RooAbsPdf wrapper
 * RooFunLibPdf.
 *
 * The kernels are pure shape kernels: NO normalization parameter N, exactly
 * mirroring the Python fn.py contract (scalar x, positional params in the
 * order given by param_names, returns a shape value; 0 / nan / inf semantics
 * preserved -- the caller screens nan/inf/negatives just like the factory).
 */
#ifndef FUNLIB_BASE_H
#define FUNLIB_BASE_H

#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <limits>

namespace funlib {

// ----------------------------------------------------------------------------
// Constants -- a desc.yaml "constants" block: numeric, string and bool values.
// Booleans are stored in the numeric map as 0.0 / 1.0.
// ----------------------------------------------------------------------------
struct Constants {
    std::map<std::string, double> nums;
    std::map<std::string, std::string> strs;

    double get(const std::string& k, double def) const {
        auto it = nums.find(k);
        return it == nums.end() ? def : it->second;
    }
    bool getb(const std::string& k, bool def) const {
        auto it = nums.find(k);
        if (it != nums.end()) return it->second != 0.0;
        auto is = strs.find(k);
        if (is != strs.end())
            return is->second == "true" || is->second == "True" || is->second == "1";
        return def;
    }
    std::string gets(const std::string& k, const std::string& def) const {
        auto it = strs.find(k);
        return it == strs.end() ? def : it->second;
    }
    bool has(const std::string& k) const {
        return nums.count(k) || strs.count(k);
    }
};

// ----------------------------------------------------------------------------
// IKernel -- pure shape kernel interface.
//   eval(x, p, npar) returns the shape value (no N).
//   param_names() returns the parameter order (for RooFit/debugging).
// ----------------------------------------------------------------------------
struct IKernel {
    virtual ~IKernel() {}
    virtual double eval(double x, const double* p, int npar) const = 0;
    virtual std::vector<std::string> param_names() const { return {}; }
};

// Factory: (constants, xmin, xmax, npar) -> new kernel.
// xmin/xmax/npar are only meaningful for range/degree-dependent kernels;
// fixed kernels ignore them.
using KernelFactory =
    std::function<IKernel*(const Constants&, double, double, int)>;

// Global registry, keyed by function directory name.
inline std::map<std::string, KernelFactory>& registry() {
    static std::map<std::string, KernelFactory> r;
    return r;
}

struct Registrar {
    Registrar(const std::string& name, KernelFactory f) {
        registry()[name] = f;
    }
};

#define FUNLIB_CAT2(a, b) a##b
#define FUNLIB_CAT(a, b) FUNLIB_CAT2(a, b)
#define FUNLIB_REGISTER(NAME, EXPR)                                           \
    static ::funlib::Registrar FUNLIB_CAT(_funlib_reg_, __COUNTER__)(         \
        NAME, [](const ::funlib::Constants& c, double xmin, double xmax,      \
                 int npar) -> ::funlib::IKernel* { return (EXPR); })

// Small numeric helpers shared by many kernels.
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

inline double clampmax(double v, double hi) { return v > hi ? hi : v; }
inline double absmax(double v, double lo) {
    double a = std::fabs(v);
    return a < lo ? lo : a;
}

}  // namespace funlib

#endif  // FUNLIB_BASE_H
