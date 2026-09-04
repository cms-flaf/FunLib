/**
 * funlib.cpp -- the single translation unit of FunLib.{so,dylib}.
 *
 * Includes every kernel header (registering all kernels), and implements
 * funlib::eval_kernel plus the RooFunLibPdf RooAbsPdf wrapper.
 */
#include "funlib_all.h"
#include "funlib_registry.h"
#include "RooFunLibPdf.h"
#include "RooFunLibBernPdf.h"

#include "RooArgList.h"
#include "RooAbsReal.h"

#include <cmath>
#include <map>
#include <memory>
#include <sstream>

namespace funlib {

// ---- eval_kernel with a small kernel cache ---------------------------------
namespace {
std::string cache_key(const std::string& dir,
                      const std::vector<std::string>& cn,
                      const std::vector<double>& cv,
                      const std::vector<std::string>& sn,
                      const std::vector<std::string>& sv, double xmin,
                      double xmax, int npar) {
    std::ostringstream o;
    o << dir << '|' << xmin << '|' << xmax << '|' << npar << '|';
    for (size_t i = 0; i < cn.size(); i++) o << cn[i] << '=' << cv[i] << ',';
    o << '|';
    for (size_t i = 0; i < sn.size(); i++) o << sn[i] << '=' << sv[i] << ',';
    return o.str();
}
std::map<std::string, std::shared_ptr<IKernel>>& kernel_cache() {
    static std::map<std::string, std::shared_ptr<IKernel>> m;
    return m;
}
}  // namespace

double eval_kernel(const std::string& dir,
                   const std::vector<std::string>& cnames,
                   const std::vector<double>& cvals,
                   const std::vector<std::string>& snames,
                   const std::vector<std::string>& svals, double xmin,
                   double xmax, int npar, double x,
                   const std::vector<double>& params) {
    std::string key =
        cache_key(dir, cnames, cvals, snames, svals, xmin, xmax, npar);
    auto& cache = kernel_cache();
    auto it = cache.find(key);
    IKernel* k = nullptr;
    if (it != cache.end()) {
        k = it->second.get();
    } else {
        Constants c = make_constants(cnames, cvals, snames, svals);
        k = make_kernel(dir, c, xmin, xmax, npar);
        cache[key] = std::shared_ptr<IKernel>(k);
    }
    if (!k) return std::numeric_limits<double>::quiet_NaN();
    return k->eval(x, params.empty() ? nullptr : params.data(),
                   (int)params.size());
}

}  // namespace funlib

// ---- RooFunLibPdf -----------------------------------------------------------
ClassImp(RooFunLibPdf)

RooFunLibPdf::RooFunLibPdf() : RooAbsPdf(), _x(), _params() {}

RooFunLibPdf::RooFunLibPdf(const char* name, const char* title, RooAbsReal& x,
                           const RooArgList& params, const char* dir,
                           const std::vector<std::string>& cnames,
                           const std::vector<double>& cvals,
                           const std::vector<std::string>& snames,
                           const std::vector<std::string>& svals, int npar,
                           double xmin, double xmax)
    : RooAbsPdf(name, title),
      _x("_x", "observable", this, x),
      _params("_params", "shape params", this),
      _dir(dir),
      _cnames(cnames),
      _cvals(cvals),
      _snames(snames),
      _svals(svals),
      _npar(npar),
      _xmin(xmin),
      _xmax(xmax) {
    _params.add(params);
}

RooFunLibPdf::RooFunLibPdf(const RooFunLibPdf& o, const char* name)
    : RooAbsPdf(o, name),
      _x("_x", this, o._x),
      _params("_params", this, o._params),
      _dir(o._dir),
      _cnames(o._cnames),
      _cvals(o._cvals),
      _snames(o._snames),
      _svals(o._svals),
      _npar(o._npar),
      _xmin(o._xmin),
      _xmax(o._xmax) {}

RooFunLibPdf::~RooFunLibPdf() { delete _kernel; }

Double_t RooFunLibPdf::evaluate() const {
    if (!_kernel) {
        funlib::Constants c =
            funlib::make_constants(_cnames, _cvals, _snames, _svals);
        _kernel = funlib::make_kernel(_dir.Data(), c, _xmin, _xmax, _npar);
    }
    if (!_kernel) return 0.0;
    int n = _params.size();
    std::vector<double> p(n);
    for (int i = 0; i < n; i++)
        p[i] = static_cast<const RooAbsReal*>(_params.at(i))->getVal();
    double v = _kernel->eval(double(_x), p.empty() ? nullptr : p.data(), n);
    // Floor the shape at a small but *normal* (non-subnormal) positive value
    // instead of returning 0.  A pdf that is exactly 0 (or underflows to a
    // subnormal) in a populated bin makes Combine's per-bin expected yield
    // non-normal -> `!std::isnormal(partialSum)` -> FASTEXIT ("PDF didn't
    // factorize"), which breaks Hesse (minimizer strategies 1 and 2).  Keeping
    // the shape strictly positive everywhere guarantees a finite NLL so Hesse
    // is robust; the floor (~1e-300) is far below any physical shape value over
    // the fit range, so it never affects the fitted minimum -- it only prevents
    // log(0) when a parameter excursion drives the analytic form to 0/negative.
    constexpr double kShapeFloor = 1e-300;
    if (!std::isfinite(v) || v < kShapeFloor) v = kShapeFloor;
    return v;
}

// ---- RooFunLibBernPdf (core x Bernstein, single native pdf) -----------------
ClassImp(RooFunLibBernPdf)

RooFunLibBernPdf::RooFunLibBernPdf()
    : RooAbsPdf(), _x(), _params(), _bern() {}

RooFunLibBernPdf::RooFunLibBernPdf(
    const char* name, const char* title, RooAbsReal& x,
    const RooArgList& coreParams, const RooArgList& bernCoefs, const char* dir,
    const std::vector<std::string>& cnames, const std::vector<double>& cvals,
    const std::vector<std::string>& snames, const std::vector<std::string>& svals,
    int npar, double xmin, double xmax)
    : RooAbsPdf(name, title),
      _x("_x", "observable", this, x),
      _params("_params", "core shape params", this),
      _bern("_bern", "per-cat Bernstein coeffs", this),
      _dir(dir),
      _cnames(cnames),
      _cvals(cvals),
      _snames(snames),
      _svals(svals),
      _npar(npar),
      _xmin(xmin),
      _xmax(xmax) {
    _params.add(coreParams);
    _bern.add(bernCoefs);
}

RooFunLibBernPdf::RooFunLibBernPdf(const RooFunLibBernPdf& o, const char* name)
    : RooAbsPdf(o, name),
      _x("_x", this, o._x),
      _params("_params", this, o._params),
      _bern("_bern", this, o._bern),
      _dir(o._dir),
      _cnames(o._cnames),
      _cvals(o._cvals),
      _snames(o._snames),
      _svals(o._svals),
      _npar(o._npar),
      _xmin(o._xmin),
      _xmax(o._xmax) {}

RooFunLibBernPdf::~RooFunLibBernPdf() { delete _kernel; }

Double_t RooFunLibBernPdf::evaluate() const {
    if (!_kernel) {
        funlib::Constants c =
            funlib::make_constants(_cnames, _cvals, _snames, _svals);
        _kernel = funlib::make_kernel(_dir.Data(), c, _xmin, _xmax, _npar);
    }
    constexpr double kShapeFloor = 1e-300;
    if (!_kernel) return kShapeFloor;
    // core(x; theta)
    int n = _params.size();
    std::vector<double> p(n);
    for (int i = 0; i < n; i++)
        p[i] = static_cast<const RooAbsReal*>(_params.at(i))->getVal();
    double core = _kernel->eval(double(_x), p.empty() ? nullptr : p.data(), n);
    // Bern_d(x; a): c0=1 fixed, coeffs c1..cd = _bern; basis over [xmin,xmax].
    int d = _bern.size();
    double bern = 1.0;
    if (d > 0) {
        double t = (double(_x) - _xmin) / (_xmax - _xmin);
        double val = 0.0;
        double binom = 1.0;  // C(d,0)
        for (int k = 0; k <= d; k++) {
            double ck =
                (k == 0)
                    ? 1.0
                    : static_cast<const RooAbsReal*>(_bern.at(k - 1))->getVal();
            val += ck * binom * std::pow(t, k) * std::pow(1.0 - t, d - k);
            binom = binom * (d - k) / (k + 1);  // -> C(d,k+1)
        }
        bern = val;
    }
    double v = core * bern;
    if (!std::isfinite(v) || v < kShapeFloor) v = kShapeFloor;
    return v;
}
