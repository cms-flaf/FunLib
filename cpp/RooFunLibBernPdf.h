/**
 * RooFunLibBernPdf.h -- a single native RooAbsPdf representing
 *
 *     f(x) = core(x; theta) * Bern_d(x; a)        (c0 = 1 fixed)
 *
 * i.e. any FunLib kernel (selected by _dir, parameterised by the streamed
 * constants + the floating core params _params) multiplied by a per-category
 * Bernstein polynomial of the floating coefficients _bern (c1..cd; c0=1 fixed,
 * Bernstein basis over [xmin,xmax]).  ``d = _bern.size()``; d=0 -> pure core.
 *
 * This is the core-PDF building block: because it is ONE RooAbsPdf whose value
 * depends on BOTH theta and a through proxy servers, RooFit's RooRealIntegral
 * tracks every parameter, so createIntegral(x, normSet, rangeName) yields the
 * correct, live per-region fraction  int_region f / int_full f  as theta and a
 * change during the fit.  (A RooProdPdf(RooFunLibPdf x RooGenericPdf) does NOT
 * -- its ranged normalisation integral is miscached and does not track the shape
 * params, giving a function-independent, frozen fraction.)
 *
 * The transient kernel pointer is rebuilt on first evaluate() after I/O.
 */
#ifndef ROOFUNLIBBERNPDF_H
#define ROOFUNLIBBERNPDF_H

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "TString.h"
#include <string>
#include <vector>

namespace funlib {
struct IKernel;
}

class RooFunLibBernPdf : public RooAbsPdf {
public:
    RooFunLibBernPdf();
    RooFunLibBernPdf(const char* name, const char* title, RooAbsReal& x,
                     const RooArgList& coreParams, const RooArgList& bernCoefs,
                     const char* dir, const std::vector<std::string>& cnames,
                     const std::vector<double>& cvals,
                     const std::vector<std::string>& snames,
                     const std::vector<std::string>& svals, int npar,
                     double xmin, double xmax);
    RooFunLibBernPdf(const RooFunLibBernPdf& other, const char* name = nullptr);
    TObject* clone(const char* newname) const override {
        return new RooFunLibBernPdf(*this, newname);
    }
    ~RooFunLibBernPdf() override;

protected:
    Double_t evaluate() const override;

private:
    RooRealProxy _x;
    RooListProxy _params;   // core shape params (theta)
    RooListProxy _bern;     // per-cat Bernstein coeffs c1..cd (c0=1 fixed)
    TString _dir;
    std::vector<std::string> _cnames;
    std::vector<double> _cvals;
    std::vector<std::string> _snames;
    std::vector<std::string> _svals;
    Int_t _npar = 0;
    Double_t _xmin = 0.0;
    Double_t _xmax = 0.0;

    mutable funlib::IKernel* _kernel = nullptr;  //!

    ClassDefOverride(RooFunLibBernPdf, 1)
};

#endif  // ROOFUNLIBBERNPDF_H
