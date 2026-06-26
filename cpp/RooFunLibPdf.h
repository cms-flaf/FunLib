/**
 * RooFunLibPdf.h -- a single streamable RooAbsPdf that represents any FunLib
 * function natively in C++ (no Python at runtime).
 *
 * One class serves every FunLib function: the function is selected by its
 * directory name (_dir) and parameterised by the streamed constants and the
 * floating shape parameters (_params).  This makes every FunLib function
 * usable through the RooAbsPdf interface and serialisable into a RooWorkspace
 * for use with Combine -- the kernel maths is pure C++ in FunLib.{so,dylib}.
 *
 * The transient kernel pointer is rebuilt on first evaluate() after I/O.
 */
#ifndef ROOFUNLIBPDF_H
#define ROOFUNLIBPDF_H

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "TString.h"
#include <string>
#include <vector>

namespace funlib {
struct IKernel;
}

class RooFunLibPdf : public RooAbsPdf {
public:
    RooFunLibPdf();
    RooFunLibPdf(const char* name, const char* title, RooAbsReal& x,
                 const RooArgList& params, const char* dir,
                 const std::vector<std::string>& cnames,
                 const std::vector<double>& cvals,
                 const std::vector<std::string>& snames,
                 const std::vector<std::string>& svals, int npar, double xmin,
                 double xmax);
    RooFunLibPdf(const RooFunLibPdf& other, const char* name = nullptr);
    TObject* clone(const char* newname) const override {
        return new RooFunLibPdf(*this, newname);
    }
    ~RooFunLibPdf() override;

protected:
    Double_t evaluate() const override;

private:
    RooRealProxy _x;
    RooListProxy _params;
    TString _dir;
    std::vector<std::string> _cnames;
    std::vector<double> _cvals;
    std::vector<std::string> _snames;
    std::vector<std::string> _svals;
    Int_t _npar = 0;
    Double_t _xmin = 0.0;
    Double_t _xmax = 0.0;

    mutable funlib::IKernel* _kernel = nullptr;  //!

    ClassDefOverride(RooFunLibPdf, 1)
};

#endif  // ROOFUNLIBPDF_H
