/**
 * RooPyFitFunction.h
 *
 * A RooAbsPdf whose evaluate() delegates to a FunLib Python kernel at runtime.
 *
 * Design
 * ------
 * The class stores two plain TString members:
 *   _metaJson  -- fn_meta JSON written by mk_workspace (contains "entry" dict
 *                 with "dir"/"initial_params", plus "xmin", "xmax", etc.)
 *   _repoRoot  -- absolute path to the repo root (prepended to sys.path)
 *
 * Both are ROOT-streamable, so the workspace is fully self-contained after
 * ws.writeToFile().  The Python interpreter state (_pyKernel, etc.) is
 * transient (marked //!) and is reconstructed from _metaJson on the first
 * evaluate() call.
 *
 * Parameter floating
 * ------------------
 * Shape parameters are held in RooListProxy _params.  During a Combine (or
 * plain RooFit) fit they are varied by the minimizer; evaluate() reads their
 * current values and passes them to the Python kernel as keyword arguments.
 *
 * Thread safety
 * -------------
 * evaluate() acquires the Python GIL via PyGILState_Ensure/Release, so it is
 * safe to call from multiple threads.  Python-level operations are serialised
 * through the GIL as usual.
 *
 * Usage (Python / mk_workspace.py)
 * ---------------------------------
 *   import ROOT
 *   ROOT.gSystem.Load("path/to/libRooPyFitFunction.dylib")
 *   ...
 *   params = ROOT.RooArgList(p_a, p_b)
 *   bkg = ROOT.RooPyFitFunction("background", "background",
 *                                x_obs, params, meta_json_str, repo_root)
 *   ws.Import(bkg)
 *
 * Compile
 * -------
 *   python3 FunLib/tools/compile_plugin.py
 *   -> FunLib/tools/lib/libRooPyFitFunction.{dylib,so}
 *   -> FunLib/tools/lib/libRooPyFitFunction.rootmap  (ROOT autoloading)
 */

#ifndef ROOPYFITFUNCTION_H
#define ROOPYFITFUNCTION_H

/* Python.h MUST come first (avoids _POSIX_C_SOURCE / sys/select.h conflicts). */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "TString.h"
#include <string>
#include <vector>

class RooArgList;

class RooPyFitFunction : public RooAbsPdf {
public:
    /* Default constructor required by ROOT I/O. */
    RooPyFitFunction();

    /**
     * Full constructor.
     *
     * @param x          Observable RooRealVar.
     * @param params     Shape-parameter RooArgList (order must match
     *                   kernel.param_names).  These variables will float
     *                   during a RooFit/Combine minimisation.
     * @param meta_json  The fn_meta JSON string written by mk_workspace:
     *                   contains "entry" (with "dir", "initial_params", …),
     *                   "xmin", "xmax", "param_names", "key", etc.
     * @param repo_root  Absolute path to the repository root.  Prepended to
     *                   sys.path so that FunLib is importable.
     */
    RooPyFitFunction(const char*  name,
                     const char*  title,
                     RooAbsReal&  x,
                     const RooArgList& params,
                     const char*  meta_json,
                     const char*  repo_root);

    /** Copy constructor.  Transient Python state is NOT copied;
     *  it is reconstructed on the first evaluate() call of the clone. */
    RooPyFitFunction(const RooPyFitFunction& other, const char* name = nullptr);

    ~RooPyFitFunction() override;

    TObject* clone(const char* newname) const override {
        return new RooPyFitFunction(*this, newname);
    }

protected:
    Double_t evaluate() const override;

private:
    /* ---- Streamed members (ROOT I/O) ------------------------------------ */
    RooRealProxy _x;       ///< observable
    RooListProxy _params;  ///< shape params (same order as kernel.param_names)
    TString      _metaJson;  ///< fn_meta JSON
    TString      _repoRoot;  ///< repo root path for sys.path

    /* ---- Transient Python state (NOT streamed) -------------------------- */
    mutable PyObject*                  _pyKernel;    //!
    mutable bool                       _pyInited;    //!
    mutable std::vector<std::string>   _paramNames;  //!

    /** Initialise Python + load kernel on first evaluate() call.
     *  Returns true iff kernel is ready. */
    bool _ensureKernel() const;

    ClassDefOverride(RooPyFitFunction, 1)
};

#endif /* ROOPYFITFUNCTION_H */
