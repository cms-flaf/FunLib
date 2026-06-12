"""mk_workspace -- Build and save a CMS Combine-compatible RooFit workspace.

Background shape is stored as a **RooPyFitFunction** (floating shape params) rather
than a frozen RooHistPdf.  Shape parameters are declared as RooRealVars in the
workspace and connected to the PDF, so Combine (or any RooFit minimiser) can float
them during the fit.

The RooPyFitFunction class is implemented in RooPyFitFunction.cxx and compiled by
``FunLib/tools/compile_plugin.py`` into ``FunLib/tools/lib/libRooPyFitFunction.dylib``.
mk_workspace compiles the library on first use if not already built.
"""

from __future__ import annotations

import json
import math
import os
import shutil as _shutil
import sys as _sys
import sysconfig as _sysconfig

__all__ = ["mk_workspace"]

# Path to the CMS Combine binary (used by the test suite).
# Auto-detected from PATH first; falls back to the local macOS build path.
COMBINE_BIN = (
    _shutil.which("combine") or "/Users/Kes/workspace/combine/build/bin/combine"
)

# Library directory (relative to this file)
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_TOOLS_DIR, "lib")

# Platform-specific extension
_SO_EXT = "dylib" if _sys.platform == "darwin" else "so"
_LIB_SO = os.path.join(_LIB_DIR, f"libRooPyFitFunction.{_SO_EXT}")

# Python shared library path -- needed for Combine (pure C++ process that has no
# Python loaded).  When the rootmap autoloads libRooPyFitFunction.dylib into
# Combine's process, the Python C API symbols (_Py_Initialize, etc.) are resolved
# via -undefined dynamic_lookup, which only works if libpython is already in the
# process.  Use DYLD_INSERT_LIBRARIES (macOS) / LD_PRELOAD (Linux) to preload it:
#   DYLD_INSERT_LIBRARIES=<LIBPYTHON_SO> combine ...
_py_lib_dir = _sysconfig.get_config_var("LIBDIR") or ""
_py_ver = f"{_sys.version_info.major}.{_sys.version_info.minor}"
LIBPYTHON_SO = os.path.join(_py_lib_dir, f"libpython{_py_ver}.{_SO_EXT}")
if not os.path.isfile(LIBPYTHON_SO):
    LIBPYTHON_SO = ""  # not available -- Combine test will skip preload


def _ensure_lib():
    """Compile the plugin library if not already built.  Returns the .so path."""
    from FunLib.tools.compile_plugin import ensure_compiled

    return ensure_compiled()


def _load_lib():
    """Load the RooPyFitFunction shared library into ROOT (idempotent)."""
    import ROOT

    if not os.path.isfile(_LIB_SO):
        _ensure_lib()
    ret = ROOT.gSystem.Load(_LIB_SO)
    if ret < 0:
        raise RuntimeError(
            f"Failed to load RooPyFitFunction library: {_LIB_SO}\n"
            f"Run:  python3 FunLib/tools/compile_plugin.py"
        )


def mk_workspace(
    entry: dict,
    hist,
    output_file: str,
    *,
    excludes=None,
    ws_name: str = "w",
    repo_root: str | None = None,
    n_sig_expected: float = 50.0,
    sig_mean: float = 125.0,
    sig_width: float = 2.0,
):
    """
    Build a CMS Combine-compatible RooFit workspace from a function entry
    and a TH1 histogram, and write it to a ROOT file.

    The background shape is a **RooPyFitFunction** with floating shape
    parameters.  Combine (or any RooFit minimiser) can float them during the
    fit; the Python kernel is re-loaded from the stored JSON metadata on the
    first ``evaluate()`` call.

    A parametrised Breit-Wigner signal peak is included so the workspace can
    be used directly with::

        DYLD_LIBRARY_PATH=FunLib/tools/lib:$DYLD_LIBRARY_PATH \\
            combine -M Significance workspace.root -m 125

    Workspace contents
    ------------------
    ``x``              -- observable RooRealVar
    ``p_{name}``       -- shape-param RooRealVars (floating, initial values from entry)
    ``background``     -- RooPyFitFunction (shape params connected and floating)
    ``n_bkg``          -- background yield RooRealVar (floating)
    ``mean_sig``       -- signal mass (fixed)
    ``width_sig``      -- signal width (fixed)
    ``signal``         -- RooBreitWigner PDF
    ``r``              -- signal strength POI [0, 10]
    ``n_sig_expected`` -- expected signal yield at r=1 (constant)
    ``n_sig``          -- RooFormulaVar: r * n_sig_expected
    ``model_s``        -- extended RooAddPdf(signal, background, [n_sig, n_bkg])
    ``data_obs``       -- RooDataHist (observed data)
    ``ModelConfig``    -- RooStats::ModelConfig (POI=r, nuisances={n_bkg, p_*})
    ``fn_meta``        -- TObjString (JSON metadata for load_workspace)

    Parameters
    ----------
    entry : dict
        One item from all_functions.yaml: ``{dir, initial_params, ...}``.
    hist : ROOT.TH1
        Source histogram.
    output_file : str
        Output .root file path.
    excludes : list of (lo, hi), optional
        Blinded sub-windows (stored in fn_meta).  Default: [(120, 130)].
    ws_name : str
        RooWorkspace name.  Default: ``"w"``.
    repo_root : str, optional
        Repo root for resolving ``entry["dir"]``.  Auto-detected if None.
    n_sig_expected : float
        Expected signal yield at r=1.  Default: 50.
    sig_mean : float
        Signal BW mean (GeV).  Default: 125.0.
    sig_width : float
        Signal BW full width (GeV).  Default: 2.0.

    Returns
    -------
    ROOT.RooWorkspace
        The workspace object (also persisted to *output_file*).
    """
    import ROOT
    from FunLib.tools.factory import _find_repo_root, load_kernel

    _load_lib()

    if excludes is None:
        excludes = [(120.0, 130.0)]

    # -- fit range / binning from histogram axis
    ax = hist.GetXaxis()
    xmin = ax.GetXmin()
    xmax = ax.GetXmax()
    nbin = ax.GetNbins()
    bin_width = (xmax - xmin) / nbin
    integral = max(hist.Integral(), 1.0)

    # -- load kernel (for metadata: key, param_names, npar, label)
    root_dir = repo_root or _find_repo_root()
    kernel = load_kernel(entry, repo_root=root_dir, xmin=xmin, xmax=xmax)

    # -- build workspace
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
    ws = ROOT.RooWorkspace(ws_name, ws_name)
    wi = getattr(ws, "import")

    # ---- Observable ----------------------------------------
    x_obs = ROOT.RooRealVar("x", "m_{#mu#mu} (GeV)", xmin, xmax)
    x_obs.setBins(nbin)
    wi(x_obs, ROOT.RooCmdArg.none())
    xw = ws.var("x")

    # ---- Shape-parameter RooRealVars (floating) ------------
    # Wide symmetric bounds centred at initial value (Combine will float them).
    # desc.yaml roofit_param_bounds override the defaults if provided.
    roo_bounds = kernel.yaml_config.get("roofit_param_bounds") or {}
    param_rvars = []
    for pname, pval in zip(kernel.param_names, kernel.initial_params):
        v0 = float(pval)
        bounds = roo_bounds.get(pname)
        if bounds and len(bounds) == 2:
            lo, hi = float(bounds[0]), float(bounds[1])
            v0 = max(lo, min(hi, v0))
            rv = ROOT.RooRealVar(f"p_{pname}", pname, v0, lo, hi)
        else:
            # generous symmetric range; minimiser can explore freely
            half = max(abs(v0) * 20.0, 10.0)
            rv = ROOT.RooRealVar(f"p_{pname}", pname, v0, v0 - half, v0 + half)
        rv.setError(max(abs(v0) * 0.1, 0.1))
        wi(rv, ROOT.RooCmdArg.none())
        param_rvars.append(rv)

    # ---- Background: RooPyFitFunction ----------------------
    # Build fn_meta JSON from workspace metadata (same format as fn_meta
    # TObjString, so load_workspace can reconstruct the kernel).
    meta = {
        "entry": _serialise_entry(entry),
        "excludes": [[float(lo), float(hi)] for lo, hi in excludes],
        "xmin": float(xmin),
        "xmax": float(xmax),
        "bin_width": float(bin_width),
        "integral": float(integral),
        "key": kernel.key,
        "label": kernel.label,
        "param_names": list(kernel.param_names),
        "npar": int(kernel.npar),
        "sig_mean": float(sig_mean),
        "sig_width": float(sig_width),
        "n_sig_expected": float(n_sig_expected),
    }
    meta_json = json.dumps(meta)

    # Build RooArgList of workspace copies of the params in kernel order
    shape_arg_list = ROOT.RooArgList()
    for rv in param_rvars:
        shape_arg_list.add(ws.var(rv.GetName()))

    bkg_pdf = ROOT.RooPyFitFunction(
        "background",
        "background PDF",
        xw,
        shape_arg_list,
        meta_json,
        root_dir,
    )
    wi(bkg_pdf, ROOT.RooCmdArg.none())

    n_bkg = ROOT.RooRealVar(
        "n_bkg", "background yield", float(integral), 0.0, float(integral) * 3.0
    )
    wi(n_bkg, ROOT.RooCmdArg.none())

    # ---- Signal: parametrised Breit-Wigner -----------------
    mean_s = ROOT.RooRealVar(
        "mean_sig",
        "signal mass",
        float(sig_mean),
        float(sig_mean) - 5.0,
        float(sig_mean) + 5.0,
    )
    mean_s.setConstant(True)
    wid_s = ROOT.RooRealVar(
        "width_sig",
        "signal width",
        float(sig_width),
        0.1,
        10.0,
    )
    wid_s.setConstant(True)
    wi(mean_s, ROOT.RooCmdArg.none())
    wi(wid_s, ROOT.RooCmdArg.none())

    sig_pdf = ROOT.RooBreitWigner(
        "signal",
        "signal BW",
        xw,
        ws.var("mean_sig"),
        ws.var("width_sig"),
    )
    wi(sig_pdf, ROOT.RooCmdArg.none())

    # ---- Signal strength (POI) -----------------------------
    r = ROOT.RooRealVar("r", "signal strength", 1.0, 0.0, 10.0)
    wi(r, ROOT.RooCmdArg.none())

    n_sig_exp_rv = ROOT.RooRealVar(
        "n_sig_expected",
        "expected signal yield",
        float(n_sig_expected),
    )
    n_sig_exp_rv.setConstant(True)
    wi(n_sig_exp_rv, ROOT.RooCmdArg.none())

    n_sig_fmla = ROOT.RooFormulaVar(
        "n_sig",
        "@0*@1",
        ROOT.RooArgList(ws.var("r"), ws.var("n_sig_expected")),
    )
    wi(n_sig_fmla, ROOT.RooCmdArg.none())

    # ---- Signal + background model -------------------------
    model_s = ROOT.RooAddPdf(
        "model_s",
        "signal+background",
        ROOT.RooArgList(ws.pdf("signal"), ws.pdf("background")),
        ROOT.RooArgList(ws.function("n_sig"), ws.var("n_bkg")),
    )
    wi(model_s, ROOT.RooCmdArg.none())

    # ---- Observed data -------------------------------------
    data_obs = ROOT.RooDataHist(
        "data_obs",
        "observed data",
        ROOT.RooArgList(xw),
        hist,
    )
    wi(data_obs, ROOT.RooCmdArg.none())

    # ---- ModelConfig (RooStats) ----------------------------
    # Shape params are nuisances (they float alongside n_bkg in the S+B fit).
    mc = ROOT.RooStats.ModelConfig("ModelConfig", ws)
    mc.SetPdf(ws.pdf("model_s"))
    mc.SetObservables(ROOT.RooArgSet(ws.var("x")))
    mc.SetParametersOfInterest(ROOT.RooArgSet(ws.var("r")))
    nuisances = ROOT.RooArgSet(ws.var("n_bkg"))
    for rv in param_rvars:
        pv = ws.var(rv.GetName())
        if pv:
            nuisances.add(pv)
    mc.SetNuisanceParameters(nuisances)
    wi(mc)  # TObject: no second argument

    # ---- Write to ROOT file --------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    ws.writeToFile(output_file)

    tf = ROOT.TFile.Open(output_file, "UPDATE")
    ROOT.TObjString(meta_json).Write("fn_meta")
    tf.Close()

    return ws


# -- internal helpers ----------------------------------------------------------


def _serialise_entry(entry: dict) -> dict:
    """Return a JSON-safe copy of an all_functions.yaml entry dict."""
    out = {}
    for k, v in entry.items():
        if k == "initial_params":
            out[k] = [float(x) for x in v]
        elif isinstance(v, dict):
            out[k] = {
                ck: float(cv) if isinstance(cv, (int, float)) else cv
                for ck, cv in v.items()
            }
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out
