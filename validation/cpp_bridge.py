#!/usr/bin/env python3
"""
cpp_bridge.py -- thin PyROOT bridge to the compiled FunLib C++ kernels.

Exposes three evaluators used by validate_functions.py to check that the
Python kernel, the standalone C++ kernel and the C++ RooAbsPdf (RooFunLibPdf)
all return the same shape value:

    cpp_eval(dir, constants, xmin, xmax, npar, x, params, funcdir)  -> float
    roo_eval(dir, constants, xmin, xmax, npar, xs, params, funcdir) -> [float]

The library is built on first use via FunLib.tools.build_funlib.ensure_compiled.
ROOT is imported lazily so that importing this module never initialises ROOT in
a parent process (keeps the validator fork-safe).
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_INCLUDE = os.path.normpath(os.path.join(_HERE, "..", "include"))

_state = {"ready": False, "ROOT": None}


def _split_constants(constants: dict):
    """Split a desc.yaml constants dict into (num_names, num_vals, str_names,
    str_vals).  bools -> 0.0/1.0; ints/floats -> float; str -> string."""
    cn, cv, sn, sv = [], [], [], []
    for k, val in constants.items():
        if isinstance(val, bool):
            cn.append(k)
            cv.append(1.0 if val else 0.0)
        elif isinstance(val, (int, float)):
            cn.append(k)
            cv.append(float(val))
        elif isinstance(val, str):
            sn.append(k)
            sv.append(val)
        # dict/list constants (e.g. minimization) are irrelevant to kernels
    return cn, cv, sn, sv


def _vec_str(ROOT, items):
    v = ROOT.std.vector("std::string")()
    for s in items:
        v.push_back(str(s))
    return v


def _vec_dbl(ROOT, items):
    v = ROOT.std.vector("double")()
    for s in items:
        v.push_back(float(s))
    return v


def _ensure():
    if _state["ready"]:
        return _state["ROOT"]
    from FunLib.tools.build_funlib import ensure_compiled

    so = ensure_compiled()
    import ROOT

    ROOT.gInterpreter.AddIncludePath(_INCLUDE)
    if ROOT.gSystem.Load(so) < 0:
        raise RuntimeError(f"Failed to load FunLib library: {so}")
    ROOT.gInterpreter.Declare('#include "funlib_registry.h"')
    _state["ROOT"] = ROOT
    _state["ready"] = True
    return ROOT


def cpp_eval(dir_name, constants, xmin, xmax, npar, x, params, funcdir=None):
    """Evaluate the standalone C++ kernel at a single x."""
    ROOT = _ensure()
    consts = dict(constants)
    if funcdir is not None:
        consts["_funcdir"] = funcdir
    cn, cv, sn, sv = _split_constants(consts)
    return float(
        ROOT.funlib.eval_kernel(
            str(dir_name),
            _vec_str(ROOT, cn),
            _vec_dbl(ROOT, cv),
            _vec_str(ROOT, sn),
            _vec_str(ROOT, sv),
            float(xmin),
            float(xmax),
            int(npar),
            float(x),
            _vec_dbl(ROOT, params),
        )
    )


def build_funlib_pdf(
    ROOT,
    x_obs,
    dir_name,
    constants,
    xmin,
    xmax,
    npar,
    params,
    funcdir=None,
    param_names=None,
    tag="",
    param_bounds=None,
):
    """Construct a RooFunLibPdf and its shape RooRealVars.

    Returns (pdf, shape_rvs).  Caller must keep both alive.  param_names (if
    given) names the RooRealVars; otherwise p0..p{n-1} are used.

    param_bounds: optional {name: [lo, hi]} from desc.yaml roofit_param_bounds.
    A named entry overrides the default heuristic range so a Combine fit cannot
    drive the parameter into an unphysical / divergent region.
    """
    consts = dict(constants)
    if funcdir is not None:
        consts["_funcdir"] = funcdir
    cn, cv, sn, sv = _split_constants(consts)
    names = param_names or [f"p{i}" for i in range(len(params))]
    bounds = param_bounds or {}
    arglist = ROOT.RooArgList()
    shape_rvs = []
    for nm, v in zip(names, params):
        v = float(v)
        if nm in bounds and len(bounds[nm]) == 2:
            # desc.yaml roofit_param_bounds: a physical range that keeps a Combine
            # fit from driving the parameter into a divergent / unphysical region.
            lo, hi = float(bounds[nm][0]), float(bounds[nm][1])
            v = min(max(v, lo), hi)
            rv = ROOT.RooRealVar(f"{nm}{tag}", nm, v, lo, hi)
        else:
            # Unbounded (no range): mirrors the roofit fit pipeline, which does
            # NOT impose Minuit limits except via roofit_param_bounds (limits
            # change the sin-transform step scales and hurt convergence).
            rv = ROOT.RooRealVar(f"{nm}{tag}", nm, v)
        shape_rvs.append(rv)
        arglist.add(rv)
    pdf = ROOT.RooFunLibPdf(
        f"funlib_pdf{tag}",
        f"funlib_pdf{tag}",
        x_obs,
        arglist,
        str(dir_name),
        _vec_str(ROOT, cn),
        _vec_dbl(ROOT, cv),
        _vec_str(ROOT, sn),
        _vec_str(ROOT, sv),
        int(npar),
        float(xmin),
        float(xmax),
    )
    pdf._keepalive = (arglist, shape_rvs)
    return pdf, shape_rvs


def roo_eval(dir_name, constants, xmin, xmax, npar, xs, params, funcdir=None):
    """Evaluate the RooFunLibPdf RooAbsPdf (raw, unnormalised) at each x in xs."""
    ROOT = _ensure()
    lo = min(float(xmin), float(min(xs)))
    hi = max(float(xmax), float(max(xs)))
    x_obs = ROOT.RooRealVar("x_funlib", "x", 0.5 * (lo + hi), lo, hi)
    pdf, shape_rvs = build_funlib_pdf(
        ROOT,
        x_obs,
        dir_name,
        constants,
        xmin,
        xmax,
        npar,
        params,
        funcdir=funcdir,
        tag="_re",
    )
    out = []
    for x in xs:
        x_obs.setVal(float(x))
        out.append(float(pdf.getVal()))
    del pdf, shape_rvs, x_obs
    return out
