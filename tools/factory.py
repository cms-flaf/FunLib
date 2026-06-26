#!/usr/bin/env python3
"""
FunLib/tools/factory.py -- Canonical factory for fit-function handles.

Design
------
fn.py files define **pure shape kernels** with no N (amplitude) parameter.
``FitFunction.__init__`` takes only ``config: dict``; ``__call__`` evaluates
the un-normalized shape and returns it directly.

The factory loads a kernel and wraps it in a :class:`FunctionHandle` that adds:

* An optional explicit normalization parameter N (``add_norm=True``, the default).
* Optional PDF normalisation so that ``sum_bins handle(x_i, p) ~= N``
  (``normalize=True``, the chi^2_norm / PDF mode).
* RooFit extended-likelihood support via ``make_roo_ext_pdf()``.

Usage
-----
    from FunLib.tools.factory import make_function, load_kernel

    # From a functions.yaml entry (has 'dir' + 'initial_params'):
    entry = {"dir": "FunLib/functions/bwz_exp", "initial_params": [0.0, 0.0]}
    handle = make_function(entry,
        add_norm=True, normalize=False,
        xmin=xmin, xmax=xmax, excludes=[(120, 130)], bin_width=0.25,
        integral=341_000)
    y   = handle([130.0], handle.initial_params)
    tf1 = handle.getFn(xmin, xmax)

    # chi^2_norm (N = event count, sum_bins handle(x_i) ~= N):
    pdf_h = make_function(entry, add_norm=True, normalize=True,
                          xmin=xmin, xmax=xmax, excludes=[(120, 130)],
                          bin_width=0.25, integral=341_000)

    # Pure shape (no N):
    shape_h = make_function(entry, add_norm=False, ...)

    # RooFit EML (always available):
    ext_pdf, N_roo, shape_rvs = handle.make_roo_ext_pdf(x_obs,
                                                         N_eml=341_000, tag="_1")

FunctionHandle attributes
-------------------------
key, label, npar, param_names, initial_params (r/w), source_file, yaml_config,
_fn (alias to raw kernel for backward compat).

Methods: __call__(xv, p), getFn(xmin, xmax), make_roo_ext_pdf(x_obs, N_eml, tag).
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import yaml

__all__ = ["FunctionHandle", "make_function", "load_kernel"]

# -- RooFit C++ RooPyCallPdf (declared once per process) ----------------------
# The #ifndef guard prevents re-declaration when fit_background.py is also imported.

_ROO_PDF_DECLARED = False

_ROO_PDF_DECL = r"""
#ifndef ROO_PY_CALL_PDF_H
#define ROO_PY_CALL_PDF_H
#include "Python.h"
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "RooRealVar.h"
#include "RooAbsReal.h"
#include <cmath>

class RooPyCallPdf : public RooAbsPdf {
public:
    RooPyCallPdf() = default;

    RooPyCallPdf(const char* name, RooRealVar& x,
                 const RooArgList& pars, PyObject* py_fn)
        : RooAbsPdf(name, name)
        , _x("_x","_x",this,x)
        , _pars("_pars","_pars",this)
        , _py_fn(py_fn)
    { _pars.add(pars); Py_XINCREF(_py_fn); }

    RooPyCallPdf(const RooPyCallPdf& o, const char* n = nullptr)
        : RooAbsPdf(o, n)
        , _x("_x",this,o._x)
        , _pars("_pars",this,o._pars)
        , _py_fn(o._py_fn)
    { Py_XINCREF(_py_fn); }

    ~RooPyCallPdf() { Py_XDECREF(_py_fn); }

    TObject* clone(const char* n) const override {
        return new RooPyCallPdf(*this, n);
    }

    Double_t evaluate() const override {
        if (!_py_fn) return 0.0;
        int n = _pars.size();
        PyObject* args = PyTuple_New(n + 1);
        PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(double(_x)));
        for (int i = 0; i < n; i++) {
            const RooAbsReal* p = static_cast<const RooAbsReal*>(_pars.at(i));
            PyTuple_SET_ITEM(args, i+1, PyFloat_FromDouble(p->getVal()));
        }
        PyObject* res = PyObject_CallObject(_py_fn, args);
        Py_DECREF(args);
        if (!res) { PyErr_Clear(); return 0.0; }
        double val = PyFloat_AsDouble(res);
        Py_DECREF(res);
        return (val > 0.0 && std::isfinite(val)) ? val : 0.0;
    }

    Int_t getAnalyticalIntegral(RooArgSet&,RooArgSet&,const char*) const override { return 0; }
    Double_t analyticalIntegral(Int_t,const char*) const override { return 1.0; }

private:
    RooRealProxy _x;
    RooListProxy _pars;
    PyObject*    _py_fn{nullptr};
    ClassDefOverride(RooPyCallPdf, 0)
};
#endif
"""

_roo_handle_counter = [0]


def _ensure_roo_pdf_declared():
    global _ROO_PDF_DECLARED
    if not _ROO_PDF_DECLARED:
        import ROOT

        ROOT.gInterpreter.Declare(_ROO_PDF_DECL)
        _ROO_PDF_DECLARED = True


# -- Repo-root detection -------------------------------------------------------


def _find_repo_root(start: str | None = None) -> str:
    """Locate the repo root by searching for the FunLib/ directory."""
    d = os.path.abspath(start or os.getcwd())
    for _ in range(12):
        if os.path.isdir(os.path.join(d, "FunLib")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        f"Cannot locate repo root (no FunLib/ directory found from {start or os.getcwd()!r})"
    )


def _load_fn_module(fn_path: str):
    spec = importlib.util.spec_from_file_location("_fn_kernel", fn_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kernel_needs_config(fn_class) -> bool:
    """Return True if FitFunction.__init__ declares a 'config' keyword-only arg."""
    try:
        sig = inspect.signature(fn_class.__init__)
        return "config" in sig.parameters
    except (ValueError, TypeError):
        return False


def _kernel_needs_range(fn_class) -> bool:
    """Return True if FitFunction.__init__ declares x_min / x_max keyword-only args."""
    try:
        sig = inspect.signature(fn_class.__init__)
        return "x_min" in sig.parameters
    except (ValueError, TypeError):
        return False


def _kernel_needs_npar(fn_class) -> bool:
    """Return True if FitFunction.__init__ declares a 'npar' keyword-only arg.

    Functions with flexible degree (e.g. Bernstein) use this to receive their
    degree at construction time instead of storing it in desc.yaml.
    """
    try:
        sig = inspect.signature(fn_class.__init__)
        return "npar" in sig.parameters
    except (ValueError, TypeError):
        return False


def _kernel_kwargs(fn_class, config: dict, xmin=None, xmax=None, npar=None) -> dict:
    """
    Build the keyword-argument dict for instantiating a FitFunction kernel.

    Handles all valid __init__ signatures; the ``npar`` positional arg is
    only included when the kernel declares it explicitly:
      (self)                                      -> {}
      (self, *, config)                           -> {config: ...}
      (self, *, x_min, x_max)                     -> {x_min: ..., x_max: ...}
      (self, *, config, x_min, x_max)             -> {config: ..., x_min: ..., x_max: ...}
      (self, *, config, x_min, x_max, npar)       -> + npar: ...
    """
    kw = {}
    if _kernel_needs_config(fn_class):
        kw["config"] = config
    if _kernel_needs_range(fn_class):
        if xmin is None or xmax is None:
            raise ValueError(
                "x_min/x_max required for this kernel but were not provided"
            )
        kw["x_min"] = float(xmin)
        kw["x_max"] = float(xmax)
    if _kernel_needs_npar(fn_class):
        if npar is None:
            raise ValueError("npar required for this kernel but was not provided")
        kw["npar"] = int(npar)
    return kw


def _inject_metadata(kernel, config: dict, initial_params: list, fn_path: str):
    """
    Inject metadata attributes onto the kernel after construction.

    For most kernels, npar/param_names come from desc.yaml (via config).
    For variable-npar kernels (those whose __init__ takes a ``npar`` kwarg),
    __init__ has already set self.npar and self.param_names; we respect those
    and only apply the desc.yaml values as fallbacks.

    key/label in desc.yaml may be format strings containing ``{npar}``
    (e.g. ``Bernstein-{npar}``); they are formatted with the resolved npar.
    """
    # Prefer npar/param_names that __init__ set (variable-npar kernels).
    # For fixed-npar kernels, derive from len(initial_params) == len(param_names).
    npar = getattr(kernel, "npar", None)
    if npar is None:
        npar = len(initial_params)

    param_names = getattr(kernel, "param_names", None) or config.get("param_names", [])

    # Format key/label templates such as "Bernstein-{npar}" or "BWZxBern-{npar-1}".
    # Supports {npar}, {npar-1}, {npar-2}, ... and kernel degree attrs {num_deg}/{den_deg}.
    def _fmt(tmpl: str) -> str:
        if not isinstance(tmpl, str) or "{" not in tmpl:
            return tmpl
        result = tmpl.replace("{npar}", str(npar))
        for i in range(1, npar + 3):
            result = result.replace(f"{{npar-{i}}}", str(npar - i))
        for attr in ("num_deg", "den_deg", "n_terms"):
            val = getattr(kernel, f"_{attr}", None)
            if val is not None:
                result = result.replace(f"{{{attr}}}", str(val))
        return result

    kernel.key = _fmt(config.get("key", ""))
    kernel.label = _fmt(config.get("label", ""))

    kernel.npar = npar
    kernel.param_names = param_names
    kernel.initial_params = list(initial_params)
    kernel.source_file = fn_path

    # Store a copy of config with the resolved (formatted) key/npar/param_names
    # so that yaml_config is self-consistent.
    resolved = dict(config)
    resolved["key"] = kernel.key
    resolved["label"] = kernel.label
    resolved["npar"] = npar
    resolved["param_names"] = param_names
    kernel.yaml_config = resolved


# -- load_kernel ---------------------------------------------------------------


def load_kernel(
    yaml_entry: dict,
    *,
    repo_root: str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
):
    """
    Load and instantiate a raw (N-free) FitFunction kernel.

    Parameters
    ----------
    yaml_entry
        One item from ``functions.yaml`` with ``dir`` + ``initial_params``
        (shape params only; no N).
    repo_root
        Repo root for resolving ``dir``.  Auto-detected if None.
    xmin, xmax
        Fit-window edges (GeV).  Required for kernels whose ``__init__``
        declares ``x_min``/``x_max`` keyword-only args; raises ValueError
        otherwise.  Pass ``None`` (or omit) only for kernels that do not
        use interval constants.

    Returns
    -------
    FitFunction  (pure shape kernel, no N in param_names)
    """
    root = repo_root or _find_repo_root()
    fn_dir = os.path.normpath(os.path.join(root, yaml_entry["dir"]))
    fn_path = os.path.join(fn_dir, "fn.py")
    desc_path = os.path.join(fn_dir, "desc.yaml")

    if not os.path.isfile(fn_path):
        raise FileNotFoundError(f"fn.py not found: {fn_path}")
    if not os.path.isfile(desc_path):
        raise FileNotFoundError(f"desc.yaml not found: {desc_path}")

    with open(desc_path) as fh:
        desc = yaml.safe_load(fh)

    config = dict(desc)
    config["initial_params"] = list(yaml_entry["initial_params"])
    config["source_file"] = fn_path

    # Per-entry overrides from all_functions.yaml: constants (deep-merged, entry
    # wins), plus key/label/param_names/npar for variant selection without a
    # separate directory.  This lets bwz_gamma + constants:{variant:running}
    # produce BWZGamma-rwForm without a bwz_gamma_rwform/ directory.
    entry_consts = yaml_entry.get("constants", {})
    if entry_consts:
        merged = dict(config.get("constants", {}))
        merged.update(entry_consts)
        config["constants"] = merged
    for _field in (
        "key",
        "label",
        "param_names",
        "npar",
        "latex",
        "description",
        "param_bounds",
        "roofit_param_bounds",
    ):
        if _field in yaml_entry:
            config[_field] = yaml_entry[_field]

    mod = _load_fn_module(fn_path)
    if not hasattr(mod, "FitFunction"):
        raise RuntimeError(f"{fn_path} does not define FitFunction")

    # For variable-npar kernels (those whose __init__ declares a ``npar`` kwarg),
    # derive npar from the length of initial_params.  This allows multiple
    # functions.yaml entries to share one dir with different degrees, e.g.
    # Bernstein-1/2/3/4 all pointing to FunLib/functions/bernstein/.
    npar_for_kernel = (
        len(yaml_entry["initial_params"])
        if _kernel_needs_npar(mod.FitFunction)
        else None
    )

    # Instantiate using only the keyword args the kernel's __init__ actually declares.
    try:
        kw = _kernel_kwargs(mod.FitFunction, config, xmin, xmax, npar=npar_for_kernel)
    except ValueError:
        raise ValueError(
            f"Function '{fn_dir}' requires x_min/x_max and/or npar but none were provided."
        )
    kernel = mod.FitFunction(**kw)

    _inject_metadata(kernel, config, yaml_entry["initial_params"], fn_path)
    return kernel


# -- N initial-value helper ----------------------------------------------------


def _compute_N_init(
    kernel,
    initial_shape_params: list,
    integral: float | None,
    xmin: float,
    xmax: float,
    excludes: list,
    bin_width: float,
    normalize: bool,
) -> float:
    """
    Estimate a good initial N value.

    For normalize=True (chi^2_norm): N ~= unblinded event count = integral x unblinded_fraction.
    For normalize=False (chi^2):     N ~= bin_content_at_midpoint / shape(midpoint).

    Falls back to ``integral`` (or 1e6) if shape at midpoint is zero/unavailable.
    """
    ref = float(integral) if integral and integral > 0 else 1e6

    # Unblinded fraction of the fit range
    blinded = sum(float(hi) - float(lo) for lo, hi in excludes)
    total = float(xmax) - float(xmin)
    unblind_frac = max(1.0 - blinded / total, 0.01) if total > 0 else 1.0
    unblind_events = ref * unblind_frac

    if normalize:
        # N = event count in the unblinded sideband
        return unblind_events

    # chi^2 mode: N x shape(x_mid) ~= average bin content at midpoint
    x_mid = 0.5 * (float(xmin) + float(xmax))
    n_unblind_bins = (total - blinded) / float(bin_width) if bin_width > 0 else 1.0
    avg_bin_content = unblind_events / max(n_unblind_bins, 1.0)

    try:
        kwargs = dict(zip(kernel.param_names, initial_shape_params))
        shape_mid = float(kernel(x_mid, **kwargs))
    except Exception:
        shape_mid = 0.0

    if shape_mid > 1e-30:
        return avg_bin_content / shape_mid
    return unblind_events


# -- make_function -------------------------------------------------------------


def make_function(
    entry: dict,
    *,
    add_norm: bool = True,
    normalize: bool = False,
    xmin: float,
    xmax: float,
    excludes: list | None = None,
    bin_width: float = 0.25,
    integral: float | None = None,
    repo_root: str | None = None,
    nan_screening: bool = True,
    inf_screening: bool = True,
) -> "FunctionHandle":
    """
    Build a :class:`FunctionHandle` from a functions.yaml entry or merged config.

    Parameters
    ----------
    entry
        Raw functions.yaml entry (has ``dir`` + ``initial_params`` shape params)
        **or** a pre-merged config dict (has ``source_file`` + full metadata).
    add_norm
        If True (default), prepend N to the parameter vector.
        ``__call__(xv, p)`` returns ``N x shape(p[1:])``.
    normalize
        If True, divide by the integral of the shape over the unblinded window
        so that ``sum_unblinded_bins handle(x_i, p) ~= N`` (chi^2_norm / PDF mode).
        Only meaningful when ``add_norm=True``.
    xmin, xmax
        Fit-window edges (GeV).
    excludes
        Blinded sub-windows as ``[(lo, hi), ...]``.
    bin_width
        Histogram bin width (GeV); used for normalisation integrals.
    integral
        Total histogram integral (event count).  Used to set N initial value.
    repo_root
        Repo root path.  Auto-detected if None.
    nan_screening
        If True (default), NaN returned by the kernel is replaced with 0.0.
    inf_screening
        If True (default), Inf returned by the kernel is replaced with 0.0.
    """
    if "dir" in entry:
        kernel = load_kernel(entry, repo_root=repo_root, xmin=xmin, xmax=xmax)
    elif "source_file" in entry:
        mod = _load_fn_module(entry["source_file"])
        if not hasattr(mod, "FitFunction"):
            raise RuntimeError(f"{entry['source_file']} does not define FitFunction")
        npar_sf = (
            len(entry.get("initial_params", []))
            if _kernel_needs_npar(mod.FitFunction)
            else None
        )
        try:
            kw = _kernel_kwargs(mod.FitFunction, entry, xmin, xmax, npar=npar_sf)
        except ValueError:
            raise ValueError(
                f"Function '{entry['source_file']}' requires x_min/x_max and/or npar but none were provided."
            )
        kernel = mod.FitFunction(**kw)
        _inject_metadata(
            kernel, entry, entry.get("initial_params", []), entry.get("source_file", "")
        )
    else:
        raise ValueError("entry must have 'dir' or 'source_file'")

    return FunctionHandle(
        kernel,
        add_norm=add_norm,
        normalize=normalize,
        xmin=xmin,
        xmax=xmax,
        excludes=excludes or [],
        bin_width=bin_width,
        integral=integral,
        nan_screening=nan_screening,
        inf_screening=inf_screening,
    )


# -- FunctionHandle ------------------------------------------------------------


class FunctionHandle:
    """
    Unified, mode-aware wrapper around a N-free fn.py kernel.

    Returned by :func:`make_function`.

    The kernel is always N-free (``param_names`` has no ``'N'``).
    This wrapper optionally prepends N, enforces PDF normalisation,
    and provides RooFit integration.

    Parameter layout (depends on *add_norm*)
    -----------------------------------------
    ``add_norm=True``  -> ``param_names = ['N', shape_p1, ...]``,
                         ``npar`` includes N,
                         ``__call__`` returns ``N x shape(p[1:])``.
    ``add_norm=False`` -> ``param_names = [shape_p1, ...]``,
                         ``npar`` excludes N,
                         ``__call__`` returns pure ``shape(p[:])``.

    RooFit
    ------
    ``make_roo_ext_pdf(x_obs, N_eml, tag)`` always available.
    """

    _QUAD_OPTS = dict(limit=200, epsabs=1e-5, epsrel=1e-5)

    def __init__(
        self,
        kernel,
        *,
        add_norm: bool,
        normalize: bool,
        xmin: float,
        xmax: float,
        excludes: list,
        bin_width: float,
        integral: float | None = None,
        nan_screening: bool = True,
        inf_screening: bool = True,
    ):
        self._kernel = kernel
        self._add_norm = add_norm
        self._normalize = normalize
        self._xmin = float(xmin)
        self._xmax = float(xmax)
        self._excludes = [(float(lo), float(hi)) for lo, hi in excludes]
        self._bin_width = max(float(bin_width), 1e-12)
        self._nan_screening = nan_screening
        self._inf_screening = inf_screening

        # The kernel is always N-free.
        kp = list(kernel.param_names)
        ki = list(kernel.initial_params)

        if add_norm:
            # Compute a sensible N initial value from the integral.
            N_init = _compute_N_init(
                kernel,
                ki,
                integral,
                xmin,
                xmax,
                self._excludes,
                self._bin_width,
                normalize,
            )
            if normalize:
                # In normalize mode N represents the event count;
                # the raw kernel N would be in "per-GeV" units.
                # N_init is already the event count from _compute_N_init.
                pass
            self._param_names = ["N"] + kp
            self._npar = kernel.npar + 1
            self._initial_params = [N_init] + ki
        else:
            self._param_names = kp
            self._npar = kernel.npar
            self._initial_params = ki

        # Normalisation integral cache
        self._cache_s = None
        self._cache_norm = 1.0
        self._cache_sref = 1.0
        if normalize:
            self._intervals = _build_unblinded(xmin, xmax, self._excludes)
            # Grid spanning the unblinded intervals, used to estimate a
            # scale reference for scale-invariant normalisation.
            grid = []
            for a, b in self._intervals:
                grid += [a + (b - a) * k / 6.0 for k in range(7)]
            self._sref_grid = grid

    # -- Properties ------------------------------------------------------------

    @property
    def key(self):
        return self._kernel.key

    @property
    def label(self):
        return self._kernel.label

    @property
    def npar(self):
        return self._npar

    @property
    def param_names(self):
        return self._param_names

    @property
    def source_file(self):
        return self._kernel.source_file

    @property
    def yaml_config(self):
        return self._kernel.yaml_config

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, value):
        self._initial_params = list(value)

    # Backward-compat: code that does fn._fn (e.g. run_all_categories checks
    # isinstance(fn, NormalizedFitFunction) then uses fn._fn) gets the kernel.
    @property
    def _fn(self):
        return self._kernel

    # -- Internal helpers ------------------------------------------------------

    def _eval_kernel(self, x: float, shape_params) -> float:
        """Raw kernel call - may return nan or inf; never raises."""
        try:
            kwargs = dict(zip(self._kernel.param_names, shape_params))
            return float(self._kernel(float(x), **kwargs))
        except Exception:
            return math.nan

    def _eval_shape(self, x: float, shape_params) -> float:
        """Always-screened evaluation used for normalisation integrals."""
        v = self._eval_kernel(x, shape_params)
        return v if math.isfinite(v) and v >= 0.0 else 0.0

    def _compute_norm(self, shape_params) -> float:
        """Cached integral_unblinded shape(x; shape_params) dx / bin_width,
        expressed in units of the shape's own scale ``s_ref``.

        The returned ``norm`` is ``(integral / bin_width) / s_ref`` and is
        scale-invariant: it stays O(1) however large or tiny the raw kernel
        is, so the protective floor never spuriously clamps a tiny-shape
        function's true integral (the old ``max(norm, 1e-30)`` on the
        un-scaled integral mis-normalised e.g. Dijet, whose raw shape ~1e-32).
        Callers must divide the kernel value by ``self._cache_sref`` to match.
        """
        from scipy.integrate import quad

        s = tuple(float(v) for v in shape_params)
        if s == self._cache_s:
            return self._cache_norm

        # Scale reference: max screened shape over a grid spanning the
        # unblinded intervals.  Makes the integral O(1) for any kernel scale.
        s_ref = max((self._eval_shape(x, s) for x in self._sref_grid), default=0.0)
        if not (math.isfinite(s_ref) and s_ref > 0.0):
            s_ref = 1.0

        def f(x_val):
            return self._eval_shape(x_val, s) / s_ref

        total = sum(quad(f, a, b, **self._QUAD_OPTS)[0] for a, b in self._intervals)
        norm = total / self._bin_width
        self._cache_s = s
        self._cache_sref = s_ref
        self._cache_norm = max(norm, 1e-30)
        return self._cache_norm

    # -- Callable --------------------------------------------------------------

    def __call__(self, xv, p):
        """
        ROOT/Minuit-compatible entry point.  Called with the ROOT convention
        (xv is a 1-element C array, p is a C parameter array).

        Converts to the kernel's clean Python interface:
          kernel(x_scalar, *, param1=v1, param2=v2, ...)

        Layout of p:
          add_norm=True  -> p[0]=N, p[1:]=shape params
          add_norm=False -> p[0:]=shape params

        With default screening (nan_screening=True, inf_screening=True):
          returns a non-negative finite float; 0.0 on any failure.
        With screening disabled: nan/inf from the kernel propagate through.
        """
        x = float(xv[0])
        if self._add_norm:
            N = float(p[0])
            shape_p = tuple(float(p[i]) for i in range(1, self._npar))
        else:
            N = 1.0
            shape_p = tuple(float(p[i]) for i in range(self._npar))

        v = self._eval_kernel(x, shape_p)
        # Always replace negative values (PDF shape must be non-negative).
        if math.isfinite(v) and v < 0.0:
            v = 0.0
        # Configurable nan/inf screening.
        if math.isnan(v) and self._nan_screening:
            v = 0.0
        elif math.isinf(v) and self._inf_screening:
            v = 0.0

        if self._normalize:
            norm = self._compute_norm(shape_p)
            # norm is in units of s_ref, so scale v the same way; the s_ref
            # factors cancel and result == N * v / (integral / bin_width),
            # but the floor in _compute_norm acts on a scale-free O(1) value.
            result = N * (v / self._cache_sref) / norm
        else:
            result = N * v

        # Final guard: non-finite result from N being pathological, or
        # nan/inf passing through when screening is disabled.
        if math.isnan(result):
            return 0.0 if self._nan_screening else result
        if math.isinf(result):
            return 0.0 if self._inf_screening else result
        if result < 0.0:
            return 0.0
        return result

    # -- ROOT.TF1 --------------------------------------------------------------

    def getFn(self, xmin: float, xmax: float):
        """Return a ROOT.TF1 backed by this handle (strong-ref via _fn_ref)."""
        import ROOT

        tf1 = ROOT.TF1(self.key, self, float(xmin), float(xmax), self._npar)
        tf1._fn_ref = self
        for i, v in enumerate(self._initial_params):
            tf1.SetParameter(i, float(v))
        return tf1

    # -- RooFit ----------------------------------------------------------------

    def make_roo_ext_pdf(self, x_obs, N_eml: float | None = None, tag: str = ""):
        """
        Build a RooFit extended-likelihood PDF.

        Always uses the raw shape (N=1 from kernel perspective).
        ``add_norm``/``normalize`` mode has no effect on the RooFit shape.

        Parameters
        ----------
        x_obs   : ROOT.RooRealVar -- the observable.
        N_eml   : Initial event count for N.  Falls back to initial_params[0]
                  (when add_norm=True) or 1e6.
        tag     : Unique string suffix for RooFit object names.

        Returns
        -------
        (ext_pdf, N_roo, shape_rvs)
            ext_pdf   : ROOT.RooExtendPdf
            N_roo     : ROOT.RooRealVar
            shape_rvs : list[ROOT.RooRealVar] (shape params, no N)

        Notes
        -----
        Keep all returned objects alive for the duration of the fit.
        ``ext_pdf._keepalive`` anchors ``shape_pdf``, ``N_roo``, and ``shape_rvs``
        so Python GC does not free them during RooFit's clone/compileForNormSet.
        ``roofit_param_bounds`` from desc.yaml are respected.
        """
        import ROOT

        _ensure_roo_pdf_declared()
        _roo_handle_counter[0] += 1
        _tag = tag or f"_{_roo_handle_counter[0]}"

        # Shape param names and initial values (kernel params, no N)
        shape_names = self._param_names[1:] if self._add_norm else self._param_names
        shape_inits = (
            self._initial_params[1:] if self._add_norm else self._initial_params
        )

        # N initial value
        if N_eml is None:
            if self._add_norm and self._initial_params:
                N_eml = float(self._initial_params[0])
            else:
                N_eml = 1e6
        N_eml = max(float(N_eml), 1.0)

        # N RooRealVar
        N_roo = ROOT.RooRealVar(f"N{_tag}", "N", N_eml, 0.0, N_eml * 3.0)

        # Shape RooRealVars (with optional bounds from desc.yaml)
        roo_bounds = self.yaml_config.get("roofit_param_bounds") or {}
        shape_rvs = []
        for name, v0 in zip(shape_names, shape_inits):
            v0f = float(v0)
            bounds = roo_bounds.get(name)
            if bounds and len(bounds) == 2:
                lo, hi = float(bounds[0]), float(bounds[1])
                v0f = max(lo, min(hi, v0f))
                rv = ROOT.RooRealVar(f"{name}{_tag}", name, v0f, lo, hi)
            else:
                rv = ROOT.RooRealVar(f"{name}{_tag}", name, v0f)
            rv.setError(max(abs(v0f) * 0.1, 0.1))
            shape_rvs.append(rv)

        # RooPyCallPdf: calls kernel with named kwargs.
        # args = (x_val, shape_p0, shape_p1, ...) from C++ evaluate().
        kernel = self._kernel
        shape_names = list(kernel.param_names)  # close over names

        def _shape_callable(*args):
            try:
                x_val = float(args[0])
                kwargs = dict(zip(shape_names, (float(a) for a in args[1:])))
                v = kernel(x_val, **kwargs)
                v = float(v)
                return v if v > 0.0 and math.isfinite(v) else 0.0
            except Exception:
                return 0.0

        pars = ROOT.RooArgList()
        for rv in shape_rvs:
            pars.add(rv)
        shape_pdf = ROOT.RooPyCallPdf(
            f"{self.key}_pdf{_tag}", x_obs, pars, _shape_callable
        )
        shape_pdf._keepalive = (kernel, _shape_callable, list(shape_rvs))

        ext_pdf = ROOT.RooExtendPdf(f"{self.key}_ext{_tag}", self.key, shape_pdf, N_roo)
        # Anchor shape_pdf and N_roo so GC doesn't free them before fitTo finishes.
        ext_pdf._keepalive = (shape_pdf, N_roo, list(shape_rvs))

        return ext_pdf, N_roo, shape_rvs


# -- Module-level helper -------------------------------------------------------


def _build_unblinded(xmin: float, xmax: float, excludes: list) -> list:
    """Return (lo, hi) unblinded sub-intervals."""
    intervals: list = []
    lo = float(xmin)
    for ex_lo, ex_hi in sorted(excludes):
        ex_lo, ex_hi = float(ex_lo), float(ex_hi)
        if ex_lo > lo:
            intervals.append((lo, ex_lo))
        lo = max(lo, ex_hi)
    if lo < float(xmax):
        intervals.append((lo, float(xmax)))
    return intervals or [(float(xmin), float(xmax))]
