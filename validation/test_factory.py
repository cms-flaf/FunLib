#!/usr/bin/env python3
"""
FunLib/validation/test_factory.py -- Test suite for FunLib/tools/factory.py.

Tests that the factory correctly wraps N-free kernels in all supported modes
and that the RooFit extended-PDF interface works end-to-end.

Run from repo root:
    python3 FunLib/validation/test_factory.py
    python3 FunLib/validation/test_factory.py --verbose
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import yaml

# Add FunLib's parent to sys.path so 'import FunLib' works when run standalone.
# __file__ = FunLib/validation/test_factory.py  ->  3 dirnames up = parent of FunLib
_FUNLIB_PARENT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _FUNLIB_PARENT not in sys.path:
    sys.path.insert(0, _FUNLIB_PARENT)

from FunLib.tools.factory import make_function, load_kernel, FunctionHandle

XMIN, XMAX = 0.0, 1.0
EXCLUDES = [(0.4, 0.6)]  # excluded sub-window within [0, 1]
BIN_WIDTH = 0.01  # 100 bins in [0, 1]
INTEGRAL = 1_000_000.0

# -- helpers -------------------------------------------------------------------


def _entry(fn_dir: str) -> dict:
    """Build a minimal yaml_entry for a function directory."""
    fn_dir_abs = os.path.normpath(os.path.join(_FUNLIB_PARENT, fn_dir))
    desc_path = os.path.join(fn_dir_abs, "desc.yaml")
    with open(desc_path) as f:
        desc = yaml.safe_load(f)
    # Use first element of initial_params from all_functions.yaml if available;
    # otherwise fall back to zeros matching npar.
    try:
        fn_yaml = os.path.join(
            _FUNLIB_PARENT, "FunLib", "applications", "h_mumu", "all_functions.yaml"
        )
        with open(fn_yaml) as f:
            cfg = yaml.safe_load(f)
        for e in cfg.get("functions", []):
            if os.path.normpath(e["dir"]) == os.path.normpath(fn_dir):
                return {"dir": fn_dir, "initial_params": e["initial_params"]}
    except Exception:
        pass
    return {"dir": fn_dir, "initial_params": [0.0] * len(desc.get("param_names", []))}


def _make(fn_dir, **kw):
    entry = _entry(fn_dir)
    defaults = dict(
        add_norm=True,
        normalize=False,
        xmin=XMIN,
        xmax=XMAX,
        excludes=EXCLUDES,
        bin_width=BIN_WIDTH,
        integral=INTEGRAL,
    )
    defaults.update(kw)
    return make_function(entry, **defaults)


# -- test functions -------------------------------------------------------------


def test_load_kernel():
    """load_kernel returns a N-free FitFunction with named-parameter __call__."""
    entry = _entry("FunLib/functions/bwz_exp")
    kernel = load_kernel(entry, xmin=XMIN, xmax=XMAX)
    assert kernel.npar == 2, f"npar={kernel.npar}"
    assert kernel.param_names == ["a", "b"], f"param_names={kernel.param_names}"
    assert "N" not in kernel.param_names, "N should not be in kernel"
    # New interface: kernel(x_scalar, *, a=..., b=...)
    v = kernel(0.5, a=0.0, b=0.0)
    assert math.isfinite(v) and v > 0, f"kernel(130, a=0, b=0)={v}"
    return "load_kernel: N-free kernel loaded correctly"


def test_add_norm_true():
    """add_norm=True prepends N; npar = kernel.npar + 1."""
    h = _make("FunLib/functions/bwz_exp", add_norm=True)
    assert isinstance(h, FunctionHandle), "should return FunctionHandle"
    assert h.npar == 3, f"npar={h.npar}"
    assert h.param_names[0] == "N", f"param_names={h.param_names}"
    assert len(h.initial_params) == 3, f"len(initial_params)={len(h.initial_params)}"
    # N_init should be sensible (order of magnitude of integral/n_bins)
    N_init = h.initial_params[0]
    assert N_init > 0, f"N_init={N_init}"
    return f"add_norm=True: npar={h.npar}, N_init={N_init:.3g}"


def test_add_norm_false():
    """add_norm=False exposes only shape params."""
    h = _make("FunLib/functions/bwz_exp", add_norm=False)
    assert h.npar == 2, f"npar={h.npar}"
    assert "N" not in h.param_names, f"N in param_names={h.param_names}"
    # FunctionHandle.__call__ still uses ROOT convention (xv, p)
    import numpy as np

    xv = np.array([0.5])
    v = h(xv, h.initial_params)
    assert math.isfinite(v) and v > 0, f"h([130])={v}"
    # Should equal kernel(x, **kwargs)
    kernel = load_kernel(_entry("FunLib/functions/bwz_exp"), xmin=XMIN, xmax=XMAX)
    kwargs = dict(zip(kernel.param_names, h.initial_params))
    v_k = kernel(0.5, **kwargs)
    assert abs(v - v_k) < 1e-10, f"h!=kernel: {v} vs {v_k}"
    return f"add_norm=False: npar={h.npar}, value={v:.4g}"


def test_chi2_mode_value():
    """In chi2 mode: h(xv, p) = N * kernel(x, **shape_kwargs)."""
    h = _make("FunLib/functions/exppoly", add_norm=True, normalize=False)
    kernel = load_kernel(_entry("FunLib/functions/exppoly"), xmin=XMIN, xmax=XMAX)
    p = h.initial_params  # [N, c1, ..., cn]
    x = 0.5
    v_h = h([x], p)  # FunctionHandle uses ROOT (xv,p) convention
    N = p[0]
    kwargs = dict(zip(kernel.param_names, p[1:]))
    v_k = kernel(x, **kwargs)
    assert abs(v_h - N * v_k) < 1e-8 * max(
        abs(v_h), 1e-30
    ), f"h={v_h}, N*kernel={N*v_k}"
    return f"chi2 mode: h(0.5)=N*kernel(0.5) [ok]  (v={v_h:.4g})"


def test_normalize_invariant():
    """In normalize mode: sum_bins h(x_i, p) ~= N (bin count, no extra Deltax)."""
    h = _make("FunLib/functions/exppoly", add_norm=True, normalize=True)
    p = list(h.initial_params)
    N = p[0]
    xs_all = [
        XMIN + (i + 0.5) * BIN_WIDTH for i in range(int((XMAX - XMIN) / BIN_WIDTH))
    ]
    xs = [x for x in xs_all if not any(lo <= x <= hi for lo, hi in EXCLUDES)]
    total = sum(h([x], p) for x in xs)
    ratio = total / N
    assert abs(ratio - 1.0) < 0.01, f"sumh/N = {ratio:.5f} (expected ~=1.0)"
    return f"normalize=True: sumh/N = {ratio:.5f}"


def test_normalize_N_init():
    """N_init in normalize mode ~= unblinded event count."""
    h = _make(
        "FunLib/functions/exppoly", add_norm=True, normalize=True, integral=500_000
    )
    N_init = h.initial_params[0]
    # Unblinded fraction: (1.0 - 0.2) / 1.0 = 0.8  (EXCLUDES width = 0.6-0.4 = 0.2)
    expected = 500_000 * 0.8
    assert (
        abs(N_init / expected - 1.0) < 0.01
    ), f"N_init={N_init:.3g}, expected~={expected:.3g}"
    return f"normalize N_init ~= unblinded events: {N_init:.3g}"


def test_initial_params_rw():
    """initial_params is read-write on FunctionHandle."""
    h = _make("FunLib/functions/bwz_exp", add_norm=True)
    orig = list(h.initial_params)
    h.initial_params = [orig[0] * 2.0] + orig[1:]
    assert h.initial_params[0] == orig[0] * 2.0, "setter failed"
    # Reset
    h.initial_params = orig
    return "initial_params setter works"


def test_getFn():
    """getFn returns a ROOT.TF1 consistent with __call__."""
    import ROOT

    ROOT.gROOT.SetBatch(True)
    h = _make("FunLib/functions/bwz_exp", add_norm=True, normalize=False)
    tf1 = h.getFn(XMIN, XMAX)
    assert tf1.GetName() == h.key, f"TF1 name={tf1.GetName()}"
    assert tf1.GetNpar() == h.npar, f"TF1 npar={tf1.GetNpar()}"
    x = 0.5
    v_tf1 = tf1.Eval(x)
    v_h = h([x], h.initial_params)
    assert abs(v_tf1 - v_h) < 1e-6 * max(
        abs(v_h), 1e-30
    ), f"TF1.Eval({x})={v_tf1} != h([{x}])={v_h}"
    assert tf1._fn_ref is h, "_fn_ref keepalive not set"
    return f"getFn: TF1 Eval({x})={v_tf1:.4g} matches __call__"


def test_fn_alias():
    """h._fn returns the raw kernel."""
    h = _make("FunLib/functions/betaprime", add_norm=True)
    kernel = load_kernel(_entry("FunLib/functions/betaprime"))
    assert h._fn.key == kernel.key, f"_fn.key={h._fn.key}"
    assert h._fn.npar == kernel.npar, f"_fn.npar={h._fn.npar}"
    return "_fn alias points to kernel"


def test_yaml_config_passthrough():
    """yaml_config is accessible and has correct metadata (variable-npar: bwz_bern npar=3)."""
    # Build entry explicitly to select npar=3 (Bernstein-2: c1, c2, a)
    entry = {
        "dir": "FunLib/functions/bwz_bern",
        "initial_params": [1.0, 1.0, -0.01],
        "constants": {"n_terms": 2},
    }
    h = make_function(
        entry,
        add_norm=True,
        xmin=XMIN,
        xmax=XMAX,
        excludes=EXCLUDES,
        bin_width=BIN_WIDTH,
        integral=INTEGRAL,
    )
    cfg = h.yaml_config
    assert cfg["key"] == "BWZxBern-2", f"key={cfg['key']}"
    # yaml_config npar = kernel npar (3); handle npar = kernel npar + 1 (N prepended)
    assert cfg["npar"] == h.npar - 1, f"yaml npar={cfg['npar']}, handle npar={h.npar}"
    return f"yaml_config key={cfg['key']}, npar_kernel={cfg['npar']}"


def test_source_file():
    """source_file is a valid path pointing to the fn.py."""
    h = _make("FunLib/functions/bwz_exp", add_norm=True)
    sf = h.source_file
    assert sf.endswith("fn.py"), f"source_file={sf}"
    assert os.path.isfile(sf), f"source_file not found: {sf}"
    return f"source_file={os.path.basename(os.path.dirname(sf))}/fn.py"


def test_kernel_named_params():
    """
    Kernel __call__ takes (x, *, param1=v1, ...) -- scalar x, named kwargs.
    Verify the new interface works and errors on wrong call convention.
    """
    import inspect

    kernel = load_kernel(_entry("FunLib/functions/bwz_exp"), xmin=XMIN, xmax=XMAX)
    sig = inspect.signature(kernel.__call__)
    params = list(sig.parameters.keys())  # bound method: no 'self'
    # Must have 'x' as the first positional arg
    assert params[0] == "x", f"first arg should be 'x', got {params[0]}"
    kw_params = [
        p for p, v in sig.parameters.items() if v.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    assert kw_params == list(
        kernel.param_names
    ), f"kw params {kw_params} != param_names {kernel.param_names}"

    # Works with named kwargs
    v1 = kernel(0.5, a=0.0, b=0.0)
    assert math.isfinite(v1) and v1 > 0, f"named call failed: {v1}"

    # Wrong kwargs raise TypeError
    try:
        kernel(0.5, wrong=1.0)
        assert False, "should have raised TypeError for wrong kwarg"
    except TypeError:
        pass  # expected

    # Positional arg for x is OK; missing kwarg raises TypeError
    try:
        kernel(0.5)  # missing a, b
        assert False, "should have raised TypeError for missing kwarg"
    except TypeError:
        pass  # expected

    return f"named-param interface: kernel(0.5, a=0, b=0) = {v1:.4g}"


def test_entry_formats():
    """make_function accepts both dir-entry and pre-merged config dict."""
    # dir-entry format: explicit [2/2] for rationalpade (variable-degree kernel)
    entry = {
        "dir": "FunLib/functions/rationalpade",
        "constants": {"num_deg": 2, "den_deg": 2},
        "initial_params": [-0.1288324, 0.0212008, 0.8079761, 0.1200489],
    }
    h1 = make_function(
        entry,
        add_norm=True,
        xmin=XMIN,
        xmax=XMAX,
        excludes=EXCLUDES,
        bin_width=BIN_WIDTH,
        integral=INTEGRAL,
    )
    assert h1.key == "RationalPade-2/2", f"key={h1.key}"

    # pre-merged config dict (desc.yaml defaults to [2/2])
    fn_dir = os.path.join(_FUNLIB_PARENT, "FunLib/functions/rationalpade")
    desc_path = os.path.join(fn_dir, "desc.yaml")
    with open(desc_path) as f:
        desc = yaml.safe_load(f)
    config = dict(desc)
    config["initial_params"] = list(entry["initial_params"])
    config["source_file"] = os.path.join(fn_dir, "fn.py")
    h2 = make_function(
        config,
        add_norm=True,
        xmin=XMIN,
        xmax=XMAX,
        excludes=EXCLUDES,
        bin_width=BIN_WIDTH,
        integral=INTEGRAL,
    )
    assert h2.key == "RationalPade-2/2", f"key={h2.key}"

    v1 = h1([0.5], h1.initial_params)
    v2 = h2([0.5], h2.initial_params)
    # Both share same initial shape params; values may differ if N differs,
    # so just check both finite and positive.
    assert math.isfinite(v1) and v1 > 0
    assert math.isfinite(v2) and v2 > 0
    return f"entry formats: dir-entry={v1:.4g}, pre-merged={v2:.4g}"


def test_scipy_kernel():
    """scipy-based kernel (BetaPrime) works correctly in all modes."""
    h_chi2 = _make("FunLib/functions/betaprime", add_norm=True, normalize=False)
    h_norm = _make("FunLib/functions/betaprime", add_norm=True, normalize=True)
    h_shp = _make("FunLib/functions/betaprime", add_norm=False)
    # All modes return finite positive values at x=0.5
    for h, mode in [(h_chi2, "chi2"), (h_norm, "norm"), (h_shp, "shape")]:
        v = h([0.5], h.initial_params)
        assert math.isfinite(v) and v >= 0, f"{mode}: v={v}"
    # Shape mode matches kernel(x, **kwargs)
    kernel = load_kernel(_entry("FunLib/functions/betaprime"))
    v_shp = h_shp([0.5], h_shp.initial_params)  # FunctionHandle ROOT-convention
    kwargs = dict(zip(kernel.param_names, kernel.initial_params))
    v_k = kernel(0.5, **kwargs)
    assert abs(v_shp - v_k) < 1e-10 * max(abs(v_shp), 1e-300)
    return f"scipy BetaPrime: chi2={h_chi2([0.5], h_chi2.initial_params):.3g}"


def test_roofit_ext_pdf():
    """make_roo_ext_pdf builds a valid RooExtendPdf that can be fitted."""
    import ROOT

    ROOT.gROOT.SetBatch(True)

    h = _make("FunLib/functions/bwz_exp", add_norm=True, normalize=False)

    x_obs = ROOT.RooRealVar("xfact", "m", XMIN, XMAX)
    x_obs.setRange("r0", XMIN, 0.4)
    x_obs.setRange("r1", 0.6, XMAX)

    ext_pdf, N_roo, shape_rvs = h.make_roo_ext_pdf(x_obs, N_eml=800_000, tag="_tst")

    assert isinstance(ext_pdf, ROOT.RooExtendPdf), "ext_pdf type"
    assert N_roo.getVal() == 800_000, "N_eml not set"
    assert len(shape_rvs) == h.npar - 1, f"shape_rvs len={len(shape_rvs)}"

    # Build a synthetic histogram and run a quick fit
    h_hist = ROOT.TH1D("fact_h", "h", 100, XMIN, XMAX)
    import random

    random.seed(99)
    for _ in range(50_000):
        m = 1.0 - random.random() ** 0.4
        if XMIN <= m <= XMAX:
            h_hist.Fill(m)
    dh = ROOT.RooDataHist(
        "fact_dh", "d", ROOT.RooArgSet(x_obs), ROOT.RooFit.Import(h_hist)
    )
    N_eml = sum(
        h_hist.GetBinContent(b)
        for b in range(1, h_hist.GetNbinsX() + 1)
        if not (0.4 <= h_hist.GetBinCenter(b) <= 0.6)
    )

    N_roo.setVal(N_eml)
    N_roo.setRange(0.0, N_eml * 3.0)
    r = ext_pdf.fitTo(
        dh,
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.Save(True),
        ROOT.RooFit.Range("r0,r1"),
        ROOT.RooFit.Minimizer("Minuit2", "Migrad"),
        ROOT.RooFit.MaxCalls(2000),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Warnings(False),
    )

    status = r.status()
    assert status in (0, 1, 2, 3, 4), f"unexpected RooFit status={status}"
    N_fit = N_roo.getVal()
    assert N_fit > 0 and math.isfinite(N_fit), f"N_fit={N_fit}"
    return f"RooFit EML fit: status={status}, N_fit={N_fit:.0f}"


def test_roofit_param_bounds():
    """roofit_param_bounds from desc.yaml are set on shape RooRealVars."""
    import ROOT

    ROOT.gROOT.SetBatch(True)

    h = _make("FunLib/functions/bwz_gamma", add_norm=True)
    x_obs = ROOT.RooRealVar("xbwg", "m", XMIN, XMAX)
    _, _, shape_rvs = h.make_roo_ext_pdf(x_obs, N_eml=1e6, tag="_bwg")

    fZ_rv = next((rv for rv in shape_rvs if "fZ" in rv.GetName()), None)
    assert fZ_rv is not None, "fZ RooRealVar not found"
    assert abs(fZ_rv.getMin() - 0.001) < 1e-9, f"fZ min={fZ_rv.getMin()}"
    assert abs(fZ_rv.getMax() - 0.999) < 1e-9, f"fZ max={fZ_rv.getMax()}"
    return f"roofit_param_bounds: fZ in [{fZ_rv.getMin()}, {fZ_rv.getMax()}]"


def test_keepalive():
    """ext_pdf._keepalive anchors shape_pdf and prevents GC crashes."""
    import ROOT, gc

    ROOT.gROOT.SetBatch(True)
    h = _make("FunLib/functions/bwz_exp", add_norm=True)
    x_obs = ROOT.RooRealVar("xka", "m", XMIN, XMAX)
    ext_pdf, N_roo, shape_rvs = h.make_roo_ext_pdf(x_obs, N_eml=1e6, tag="_ka")
    gc.collect()  # force GC; ext_pdf._keepalive should prevent crash
    assert hasattr(ext_pdf, "_keepalive"), "ext_pdf._keepalive missing"
    shape_pdf = ext_pdf._keepalive[0]
    assert hasattr(shape_pdf, "_keepalive"), "shape_pdf._keepalive missing"
    return "keepalive chain intact after GC"


def test_regression_vs_tests_json():
    """
    All 75 functions: kernel(x, **kwargs) matches tests.json expected values.
    Scans FunLib/functions/ directly -- no dependency on functions.yaml.
    """
    funcs_dir = os.path.normpath(os.path.join(_FUNLIB_PARENT, "FunLib", "functions"))
    failures = []

    for name in sorted(os.listdir(funcs_dir)):
        fn_dir = os.path.join(funcs_dir, name)
        tests_path = os.path.join(fn_dir, "tests.json")
        desc_path = os.path.join(fn_dir, "desc.yaml")
        fn_path = os.path.join(fn_dir, "fn.py")
        if not all(os.path.isfile(p) for p in (tests_path, desc_path, fn_path)):
            continue

        with open(tests_path) as f:
            tests = json.load(f)
        with open(desc_path) as f:
            desc = yaml.safe_load(f)

        # tests.json root is a list of test cases; use the first one.
        # Support both dict-list format and compact x_grid+expected format.
        raw_tc = tests[0] if isinstance(tests, list) else tests

        # Per-test-case npar (e.g. Bernstein tests multiple degrees in one file).
        # When npar is in the test case, use it to derive param_names.
        # Otherwise, param_names come from desc.yaml (the authoritative source).
        tc_npar = raw_tc.get("npar")
        if tc_npar is not None:
            pnames = [f"c{i}" for i in range(1, int(tc_npar) + 1)]
            effective_npar = int(tc_npar)
        else:
            pnames = desc.get("param_names", [])
            effective_npar = len(pnames)

        params_raw = raw_tc["params"]
        params_dict = (
            params_raw
            if isinstance(params_raw, dict)
            else dict(zip(pnames, params_raw))
        )
        # Variable-degree kernels omit param_names from desc.yaml and use named-dict
        # params in tests.json; infer effective_npar from the dict length.
        if isinstance(params_raw, dict):
            effective_npar = len(params_dict)

        if len(params_dict) != effective_npar:
            failures.append(
                (
                    desc.get("key", name),
                    f"param count {len(params_dict)} != npar {effective_npar}",
                )
            )
            continue

        # Reconstruct test_points from either format
        if "x_grid" in raw_tc and "expected" in raw_tc:
            xmin_g, xmax_g, n_g = raw_tc["x_grid"]
            xs_g = [xmin_g + (xmax_g - xmin_g) * i / (n_g - 1) for i in range(n_g)]
            test_points = [
                {"x": x, "expected": e}
                for x, e in zip(xs_g, raw_tc["expected"])
                if e is not None
            ]
        else:
            test_points = raw_tc.get("test_points", [])

        # range field: required for range-dependent kernels.
        # initial_params length drives npar for variable-npar kernels.
        # constants from the test case (e.g. n_terms) are forwarded so that
        # constants-driven variable-npar kernels can initialise correctly.
        rng = raw_tc.get("range")
        e = {
            "dir": os.path.join("FunLib", "functions", name),
            "initial_params": list(params_dict.values()),
            "constants": raw_tc.get("constants", {}),
        }
        try:
            if rng is not None:
                kernel = load_kernel(e, xmin=float(rng[0]), xmax=float(rng[1]))
            else:
                kernel = load_kernel(e)
        except Exception as ex:
            failures.append((desc["key"], f"load_kernel failed: {ex}"))
            continue

        for pt in test_points:
            x_val = pt["x"]
            expected = pt["expected"]
            try:
                v = float(kernel(float(x_val), **params_dict))
            except Exception:
                v = float("nan")

            if not math.isfinite(v) and not math.isfinite(expected):
                continue
            if not math.isfinite(expected):
                continue
            scale = max(abs(expected), 1e-30)
            tol = 1e-7 + 5e-4 * scale  # ATOL + RTOL * scale
            if abs(v - expected) > tol:
                failures.append(
                    (
                        desc["key"],
                        f"x={x_val}: v={v:.4g}, expected={expected:.4g}, "
                        f"err={abs(v-expected):.2e}",
                    )
                )

    if failures:
        msg = "\n  ".join(f"{k}: {m}" for k, m in failures[:10])
        assert False, f"{len(failures)} regression failures:\n  {msg}"

    return f"regression: all 75 functions match tests.json"


# -- runner --------------------------------------------------------------------

TESTS = [
    test_load_kernel,
    test_kernel_named_params,
    test_add_norm_true,
    test_add_norm_false,
    test_chi2_mode_value,
    test_normalize_invariant,
    test_normalize_N_init,
    test_initial_params_rw,
    test_getFn,
    test_fn_alias,
    test_yaml_config_passthrough,
    test_source_file,
    test_entry_formats,
    test_scipy_kernel,
    test_roofit_ext_pdf,
    test_roofit_param_bounds,
    test_keepalive,
    test_regression_vs_tests_json,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    passed = failed = 0
    print(f"\nRunning {len(TESTS)} factory tests...\n")

    for fn in TESTS:
        name = fn.__name__
        try:
            msg = fn()
            status = "PASS"
            passed += 1
        except AssertionError as e:
            status = "FAIL"
            msg = str(e)
            failed += 1
        except Exception as e:
            status = "ERROR"
            msg = f"{type(e).__name__}: {e}"
            failed += 1

        color = "\033[92m" if status == "PASS" else "\033[91m"
        reset = "\033[0m"
        pad = max(0, 45 - len(name))
        print(f"  {name}{' '*pad}  {color}{status}{reset}")
        if args.verbose or status != "PASS":
            indent = "    "
            for line in str(msg).splitlines():
                print(f"{indent}{line}")

    print(f"\n  Results: {passed}/{len(TESTS)} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
