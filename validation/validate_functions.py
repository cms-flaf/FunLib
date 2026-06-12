#!/usr/bin/env python3
"""
validate_functions.py -- Cross-validate each function's LaTeX formula against
its Python FitFunction implementation.

Two checks per function (run on test_points from tests.json):
  1. Regression:  Python FitFunction vs stored expected values in tests.json
  2. Formula:     SymPy (from LaTeX) vs stored expected values in tests.json

Usage:
  python3 FunLib/validation/validate_functions.py
  python3 FunLib/validation/validate_functions.py --key ExpPoly-3
  python3 FunLib/validation/validate_functions.py --update-expected
"""

import argparse, importlib.util, json, math, os, re, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
FUNCTIONS_DIR = os.path.normpath(os.path.join(_HERE, "..", "functions"))
RTOL = 5e-4  # relative tolerance (allows for ROOT vs scipy precision differences)
ATOL = 1e-7  # absolute tolerance (for near-zero values)

# Add FunLib's parent directory to sys.path so 'import FunLib' works when run standalone.
# _HERE = FunLib/validation/  ->  '../..' = parent of FunLib
_FUNLIB_PARENT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _FUNLIB_PARENT not in sys.path:
    sys.path.insert(0, _FUNLIB_PARENT)
from FunLib.tools.latex_parser import build_ref_fn

# -- Load FitFunction Python code ----------------------------------------------


def load_fit_fn(entry, xmin=None, xmax=None):
    """Load and instantiate FitFunction from entry['source_file'].

    Checks the __init__ signature: if x_min/x_max are required keyword args they
    are passed explicitly; for range-dependent functions xmin/xmax must be provided
    (no defaults -- raises ValueError if missing).
    Metadata (param_names etc.) is injected onto the instance after construction.
    """
    import inspect as _inspect

    abs_path = entry.get("source_file", "")
    if not os.path.isfile(abs_path):
        return None
    mn = os.path.splitext(os.path.basename(abs_path))[0]
    spec = importlib.util.spec_from_file_location(mn, abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = dict(entry)
    try:
        sig = _inspect.signature(mod.FitFunction.__init__)
        needs_config = "config" in sig.parameters
        needs_range = "x_min" in sig.parameters
        needs_npar = "npar" in sig.parameters
    except (ValueError, TypeError):
        needs_config = False
        needs_range = False
        needs_npar = False
    kw = {}
    if needs_config:
        kw["config"] = cfg
    if needs_range:
        if xmin is None or xmax is None:
            raise ValueError(
                f"load_fit_fn: '{abs_path}' requires x_min/x_max but none provided"
            )
        kw["x_min"] = float(xmin)
        kw["x_max"] = float(xmax)
    if needs_npar:
        npar = cfg.get("npar")
        if npar is None:
            raise ValueError(
                f"load_fit_fn: '{abs_path}' requires npar but entry has none"
            )
        kw["npar"] = int(npar)
    fn = mod.FitFunction(**kw)
    # For variable-npar kernels __init__ already set param_names; only override
    # from entry when the kernel didn't set it itself.
    if not getattr(fn, "param_names", None):
        fn.param_names = entry.get("param_names", [])
    return fn


# -- Regression check against tests.json --------------------------------------


def _raw_to_params(raw, pnames):
    """Convert raw params (list or dict) to a named dict."""
    if isinstance(raw, list):
        return dict(zip(pnames, raw))
    return dict(raw)


def _constants_match(when_dict, constants_dict):
    """Return True if every item in *when_dict* matches the corresponding value
    in *constants_dict*.  Missing keys in *constants_dict* -> no match."""
    for k, v in when_dict.items():
        if constants_dict.get(k) != v:
            return False
    return True


def _select_latex_entry(latex, merged_consts):
    """Select the formula entry that matches *merged_consts*.

    *latex* can be:
      - a plain string  -> returns ``{"formula": latex}`` (backward compat)
      - a list of dicts each with ``"when": {...}`` and ``"formula": "..."``
        (and optionally ``"latex_var_map": {...}``) -> returns the first entry
        whose ``when`` dict is a subset-match of *merged_consts*.

    Returns a dict with at least a ``"formula"`` key (empty string on no match).
    """
    if isinstance(latex, str):
        return {"formula": latex}
    if isinstance(latex, list):
        for entry in latex:
            when = entry.get("when", {})
            if _constants_match(when, merged_consts):
                return entry
    return {"formula": ""}


def _parse_test_case(tc, pnames):
    """
    Normalise one raw test-case dict to {'range', 'params', 'test_points'[, 'npar']}.

    Supports two test_points formats:
      Dict list:  "test_points": [{"x": 1.0, "expected": 2.0}, ...]
      Compact:    "x_grid": [xmin, xmax, n], "expected": [v0, v1, ...]
                  x is reconstructed as linspace(xmin, xmax, n).

    If the test case carries a ``npar`` field (for variable-degree kernels such as
    Bernstein), param_names are auto-generated as c1..cn instead of using the
    desc.yaml default.  The ``npar`` value is forwarded into the returned dict so
    that validate_one() can use a per-test-case kernel instantiation.
    """
    raw_params = tc.get("params", {})
    # Per-test-case npar: auto-generate c1..cn param names for that degree.
    if "npar" in tc and isinstance(raw_params, list):
        effective_pnames = [f"c{i}" for i in range(1, int(tc["npar"]) + 1)]
    else:
        effective_pnames = pnames
    params = _raw_to_params(raw_params, effective_pnames)
    rng = tc.get("range")
    if "x_grid" in tc and "expected" in tc:
        xmin, xmax, n = tc["x_grid"]
        xs = [xmin + (xmax - xmin) * i / (n - 1) for i in range(n)]
        test_points = [
            {"x": x, "expected": e} for x, e in zip(xs, tc["expected"]) if e is not None
        ]
    else:
        test_points = tc.get("test_points", [])
    result = {"range": rng, "params": params, "test_points": test_points}
    if "npar" in tc:
        result["npar"] = int(tc["npar"])
    # Preserve original raw_params list so validate_one can re-map to the
    # kernel's actual param_names after instantiation (list pnames may differ
    # from auto-generated c1..cn for kernels like BWZxBern or LandauxBern).
    if isinstance(raw_params, list):
        result["_raw_params_list"] = list(raw_params)
    if "constants" in tc:
        result["constants"] = dict(tc["constants"])
    if tc.get("skip_formula"):
        result["skip_formula"] = True
    return result


def _load_tests(tests_path, pnames):
    """
    Load tests.json.  Returns a list of test-case dicts.

    Each test case has keys:
      'params'      -- named dict of shape parameters
      'test_points' -- list of {x, expected}
      'range'       -- [xmin, xmax] or None

    Format: JSON array at root level, one element per test case.
    Legacy: JSON object with top-level params/test_points/range -> treated as 1 case.
    """
    if not tests_path or not os.path.isfile(tests_path):
        return []
    with open(tests_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [_parse_test_case(tc, pnames) for tc in data]
    # Legacy dict format -- single test case
    return [_parse_test_case(data, pnames)]


def _pnames_for_tc(tc, default_pnames):
    """Resolve effective param_names for a test case.

    For variable-npar kernels:
    - If params is a dict: use dict keys as pnames (preserves non-c1..cn names).
    - If params is a list: auto-generate c1..cn from the npar field.
    For fixed-npar kernels (no npar field): use desc.yaml default_pnames.
    """
    tc_npar = tc.get("npar")
    raw = tc.get("params", {})
    if tc_npar is not None:
        if isinstance(raw, dict):
            return list(raw.keys())
        return [f"c{i}" for i in range(1, int(tc_npar) + 1)]
    return default_pnames


# -- Update expected values from SymPy ----------------------------------------


def _make_ref_fn(entry, consts_eval):
    """Build a SymPy reference callable for entry, given evaluated constants."""
    key = entry["key"]
    latex = entry.get("latex", "")
    pnames = entry.get("param_names", [])
    lvm = entry.get("latex_var_map", {})
    # Exclude dict-valued latex_* keys (latex_var_map, latex_fn_defs) from laux;
    # only string-valued latex_* entries are auxiliary LaTeX definitions.
    _LATEX_DICT_KEYS = {"latex_var_map", "latex_fn_defs"}
    laux = [
        v
        for k, v in sorted(entry.items())
        if k.startswith("latex_") and k not in _LATEX_DICT_KEYS and isinstance(v, str)
    ]
    fn_defs = entry.get("latex_fn_defs")
    kernel_csv = entry.get("validate_kernel_csv")
    if kernel_csv and not os.path.isabs(kernel_csv):
        kernel_csv = os.path.join(
            os.path.dirname(entry.get("source_file", "")), kernel_csv
        )
    if kernel_csv and os.path.isfile(kernel_csv):
        import csv as _csv_mod
        from sympy import (
            Symbol as _Symbol,
            interpolating_spline as _isp,
            lambdify as _lfy,
        )

        rows = list(_csv_mod.DictReader(open(kernel_csv)))
        all_xs = [float(r["mass"]) for r in rows]
        all_ys = [float(r["sigma"]) for r in rows]

        # Subsample to ~1 GeV resolution over test range +/-10 GeV for a fast,
        # independent SymPy spline.  scipy CubicSpline and SymPy interpolating_spline
        # are different algorithms; agreement at 1 GeV resolution is ~1e-8 relative.
        raw_step = all_xs[1] - all_xs[0] if len(all_xs) > 1 else 1.0
        subsample = max(1, round(1.0 / raw_step))
        lo = consts_eval.get("XMIN", all_xs[0]) - 10.0
        hi = consts_eval.get("XMAX", all_xs[-1]) + 10.0
        pairs = [(x, y) for x, y in zip(all_xs, all_ys) if lo <= x <= hi]
        pairs = pairs[::subsample]
        _xs_sub = [p[0] for p in pairs]
        _ys_sub = [p[1] for p in pairs]

        _xsym = _Symbol("x")
        _spl_expr = _isp(3, _xsym, _xs_sub, _ys_sub)
        _kernel_fn = _lfy(_xsym, _spl_expr, modules=["numpy"])

        # \mathrm{CubicSpline} is stripped of \mathrm{} by the LaTeX preprocessor,
        # leaving "CubicSpline".  Inject the SymPy spline as that callable so
        # build_ref_fn tests the full formula f = CubicSpline(x) * Bernstein(x, p).
        return build_ref_fn(
            key,
            latex,
            pnames,
            consts_eval,
            lvm,
            laux,
            fn_defs=fn_defs,
            extra_fns={"CubicSpline": lambda xi, _k=_kernel_fn: float(_k(float(xi)))},
        )
    return build_ref_fn(key, latex, pnames, consts_eval, lvm, laux, fn_defs=fn_defs)


def _consts_for_range(base_consts, tc_range):
    """Merge base constants with interval values derived from tc_range."""
    c = dict(base_consts)
    if tc_range is not None:
        xmin, xmax = float(tc_range[0]), float(tc_range[1])
        c["XMIN"] = xmin
        c["XMAX"] = xmax
        c["X0"] = (xmin + xmax) / 2.0
        c["XHALF"] = (xmax - xmin) / 2.0
    return c


def update_expected(entry):
    """
    Recompute expected values in tests.json using SymPy.
    SymPy is treated as the authoritative formula; Python must agree afterwards.
    Returns the number of updated points, or None if SymPy is unavailable.
    """
    tests_path = entry.get("tests_path", "")
    if not tests_path or not os.path.isfile(tests_path):
        return None

    pnames = entry.get("param_names", [])
    base_consts = dict(entry.get("constants", {}))
    test_cases = _load_tests(tests_path, pnames)

    updated = 0
    out_cases = []
    for tc in test_cases:
        consts_eval = _consts_for_range(base_consts, tc["range"])
        try:
            ref_fn = _make_ref_fn(entry, consts_eval)
        except ValueError:
            return None
        params_list = [tc["params"][k] for k in pnames]
        for tp in tc["test_points"]:
            x = float(tp["x"])
            try:
                v = float(ref_fn(x, *params_list))
            except Exception:
                v = float("nan")
            if math.isfinite(v):
                tp["expected"] = round(v, 10)
                updated += 1
        out_cases.append(
            {
                "range": tc["range"],
                "params": tc["params"],
                "test_points": tc["test_points"],
            }
            if tc["range"] is not None
            else {"params": tc["params"], "test_points": tc["test_points"]}
        )

    with open(tests_path, "w") as f:
        json.dump(out_cases, f, indent=2)

    return updated


# -- Validate one function -----------------------------------------------------


def validate_one(entry):
    """
    Validate one function.  entry is the fully-merged config dict (desc.yaml +
    initial_params + source_file).  Returns (status, message).

    status is 'pass' or 'fail'.  Never 'skip' -- an unparseable formula is a failure.
    """
    key = entry["key"]
    latex = entry.get("latex", "")
    pnames = entry.get("param_names", [])
    consts = entry.get("constants", {})
    lvm = entry.get("latex_var_map", {})
    laux = [
        v
        for k, v in sorted(entry.items())
        if k.startswith("latex_") and k not in ("latex_var_map",)
    ]

    # -- 0. Invalid entry (missing fn.py or desc.yaml) ---------------------------
    if entry.get("_invalid"):
        return "fail", entry.get("_invalid_reason", "missing required files")

    # -- 0b. Formula variable check -----------------------------------------------
    # The independent variable must be f(x)=.  Detect undeclared use of 'm' as a
    # standalone variable -- the most common mistake is writing the dimuon mass as
    # m instead of x.  'm' is declared when it appears in param_names, constants,
    # or latex_var_map (e.g. as an alias for a degree constant like num_deg).
    def _check_formula_str(formula_str):
        """Check a single formula string; return error message or None."""
        if not formula_str:
            return None
        if not re.match(r"^f\s*\(x\)\s*=", formula_str):
            return f"latex must start with f(x)= -- got: {formula_str[:40]!r}"
        if "m" not in pnames and "m" not in lvm and "m" not in consts:
            _body = formula_str[formula_str.index("=") + 1 :]
            _body = re.sub(r"m_\{[^}]+\}", "", _body)
            _body = re.sub(r"m_[a-zA-Z0-9]+", "", _body)
            if re.search(r"(?<!\\)(?<![a-zA-Z])m(?![a-zA-Z_{\\])", _body) or re.search(
                r"\{m(?![a-zA-Z_{])", _body
            ):
                return (
                    f"latex uses undeclared variable m -- use x for the independent "
                    f"variable, or declare m in param_names/constants/latex_var_map"
                )
        return None

    if latex:
        formula_strings = (
            [e.get("formula", "") for e in latex]
            if isinstance(latex, list)
            else [latex]
        )
        for fs in formula_strings:
            err = _check_formula_str(fs)
            if err:
                return "fail", err

    # -- 1. Load test cases -------------------------------------------------------
    tests_path = entry.get("tests_path", "")
    test_cases = _load_tests(tests_path, pnames)
    if not test_cases:
        return "fail", "no test cases in tests.json"

    # -- 2-4. Per test case: instantiate kernel, build ref_fn, check points -------
    failures = []
    total_points = 0
    for tc in test_cases:
        tc_range = tc["range"]
        params_dict = tc["params"]
        test_points = tc["test_points"]
        xmin = float(tc_range[0]) if tc_range is not None else None
        xmax = float(tc_range[1]) if tc_range is not None else None

        # Per-test-case constants override: merge desc.yaml base constants with any
        # ``constants`` field on the test case (used for multi-variant functions).
        tc_constants = tc.get("constants", {})
        merged_consts = {**consts, **tc_constants}

        # Always build tc_entry as a dict so we can inject merged constants for
        # variant-dispatch kernels (e.g. CrystalBall left/right/double_sided).
        tc_npar = tc.get("npar")
        tc_pnames = _pnames_for_tc(tc, pnames)
        tc_entry = dict(entry)
        tc_entry["constants"] = merged_consts
        if tc_npar is not None:
            tc_entry["npar"] = tc_npar
            tc_entry["param_names"] = tc_pnames

        fn_py = load_fit_fn(tc_entry, xmin=xmin, xmax=xmax)
        if fn_py is None:
            return (
                "fail",
                f'cannot load FitFunction from {entry.get("source_file","?")}',
            )

        # For list-valued params with a npar field: the auto-generated c1..cn pnames
        # may not match the kernel's actual param_names (e.g. BWZxBern uses c1..cn-1+a).
        # Re-map positionally to the kernel's own param_names after instantiation.
        raw_list = tc.get("_raw_params_list")
        if raw_list is not None and tc_npar is not None:
            actual_pnames = list(fn_py.param_names)
            params_dict = dict(zip(actual_pnames, raw_list))
            tc_pnames = actual_pnames
        else:
            params_dict = tc["params"]
        # For variable-degree kernels with named-dict params and no npar in the test
        # case, fall back to the instantiated kernel's own param_names.
        if not tc_pnames and fn_py is not None:
            tc_pnames = list(getattr(fn_py, "param_names", []))

        # Use merged constants (desc.yaml base + test-case override) for the
        # formula constant evaluator so variant-specific formulas get correct vals.
        consts_eval = _consts_for_range(merged_consts, tc_range)
        # For variable-npar functions add 'n' to constants so the LaTeX parser
        # can expand sum patterns \sum_{k=1}^{n}... for that degree.
        # Functions using latex_var_map: {n: n_terms} already populate 'n' via
        # augmented_constants inside build_ref_fn; the injection below only matters
        # for Bernstein (tc_npar = polynomial degree) and similar plain-npar cases.
        if tc_npar is not None:
            n_val = tc_npar
        else:
            n_val = getattr(fn_py, "npar", None)
        # Only inject n as a degree constant when it is NOT itself a shape
        # parameter (e.g. Crystal Ball / Hagedorn / Sersic all have a param
        # named "n" -- those must not be shadowed by the Bernstein npar value).
        if n_val is not None and "n" not in consts_eval and "n" not in tc_pnames:
            consts_eval["n"] = int(n_val)

        # Build the per-test-case latex entry: select the formula that matches
        # merged_consts (variant-dispatch).  Merge any per-variant latex_var_map.
        latex_sel = _select_latex_entry(latex, merged_consts)
        tc_entry_for_latex = dict(tc_entry)
        tc_entry_for_latex["param_names"] = tc_pnames
        tc_entry_for_latex["latex"] = latex_sel.get("formula", "")
        tc_entry_for_latex["latex_var_map"] = {
            **lvm,
            **latex_sel.get("latex_var_map", {}),
        }

        ref_fn = None
        if tc_entry_for_latex.get("latex", "") and not tc.get("skip_formula", False):
            try:
                ref_fn = _make_ref_fn(tc_entry_for_latex, consts_eval)
            except ValueError as e:
                return "fail", f"LaTeX parse error (npar={tc_npar}): {e}"

        if not test_points:
            failures.append("  test case has no test_points")
            continue

        params_list = [params_dict[k] for k in tc_pnames]
        total_points += len(test_points)
        for tp in test_points:
            x, expected = float(tp["x"]), float(tp["expected"])
            scale = max(abs(expected), 1e-30)
            tol = ATOL + RTOL * scale

            try:
                v_py = float(fn_py(x, **params_dict))
            except Exception:
                v_py = float("nan")
            if not math.isfinite(v_py) or abs(v_py - expected) > tol:
                failures.append(
                    f"  x={x:.3f}: python={v_py:.6g}  expected={expected:.6g}  "
                    f"(rel={abs(v_py-expected)/scale:.3e})"
                )

            if ref_fn is not None:
                try:
                    v_sym = float(ref_fn(x, *params_list))
                except Exception:
                    v_sym = float("nan")
                if not math.isfinite(v_sym) or abs(v_sym - expected) > tol:
                    failures.append(
                        f"  x={x:.3f}: sympy={v_sym:.6g}  expected={expected:.6g}  "
                        f"(rel={abs(v_sym-expected)/scale:.3e})"
                    )

    if failures:
        return "fail", (
            f"{len(failures)} test-point failures:\n" + "\n".join(failures[:6])
        )

    return (
        "pass",
        f"tested {len(test_cases)} point(s) in param space, {total_points} function values",
    )


# -- Main ----------------------------------------------------------------------


def load_all_entries(functions_dir=FUNCTIONS_DIR):
    """
    Scan every subdirectory of functions_dir for desc.yaml + fn.py + tests.json.
    Returns list of fully-merged config dicts ready for validate_one(),
    sorted by function key.

    Directories that are missing required files (fn.py or desc.yaml) are
    included as invalid entries so they appear as failures, not silent skips.
    """
    entries = []
    for name in sorted(os.listdir(functions_dir)):
        func_dir = os.path.join(functions_dir, name)
        if not os.path.isdir(func_dir):
            continue
        desc_path = os.path.join(func_dir, "desc.yaml")
        fn_path = os.path.join(func_dir, "fn.py")
        missing = [
            f
            for f, p in [("desc.yaml", desc_path), ("fn.py", fn_path)]
            if not os.path.isfile(p)
        ]
        if missing:
            entries.append(
                {
                    "key": name,
                    "_invalid": True,
                    "_invalid_reason": f"missing required file(s): {', '.join(missing)} in {func_dir}",
                }
            )
            continue
        with open(desc_path) as f:
            desc = yaml.safe_load(f)
        entry = dict(desc)
        entry["source_file"] = fn_path
        entry["tests_path"] = os.path.join(func_dir, "tests.json")
        # Format key/label templates (e.g. "RationalPade-{num_deg}/{den_deg}")
        # using default constants from desc.yaml so the key is fully resolved.
        consts = entry.get("constants", {})
        for _field in ("key", "label"):
            tmpl = entry.get(_field, "")
            if isinstance(tmpl, str) and "{" in tmpl:
                for _k, _v in consts.items():
                    tmpl = tmpl.replace(f"{{{_k}}}", str(_v))
                entry[_field] = tmpl
        entries.append(entry)
    return entries


def _validate_one_entry(entry):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    return entry["key"], validate_one(entry)


def main():
    ap = argparse.ArgumentParser(description="Validate LaTeX formulas vs Python code.")
    ap.add_argument("--key", default=None, help="Validate only the named function")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Parallel worker processes (default 8; 1 = serial)",
    )
    ap.add_argument(
        "--update-expected",
        action="store_true",
        help="Recompute expected values in tests.json from SymPy results",
    )
    args = ap.parse_args()

    functions = load_all_entries()
    if args.key:
        functions = [e for e in functions if e["key"] == args.key]
        if not functions:
            sys.exit(f"Function {args.key!r} not found")

    if args.update_expected:
        n_ok = n_skip = 0
        for entry in functions:
            updated = update_expected(entry)
            key = entry["key"]
            if updated is None:
                print(f"  {key:30s}  skip")
                n_skip += 1
            else:
                print(f"  {key:30s}  updated {updated} points")
                n_ok += 1
        print(f"\n  Updated: {n_ok}  skipped: {n_skip}")
        return

    n_pass = n_fail = 0
    results = (
        {}
    )  # key -> (status, msg) -- collected in arrival order, printed in function order

    workers = min(args.workers, len(functions))
    if workers <= 1:
        for entry in functions:
            results[entry["key"]] = validate_one(entry)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_validate_one_entry, e): e["key"] for e in functions}
            for fut in as_completed(futures):
                key, result = fut.result()
                results[key] = result

    for entry in functions:
        key = entry["key"]
        status, msg = results[key]
        print(f"  {key:30s}  ", end="")
        if status == "pass":
            n_pass += 1
            print(f"pass  {msg}")
        else:
            n_fail += 1
            print(f"FAIL\n{msg}")

    total = n_pass + n_fail
    print(f"\n  Results: {n_pass}/{total} pass, {n_fail} fail")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
