#!/usr/bin/env python3
"""
FunLib/validation/test_workspace.py -- Test suite for mk_workspace + load_workspace.

Three test groups:

1. Roundtrip (all 94 entries from all_functions.yaml)
   mk_workspace -> load_workspace -> make_handle -> eval
   Checks: file written, all objects in workspace, ModelConfig present,
   signal/background/model_s/data_obs present, FunctionHandle callable.

2. FitTo smoke (5 simple functions, <= 2 shape params)
   mk_workspace -> load_workspace -> make_handle ->
   make_roo_ext_pdf -> fitTo(data_obs).
   Verifies the reconstructed Python-backed PDF can be minimised end-to-end.

3. Combine significance (selected functions + a quick sanity entry)
   Runs ``combine -M Significance ws.root -m 125`` via subprocess for each
   selected workspace and checks for rc=0 + ``Significance:`` in output.
   Skipped if the Combine binary is not found.

Run from repo root:
    python3 FunLib/validation/test_workspace.py
    python3 FunLib/validation/test_workspace.py --verbose
    python3 FunLib/validation/test_workspace.py --no-combine   # skip Combine tests
"""

import argparse
import os
import subprocess
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import yaml

_YAML = os.path.join(_REPO, "FunLib", "applications", "h_mumu", "all_functions.yaml")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


# -- shared fixtures -----------------------------------------------------------


def _load_entries():
    with open(_YAML) as fh:
        cfg = yaml.safe_load(fh)
    return cfg["functions"], cfg


def _make_test_hist():
    """Synthetic falling-exponential TH1D: [110, 150] GeV, 160 bins, ~20k events."""
    import ROOT

    hist = ROOT.TH1D("h_ws_test", "test", 160, 110.0, 150.0)
    f_exp = ROOT.TF1("f_ws_exp", "exp(-0.08*(x-110))", 110.0, 150.0)
    hist.FillRandom("f_ws_exp", 20_000)
    hist.SetDirectory(0)
    return hist


def _entry_label(entry: dict) -> str:
    key = entry.get("key", "") or os.path.basename(entry.get("dir", "?"))
    return f"{key}[{len(entry.get('initial_params', []))}p]"


# -- test 1: roundtrip for all entries -----------------------------------------


def test_roundtrip(entries, hist, tmpdir, verbose=False):
    """
    mk_workspace -> load_workspace -> make_handle -> eval for every entry.
    Also checks that the Combine-compatible objects are present in the workspace.
    """
    from FunLib.tools.mk_workspace import mk_workspace
    from FunLib.tools.load_workspace import load_workspace

    passed = failed = 0
    failures = []

    for i, entry in enumerate(entries):
        label = _entry_label(entry)
        npar = len(entry.get("initial_params", []))
        dir_tag = os.path.basename(entry.get("dir", f"entry{i}"))
        ws_file = os.path.join(tmpdir, f"ws_{dir_tag}_{npar}p.root")

        try:
            ws = mk_workspace(entry, hist, ws_file)
            assert ws is not None, "mk_workspace returned None"
            assert os.path.isfile(ws_file), "output file not created"

            with load_workspace(ws_file) as wh:

                # -- observable
                x = wh.x_obs
                assert x is not None, "observable 'x' not in workspace"
                assert abs(x.getMin() - 110.0) < 0.01
                assert abs(x.getMax() - 150.0) < 0.01

                # -- combine-required objects
                assert wh.data is not None, "data_obs not in workspace"
                assert wh.data.sumEntries() > 0, "data_obs is empty"
                assert wh.model is not None, "model_s not in workspace"

                # -- background and signal PDFs
                assert wh.ws.pdf("background") is not None, "background PDF missing"
                assert wh.ws.pdf("signal") is not None, "signal PDF missing"

                # -- ModelConfig
                mc = wh.ws.obj("ModelConfig")
                assert mc is not None, "ModelConfig missing"

                # -- POI and nuisance
                r_var = wh.ws.var("r")
                assert r_var is not None, "POI 'r' missing"
                n_bkg = wh.ws.var("n_bkg")
                assert n_bkg is not None, "'n_bkg' missing"
                assert n_bkg.getVal() > 0

                # -- metadata
                assert wh.key, "meta key is empty"
                assert (
                    wh.meta["npar"] == npar
                ), f"npar mismatch: meta={wh.meta['npar']} entry={npar}"
                assert len(wh.param_names) == npar

                # -- shape param RooRealVars present
                for pname in wh.param_names:
                    rv = wh.get_param(pname)
                    assert rv is not None, f"p_{pname} not in workspace"

                # -- reconstruct handle and evaluate at midpoint
                handle = wh.make_handle()
                x_mid = 0.5 * (wh.meta["xmin"] + wh.meta["xmax"])
                val = handle([x_mid], list(handle.initial_params))
                assert val >= 0.0, f"handle({x_mid}) = {val} < 0"

            passed += 1
            if verbose:
                print(f"  {label:<55} {PASS}")

        except Exception as exc:
            failed += 1
            msg = str(exc)
            failures.append((label, msg))
            print(f"  {label:<55} {FAIL}  {msg}")
            if verbose:
                traceback.print_exc()

    return passed, failed, failures


# -- test 2: fitTo smoke -------------------------------------------------------

_SMOKE_DIRS = {
    "FunLib/functions/bwz_exp",
    "FunLib/functions/s_exp",
    "FunLib/functions/chebyshev",
    "FunLib/functions/exppoly",
    "FunLib/functions/bernstein",
}
_SMOKE_MAX_NPAR = 2


def test_fitTo_smoke(entries, hist, tmpdir, verbose=False):
    """
    mk_workspace -> load_workspace -> make_handle ->
    make_roo_ext_pdf -> fitTo(data_obs).
    Confirms the reconstructed Python-backed PDF can be used for fitting
    (independent of the RooHistPdf model stored in the workspace).
    """
    import ROOT
    from FunLib.tools.mk_workspace import mk_workspace
    from FunLib.tools.load_workspace import load_workspace

    passed = failed = skipped = 0
    failures = []

    for entry in entries:
        fn_dir = entry.get("dir", "")
        npar = len(entry.get("initial_params", []))
        if fn_dir not in _SMOKE_DIRS or npar > _SMOKE_MAX_NPAR:
            skipped += 1
            continue

        label = _entry_label(entry)
        dir_tag = os.path.basename(fn_dir)
        ws_file = os.path.join(tmpdir, f"ws_fit_{dir_tag}_{npar}p.root")

        try:
            mk_workspace(entry, hist, ws_file)

            with load_workspace(ws_file) as wh:
                handle = wh.make_handle()
                x_obs = wh.x_obs
                data = wh.data

                ext_pdf, N_roo, shape_rvs = handle.make_roo_ext_pdf(
                    x_obs,
                    N_eml=wh.meta["integral"],
                    tag=f"_smoke_{dir_tag}_{npar}",
                )
                result = ext_pdf.fitTo(
                    data,
                    ROOT.RooFit.Save(True),
                    ROOT.RooFit.PrintLevel(-1),
                    ROOT.RooFit.Warnings(False),
                    ROOT.RooFit.Strategy(0),
                )
                assert result is not None, "fitTo returned None"
                status = result.status()

            passed += 1
            print(f"  {label:<55} {PASS}  status={status}")

        except Exception as exc:
            failed += 1
            failures.append((label, str(exc)))
            print(f"  {label:<55} {FAIL}  {exc}")
            if verbose:
                traceback.print_exc()

    return passed, failed, skipped, failures


# -- test 3: combine -M Significance ------------------------------------------

# Representative subset: one entry from several different function families
_COMBINE_DIRS = {
    "FunLib/functions/bwz_exp",  # BWZ-based
    "FunLib/functions/exppoly",  # exp-polynomial
    "FunLib/functions/bernstein",  # Bernstein
    "FunLib/functions/chebyshev",  # Chebyshev
    "FunLib/functions/s_exp",  # power-law exp
    "FunLib/functions/bwz_bern",  # BWZ x Bernstein
    "FunLib/functions/fewz_bern",  # FEWZ tabulated kernel
    "FunLib/functions/johnsonsu",  # scipy-backed
    "FunLib/functions/rationalpade",  # rational function
    "FunLib/functions/variancegamma",  # special function
}
_COMBINE_MAX_NPAR = 3  # keep tests fast -- skip very high-npar entries


def test_combine_significance(entries, hist, tmpdir, combine_bin, verbose=False):
    """
    For each selected function, write workspace and run::

        combine -M Significance workspace.root -m 125

    Checks: rc=0, ``Significance:`` present in stdout, value is finite.
    """
    from FunLib.tools.mk_workspace import mk_workspace

    passed = failed = skipped = 0
    failures = []

    for entry in entries:
        fn_dir = entry.get("dir", "")
        npar = len(entry.get("initial_params", []))
        if fn_dir not in _COMBINE_DIRS or npar > _COMBINE_MAX_NPAR:
            skipped += 1
            continue

        label = _entry_label(entry)
        dir_tag = os.path.basename(fn_dir)
        ws_file = os.path.join(tmpdir, f"ws_combine_{dir_tag}_{npar}p.root")

        try:
            mk_workspace(entry, hist, ws_file)

            # Combine must find our RooPyFitFunction library to deserialise the
            # workspace.  The rootmap in the lib directory enables ROOT autoload;
            # we also set PYTHONPATH so the embedded Python can import FunLib.
            #
            # CRITICAL (macOS / Linux): libRooPyFitFunction uses -undefined
            # dynamic_lookup for Python C API symbols.  When loaded into Combine
            # (a pure C++ process with no libpython), those symbols are NULL ->
            # SIGSEGV on first evaluate().  We must preload libpython3.xx into
            # Combine's process space so the symbols are resolved at load time.
            # macOS: DYLD_INSERT_LIBRARIES   Linux: LD_PRELOAD
            from FunLib.tools.mk_workspace import _LIB_DIR, LIBPYTHON_SO

            env = os.environ.copy()
            lib_path_var = (
                "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
            )
            py_lib_dir = os.path.dirname(LIBPYTHON_SO) if LIBPYTHON_SO else ""
            # Add both the plugin lib dir and the Python lib dir to the search path
            path_parts = [
                p for p in [_LIB_DIR, py_lib_dir, env.get(lib_path_var, "")] if p
            ]
            env[lib_path_var] = ":".join(path_parts)
            # Preload libpython so Combine's embedded call to Py_Initialize works
            if LIBPYTHON_SO:
                if sys.platform == "darwin":
                    preload_var = "DYLD_INSERT_LIBRARIES"
                    existing_ins = env.get(preload_var, "")
                    env[preload_var] = (
                        f"{LIBPYTHON_SO}:{existing_ins}"
                        if existing_ins
                        else LIBPYTHON_SO
                    )
                else:
                    preload_var = "LD_PRELOAD"
                    existing_ins = env.get(preload_var, "")
                    env[preload_var] = (
                        f"{LIBPYTHON_SO}:{existing_ins}"
                        if existing_ins
                        else LIBPYTHON_SO
                    )
            # Ensure FunLib and all installed packages (scipy etc.) are importable
            # from the embedded Python inside Combine.  site.getsitepackages()
            # returns the active environment's site-packages (venv-aware on 3.9+).
            import site as _site

            _sp = getattr(_site, "getsitepackages", lambda: [])()
            existing_py = env.get("PYTHONPATH", "")
            _py_parts = [_REPO] + list(_sp) + ([existing_py] if existing_py else [])
            env["PYTHONPATH"] = ":".join(filter(None, _py_parts))

            result = subprocess.run(
                [
                    combine_bin,
                    "-M",
                    "Significance",
                    ws_file,
                    "-m",
                    "125",
                    "--cminDefaultMinimizerType",
                    "Minuit2",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=tmpdir,
                env=env,
            )

            assert (
                result.returncode == 0
            ), f"combine exited with rc={result.returncode}\n{result.stderr[-300:]}"

            # Parse significance value from stdout
            sig_val = None
            for line in result.stdout.splitlines():
                if line.strip().startswith("Significance:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            sig_val = float(parts[1])
                        except ValueError:
                            pass
            assert (
                sig_val is not None
            ), f"'Significance:' not found in combine stdout:\n{result.stdout}"
            import math as _math

            assert _math.isfinite(sig_val), f"Significance is not finite: {sig_val}"

            passed += 1
            print(f"  {label:<55} {PASS}  significance={sig_val:.3f}")

        except subprocess.TimeoutExpired:
            failed += 1
            msg = "combine timed out (>120 s)"
            failures.append((label, msg))
            print(f"  {label:<55} {FAIL}  {msg}")

        except Exception as exc:
            failed += 1
            msg = str(exc)
            failures.append((label, msg))
            print(f"  {label:<55} {FAIL}  {msg}")
            if verbose:
                traceback.print_exc()

    return passed, failed, skipped, failures


# -- main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test mk_workspace + load_workspace + CMS Combine compatibility"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all results, not just failures",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Skip the combine -M Significance tests",
    )
    args = parser.parse_args()

    import ROOT
    from FunLib.tools.mk_workspace import COMBINE_BIN

    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    entries, cfg = _load_entries()
    hist = _make_test_hist()

    total_pass = total_fail = 0

    with tempfile.TemporaryDirectory() as tmpdir:

        # ---- test 1: roundtrip --------------------------------------------
        print(
            f"\nRoundtrip test: mk_workspace + Combine objects + load_workspace"
            f"  ({len(entries)} entries)\n"
        )
        p1, f1, fail1 = test_roundtrip(entries, hist, tmpdir, verbose=args.verbose)
        if not args.verbose:
            if f1 == 0:
                print(f"  All {p1} entries  {PASS}")
            else:
                print(f"  {p1} passed, {f1} failed")
        total_pass += p1
        total_fail += f1

        # ---- test 2: fitTo smoke ------------------------------------------
        print(
            f"\nFitTo smoke test (Python-backed PDF, dirs in "
            f"{{{', '.join(os.path.basename(d) for d in sorted(_SMOKE_DIRS))}}},"
            f" npar<={_SMOKE_MAX_NPAR})\n"
        )
        p2, f2, sk2, fail2 = test_fitTo_smoke(
            entries, hist, tmpdir, verbose=args.verbose
        )
        total_pass += p2
        total_fail += f2

        # ---- test 3: combine Significance ---------------------------------
        combine_available = (
            not args.no_combine
            and os.path.isfile(COMBINE_BIN)
            and os.access(COMBINE_BIN, os.X_OK)
        )
        if combine_available:
            n_combine = sum(
                1
                for e in entries
                if e.get("dir", "") in _COMBINE_DIRS
                and len(e.get("initial_params", [])) <= _COMBINE_MAX_NPAR
            )
            print(f"\ncombine -M Significance test  ({n_combine} workspaces)\n")
            p3, f3, sk3, fail3 = test_combine_significance(
                entries, hist, tmpdir, COMBINE_BIN, verbose=args.verbose
            )
            total_pass += p3
            total_fail += f3
        else:
            p3 = f3 = sk3 = 0
            fail3 = []
            reason = (
                "--no-combine"
                if args.no_combine
                else f"binary not found: {COMBINE_BIN}"
            )
            print(f"\ncombine -M Significance test  [{SKIP}  {reason}]\n")

    # ---- summary ----------------------------------------------------------
    print(f"\n  Roundtrip:   {p1}/{p1 + f1} passed")
    print(f"  FitTo smoke: {p2}/{p2 + f2} passed  ({sk2} skipped)")
    if combine_available:
        print(f"  Combine sig: {p3}/{p3 + f3} passed  ({sk3} skipped)")
    print(
        f"\n  Results: {total_pass}/{total_pass + total_fail} passed, "
        f"{total_fail} failed\n"
    )

    all_failures = fail1 + fail2 + (fail3 if combine_available else [])
    if all_failures:
        print("  Failures:")
        for label, msg in all_failures:
            print(f"    {label}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
