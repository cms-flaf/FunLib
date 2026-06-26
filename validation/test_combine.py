#!/usr/bin/env python3
"""
test_combine.py -- verify FunLib.{so,dylib} works with Combine for every
registered function.

Combine is REQUIRED: if the `combine` binary is not on PATH the test fails.

For each entry in all_functions.yaml this test:

  1. Builds a RooWorkspace whose single process is the NATIVE C++ RooFunLibPdf
     (the function under test, no Python at fit time), plus an Asimov
     RooDataHist generated from that shape.  Shape parameters that have a
     desc.yaml ``roofit_param_bounds`` range are bounded by it (so a fit cannot
     drive them into a divergent / unphysical region); the rest float unbounded,
     exactly as in the roofit fit pipeline.
  2. Converts the datacard with `text2workspace.py` (`--X-allow-no-background`)
     and runs `combine -M MultiDimFit` with the process NORMALISATION as the POI
     (`r`).  Combine autoloads RooFunLibPdf from the rootmap (exactly how a real
     analysis uses a custom PDF).  The test passes iff Combine converges
     (rc == 0) to a finite r ~= 1 (the Asimov truth) with no NaN/Inf.

Usage:
  python3 FunLib/validation/test_combine.py                # all functions
  python3 FunLib/validation/test_combine.py --key BWZ      # one function
  python3 FunLib/validation/test_combine.py --workers 4
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_LIB_DIR = os.path.normpath(os.path.join(_HERE, "..", "tools", "lib"))
_INCLUDE = os.path.normpath(os.path.join(_HERE, "..", "include"))
_CONFIG = os.path.normpath(
    os.path.join(_HERE, "..", "applications", "h_mumu", "all_functions.yaml")
)
_R_TOL = 0.05  # |r - 1| tolerance for the Asimov normalisation POI


def _load_entries(key=None):
    """Return picklable task tuples (entry, xmin, xmax, integral)."""
    import yaml

    cfg = yaml.safe_load(open(_CONFIG))
    fr = cfg.get("fit_range", {})
    xmin = float(fr.get("min", 110.0))
    xmax = float(fr.get("max", 150.0))
    integral = float(cfg.get("reference_integral", 100000.0))
    entries = cfg["functions"]
    out = [(e, xmin, xmax, integral) for e in entries]
    if key:
        from FunLib.tools.factory import load_kernel

        filt = []
        for t in out:
            kern = load_kernel(t[0], repo_root=_REPO, xmin=xmin, xmax=xmax)
            if kern.key == key:
                filt.append(t)
        out = filt
    return out


def _combine_env():
    env = dict(os.environ)
    key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    env[key] = _LIB_DIR + os.pathsep + env.get(key, "")
    env["ROOT_INCLUDE_PATH"] = _INCLUDE + os.pathsep + env.get("ROOT_INCLUDE_PATH", "")
    return env


def _build_workspace(entry, kern, xmin, xmax, integral, out_path):
    """Build + write a RooWorkspace whose single process is a RooFunLibPdf."""
    import ROOT
    from FunLib.validation.cpp_bridge import _ensure, build_funlib_pdf

    _ensure()
    ROOT.gErrorIgnoreLevel = ROOT.kError

    dir_name = os.path.basename(entry["dir"].rstrip("/"))
    funcdir = os.path.normpath(os.path.join(_REPO, entry["dir"]))
    constants = dict(kern.yaml_config.get("constants", {}))
    param_names = list(kern.param_names)
    inits = list(kern.initial_params)
    npar = len(inits)
    pbounds = kern.yaml_config.get("roofit_param_bounds") or {}

    nbins = 80
    x = ROOT.RooRealVar("x", "m_{#mu#mu}", xmin, xmax)
    x.setBins(nbins)

    pdf, shape_rvs = build_funlib_pdf(
        ROOT,
        x,
        dir_name,
        constants,
        xmin,
        xmax,
        npar,
        inits,
        funcdir=funcdir,
        param_names=param_names,
        tag="",
        param_bounds=pbounds,
    )
    pdf.SetName("bkg")
    pdf.SetTitle("bkg")

    # The shape must be positive somewhere over the range.
    probe = 0.0
    for i in range(nbins):
        x.setVal(xmin + (i + 0.5) * (xmax - xmin) / nbins)
        probe = max(probe, pdf.getVal(ROOT.RooArgSet(x)))
    if not (probe > 0.0):
        return None, "model is non-positive over the whole range"

    # Asimov data: histogram of the shape, scaled to the reference yield.
    h = pdf.createHistogram("h_asimov", x)
    if h.Integral() <= 0.0:
        return None, "pdf integral <= 0"
    h.Scale(integral / h.Integral())
    data = ROOT.RooDataHist("data_obs", "data_obs", ROOT.RooArgList(x), h)

    ws = ROOT.RooWorkspace("w", "w")
    getattr(ws, "import")(pdf)
    getattr(ws, "import")(data)
    ws.writeToFile(out_path)
    # keep shape_rvs referenced until after writeToFile
    del shape_rvs
    return ("w", "bkg", "data_obs"), None


def _datacard(ws_path, integral):
    return f"""# FunLib combine test datacard (single process, normalisation = POI r)
imax 1
jmax 0
kmax *
---------------
shapes bkg      bin1 {ws_path} w:bkg
shapes data_obs bin1 {ws_path} w:data_obs
---------------
bin bin1
observation {integral:.1f}
---------------
bin          bin1
process      bkg
process      0
rate         {integral:.1f}
---------------
"""


def _run_combine(ws_path, integral, tmpdir):
    """Convert + run combine MultiDimFit with the yield as POI.  Returns (ok, msg)."""
    combine = shutil.which("combine")
    if not combine:
        return False, "combine binary not found on PATH (required)"
    t2w = os.path.join(os.path.dirname(combine), "text2workspace.py")
    if not os.path.isfile(t2w):
        return False, f"text2workspace.py not found next to combine ({t2w})"

    env = _combine_env()
    card = os.path.join(tmpdir, "datacard.txt")
    with open(card, "w") as f:
        f.write(_datacard(ws_path, integral))
    card_ws = os.path.join(tmpdir, "card_ws.root")

    # text2workspace.py needs ROOT + pandas: run it with THIS interpreter (the
    # hep env python), not combine's default `/usr/bin/env python3`.
    r1 = subprocess.run(
        [sys.executable, t2w, card, "-o", card_ws, "--X-allow-no-background"],
        env=env,
        cwd=tmpdir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r1.returncode != 0 or not os.path.isfile(card_ws):
        tail = (r1.stdout + r1.stderr).strip().splitlines()
        return False, "text2workspace failed: " + (tail[-1] if tail else "?")

    r2 = subprocess.run(
        [combine, "-M", "MultiDimFit", card_ws, "--algo", "singles"],
        env=env,
        cwd=tmpdir,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = r2.stdout + "\n" + r2.stderr
    if r2.returncode != 0:
        line = next((l for l in out.splitlines() if "error" in l.lower()), "")
        return False, f"combine rc={r2.returncode} {line[:80]}"
    m = re.search(r"\br\s*:\s*([+-]?[0-9.eEnaN]+)", out)
    if not m:
        return False, "combine produced no best-fit r"
    try:
        rval = float(m.group(1))
    except ValueError:
        return False, f"combine r non-numeric ({m.group(1)})"
    if not math.isfinite(rval):
        return False, f"combine r non-finite ({rval})"
    if abs(rval - 1.0) > _R_TOL:
        return False, f"combine r={rval:.4f} not ~1 (Asimov truth)"
    return True, f"combine MultiDimFit OK, r={rval:.4f}"


def test_one(args_tuple):
    entry, xmin, xmax, integral = args_tuple
    from FunLib.tools.factory import load_kernel

    kern = load_kernel(entry, repo_root=_REPO, xmin=xmin, xmax=xmax)
    key = kern.key
    tmpdir = tempfile.mkdtemp(prefix="funlib_combine_")
    try:
        ws_path = os.path.join(tmpdir, "ws.root")
        info, err = _build_workspace(entry, kern, xmin, xmax, integral, ws_path)
        if info is None:
            return key, "fail", f"workspace build: {err}"
        ok, msg = _run_combine(ws_path, integral, tmpdir)
        return key, ("pass" if ok else "fail"), msg
    except Exception as e:
        return key, "fail", f"exception: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--key", default=None, help="Test only the named function")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default 1; one combine subprocess per test)",
    )
    args = ap.parse_args()

    if not shutil.which("combine"):
        sys.exit("FAIL: `combine` binary not found on PATH (the test requires it)")

    # Build the library once up front.
    from FunLib.tools.build_funlib import ensure_compiled

    ensure_compiled()

    entries = _load_entries(args.key)
    if not entries:
        sys.exit(f"No functions matched key={args.key!r}")

    results = {}
    if args.workers <= 1:
        for t in entries:
            k, st, m = test_one(t)
            results[k] = (st, m)
            print(f"  {k:30s}  {st.upper():4s}  {m}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(test_one, t): i for i, t in enumerate(entries)}
            for fut in as_completed(futs):
                k, st, m = fut.result()
                results[k] = (st, m)
                print(f"  {k:30s}  {st.upper():4s}  {m}")

    n_pass = sum(1 for st, _ in results.values() if st == "pass")
    n = len(results)
    print(f"\n  Results: {n_pass}/{n} pass, {n - n_pass} fail")
    if n_pass != n:
        sys.exit(1)


if __name__ == "__main__":
    main()
