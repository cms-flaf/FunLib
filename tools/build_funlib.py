#!/usr/bin/env python3
"""
FunLib/tools/build_funlib.py -- Compile ALL FunLib C++ kernels + the streamable
RooFunLibPdf into a single shared library.

Output (in FunLib/tools/lib/):
  FunLib.{dylib,so}        -- the shared library (all 69 kernels + RooFunLibPdf)
  FunLib.rootmap           -- ROOT autoload map (RooFunLibPdf -> FunLib.so)
  FunLib_rdict.pcm         -- ROOT reflection module

Usage
-----
    python3 FunLib/tools/build_funlib.py [--force] [--print-lib]

Then use the kernels natively from C++/PyROOT/Combine:

    import ROOT
    ROOT.gSystem.Load("FunLib/tools/lib/FunLib.so")
    # funlib::eval_kernel(...) and RooFunLibPdf are now available.
"""

import argparse
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))  # repo root
_INC = os.path.join(_HERE, "..", "include")
_CPP = os.path.join(_HERE, "..", "cpp")
_LIB_DIR = os.path.join(_HERE, "lib")
_SO_EXT = "dylib" if sys.platform == "darwin" else "so"
_STEM = "FunLib"
_SO = os.path.join(_LIB_DIR, f"{_STEM}.{_SO_EXT}")
_MAP = os.path.join(_LIB_DIR, f"{_STEM}.rootmap")
_PCM = os.path.join(_LIB_DIR, f"{_STEM}_rdict.pcm")

_SRC = os.path.join(_CPP, "funlib.cpp")
_HDR = os.path.join(_CPP, "RooFunLibPdf.h")
_LINKDEF = os.path.join(_CPP, "FunLib_LinkDef.h")
_DICT = os.path.join(_LIB_DIR, "FunLib_dict.cxx")


def _run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"\nCommand failed (rc={r.returncode})")


def _tool(name):
    for c in [os.path.join(sys.prefix, "bin", name), shutil.which(name)]:
        if c and os.path.isfile(c):
            return c
    sys.exit(f"{name} not found; activate the hep environment first.")


def _root_config(*args):
    rc = shutil.which("root-config") or os.path.join(sys.prefix, "bin", "root-config")
    return subprocess.check_output([rc, *args]).decode().strip().split()


def _cxx():
    # Use the compiler ROOT was built with so the C++ standard library and ABI
    # match what we link against (RooFit, MathMore, ...). On lxplus/LCG this is
    # the LCG gcc; on macOS it is the system clang. Picking the system clang++
    # on el9 breaks: it has no libc++ headers and mismatches the gcc-built ROOT.
    cxx_name = _root_config("--cxx")[0] if _root_config("--cxx") else ""
    if cxx_name:
        cxx = cxx_name if os.path.isabs(cxx_name) else shutil.which(cxx_name)
        if cxx:
            return cxx, "clang" in os.path.basename(cxx)
    for n in ("g++", "clang++"):
        p = shutil.which(n)
        if p:
            return p, "clang" in n
    sys.exit("No C++ compiler found.")


def _uptodate():
    if not (os.path.isfile(_SO) and os.path.isfile(_MAP)):
        return False
    lib_m = os.path.getmtime(_SO)
    srcs = [_SRC, _HDR, _LINKDEF]
    # every fn.h + base headers
    for root, _, files in os.walk(os.path.join(_ROOT_DIR, "FunLib", "functions")):
        for f in files:
            if f == "fn.h":
                srcs.append(os.path.join(root, f))
    for f in ("funlib_base.h", "funlib_math.h", "funlib_registry.h", "funlib_all.h"):
        srcs.append(os.path.join(_INC, f))
    return all(os.path.getmtime(s) < lib_m for s in srcs if os.path.isfile(s))


def compile(force=False):
    if not force and _uptodate():
        print(f"FunLib library up-to-date: {_SO}")
        return _SO
    os.makedirs(_LIB_DIR, exist_ok=True)
    root_inc = _root_config("--incdir")[0]
    root_libdir = _root_config("--libdir")[0]
    root_cflags = _root_config("--cflags")
    root_libs = _root_config("--libs")

    print(f"\n[build_funlib] Building {_SO}\n")
    print("[1/2] Generating ROOT dictionary (RooFunLibPdf)...")
    _run(
        [
            _tool("rootcling"),
            "-f",
            _DICT,
            "-s",
            _SO,
            "--rmf",
            _MAP,
            "--rml",
            f"{_STEM}.{_SO_EXT}",
            f"-I{_CPP}",
            f"-I{_INC}",
            f"-I{root_inc}",
            _HDR,
            _LINKDEF,
        ]
    )
    raw_pcm = os.path.join(_LIB_DIR, "FunLib_dict_rdict.pcm")
    if os.path.isfile(raw_pcm) and raw_pcm != _PCM:
        shutil.move(raw_pcm, _PCM)

    print("[2/2] Compiling shared library...")
    cxx, is_clang = _cxx()
    print(f"[build_funlib] Compiler: {cxx}")
    common = [
        # libc++ is the default stdlib for clang on macOS; on Linux the system
        # clang usually lacks it, and we compile with gcc there anyway.
        *(["-stdlib=libc++"] if (is_clang and sys.platform == "darwin") else []),
        "-std=c++17",
        "-O2",
        "-fPIC",
        f"-I{_CPP}",
        f"-I{_INC}",
        f"-I{root_inc}",
        *root_cflags,
    ]
    if sys.platform == "darwin":
        link = [
            "-dynamiclib",
            "-install_name",
            f"@rpath/{_STEM}.{_SO_EXT}",
            f"-Wl,-rpath,{root_libdir}",
        ]
    else:
        link = ["-shared", f"-Wl,-rpath,{root_libdir}"]
    _run(
        [
            cxx,
            *common,
            *link,
            _SRC,
            _DICT,
            f"-L{root_libdir}",
            *root_libs,
            "-lMathMore",
            "-lRooFit",
            "-lRooFitCore",
            "-o",
            _SO,
        ]
    )
    print(f"\nBuilt:  {_SO}\nMap:    {_MAP}")
    return _SO


def ensure_compiled(force=False):
    return compile(force=force)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print-lib", action="store_true")
    args = ap.parse_args()
    if args.print_lib:
        print(_SO)
        return
    compile(force=args.force)


if __name__ == "__main__":
    main()
