#!/usr/bin/env python3
"""
FunLib/tools/compile_plugin.py
Compile RooPyFitFunction.cxx into a ROOT-loadable shared library.

Output (in FunLib/tools/lib/):
  libRooPyFitFunction.{dylib,so}   -- shared library
  libRooPyFitFunction.rootmap      -- ROOT autoload map
  libRooPyFitFunction_rdict.pcm    -- ROOT reflection module

The rootmap enables ROOT to autoload the library whenever it encounters the
class name ``RooPyFitFunction`` in a ROOT file -- so Combine does not need an
explicit ``gSystem->Load()`` call, as long as the lib directory is in
DYLD_LIBRARY_PATH (macOS) or LD_LIBRARY_PATH (Linux).

Usage
-----
    # Compile (or recompile if sources changed):
    python3 FunLib/tools/compile_plugin.py

    # Force recompile:
    python3 FunLib/tools/compile_plugin.py --force

    # Print library path (for scripting):
    python3 FunLib/tools/compile_plugin.py --print-lib

Then run Combine with the library in its search path:

    DYLD_LIBRARY_PATH=FunLib/tools/lib:$DYLD_LIBRARY_PATH \\
        combine -M Significance workspace.root -m 125
"""

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_HERE, "lib")
_SO_EXT = "dylib" if sys.platform == "darwin" else "so"
_LIB_STEM = "libRooPyFitFunction"
_SO = os.path.join(_LIB_DIR, f"{_LIB_STEM}.{_SO_EXT}")
_MAP = os.path.join(_LIB_DIR, f"{_LIB_STEM}.rootmap")
_PCM = os.path.join(_LIB_DIR, f"{_LIB_STEM}_rdict.pcm")

_SRC_H = os.path.join(_HERE, "RooPyFitFunction.h")
_SRC_CXX = os.path.join(_HERE, "RooPyFitFunction.cxx")
_LINKDEF = os.path.join(_HERE, "RooPyFitFunction_LinkDef.h")
_DICT_CXX = os.path.join(_LIB_DIR, "RooPyFitFunction_dict.cxx")


# -- utilities -----------------------------------------------------------------


def _run(cmd, *, check=True, **kwargs):
    """Run a command, print it, raise on failure."""
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, **kwargs)
    if check and r.returncode != 0:
        sys.exit(f"\nCommand failed (rc={r.returncode})")
    return r


def _rootcling():
    """Return full path to rootcling executable."""
    for candidate in [
        os.path.join(sys.prefix, "bin", "rootcling"),
        shutil.which("rootcling"),
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    sys.exit("rootcling not found.  Activate the hep conda environment first.")


def _cxx_compiler():
    """Return (path, is_clang) for the best available C++ compiler.

    Prefers clang++ (macOS / LLVM), falls back to g++ (Linux / GCC).
    """
    for name in ("clang++", "g++"):
        path = shutil.which(name)
        if path:
            return path, (name == "clang++")
    for candidate in [os.path.join(sys.prefix, "bin", "clang++"), "/usr/bin/g++"]:
        if os.path.isfile(candidate):
            return candidate, ("clang" in candidate)
    sys.exit("No C++ compiler found (tried clang++, g++).")


def _root_config(*args):
    """Call root-config and return output as a list of tokens."""
    rc = shutil.which("root-config") or os.path.join(sys.prefix, "bin", "root-config")
    out = subprocess.check_output([rc, *args]).decode().strip()
    return out.split()


def _is_uptodate():
    """Return True iff the library is newer than both source files."""
    if not os.path.isfile(_SO) or not os.path.isfile(_MAP):
        return False
    lib_mtime = os.path.getmtime(_SO)
    for src in (_SRC_H, _SRC_CXX):
        if os.path.getmtime(src) >= lib_mtime:
            return False
    return True


# -- public API ----------------------------------------------------------------


def ensure_compiled(*, force: bool = False) -> str:
    """
    Compile the plugin if needed.  Returns the path to the shared library.
    Safe to call from mk_workspace.py at import time.
    """
    if not force and _is_uptodate():
        return _SO
    compile(force=force)
    return _SO


def compile(*, force: bool = False):  # noqa: A001
    """Compile RooPyFitFunction into FunLib/tools/lib/."""
    if not force and _is_uptodate():
        print(f"Library is up-to-date: {_SO}")
        return

    os.makedirs(_LIB_DIR, exist_ok=True)

    # ---- Python paths -------------------------------------------------------
    py_inc = sysconfig.get_path("include")  # e.g. .../include/python3.11
    py_lib_dir = sysconfig.get_config_var("LIBDIR") or os.path.join(sys.prefix, "lib")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    # On macOS conda, there's a .dylib; on Linux there may be only a .a.
    # Link against the .dylib/.so if available, otherwise fall back.
    py_lib_name = f"python{py_ver}"

    # ---- ROOT paths ---------------------------------------------------------
    root_inc = _root_config("--incdir")[0]
    root_libdir = _root_config("--libdir")[0]
    root_cflags = _root_config("--cflags")
    root_libs = _root_config("--libs")

    print(f"\n[compile_plugin] Building {_SO}\n")

    # ---- Step 1: generate ROOT dictionary + rootmap -------------------------
    print("[1/2] Generating ROOT dictionary...")
    _run(
        [
            _rootcling(),
            "-f",
            _DICT_CXX,
            "-s",
            _SO,
            "--rmf",
            _MAP,
            "--rml",
            f"{_LIB_STEM}.{_SO_EXT}",
            f"-I{_HERE}",
            f"-I{root_inc}",
            f"-I{py_inc}",
            _SRC_H,
            _LINKDEF,
        ]
    )

    # rootcling writes the PCM alongside _DICT_CXX; rename to final location.
    raw_pcm = os.path.join(_LIB_DIR, f"RooPyFitFunction_dict_rdict.pcm")
    if os.path.isfile(raw_pcm) and raw_pcm != _PCM:
        shutil.move(raw_pcm, _PCM)

    # ---- Step 2: compile shared library -------------------------------------
    print("[2/2] Compiling shared library...")

    compiler, is_clang = _cxx_compiler()
    print(f"[compile_plugin] Compiler: {compiler}")

    common_flags = [
        # -stdlib=libc++ is Clang-specific; GCC uses libstdc++ by default
        *(["-stdlib=libc++"] if is_clang else []),
        "-std=c++17",
        "-O2",
        "-fPIC",
        f"-I{_HERE}",
        f"-I{root_inc}",
        f"-I{py_inc}",
        *root_cflags,
    ]

    # Python symbols: do NOT link libpython explicitly.
    # When loaded into an already-running Python process (from Python itself or
    # from Combine which ROOT initialised), we must use the Python runtime that
    # is already in memory -- not spin up a second one (causes heap corruption).
    # On macOS: -undefined dynamic_lookup lets the linker leave Python symbols
    # unresolved at link time; they bind to the running interpreter at load time.
    # On Linux: omitting -lpython achieves the same; LD_PRELOAD supplies libpython
    # when running combine (a pure C++ binary).
    if sys.platform == "darwin":
        link_mode = [
            "-dynamiclib",
            "-install_name",
            f"@rpath/{_LIB_STEM}.{_SO_EXT}",
            f"-Wl,-rpath,{root_libdir}",
            "-undefined",
            "dynamic_lookup",
        ]
    else:
        link_mode = ["-shared", f"-Wl,-rpath,{root_libdir}", f"-Wl,-rpath,{py_lib_dir}"]

    _run(
        [
            compiler,
            *common_flags,
            *link_mode,
            _SRC_CXX,
            _DICT_CXX,
            f"-L{root_libdir}",
            *root_libs,
            "-lRooFit",
            "-lRooFitCore",
            "-o",
            _SO,
        ]
    )

    print(f"\nBuilt:  {_SO}")
    print(f"Map:    {_MAP}")
    print(f"\nTo use with Combine add the lib dir to the library path:")
    if sys.platform == "darwin":
        print(f"  export DYLD_LIBRARY_PATH={_LIB_DIR}:$DYLD_LIBRARY_PATH")
    else:
        print(f"  export LD_LIBRARY_PATH={_LIB_DIR}:$LD_LIBRARY_PATH")


# -- main ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--force", action="store_true", help="Recompile even if up-to-date")
    ap.add_argument(
        "--print-lib", action="store_true", help="Print library path and exit"
    )
    args = ap.parse_args()

    if args.print_lib:
        print(_SO)
        return

    compile(force=args.force)


if __name__ == "__main__":
    main()
