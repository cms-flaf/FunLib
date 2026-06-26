# FunLib C++ kernels & RooFunLibPdf

Native C++ implementations of every FunLib fit function, mirroring the Python
`fn.py` kernels exactly (scalar `x`, positional shape params in `param_names`
order, no normalization `N`).

## Layout

| Path | Purpose |
|------|---------|
| `FunLib/functions/<dir>/fn.h` | Header-only C++ kernel (`funlib::<Name> : IKernel`) |
| `FunLib/include/<dir>.h` | Symlink → `../functions/<dir>/fn.h` |
| `FunLib/include/funlib_base.h` | `IKernel`, `Constants`, registry, `FUNLIB_REGISTER` |
| `FunLib/include/funlib_math.h` | special functions + scipy.stats PDF closed forms |
| `FunLib/include/funlib_registry.h` | `make_kernel` / `eval_kernel` |
| `FunLib/include/funlib_all.h` | includes every kernel (auto-generated list) |
| `FunLib/cpp/RooFunLibPdf.{h,cpp}` | single streamable `RooAbsPdf` for any function |
| `FunLib/cpp/funlib.cpp` | the one TU compiled into the library |

## Build

```bash
python3 FunLib/tools/build_funlib.py [--force]
# -> FunLib/tools/lib/FunLib.{so,dylib} + FunLib.rootmap + FunLib_rdict.pcm
```

All 69 kernels + `RooFunLibPdf` compile into a **single** shared library.

## Use from C++ / PyROOT

```python
import ROOT
ROOT.gSystem.Load("FunLib/tools/lib/FunLib.so")
ROOT.gInterpreter.Declare('#include "funlib_registry.h"')   # for eval_kernel
# RooFunLibPdf(name, title, x, params, dir, cnames, cvals, snames, svals, npar, xmin, xmax)
```

`RooFunLibPdf` selects the function by its **directory name** (`dir`) and is
parameterised by the streamed `constants` + the floating shape params.  One
class represents every function, is fully streamable into a `RooWorkspace`, and
autoloads from the rootmap — so it works directly with **Combine**.

## Validation

```bash
# Python kernel == C++ kernel == C++ RooAbsPdf (+ existing LaTeX/regression):
python3 FunLib/validation/validate_functions.py            # 69/69
python3 FunLib/validation/validate_functions.py --no-cpp   # skip C++ check

# FunLib.so works with Combine (REQUIRED): builds a RooWorkspace per function
# with the native RooFunLibPdf as the single process, converts it with
# text2workspace.py (--X-allow-no-background) and runs `combine -M MultiDimFit`
# with the process normalisation as the POI (r); passes iff combine converges
# to r ~= 1 on Asimov data.  Fails if `combine` is not on PATH.
python3 FunLib/validation/test_combine.py --workers 4   # 114/114
```

Notes:
- `combine` must be on `PATH` (e.g. `/Users/Kes/workspace/combine/build/bin`).
- `text2workspace.py` is run with the hep-env Python (it needs `ROOT` **and**
  `pandas`); Combine autoloads `RooFunLibPdf` from the rootmap via
  `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` pointing at `FunLib/tools/lib`.
- Shape params with a desc.yaml `roofit_param_bounds` range are bounded by it;
  the rest float unbounded (Minuit limits change the sin-transform step scales
  and hurt convergence — matching the roofit fit pipeline).
