"""SplineBern: tabulated theory kernel x Bernstein-n (variable Bernstein degree).

Supports two variants selected via ``constants["variant"]`` (default ``"fewz"``):

  fewz     -- FEWZ NLO dsigma/dm spline (fewz_smooth.csv, 0.1 GeV steps).
               CMS Run 2 H->mumu primary baseline (CMS-HIG-19-006).
  dyturbo  -- DYTurbo N^3LO dsigma/dm spline (dyturbo_smooth.csv, 0.01 GeV steps).
               Higher-order theory alternative (Camarda et al. 2020).

npar = Bernstein degree n (set via initial_params length in functions.yaml).
"""

import math
import os

from FunLib.functions.bernstein.fn import FitFunction as Bernstein

_CSV_FILES = {
    "fewz": "fewz_smooth.csv",
    "dyturbo": "dyturbo_smooth.csv",
}


class FitFunction:
    _spline_cache: dict = {}

    def __init__(self, *, config: dict, x_min: float, x_max: float, npar: int):
        c = config.get("constants", {})
        variant = str(c.get("variant", "fewz"))
        if variant not in _CSV_FILES:
            raise ValueError(
                f"Unknown SplineBern variant {variant!r}; choose from {list(_CSV_FILES)}"
            )
        self.npar = int(npar)
        self.param_names = [f"c{i}" for i in range(1, self.npar + 1)]
        self._bern = Bernstein(config={}, x_min=x_min, x_max=x_max, npar=self.npar)
        _here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(_here, _CSV_FILES[variant])
        self._spline = self._load_spline(csv_path)

    @classmethod
    def _load_spline(cls, csv_path: str):
        if csv_path in cls._spline_cache:
            return cls._spline_cache[csv_path]
        import csv as _csv
        from scipy.interpolate import CubicSpline

        rows = list(_csv.DictReader(open(csv_path)))
        xs = [float(r["mass"]) for r in rows]
        ys = [float(r["sigma"]) for r in rows]
        spl = CubicSpline(xs, ys, extrapolate=False)
        cls._spline_cache[csv_path] = spl
        return spl

    def __call__(self, x, **kwargs):
        spline_val = float(self._spline(x))
        if not math.isfinite(spline_val):
            return spline_val
        if spline_val <= 0.0:
            return math.nan
        bern_kw = {f"c{i}": float(kwargs[f"c{i}"]) for i in range(1, self.npar + 1)}
        return spline_val * self._bern(x, **bern_kw)
