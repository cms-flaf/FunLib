"""BWZ x Bernstein-n: Z propagator x Bernstein-n x exponential slope (variable Bernstein degree)."""

import math
from FunLib.functions.bernstein.fn import FitFunction as Bernstein
from FunLib.functions.bwz.fn import FitFunction as BW


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        c = config["constants"]
        n_terms = int(c["n_terms"])  # Bernstein polynomial degree
        self.npar = n_terms + 1  # n_terms Bernstein coeffs + exponential slope a
        self.param_names = [f"c{i}" for i in range(1, n_terms + 1)] + ["a"]
        self._exp_max = float(c["exp_max"])
        self._bern = Bernstein(config={}, x_min=x_min, x_max=x_max, npar=n_terms)
        self._bwz = BW(config={"constants": c})
        self._n_terms = n_terms

    def __call__(self, x, **kwargs):
        a = float(kwargs["a"])
        exp_arg = min(a * x, self._exp_max)
        bern_kw = {f"c{i}": float(kwargs[f"c{i}"]) for i in range(1, self._n_terms + 1)}
        try:
            return self._bwz(x) * self._bern(x, **bern_kw) * math.exp(exp_arg)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
