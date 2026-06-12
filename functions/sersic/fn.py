"""Sersic profile  -- generalised galaxy surface-brightness law with free Sersic index n."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])
        self._param_min = float(config["constants"]["param_min"])
        self._n_min = float(config["constants"]["n_min"])

    def __call__(self, x, *, b, re, n):
        re = max(abs(re), self._param_min)
        n = max(abs(n), self._n_min)
        try:
            arg = -b * (x / re) ** (1.0 / n)
            arg = min(arg, self._exp_max)
            return math.exp(arg)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
