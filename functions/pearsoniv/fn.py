"""Pearson type IV distribution  -- asymmetric power-law body with exponential tilt (spectroscopy)."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, lam, a, m, nu):
        a = max(abs(a), self._param_min)
        z = (x - lam) / a
        arg = -nu * math.atan(z)
        arg = min(arg, self._exp_max)
        try:
            return (1.0 + z * z) ** (-m) * math.exp(arg)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
