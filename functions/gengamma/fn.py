"""Generalized Gamma (Stacy) distribution  -- three-shape-parameter family unifying Gamma, Weibull, exponential."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, p, a, d):
        a = max(abs(a), self._param_min)
        try:
            arg = -((x / a) ** d)
            if arg < -self._exp_max:
                return 0.0
            return x**p * math.exp(arg)
        except OverflowError:
            return 0.0
        except ValueError:
            return math.nan
