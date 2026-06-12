"""Weibull (stretched-exponential) distribution  -- flexible reliability model with free hazard-rate shape."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._exp_max = float(c["exp_max"])
        self._lam_min = float(c["lam_min"])

    def __call__(self, x, *, lam, k):
        lam = max(abs(lam), self._lam_min)
        t = x / lam
        try:
            arg = -(t**k)
            if arg < -self._exp_max:
                return 0.0
            return t ** (k - 1.0) * math.exp(arg)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
