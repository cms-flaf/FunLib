"""Frechet (inverse-Weibull) distribution  -- Type-II extreme value with power-law right tail."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._s_min = float(config["constants"]["s_min"])

    def __call__(self, x, *, s, alpha):
        s = max(abs(s), self._s_min)
        u = x / s
        try:
            return u ** (-1.0 - alpha) * math.exp(-(u ** (-alpha)))
        except OverflowError:
            return 0.0
        except ValueError:
            return math.nan
