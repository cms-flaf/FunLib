"""Planck/Bose-Einstein spectral shape  -- power-law rise cut off by a Bose-Einstein thermal factor."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])
        self._T_min = float(config["constants"]["T_min"])

    def __call__(self, x, *, p, T):
        T = max(abs(T), self._T_min)
        a = x / T
        if a > self._exp_max:
            return 0.0
        denom = math.expm1(a)  # exp(a) - 1, accurate for small a
        if denom <= 1e-300:
            return math.nan
        try:
            return x**p / denom
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
