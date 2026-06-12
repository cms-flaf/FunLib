"""Hagedorn/Tsallis-Pareto QCD power law  -- thermal-to-power-law transition in hadronic spectra."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._p0_min = float(config["constants"]["p0_min"])

    def __call__(self, x, *, p0, n):
        p0 = max(abs(p0), self._p0_min)
        base = 1.0 + x / p0
        if base <= 0.0:
            return math.nan
        try:
            return base ** (-n)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
