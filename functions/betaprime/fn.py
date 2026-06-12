"""Beta-prime (inverted beta) distribution  -- power-law body x power-law tail."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, a, b, loc, scale):
        a = max(abs(a), self._param_min)
        b = max(abs(b), self._param_min)
        scale = max(abs(scale), self._param_min)
        val = _st.betaprime.pdf(x, a, b, loc=loc, scale=scale)
        return float(val) if math.isfinite(val) else 0.0
