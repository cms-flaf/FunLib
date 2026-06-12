"""Inverse-Gamma distribution  -- conjugate prior for Gaussian variance; power-law with exponential cutoff."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, a, loc, scale):
        a = max(abs(a), self._param_min)
        scale = max(abs(scale), self._param_min)
        try:
            val = _st.invgamma.pdf(x, a, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
