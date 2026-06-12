"""Exponentiated Weibull  -- two-shape-parameter generalisation of the Weibull distribution."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, a, c, loc, scale):
        a = max(abs(a), self._param_min)
        c = max(abs(c), self._param_min)
        scale = max(abs(scale), self._param_min)
        try:
            val = _st.exponweib.pdf(x, a, c, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
