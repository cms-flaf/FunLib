"""Johnson S_U distribution  -- sinh^(-1)-transformed normal covering the full skewness-kurtosis plane."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, a, b, loc, scale):
        b = max(abs(b), self._param_min)
        scale = max(abs(scale), self._param_min)
        try:
            val = _st.johnsonsu.pdf(x, a, b, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
