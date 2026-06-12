"""Exponentially-Modified Gaussian (EMG)  -- Gaussian convolved with a one-sided exponential."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, K, loc, scale):
        K = max(abs(K), self._param_min)
        scale = max(abs(scale), self._param_min)
        try:
            val = _st.exponnorm.pdf(x, K, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
