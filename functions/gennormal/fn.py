"""Generalized normal (Subbotin) distribution  -- exponential power, interpolates Laplace and Gaussian."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._beta_min = float(config["constants"]["beta_min"])
        self._scale_min = float(config["constants"]["scale_min"])

    def __call__(self, x, *, beta, loc, scale):
        beta = max(abs(beta), self._beta_min)
        scale = max(abs(scale), self._scale_min)
        try:
            val = _st.gennorm.pdf(x, beta, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
