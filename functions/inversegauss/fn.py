"""Inverse Gaussian (Wald) distribution  -- Brownian first-passage-time density with heavy right tail."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, mu, loc, scale):
        mu = max(abs(mu), self._param_min)
        scale = max(abs(scale), self._param_min)
        try:
            val = _st.invgauss.pdf(x, mu, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
