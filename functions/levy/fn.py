"""Levy stable distribution  -- one-sided alpha=1/2 stable law with x^{-3/2} power-law right tail."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._scale_min = float(config["constants"]["scale_min"])

    def __call__(self, x, *, loc, scale):
        scale = max(abs(scale), self._scale_min)
        try:
            val = _st.levy.pdf(x, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
