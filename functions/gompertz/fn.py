"""Gompertz distribution  -- double-exponential (Gumbel-type) survival law from mortality modelling."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, c, loc, scale):
        c = max(abs(c), self._param_min)
        scale = max(abs(scale), self._param_min)
        try:
            val = _st.gompertz.pdf(x, c, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
