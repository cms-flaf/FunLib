"""Normal-Inverse-Gaussian distribution  -- Brownian motion at inverse-Gaussian stopping time; Levy process."""

import math
import scipy.stats as _st


class FitFunction:
    def __init__(self, *, config: dict):
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, a, b, loc, scale):
        a = max(abs(a), self._param_min)
        b = a * math.tanh(b)  # smooth reparametrisation; tanh keeps |b|<a always
        scale = max(abs(scale), self._param_min)
        try:
            val = _st.norminvgauss.pdf(x, a, b, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
