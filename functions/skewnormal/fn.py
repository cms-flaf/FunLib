"""Skew-normal distribution  -- Gaussian asymmetrized by a shape parameter (Azzalini, 1985)."""

import math
import scipy.stats as _st


class FitFunction:

    def __call__(self, x, *, a, loc, scale):
        scale = max(abs(scale), 1e-9)
        try:
            val = _st.skewnorm.pdf(x, a, loc=loc, scale=scale)
        except Exception:
            return math.nan
        return float(val)
