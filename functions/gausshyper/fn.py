"""Gauss hypergeometric 2F1  -- most general classical special function, solution of the hypergeometric ODE."""

import math
import scipy.special as _sp


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        self.x_0 = (x_min + x_max) / 2.0
        c = config["constants"]
        self._ab_bound = float(c["ab_bound"])
        self._c_min = float(c["c_min"])
        self._c_max = float(c["c_max"])

    def __call__(self, x, *, a, b, c):
        # Clamp a, b, c: hyp2f1(a,b;c;z) converges slowly for large |a|,|b|
        # c must not be 0 or negative integer (poles of the function)
        a = max(-self._ab_bound, min(a, self._ab_bound))
        b = max(-self._ab_bound, min(b, self._ab_bound))
        c = max(self._c_min, min(c, self._c_max))
        try:
            val = _sp.hyp2f1(a, b, c, -x / self.x_0)
        except Exception:
            return math.nan
        return float(val)
