"""Incomplete gamma function: upper Q(s,bx) or lower P(s,bx) variant."""

import math
import scipy.special as _sp


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._s_min = float(c["s_min"])
        self._variant = str(c.get("variant", "upper"))

    def __call__(self, x, *, s, b):
        s = max(abs(s), self._s_min)
        z = b * x
        if z <= 0.0:
            return math.nan
        try:
            if self._variant == "lower":
                val = _sp.gammainc(s, z)
            else:
                val = _sp.gammaincc(s, z)
        except Exception:
            return math.nan
        return float(val)
