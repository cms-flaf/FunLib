"""Kummer M (1F1, confluent hypergeometric)  -- power-series ODE solution, complement to Tricomi U."""

import math
import scipy.special as _sp


class FitFunction:

    def __init__(self, config: dict):
        c = config["constants"]
        self.a_lo = c[
            "a_lo"
        ]  # hyp1f1 converges slowly for a outside [a_lo, a_hi] -- clamp
        self.a_hi = c["a_hi"]
        self.b_lo = c["b_lo"]  # b must not be 0 or a negative integer -- clamp to b_lo
        self.b_hi = c["b_hi"]  # hyp1f1 converges slowly for large b -- clamp
        self.z_abs_max = c["z_abs_max"]  # clamp |c*x| to prevent divergence

    def __call__(self, x, *, a, b, c):
        a = max(self.a_lo, min(a, self.a_hi))
        b = max(self.b_lo, min(b, self.b_hi))
        z = -c * x
        if abs(z) > self.z_abs_max:
            return math.nan
        val = _sp.hyp1f1(a, b, z)
        return float(val)
