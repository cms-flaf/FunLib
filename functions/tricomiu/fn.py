"""Tricomi U (confluent hypergeometric)  -- decaying ODE solution regular at infinity; complements Kummer M."""

import math
import ROOT


class FitFunction:

    def __init__(self, config: dict):
        c = config["constants"]
        self.a_lo = c["a_lo"]  # conf_hypergU hangs for a outside [a_lo, a_hi] -- clamp
        self.a_hi = c["a_hi"]
        self.b_lo = c["b_lo"]  # conf_hypergU hangs for b outside [b_lo, b_hi] -- clamp
        self.b_hi = c["b_hi"]

    def __call__(self, x, *, a, b, c):
        z = c * x
        if z <= 0.0:
            return math.nan
        a = max(self.a_lo, min(a, self.a_hi))
        b = max(self.b_lo, min(b, self.b_hi))
        val = ROOT.Math.conf_hypergU(a, b, z)
        return val
