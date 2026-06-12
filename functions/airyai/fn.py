"""Airy function Ai  -- solution of the Airy equation y'' = x*y; super-exponential decaying tail."""

import math
import ROOT


class FitFunction:

    def __init__(self, config: dict):
        c = config["constants"]
        self.arg_lo = c["arg_lo"]  # Ai oscillates for arg < arg_lo -- clamp to 0
        self.arg_hi = c["arg_hi"]  # Ai underflows to ~0 for arg > arg_hi -- clamp to 0

    def __call__(self, x, *, a, b):
        arg = a * x + b
        if arg < self.arg_lo or arg > self.arg_hi:
            return math.nan
        val = ROOT.Math.airy_Ai(arg)
        return val
