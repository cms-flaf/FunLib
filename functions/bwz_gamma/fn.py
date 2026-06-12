"""BWZGamma: Z + gamma* mixture with exponential slope.

Formula (arXiv:2009.04363 / AN-19-090, relativistic BW form):
  f(x) = exp(a * x') * [fZ * BW_norm(x) + (1 - fZ) * (x_mid / x)^2]

where:
  x'        = (x - x_mid) / x_half        (normalised mass)
  BW_norm(x) = BW(x) / BW(x_mid)          (BW normalised to 1 at range midpoint)
  fZ        in (0, 1)                     (Z fraction; 1-fZ is gamma* fraction)

The BW form is selected by the same boolean constants as breit_wigner / bwz:
  is_running=False (default): fixed-width relativistic BW in denominator
  is_running=True:            running-width relativistic BW (x^2 numerator)
  is_lorentzian=True:         non-relativistic Lorentzian denominator

In all variants the two free shape parameters are (a, fZ).
"""

import math

from FunLib.functions.bwz.fn import FitFunction as BW


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        c = config["constants"]
        self._bwz = BW(config={"constants": c})
        self.x_0 = (x_min + x_max) / 2.0
        self.x_half = (x_max - x_min) / 2.0
        self._exp_max = float(c["exp_max"])
        bwz0 = self._bwz(self.x_0)
        self._bwz0 = bwz0 if bwz0 != 0.0 else 1.0

    def __call__(self, x, *, a, fZ):
        xn = (x - self.x_0) / self.x_half
        exp_arg = a * xn
        exp_arg = min(exp_arg, self._exp_max)
        bw_n = self._bwz(x) / self._bwz0  # Z term, normalised to 1 at x_0
        gamma_n = (self.x_0 / x) ** 2  # gamma* term, normalised to 1 at x_0
        fZ_c = max(0.0, min(1.0, fZ))  # guard against out-of-bounds optimiser steps
        val = fZ_c * bw_n + (1.0 - fZ_c) * gamma_n
        if val <= 0.0:
            return math.nan
        return math.exp(exp_arg) * val
