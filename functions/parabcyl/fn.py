"""Parabolic cylinder function D_nu  -- solution of the quantum harmonic-oscillator (Weber) equation."""

import math
import scipy.special as _sp


class FitFunction:

    def __init__(self, config: dict):
        c = config["constants"]
        self.z_abs_max = c[
            "z_abs_max"
        ]  # pbdv segfaults (SIGSEGV) for |z| > z_abs_max -- clamp to 0
        self.nu_lo = c["nu_lo"]  # pbdv segfaults for nu outside [nu_lo, nu_hi] -- clamp
        self.nu_hi = c["nu_hi"]

    def __call__(self, x, *, nu, a, b):
        z = a * x + b
        if abs(z) > self.z_abs_max:
            return math.nan
        nu = max(self.nu_lo, min(nu, self.nu_hi))
        d = float(_sp.pbdv(nu, z)[0])  # pbdv returns (D_v, D_v')
        return d
