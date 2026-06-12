"""Modified Bessel K_nu of free real order  -- generalises SumBesselK0 with a free order parameter."""

import math
import ROOT


class FitFunction:

    def __init__(self, config: dict):
        c = config["constants"]
        self.z_lo = c["z_lo"]  # K_nu(z) diverges as z->0 -- clamp to 0
        self.nu_max = c[
            "nu_max"
        ]  # cyl_bessel_k hangs for large non-integer nu -- clamp abs(nu)

    def __call__(self, x, *, nu, a):
        z = a * x
        if z <= 0.0:
            return math.nan
        if z <= self.z_lo:
            return math.inf
        nu = min(abs(nu), self.nu_max)
        val = ROOT.Math.cyl_bessel_k(nu, z)
        return val
