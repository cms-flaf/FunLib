"""Generic Breit-Wigner propagator with free mass and width.

Three independent boolean constants control the formula (all default False):

  is_lorentzian  -- True:  non-relativistic Lorentzian denominator (x-m)^p + (Gamma/2)^p
                    False: relativistic denominator (x^2-m^2)^p + (m*Gamma)^p
  is_running     -- True:  x^2 in numerator (running-width spin-1 form)
                    False: 1 in numerator (constant-width form)
  free_power     -- True:  exponent p is a free shape parameter (adds 'p' to param_names)
                    False: fixed exponent p=2 (standard BW denominator)
"""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        c = config.get("constants", {})
        self._Gamma_min = float(c.get("Gamma_min", 1e-9))
        self.is_lorentzian = bool(c.get("is_lorentzian", False))
        self.is_running = bool(c.get("is_running", False))
        self.free_power = bool(c.get("free_power", False))

        if self.free_power:
            self.param_names = list(config.get("param_names", ["m", "Gamma", "p"]))
        else:
            self.param_names = list(config.get("param_names", ["m", "Gamma"]))
        self.npar = len(self.param_names)

    def __call__(self, x, *, m, Gamma, p=2.0):
        # Enforce physical constraints: mass >= 0, width > 0.
        m = abs(m)
        Gamma = max(abs(Gamma), self._Gamma_min)
        try:
            if self.is_lorentzian:
                d = abs(x - m) ** p + (Gamma / 2.0) ** p
            else:
                Gm = Gamma * m
                d = abs(x * x - m * m) ** p + Gm**p
            numerator = x * x if self.is_running else 1.0
            result = numerator / d
            return result
        except (ZeroDivisionError, OverflowError):
            return math.inf
