"""Doniach-Sunjic line shape  -- asymmetric XPS core-level photoemission profile."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._gamma_min = float(config["constants"]["gamma_min"])

    def __call__(self, x, *, x0, gamma, alpha):
        g = max(abs(gamma), self._gamma_min)
        al = alpha
        d = x - x0
        try:
            num = math.cos(math.pi * al / 2.0 + (1.0 - al) * math.atan(d / g))
            den = (g * g + d * d) ** ((1.0 - al) / 2.0)
            return num / den
        except (OverflowError, ZeroDivisionError):
            return math.inf
        except ValueError:
            return math.nan
