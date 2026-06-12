"""GammaTail: Gamma-distribution-shaped falling tail."""

import math


class FitFunction:

    def __call__(self, x, *, a, b, x_s):
        dx = x - x_s
        if dx <= 0.0:
            return 0.0
        return math.pow(dx, a) * math.exp(-b * x)
