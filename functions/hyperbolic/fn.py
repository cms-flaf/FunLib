"""Hyperbolic distribution  -- generalized-hyperbolic member (lambda=1), used in finance and turbulence."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])

    def __call__(self, x, *, alpha, delta, beta, mu):
        d = x - mu
        alpha = abs(alpha)
        delta = abs(delta)
        arg = -alpha * math.sqrt(delta * delta + d * d) + beta * d
        arg = min(arg, self._exp_max)
        return math.exp(arg)
