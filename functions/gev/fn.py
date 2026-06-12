"""Generalized Extreme Value distribution  -- unifies Gumbel, Frechet, and Weibull limits."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._param_min = float(c["param_min"])
        self._xi_eps = float(c["xi_eps"])

    def __call__(self, x, *, mu, sigma, xi):
        sigma = max(abs(sigma), self._param_min)
        z = (x - mu) / sigma
        if abs(xi) < self._xi_eps:  # Gumbel limit
            t = math.exp(-z)
            return t * math.exp(-t)
        base = 1.0 + xi * z
        if base <= 0.0:
            return math.nan
        tinv = base ** (-1.0 / xi)
        return base ** (-1.0) * tinv * math.exp(-tinv)
