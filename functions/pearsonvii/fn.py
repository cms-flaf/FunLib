"""Pearson type VII (generalised Lorentzian)  -- controllable tail exponent between Cauchy and Gaussian."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._sigma_min = float(config["constants"]["sigma_min"])

    def __call__(self, x, *, mu, sigma, m):
        sigma = max(abs(sigma), self._sigma_min)
        z = (x - mu) / sigma
        try:
            return (1.0 + z * z) ** (-m)
        except OverflowError:
            return 0.0
        except ValueError:
            return math.nan
