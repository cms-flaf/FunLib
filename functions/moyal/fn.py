"""Moyal distribution  -- closed-form Landau approximation; Wigner function of the harmonic oscillator."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, mu, sigma):
        sigma = max(abs(sigma), self._param_min)
        z = (x - mu) / sigma
        ez = math.exp(min(-z, self._exp_max))
        return math.exp(-0.5 * (z + ez))
