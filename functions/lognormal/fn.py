"""Log-normal distribution  -- normal in log(x); multiplicative-process central-limit result."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._sigma_min = float(config["constants"]["sigma_min"])

    def __call__(self, x, *, mu, sigma):
        sigma = max(abs(sigma), self._sigma_min)
        z = (math.log(x) - mu) / sigma
        return (mu / x) * math.exp(-0.5 * z * z)
