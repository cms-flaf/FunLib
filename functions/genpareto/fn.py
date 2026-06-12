"""Generalized Pareto distribution  -- peaks-over-threshold tail with free tail-index parameter."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])
        self._param_min = float(config["constants"]["param_min"])
        self._xi_eps = float(config["constants"]["xi_eps"])

    def __call__(self, x, *, mu, sigma, xi):
        sigma = max(abs(sigma), self._param_min)
        z = (x - mu) / sigma
        if abs(xi) < self._xi_eps:  # exponential limit
            return math.exp(-z) if -z < self._exp_max else 0.0
        base = 1.0 + xi * z
        if base <= 0.0:
            return math.nan
        try:
            return base ** (-1.0 / xi - 1.0)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
