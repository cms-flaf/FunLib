"""Fermi-Dirac distribution  -- smooth falling step from quantum statistical mechanics."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])
        self._T_min = float(config["constants"]["T_min"])

    def __call__(self, x, *, mu, T):
        T = max(abs(T), self._T_min)
        arg = (x - mu) / T
        if arg > self._exp_max:
            return 0.0
        if arg < -self._exp_max:
            return 1.0
        return 1.0 / (math.exp(arg) + 1.0)
