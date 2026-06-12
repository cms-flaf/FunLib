"""Generalized logistic (Richards) curve  -- asymmetric sigmoid with free inflection-point shape."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._exp_max = float(config["constants"]["exp_max"])
        self._param_min = float(config["constants"]["param_min"])

    def __call__(self, x, *, mu, s, a):
        s = max(abs(s), self._param_min)
        z = (x - mu) / s
        if z > self._exp_max:
            return 0.0
        try:
            base = 1.0 + math.exp(z)
            return 1.0 / base**a
        except OverflowError:
            return 0.0
        except ValueError:
            return math.nan
