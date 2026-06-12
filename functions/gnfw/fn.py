"""Generalized NFW dark-matter halo profile  -- doubly-broken power law with free inner/outer slopes."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._rs_min = float(config["constants"]["rs_min"])

    def __call__(self, x, *, rs, gamma, beta):
        rs = max(abs(rs), self._rs_min)
        u = x / rs
        try:
            return 1.0 / (u**gamma * (1.0 + u) ** (beta - gamma))
        except OverflowError:
            return 0.0
        except ZeroDivisionError:
            return math.inf
        except ValueError:
            return math.nan
