"""Butterworth filter magnitude response  -- maximally-flat low-pass roll-off from signal processing."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._xc_min = float(config["constants"]["xc_min"])

    def __call__(self, x, *, xc, k):
        xc = max(abs(xc), self._xc_min)
        u = x / xc
        if u <= 0.0:
            return math.nan
        log_u2k = 2.0 * k * math.log(u)
        # log(1 + u^(2k)): stable for large log_u2k where exp(log_u2k) would overflow
        log1p_u2k = log_u2k if log_u2k > 30.0 else math.log1p(math.exp(log_u2k))
        return math.exp(-0.5 * log1p_u2k)
