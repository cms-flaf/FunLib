"""Burr type XII (Singh-Maddala) distribution  -- flexible heavy-tailed reliability model."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._exp_max = float(c["exp_max"])
        self._log1p_cutoff = float(c["log1p_cutoff"])
        self._param_min = float(c["param_min"])

    def __call__(self, x, *, s, c, k):
        s = max(abs(s), self._param_min)
        c = max(abs(c), self._param_min)
        k = max(abs(k), self._param_min)
        u = x / s
        if u <= 0.0:
            return math.nan
        log_u = math.log(u)
        log_uc = c * log_u
        # log(1 + u^c): stable for large log_uc where exp(log_uc) would overflow
        log1p_uc = (
            log_uc if log_uc > self._log1p_cutoff else math.log1p(math.exp(log_uc))
        )
        ln_val = math.log(c * k) + (c - 1.0) * log_u - (k + 1.0) * log1p_uc
        return math.exp(ln_val) if ln_val > -self._exp_max else 0.0
