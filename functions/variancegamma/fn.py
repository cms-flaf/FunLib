"""Variance-Gamma (generalised Laplace) distribution  -- Brownian motion at Gamma stopping time."""

import math
import scipy.special as _sp


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._exp_max = float(c["exp_max"])
        self._alpha_min = float(c["alpha_min"])
        self._z_min = float(c["z_min"])

    def __call__(self, x, *, lam, alpha, beta, mu):
        lam, alpha, beta, mu = lam, max(abs(alpha), self._alpha_min), beta, mu
        z = x - mu
        az = max(abs(z), self._z_min)
        arg = alpha * az
        try:
            k = _sp.kv(lam - 0.5, arg)
            ex = beta * z
            if ex > self._exp_max:
                return math.inf
            val = az ** (lam - 0.5) * float(k) * math.exp(ex)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
        return val
