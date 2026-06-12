"""Whittaker W function  -- decaying confluent hypergeometric solution; special cases include Kummer U."""

import math
import scipy.special as _sp


class FitFunction:

    def __init__(self, config: dict):
        c = config["constants"]
        self.z_hi = c["z_hi"]  # W negligible and may overflow for z > z_hi
        self.kappa_lo = c[
            "kappa_lo"
        ]  # hyperu slow/divergent for kappa outside these bounds -- clamp
        self.kappa_hi = c["kappa_hi"]
        self.mu_lo = c["mu_lo"]  # mu < -0.5 makes z^(mu+0.5) complex -- clamp
        self.mu_hi = c["mu_hi"]

    def __call__(self, x, *, kappa, mu, a):
        z = a * x
        if z <= 0.0 or z > self.z_hi:
            return math.nan
        kappa = max(self.kappa_lo, min(kappa, self.kappa_hi))
        mu = max(self.mu_lo, min(mu, self.mu_hi))
        try:
            u = _sp.hyperu(mu - kappa + 0.5, 1.0 + 2.0 * mu, z)
            w = math.exp(-0.5 * z) * z ** (mu + 0.5) * float(u)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
        return w
