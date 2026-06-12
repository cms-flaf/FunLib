"""Mittag-Leffler function E_alpha  -- fractional-calculus generalisation of the exponential."""

import math


def _mittag_leffler(a, z, series_limit, lt_cutoff):
    az = abs(z)
    if az < 1e-12:
        return 1.0
    if az <= series_limit:  # convergent alternating series
        s = 0.0
        for k in range(0, 130):
            try:
                lt = k * math.log(az) - math.lgamma(a * k + 1.0)
            except (ValueError, OverflowError):
                break
            if lt < -lt_cutoff and k > 5:
                continue
            t = math.exp(lt)
            s += t if (z > 0 or k % 2 == 0) else -t
        return s
    # asymptotic expansion for z -> -inf : E_alpha(z) ~ -Sum_{k=1}^{4} z^{-k}/Gamma(1-alphak)
    s = 0.0
    for k in range(1, 5):
        g = 1.0 - a * k
        try:
            inv = 1.0 / math.gamma(g)
        except (ValueError, OverflowError, ZeroDivisionError):
            # gamma underflows to 0.0 for large |negative| non-integer args
            inv = 0.0
        s -= inv * z ** (-k)
    return s


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._series_limit = float(c["series_limit"])
        self._lt_cutoff = float(c["lt_cutoff"])
        self._alpha_min = float(c["alpha_min"])
        self._tau_min = float(c["tau_min"])

    def __call__(self, x, *, alpha, tau):
        a = max(abs(alpha), self._alpha_min)
        tau = max(abs(tau), self._tau_min)
        try:
            val = _mittag_leffler(a, -x / tau, self._series_limit, self._lt_cutoff)
        except (OverflowError, ZeroDivisionError):
            return math.inf
        except ValueError:
            return math.nan
        return val
