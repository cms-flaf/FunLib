"""Moffat profile  -- power-law-tailed atmospheric-seeing PSF (Kolmogorov turbulence model)."""

import math


class FitFunction:

    def __call__(self, x, *, alpha, beta):
        al = max(abs(alpha), 1e-9)
        u = x / al
        try:
            return (1.0 + u * u) ** (-beta)
        except OverflowError:
            return 0.0
        except ValueError:
            return math.nan
