"""Standard CMS Bernstein polynomial (c0=1 fixed, npar free coefficients)."""

from math import comb


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float, npar: int):
        self.x_min = x_min
        self.x_max = x_max
        self.npar = int(npar)
        self.param_names = [f"c{i}" for i in range(1, self.npar + 1)]
        self._binom = [comb(self.npar, k) for k in range(self.npar + 1)]

    def __call__(self, x, **kwargs):
        t = (x - self.x_min) / (self.x_max - self.x_min)
        n = self.npar
        val = 0.0
        for k in range(n + 1):
            ck = 1.0 if k == 0 else float(kwargs[self.param_names[k - 1]])
            val += ck * self._binom[k] * (t**k) * ((1.0 - t) ** (n - k))
        return max(val, 0.0)
