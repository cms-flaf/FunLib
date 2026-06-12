"""ExpPoly: exp of a degree-n polynomial in the normalised mass variable (variable npar)."""

import math


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float, npar: int):
        self.x_0 = (x_min + x_max) / 2.0
        self.x_half = (x_max - x_min) / 2.0
        self.npar = int(npar)
        self.param_names = [f"c{i}" for i in range(1, self.npar + 1)]
        self._exp_max = float(config["constants"]["exp_max"])

    def __call__(self, x, **kwargs):
        xn = (x - self.x_0) / self.x_half
        arg = sum(float(kwargs[f"c{i}"]) * xn**i for i in range(1, self.npar + 1))
        arg = min(arg, self._exp_max)
        return math.exp(arg)
