"""BWZ x exp(quad): fixed-width Z propagator x quadratic exponential correction."""

import math
from FunLib.functions.bwz.fn import FitFunction as BW


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        c = config["constants"]
        self.x_0 = (x_min + x_max) / 2.0
        self.x_half = (x_max - x_min) / 2.0
        self._exp_max = float(c["exp_max"])
        self._bwz = BW(config={"constants": c})

    def __call__(self, x, *, a, b):
        xn = (x - self.x_0) / self.x_half
        arg = a * xn + b * xn * xn
        arg = min(arg, self._exp_max)
        return self._bwz(x) * math.exp(arg)
