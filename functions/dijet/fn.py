"""Dijet: QCD power-law background (variable number of logarithmic correction terms)."""

import math


class FitFunction:
    def __init__(self, *, config: dict, npar: int = 3):
        c = config["constants"]
        self.sqrt_s = math.sqrt(c["s"])
        self.exp_max = c["exp_max"]
        self.npar = int(npar)
        # param_names: p1, p2, p3[, p4, p5, ...]
        self.param_names = [f"p{i}" for i in range(1, self.npar + 1)]

    def __call__(self, x, **kwargs):
        y = x / self.sqrt_s
        if not (0.0 < y < 1.0):
            return math.nan
        lny = math.log(y)
        # Build the exponent: -(p2 + p3*lny + p4*lny^2 + ...)
        expo = float(kwargs["p2"]) + float(kwargs["p3"]) * lny
        for k in range(4, self.npar + 1):
            expo += float(kwargs[f"p{k}"]) * lny ** (k - 2)
        arg = float(kwargs["p1"]) * math.log(1.0 - y) - expo * lny
        arg = min(arg, self.exp_max)
        return math.exp(arg)
