"""UA2 function  -- classic QCD dijet/continuum background from the UA2 collaboration."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self.sqrt_s = math.sqrt(c["s"])
        self._exp_max = float(c["exp_max"])

    def __call__(self, x, *, p1, p2, p3):
        y = x / self.sqrt_s
        arg = p2 * y + p3 * y * y
        arg = min(arg, self._exp_max)
        try:
            return y**p1 * math.exp(arg)
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
