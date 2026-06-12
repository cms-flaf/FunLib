"""RationalPade: ratio of two polynomials -- variable-degree Pade [m/n] approximant."""

import math


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        self.x_0 = (x_min + x_max) / 2.0
        self.x_half = (x_max - x_min) / 2.0
        c = config["constants"]
        self._num_deg = int(c["num_deg"])
        self._den_deg = int(c["den_deg"])
        self.npar = self._num_deg + self._den_deg
        self.param_names = [f"a{i}" for i in range(1, self._num_deg + 1)] + [
            f"b{i}" for i in range(1, self._den_deg + 1)
        ]

    def __call__(self, x, **kwargs):
        t = (x - self.x_0) / self.x_half
        num = 1.0 + sum(
            float(kwargs[f"a{i}"]) * t**i for i in range(1, self._num_deg + 1)
        )
        den = 1.0 + sum(
            float(kwargs[f"b{i}"]) * t**i for i in range(1, self._den_deg + 1)
        )
        if abs(den) < 1e-12:
            return math.inf
        val = num / den
        return val
