"""BWZRedux: Z propagator with free denominator exponent + polynomial exp numerator."""

import math
from FunLib.functions.bwz.fn import FitFunction as BW


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._bwz = BW(config={"constants": c})
        n_terms = int(c["n_terms"])
        self.npar = n_terms + 1  # n_terms polynomial coeffs + free exponent p
        self.param_names = [f"c{i}" for i in range(1, n_terms + 1)] + ["p"]
        self._n_terms = n_terms
        self._exp_max = float(c["exp_max"])

    def __call__(self, x, **kwargs):
        exp_arg = sum(
            float(kwargs[f"c{i}"]) * x**i for i in range(1, self._n_terms + 1)
        )
        exp_arg = min(exp_arg, self._exp_max)
        return math.exp(exp_arg) * self._bwz(x, p=float(kwargs["p"]))
