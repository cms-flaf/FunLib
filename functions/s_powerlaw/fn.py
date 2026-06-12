"""S-Powerlaw: sum of n_terms power-law terms (2*n_terms-1 shape parameters)."""

import math


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        c = config["constants"]
        n_terms = int(c["n_terms"])
        self._n_terms = n_terms
        self.x_mid = (x_min + x_max) / 2.0
        self.npar = 2 * n_terms - 1
        if n_terms == 1:
            self.param_names = ["b1"]
        else:
            names = ["b1"]
            for k in range(2, n_terms + 1):
                names += [f"A{k}", f"b{k}"]
            self.param_names = names

    def __call__(self, x, **kwargs):
        xn = x / self.x_mid
        try:
            total = math.pow(xn, float(kwargs["b1"]))
            for k in range(2, self._n_terms + 1):
                total += float(kwargs[f"A{k}"]) * math.pow(xn, float(kwargs[f"b{k}"]))
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
        return total
