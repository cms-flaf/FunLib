"""S-Exp: sum of n_terms exponential terms (2*n_terms-1 shape parameters)."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._exp_max = float(c["exp_max"])
        n_terms = int(c["n_terms"])
        self._n_terms = n_terms  # exposed as {n_terms} factory template variable
        self.npar = 2 * n_terms - 1
        if n_terms == 1:
            self.param_names = ["b1"]
        else:
            names = ["b1"]
            for k in range(2, n_terms + 1):
                names += [f"A{k}", f"b{k}"]
            self.param_names = names

    def __call__(self, x, **kwargs):
        try:
            total = math.exp(min(float(kwargs["b1"]) * x, self._exp_max))
            for k in range(2, self._n_terms + 1):
                total += float(kwargs[f"A{k}"]) * math.exp(
                    min(float(kwargs[f"b{k}"]) * x, self._exp_max)
                )
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
        return total
