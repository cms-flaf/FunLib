"""Laurent series: (x_max/x)^p x (1 + sum_{i=1}^{n} c_i*(x_max/x)^i)."""

import math


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        self.x_max = x_max
        n_terms = int(config["constants"]["n_terms"])  # degree of polynomial correction
        self.npar = n_terms + 1  # leading power p + n_terms correction coefficients
        self.param_names = ["p"] + [f"c{i}" for i in range(1, n_terms + 1)]
        self._n_terms = n_terms

    def __call__(self, x, **kwargs):
        r = self.x_max / x
        p = float(kwargs["p"])
        try:
            poly = 1.0 + sum(
                float(kwargs[f"c{i}"]) * r**i for i in range(1, self._n_terms + 1)
            )
            return r**p * poly
        except OverflowError:
            return math.inf
        except ValueError:
            return math.nan
