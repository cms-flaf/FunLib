"""SumBesselK0: sum of n_terms K0 modified Bessel terms (2*n_terms-1 shape parameters)."""

import math

import ROOT


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        c = config["constants"]
        n_terms = int(c["n_terms"])
        self._n_terms = n_terms
        self.npar = 2 * n_terms - 1
        if n_terms == 1:
            self.param_names = ["a1"]
        else:
            names = ["a1"]
            for k in range(2, n_terms + 1):
                names += [f"R{k}", f"a{k}"]
            self.param_names = names

    def _k0(self, a, x):
        arg = a * x
        if arg <= 0.0:
            return math.nan
        if arg < 1e-6:
            return math.inf
        return ROOT.TMath.BesselK0(arg)

    def __call__(self, x, **kwargs):
        try:
            total = self._k0(float(kwargs["a1"]), x)
            for k in range(2, self._n_terms + 1):
                total += float(kwargs[f"R{k}"]) * self._k0(float(kwargs[f"a{k}"]), x)
        except (OverflowError, ValueError):
            return math.nan
        return total
