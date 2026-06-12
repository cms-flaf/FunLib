"""ExpRational: exp of a Pade-rational [m/n] function (variable degree)."""

import math

from FunLib.functions.rationalpade.fn import FitFunction as RationalPade


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        self._pade = RationalPade(config=config, x_min=x_min, x_max=x_max)
        self._exp_max = float(config["constants"]["exp_max"])
        self._num_deg = self._pade._num_deg
        self._den_deg = self._pade._den_deg
        self.npar = self._pade.npar
        self.param_names = list(self._pade.param_names)

    def __call__(self, x, **kwargs):
        arg = self._pade(x, **kwargs)
        arg = min(arg, self._exp_max)
        return math.exp(arg)
