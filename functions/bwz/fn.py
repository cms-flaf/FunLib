"""Z-boson Breit-Wigner propagator with M_Z and Gamma_Z fixed from constants.

Wraps the generic breit_wigner kernel, binding the resonance mass and width
to the Z-boson values stored in constants.  Zero free shape parameters; used
as a component kernel.
"""

from FunLib.functions.breit_wigner.fn import FitFunction as BW


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._m = float(c["m"])
        self._Gamma = float(c["Gamma"])
        self._bw = BW(config={"constants": c})

    def __call__(self, x, *, p=2.0):
        return self._bw(x, m=self._m, Gamma=self._Gamma, p=p)
