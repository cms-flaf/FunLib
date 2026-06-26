"""Z-boson Breit-Wigner propagator with M_Z and Gamma_Z fixed from constants.

Wraps the generic breit_wigner kernel, binding the resonance mass and width
to the Z-boson values stored in constants.  The denominator form is selected
by the same three boolean constants as breit_wigner (all default False):

  is_lorentzian -- True: non-relativistic Lorentzian denominator
  is_running    -- True: x^2 numerator (running-width spin-1 form)
  free_power    -- True: the denominator exponent p is a free shape parameter
                   (adds 'p' to param_names -> npar=1); False: fixed p=2 (npar=0)

With free_power=False (default) this is a zero-parameter component kernel.
With free_power=True it is a one-parameter (p) fittable shape.
"""

from FunLib.functions.breit_wigner.fn import FitFunction as BW


class FitFunction:
    def __init__(self, *, config: dict):
        c = config["constants"]
        self._m = float(c["m"])
        self._Gamma = float(c["Gamma"])
        self._bw = BW(config={"constants": c})
        self.free_power = bool(c.get("free_power", False))
        if self.free_power:
            # desc.yaml ships param_names: [] (the DOF0 default); honour an
            # explicit non-empty override but otherwise expose 'p'.
            self.param_names = list(config.get("param_names") or ["p"])
        else:
            self.param_names = list(config.get("param_names") or [])
        self.npar = len(self.param_names)

    def __call__(self, x, *, p=2.0):
        return self._bw(x, m=self._m, Gamma=self._Gamma, p=p)
