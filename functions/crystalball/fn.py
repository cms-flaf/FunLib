"""Crystal Ball function -- three variants.

Selected via ``constants["variant"]`` (default: ``"right_sided"``):

  right_sided  -- reversed / mirrored form: power-law tail on the HIGH-mass
                  side (t > |alpha|).  Useful for background modelling when the
                  mean is placed below the fit range so that the entire range
                  sits in the slowly-falling power-law region.

  left_sided   -- standard Crystal Ball (T. Skwarnicki, DESY thesis 1986):
                  power-law tail on the LOW-mass side (t <= -|alpha|).  Matches
                  Wikipedia, SciPy ``crystalball``, and ROOT ``RooCBShape``.
                  Used to model bremsstrahlung energy-loss tails on signal
                  peaks.

  double_sided -- Double Crystal Ball (DSCB): independent power-law tails on
                  BOTH sides of the Gaussian core.  Parameters:
                  mean, sigma, alpha_l, n_l, alpha_r, n_r  (npar=6).
                  Ref: Kenzie, LHCb PhD thesis (CERN-THESIS-2012-039);
                  ATLAS arXiv:1207.7214.

All three variants are continuous and C^1 at every transition point.
"""

import math


def _cb_tail(t: float, a: float, n: float, sign: int) -> float:
    """Crystal Ball power-law tail.

    Parameters
    ----------
    t    : reduced variable (x - mean) / sigma
    a    : |alpha| -- transition point in units of sigma
    n    : tail exponent
    sign : +1 for the right-side tail  A * (B + t)^{-n}
           -1 for the left-side tail   A * (B - t)^{-n}

    With B = n/|a| - |a| the formula is A * (B + sign*t)^{-n}.
    Continuity at t = sign*|a| follows from A * (n/|a|)^{-n} = exp(-|a|^2/2).
    """
    try:
        A = (n / a) ** n * math.exp(-0.5 * a * a)
        B = n / a - a
        arg = B + sign * t
        if arg <= 0.0:
            return math.nan
        val = A * arg ** (-n)
    except (OverflowError, ZeroDivisionError):
        return math.inf
    except ValueError:
        return math.nan
    return val


class FitFunction:
    def __init__(self, *, config: dict):
        c = config.get("constants", {})
        self._param_min = float(c.get("param_min", 1e-9))
        self.variant = str(c.get("variant", "right_sided"))
        if self.variant in ("left_sided", "right_sided"):
            self.npar = 4
            self.param_names = ["mean", "sigma", "alpha", "n"]
        elif self.variant == "double_sided":
            self.npar = 6
            self.param_names = ["mean", "sigma", "alpha_l", "n_l", "alpha_r", "n_r"]
        else:
            raise ValueError(
                f"Unknown CrystalBall variant {self.variant!r}; "
                "choose from 'left_sided', 'right_sided', 'double_sided'"
            )

    def __call__(self, x, **kwargs):
        if self.variant == "right_sided":
            sigma = max(abs(kwargs["sigma"]), self._param_min)
            a = max(abs(kwargs["alpha"]), self._param_min)
            t = (x - kwargs["mean"]) / sigma
            if t <= a:
                return math.exp(-0.5 * t * t)
            return _cb_tail(t, a, kwargs["n"], sign=+1)

        if self.variant == "left_sided":
            sigma = max(abs(kwargs["sigma"]), self._param_min)
            a = max(abs(kwargs["alpha"]), self._param_min)
            t = (x - kwargs["mean"]) / sigma
            if t > -a:
                return math.exp(-0.5 * t * t)
            return _cb_tail(t, a, kwargs["n"], sign=-1)

        # double_sided
        sigma = max(abs(kwargs["sigma"]), self._param_min)
        aL = max(abs(kwargs["alpha_l"]), self._param_min)
        aR = max(abs(kwargs["alpha_r"]), self._param_min)
        t = (x - kwargs["mean"]) / sigma
        if t < -aL:
            return _cb_tail(t, aL, kwargs["n_l"], sign=-1)
        if t > aR:
            return _cb_tail(t, aR, kwargs["n_r"], sign=+1)
        return math.exp(-0.5 * t * t)
