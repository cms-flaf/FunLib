"""VoigtZ: Voigt Z line-shape tail (Z-propagator conv Gaussian) x quadratic exponential correction."""

import math
import ROOT


class FitFunction:
    def __init__(self, *, config: dict, x_min: float, x_max: float):
        c = config["constants"]
        self.m_z = c["M_Z"]
        self.g_z = c["G_Z"]
        self._exp_max = float(c["exp_max"])
        self._sigma_min = float(c["sigma_min"])
        self.x_0 = (x_min + x_max) / 2.0
        self.x_half = (x_max - x_min) / 2.0

    def __call__(self, x, *, sigma, a, b):
        sigma = max(abs(sigma), self._sigma_min)
        voigt = ROOT.TMath.Voigt(x - self.m_z, sigma, self.g_z)
        xn = (x - self.x_0) / self.x_half
        arg = a * xn + b * xn**2
        arg = min(arg, self._exp_max)
        return voigt * math.exp(arg)
