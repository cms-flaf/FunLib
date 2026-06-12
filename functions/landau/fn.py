"""Landau PDF: ROOT TMath.Landau(x, mpv, sigma, normalised=True)."""

import ROOT


class FitFunction:

    def __init__(self, *, config: dict):
        c = config.get("constants", {})
        self._param_min = float(c.get("param_min", 1.0e-9))

    def __call__(self, x, *, mpv, sigma):
        sigma = max(abs(sigma), self._param_min)
        return float(ROOT.TMath.Landau(x, mpv, sigma, True))
