"""Complementary error function  -- Gaussian tail integral, smooth falling step."""

import ROOT


class FitFunction:
    def __init__(self, *, config: dict):
        self._sigma_min = float(config["constants"]["sigma_min"])

    def __call__(self, x, *, mu, sigma):
        sigma = max(abs(sigma), self._sigma_min)
        return ROOT.TMath.Erfc((x - mu) / sigma)
