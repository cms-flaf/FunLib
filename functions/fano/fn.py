"""Fano resonance profile  -- quantum interference of a discrete state with a continuum."""


class FitFunction:
    def __init__(self, *, config: dict):
        self._Gamma_min = float(config["constants"]["Gamma_min"])

    def __call__(self, x, *, x0, Gamma, q):
        G = max(abs(Gamma), self._Gamma_min)
        eps = (x - x0) / G
        return (q + eps) ** 2 / (1.0 + eps * eps)
