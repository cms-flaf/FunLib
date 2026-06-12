"""Landau x Bernstein-n, with either fixed MPV (landau_mpv) or free MPV (variable npar)."""

from FunLib.functions.bernstein.fn import FitFunction as Bernstein
from FunLib.functions.landau.fn import FitFunction as Landau


class FitFunction:
    FIXED_MPV = "fixed_mpv"
    FREE_MPV = "free_mpv"

    def __init__(self, *, config: dict, x_min: float, x_max: float):
        c = config["constants"]
        self.variant = c.get("variant", self.FIXED_MPV)
        n_terms = int(c["n_terms"])  # Bernstein polynomial degree

        if self.variant == self.FREE_MPV:
            self.npar = n_terms + 2  # mpv and sigma consume 2 slots
            self.param_names = ["mpv", "sigma"] + [
                f"c{i}" for i in range(1, n_terms + 1)
            ]
        else:  # fixed_mpv (default)
            self.landau_mpv = c["landau_mpv"]
            self.npar = n_terms + 1  # sigma consumes 1 slot
            self.param_names = [f"c{i}" for i in range(1, n_terms + 1)] + ["sigma"]

        self._n_terms = n_terms
        self._bern = Bernstein(config={}, x_min=x_min, x_max=x_max, npar=n_terms)
        self._landau = Landau(config={})

    def __call__(self, x, **kwargs):
        sigma = float(kwargs["sigma"])
        bern_kw = {f"c{i}": float(kwargs[f"c{i}"]) for i in range(1, self._n_terms + 1)}
        mpv = float(kwargs["mpv"]) if self.variant == self.FREE_MPV else self.landau_mpv
        return self._bern(x, **bern_kw) * self._landau(x, mpv=mpv, sigma=sigma)
