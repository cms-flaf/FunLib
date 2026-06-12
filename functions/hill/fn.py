"""Hill equation  -- cooperative dose-response sigmoid from biochemistry and pharmacology."""

import math


class FitFunction:
    def __init__(self, *, config: dict):
        self._K_min = float(config["constants"]["K_min"])

    def __call__(self, x, *, K, h):
        K = max(abs(K), self._K_min)
        u = x / K
        try:
            return 1.0 / (1.0 + u**h)
        except OverflowError:
            return 0.0
        except ValueError:
            return math.nan
