"""Chebyshev polynomial expansion of degree n (variable npar)."""


class FitFunction:
    def __init__(self, *, x_min: float, x_max: float, npar: int):
        self.x_min = x_min
        self.x_max = x_max
        self.npar = int(npar)
        self.param_names = [f"c{i}" for i in range(1, self.npar + 1)]

    def __call__(self, x, **kwargs):
        u = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        # Chebyshev recurrence: T_0=1, T_1=u, T_{k+1}=2u*T_k-T_{k-1}
        T_prev = 1.0  # T_0
        T_curr = u  # T_1
        val = 1.0  # constant term (c_0 = 1, fixed)
        for i in range(1, self.npar + 1):
            val += float(kwargs[f"c{i}"]) * T_curr
            T_next = 2.0 * u * T_curr - T_prev
            T_prev, T_curr = T_curr, T_next
        return val
