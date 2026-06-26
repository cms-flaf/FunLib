// Chebyshev expansion (c0=1 fixed). Mirror of chebyshev/fn.py.
#ifndef FUNLIB_CHEBYSHEV_H
#define FUNLIB_CHEBYSHEV_H
#include "funlib_base.h"
namespace funlib {
struct Chebyshev : IKernel {
    double x_min, x_max;
    int n;
    Chebyshev(double xmin, double xmax, int npar)
        : x_min(xmin), x_max(xmax), n(npar) {}
    double eval(double x, const double* p, int) const override {
        double u = 2.0 * (x - x_min) / (x_max - x_min) - 1.0;
        double T_prev = 1.0, T_curr = u, val = 1.0;
        for (int i = 1; i <= n; i++) {
            val += p[i - 1] * T_curr;
            double T_next = 2.0 * u * T_curr - T_prev;
            T_prev = T_curr;
            T_curr = T_next;
        }
        return val;
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        for (int i = 1; i <= n; i++) v.push_back("c" + std::to_string(i));
        return v;
    }
};
FUNLIB_REGISTER("chebyshev", new Chebyshev(xmin, xmax, npar));
}
#endif
