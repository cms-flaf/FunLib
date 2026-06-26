// Bernstein polynomial (c0=1 fixed, npar free coeffs). Mirror of bernstein/fn.py.
#ifndef FUNLIB_BERNSTEIN_H
#define FUNLIB_BERNSTEIN_H
#include "funlib_base.h"
namespace funlib {
struct Bernstein : IKernel {
    double x_min, x_max;
    int n;
    std::vector<double> binom;
    Bernstein(double xmin, double xmax, int npar)
        : x_min(xmin), x_max(xmax), n(npar) {
        binom.resize(n + 1);
        binom[0] = 1.0;
        for (int k = 1; k <= n; k++) binom[k] = binom[k - 1] * (n - k + 1) / k;
    }
    double call(double x, const double* c) const {
        double t = (x - x_min) / (x_max - x_min);
        double val = 0.0;
        for (int k = 0; k <= n; k++) {
            double ck = (k == 0) ? 1.0 : c[k - 1];
            val += ck * binom[k] * std::pow(t, k) * std::pow(1.0 - t, n - k);
        }
        return val > 0.0 ? val : 0.0;
    }
    double eval(double x, const double* p, int) const override { return call(x, p); }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        for (int i = 1; i <= n; i++) v.push_back("c" + std::to_string(i));
        return v;
    }
};
FUNLIB_REGISTER("bernstein", new Bernstein(xmin, xmax, npar));
}  // namespace funlib
#endif
