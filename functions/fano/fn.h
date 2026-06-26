// Fano resonance. Mirror of fano/fn.py.
#ifndef FUNLIB_FANO_H
#define FUNLIB_FANO_H
#include "funlib_base.h"
namespace funlib {
struct Fano : IKernel {
    double Gamma_min;
    Fano(const Constants& c) { Gamma_min = c.get("Gamma_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double x0 = p[0];
        double G = std::max(std::fabs(p[1]), Gamma_min);
        double q = p[2];
        double eps = (x - x0) / G;
        return (q + eps) * (q + eps) / (1.0 + eps * eps);
    }
    std::vector<std::string> param_names() const override {
        return {"x0", "Gamma", "q"};
    }
};
FUNLIB_REGISTER("fano", new Fano(c));
}
#endif
