#ifndef FUNLIB_GENPARETO_H
#define FUNLIB_GENPARETO_H
#include "funlib_base.h"
namespace funlib {
struct GenPareto : IKernel {
    double exp_max, param_min, xi_eps;
    GenPareto(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        param_min = c.get("param_min", 1e-9);
        xi_eps = c.get("xi_eps", 1e-8);
    }
    double eval(double x, const double* p, int) const override {
        double mu = p[0];
        double sigma = std::max(std::fabs(p[1]), param_min);
        double xi = p[2];
        double z = (x - mu) / sigma;
        if (std::fabs(xi) < xi_eps) return (-z < exp_max) ? std::exp(-z) : 0.0;
        double base = 1.0 + xi * z;
        if (base <= 0.0) return kNaN;
        return std::pow(base, -1.0 / xi - 1.0);
    }
    std::vector<std::string> param_names() const override {
        return {"mu", "sigma", "xi"};
    }
};
FUNLIB_REGISTER("genpareto", new GenPareto(c));
}
#endif
