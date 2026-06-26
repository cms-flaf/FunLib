#ifndef FUNLIB_SERSIC_H
#define FUNLIB_SERSIC_H
#include "funlib_base.h"
namespace funlib {
struct Sersic : IKernel {
    double exp_max, param_min, n_min;
    Sersic(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        param_min = c.get("param_min", 1e-9);
        n_min = c.get("n_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double b = p[0];
        double re = std::max(std::fabs(p[1]), param_min);
        double n = std::max(std::fabs(p[2]), n_min);
        double arg = -b * std::pow(x / re, 1.0 / n);
        if (arg > exp_max) arg = exp_max;
        return std::exp(arg);
    }
    std::vector<std::string> param_names() const override { return {"b", "re", "n"}; }
};
FUNLIB_REGISTER("sersic", new Sersic(c));
}
#endif
