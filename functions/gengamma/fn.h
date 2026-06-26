#ifndef FUNLIB_GENGAMMA_H
#define FUNLIB_GENGAMMA_H
#include "funlib_base.h"
namespace funlib {
struct GenGamma : IKernel {
    double exp_max, param_min;
    GenGamma(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        param_min = c.get("param_min", 1e-9);
    }
    double eval(double x, const double* pp, int) const override {
        double p = pp[0];
        double a = std::max(std::fabs(pp[1]), param_min);
        double d = pp[2];
        double arg = -std::pow(x / a, d);
        if (arg < -exp_max) return 0.0;
        return std::pow(x, p) * std::exp(arg);
    }
    std::vector<std::string> param_names() const override { return {"p", "a", "d"}; }
};
FUNLIB_REGISTER("gengamma", new GenGamma(c));
}
#endif
