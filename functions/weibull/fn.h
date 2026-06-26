#ifndef FUNLIB_WEIBULL_H
#define FUNLIB_WEIBULL_H
#include "funlib_base.h"
namespace funlib {
struct Weibull : IKernel {
    double exp_max, lam_min;
    Weibull(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        lam_min = c.get("lam_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double lam = std::max(std::fabs(p[0]), lam_min);
        double k = p[1];
        double t = x / lam;
        double arg = -std::pow(t, k);
        if (arg < -exp_max) return 0.0;
        return std::pow(t, k - 1.0) * std::exp(arg);
    }
    std::vector<std::string> param_names() const override { return {"lam", "k"}; }
};
FUNLIB_REGISTER("weibull", new Weibull(c));
}
#endif
