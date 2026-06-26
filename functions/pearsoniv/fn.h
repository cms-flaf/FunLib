#ifndef FUNLIB_PEARSONIV_H
#define FUNLIB_PEARSONIV_H
#include "funlib_base.h"
namespace funlib {
struct PearsonIV : IKernel {
    double exp_max, param_min;
    PearsonIV(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        param_min = c.get("param_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double lam = p[0];
        double a = std::max(std::fabs(p[1]), param_min);
        double m = p[2], nu = p[3];
        double z = (x - lam) / a;
        double arg = -nu * std::atan(z);
        if (arg > exp_max) arg = exp_max;
        return std::pow(1.0 + z * z, -m) * std::exp(arg);
    }
    std::vector<std::string> param_names() const override {
        return {"lam", "a", "m", "nu"};
    }
};
FUNLIB_REGISTER("pearsoniv", new PearsonIV(c));
}
#endif
