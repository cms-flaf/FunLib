#ifndef FUNLIB_GENLOGISTIC_H
#define FUNLIB_GENLOGISTIC_H
#include "funlib_base.h"
namespace funlib {
struct GenLogistic : IKernel {
    double exp_max, param_min;
    GenLogistic(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        param_min = c.get("param_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double mu = p[0];
        double s = std::max(std::fabs(p[1]), param_min);
        double a = p[2];
        double z = (x - mu) / s;
        if (z > exp_max) return 0.0;
        double base = 1.0 + std::exp(z);
        return 1.0 / std::pow(base, a);
    }
    std::vector<std::string> param_names() const override { return {"mu", "s", "a"}; }
};
FUNLIB_REGISTER("genlogistic", new GenLogistic(c));
}
#endif
