#ifndef FUNLIB_MOYAL_H
#define FUNLIB_MOYAL_H
#include "funlib_base.h"
namespace funlib {
struct Moyal : IKernel {
    double exp_max, param_min;
    Moyal(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        param_min = c.get("param_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double mu = p[0];
        double sigma = std::max(std::fabs(p[1]), param_min);
        double z = (x - mu) / sigma;
        double ez = std::exp(std::min(-z, exp_max));
        return std::exp(-0.5 * (z + ez));
    }
    std::vector<std::string> param_names() const override { return {"mu", "sigma"}; }
};
FUNLIB_REGISTER("moyal", new Moyal(c));
}
#endif
