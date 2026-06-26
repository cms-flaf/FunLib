#ifndef FUNLIB_HYPERBOLIC_H
#define FUNLIB_HYPERBOLIC_H
#include "funlib_base.h"
namespace funlib {
struct Hyperbolic : IKernel {
    double exp_max;
    Hyperbolic(const Constants& c) { exp_max = c.get("exp_max", 700); }
    double eval(double x, const double* p, int) const override {
        double alpha = std::fabs(p[0]);
        double delta = std::fabs(p[1]);
        double beta = p[2], mu = p[3];
        double d = x - mu;
        double arg = -alpha * std::sqrt(delta * delta + d * d) + beta * d;
        if (arg > exp_max) arg = exp_max;
        return std::exp(arg);
    }
    std::vector<std::string> param_names() const override {
        return {"alpha", "delta", "beta", "mu"};
    }
};
FUNLIB_REGISTER("hyperbolic", new Hyperbolic(c));
}
#endif
