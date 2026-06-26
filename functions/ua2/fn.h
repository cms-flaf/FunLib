#ifndef FUNLIB_UA2_H
#define FUNLIB_UA2_H
#include "funlib_base.h"
namespace funlib {
struct UA2 : IKernel {
    double sqrt_s, exp_max;
    UA2(const Constants& c) {
        sqrt_s = std::sqrt(c.get("s", 1.0));
        exp_max = c.get("exp_max", 700);
    }
    double eval(double x, const double* p, int) const override {
        double p1 = p[0], p2 = p[1], p3 = p[2];
        double y = x / sqrt_s;
        double arg = p2 * y + p3 * y * y;
        if (arg > exp_max) arg = exp_max;
        return std::pow(y, p1) * std::exp(arg);
    }
    std::vector<std::string> param_names() const override {
        return {"p1", "p2", "p3"};
    }
};
FUNLIB_REGISTER("ua2", new UA2(c));
}
#endif
