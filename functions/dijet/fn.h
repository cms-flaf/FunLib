#ifndef FUNLIB_DIJET_H
#define FUNLIB_DIJET_H
#include "funlib_base.h"
namespace funlib {
struct Dijet : IKernel {
    double sqrt_s, exp_max;
    int n;
    Dijet(const Constants& c, int npar) {
        sqrt_s = std::sqrt(c.get("s", 1.0));
        exp_max = c.get("exp_max", 700);
        n = npar > 0 ? npar : 3;
    }
    double eval(double x, const double* p, int) const override {
        double y = x / sqrt_s;
        if (!(y > 0.0 && y < 1.0)) return kNaN;
        double lny = std::log(y);
        double expo = p[1] + p[2] * lny;  // p2 + p3*lny
        for (int k = 4; k <= n; k++) expo += p[k - 1] * std::pow(lny, k - 2);
        double arg = p[0] * std::log(1.0 - y) - expo * lny;
        if (arg > exp_max) arg = exp_max;
        return std::exp(arg);
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        for (int i = 1; i <= n; i++) v.push_back("p" + std::to_string(i));
        return v;
    }
};
FUNLIB_REGISTER("dijet", new Dijet(c, npar));
}
#endif
