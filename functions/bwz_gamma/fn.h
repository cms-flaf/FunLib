#ifndef FUNLIB_BWZ_GAMMA_H
#define FUNLIB_BWZ_GAMMA_H
#include "bwz.h"
namespace funlib {
struct BWZGamma : IKernel {
    BWZ bwz;
    double x0, xhalf, exp_max, bwz0;
    BWZGamma(const Constants& c, double xmin, double xmax) : bwz(c) {
        x0 = (xmin + xmax) / 2.0;
        xhalf = (xmax - xmin) / 2.0;
        exp_max = c.get("exp_max", 700);
        double b0 = bwz.call(x0);
        bwz0 = (b0 != 0.0) ? b0 : 1.0;
    }
    double eval(double x, const double* p, int) const override {
        double xn = (x - x0) / xhalf;
        double exp_arg = std::min(p[0] * xn, exp_max);
        double bw_n = bwz.call(x) / bwz0;
        double gamma_n = (x0 / x) * (x0 / x);
        double fZ_c = std::max(0.0, std::min(1.0, p[1]));
        double val = fZ_c * bw_n + (1.0 - fZ_c) * gamma_n;
        if (val <= 0.0) return kNaN;
        return std::exp(exp_arg) * val;
    }
    std::vector<std::string> param_names() const override { return {"a", "fZ"}; }
};
FUNLIB_REGISTER("bwz_gamma", new BWZGamma(c, xmin, xmax));
}
#endif
