#ifndef FUNLIB_BWZ_EXP_H
#define FUNLIB_BWZ_EXP_H
#include "bwz.h"
namespace funlib {
struct BWZExp : IKernel {
    BWZ bwz;
    double x0, xhalf, exp_max;
    BWZExp(const Constants& c, double xmin, double xmax) : bwz(c) {
        x0 = (xmin + xmax) / 2.0;
        xhalf = (xmax - xmin) / 2.0;
        exp_max = c.get("exp_max", 700);
    }
    double eval(double x, const double* p, int) const override {
        double xn = (x - x0) / xhalf;
        double arg = p[0] * xn + p[1] * xn * xn;
        if (arg > exp_max) arg = exp_max;
        return bwz.call(x) * std::exp(arg);
    }
    std::vector<std::string> param_names() const override { return {"a", "b"}; }
};
FUNLIB_REGISTER("bwz_exp", new BWZExp(c, xmin, xmax));
}
#endif
