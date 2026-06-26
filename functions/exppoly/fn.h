// exp(deg-n polynomial in normalised mass). Mirror of exppoly/fn.py.
#ifndef FUNLIB_EXPPOLY_H
#define FUNLIB_EXPPOLY_H
#include "funlib_base.h"
namespace funlib {
struct ExpPoly : IKernel {
    double x0, xhalf, exp_max;
    int n;
    ExpPoly(const Constants& c, double xmin, double xmax, int npar) {
        x0 = (xmin + xmax) / 2.0;
        xhalf = (xmax - xmin) / 2.0;
        exp_max = c.get("exp_max", 700);
        n = npar;
    }
    double eval(double x, const double* p, int) const override {
        double xn = (x - x0) / xhalf;
        double arg = 0.0;
        for (int i = 1; i <= n; i++) arg += p[i - 1] * std::pow(xn, i);
        if (arg > exp_max) arg = exp_max;
        return std::exp(arg);
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        for (int i = 1; i <= n; i++) v.push_back("c" + std::to_string(i));
        return v;
    }
};
FUNLIB_REGISTER("exppoly", new ExpPoly(c, xmin, xmax, npar));
}
#endif
