#ifndef FUNLIB_EXPRATIONAL_H
#define FUNLIB_EXPRATIONAL_H
#include "rationalpade.h"
namespace funlib {
struct ExpRational : IKernel {
    RationalPade pade;
    double exp_max;
    ExpRational(const Constants& c, double xmin, double xmax) : pade(c, xmin, xmax) {
        exp_max = c.get("exp_max", 700);
    }
    double eval(double x, const double* p, int) const override {
        double arg = pade.call(x, p);
        if (arg > exp_max) arg = exp_max;
        return std::exp(arg);
    }
    std::vector<std::string> param_names() const override {
        return pade.param_names();
    }
};
FUNLIB_REGISTER("exprational", new ExpRational(c, xmin, xmax));
}
#endif
