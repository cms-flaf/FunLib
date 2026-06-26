// GammaTail. Mirror of gammatail/fn.py.
#ifndef FUNLIB_GAMMATAIL_H
#define FUNLIB_GAMMATAIL_H
#include "funlib_base.h"
namespace funlib {
struct GammaTail : IKernel {
    GammaTail(const Constants&) {}
    double eval(double x, const double* p, int) const override {
        double a = p[0], b = p[1], x_s = p[2];
        double dx = x - x_s;
        if (dx <= 0.0) return 0.0;
        return std::pow(dx, a) * std::exp(-b * x);
    }
    std::vector<std::string> param_names() const override {
        return {"a", "b", "x_s"};
    }
};
FUNLIB_REGISTER("gammatail", new GammaTail(c));
}
#endif
