// Doniach-Sunjic. Mirror of doniachsunjic/fn.py.
#ifndef FUNLIB_DONIACHSUNJIC_H
#define FUNLIB_DONIACHSUNJIC_H
#include "funlib_base.h"
namespace funlib {
struct DoniachSunjic : IKernel {
    double gamma_min;
    DoniachSunjic(const Constants& c) { gamma_min = c.get("gamma_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double x0 = p[0];
        double g = std::max(std::fabs(p[1]), gamma_min);
        double al = p[2];
        double d = x - x0;
        double num = std::cos(M_PI * al / 2.0 + (1.0 - al) * std::atan(d / g));
        double den = std::pow(g * g + d * d, (1.0 - al) / 2.0);
        if (den == 0.0) return kInf;
        return num / den;
    }
    std::vector<std::string> param_names() const override {
        return {"x0", "gamma", "alpha"};
    }
};
FUNLIB_REGISTER("doniachsunjic", new DoniachSunjic(c));
}
#endif
