#ifndef FUNLIB_PLANCK_H
#define FUNLIB_PLANCK_H
#include "funlib_base.h"
namespace funlib {
struct Planck : IKernel {
    double exp_max, T_min;
    Planck(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        T_min = c.get("T_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double pw = p[0];
        double T = std::max(std::fabs(p[1]), T_min);
        double a = x / T;
        if (a > exp_max) return 0.0;
        double denom = std::expm1(a);
        if (denom <= 1e-300) return kNaN;
        return std::pow(x, pw) / denom;
    }
    std::vector<std::string> param_names() const override { return {"p", "T"}; }
};
FUNLIB_REGISTER("planck", new Planck(c));
}
#endif
