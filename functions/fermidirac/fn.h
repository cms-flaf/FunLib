// Fermi-Dirac. Mirror of fermidirac/fn.py.
#ifndef FUNLIB_FERMIDIRAC_H
#define FUNLIB_FERMIDIRAC_H
#include "funlib_base.h"
namespace funlib {
struct FermiDirac : IKernel {
    double exp_max, T_min;
    FermiDirac(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        T_min = c.get("T_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double mu = p[0];
        double T = std::max(std::fabs(p[1]), T_min);
        double arg = (x - mu) / T;
        if (arg > exp_max) return 0.0;
        if (arg < -exp_max) return 1.0;
        return 1.0 / (std::exp(arg) + 1.0);
    }
    std::vector<std::string> param_names() const override { return {"mu", "T"}; }
};
FUNLIB_REGISTER("fermidirac", new FermiDirac(c));
}
#endif
