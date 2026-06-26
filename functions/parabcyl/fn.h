#ifndef FUNLIB_PARABCYL_H
#define FUNLIB_PARABCYL_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct ParabCyl : IKernel {
    double z_abs_max, nu_lo, nu_hi;
    ParabCyl(const Constants& c) {
        z_abs_max = c.get("z_abs_max", 0);
        nu_lo = c.get("nu_lo", 0); nu_hi = c.get("nu_hi", 0);
    }
    double eval(double x, const double* p, int) const override {
        double z = p[1] * x + p[2];
        if (std::fabs(z) > z_abs_max) return kNaN;
        double nu = std::max(nu_lo, std::min(p[0], nu_hi));
        return fmath::pbdv(nu, z);
    }
    std::vector<std::string> param_names() const override {
        return {"nu", "a", "b"};
    }
};
FUNLIB_REGISTER("parabcyl", new ParabCyl(c));
}
#endif
