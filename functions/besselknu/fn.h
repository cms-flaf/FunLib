// Modified Bessel K_nu free order. Mirror of besselknu/fn.py.
#ifndef FUNLIB_BESSELKNU_H
#define FUNLIB_BESSELKNU_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct BesselKnu : IKernel {
    double z_lo, nu_max;
    BesselKnu(const Constants& c) {
        z_lo = c.get("z_lo", 0);
        nu_max = c.get("nu_max", 0);
    }
    double eval(double x, const double* p, int) const override {
        double nu = p[0], a = p[1];
        double z = a * x;
        if (z <= 0.0) return kNaN;
        if (z <= z_lo) return kInf;
        nu = std::min(std::fabs(nu), nu_max);
        return ROOT::Math::cyl_bessel_k(nu, z);
    }
    std::vector<std::string> param_names() const override { return {"nu", "a"}; }
};
FUNLIB_REGISTER("besselknu", new BesselKnu(c));
}
#endif
