#ifndef FUNLIB_INCGAMMA_H
#define FUNLIB_INCGAMMA_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct IncGamma : IKernel {
    double s_min;
    bool lower;
    IncGamma(const Constants& c) {
        s_min = c.get("s_min", 1e-9);
        lower = (c.gets("variant", "upper") == "lower");
    }
    double eval(double x, const double* p, int) const override {
        double s = std::max(std::fabs(p[0]), s_min);
        double z = p[1] * x;
        if (z <= 0.0) return kNaN;
        return lower ? ROOT::Math::inc_gamma(s, z) : ROOT::Math::inc_gamma_c(s, z);
    }
    std::vector<std::string> param_names() const override { return {"s", "b"}; }
};
FUNLIB_REGISTER("incgamma", new IncGamma(c));
}
#endif
