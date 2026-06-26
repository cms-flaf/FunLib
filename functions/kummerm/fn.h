#ifndef FUNLIB_KUMMERM_H
#define FUNLIB_KUMMERM_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct KummerM : IKernel {
    double a_lo, a_hi, b_lo, b_hi, z_abs_max;
    KummerM(const Constants& c) {
        a_lo = c.get("a_lo", 0); a_hi = c.get("a_hi", 0);
        b_lo = c.get("b_lo", 0); b_hi = c.get("b_hi", 0);
        z_abs_max = c.get("z_abs_max", 0);
    }
    double eval(double x, const double* p, int) const override {
        double a = std::max(a_lo, std::min(p[0], a_hi));
        double b = std::max(b_lo, std::min(p[1], b_hi));
        double z = -p[2] * x;
        if (std::fabs(z) > z_abs_max) return kNaN;
        return ROOT::Math::conf_hyperg(a, b, z);
    }
    std::vector<std::string> param_names() const override { return {"a", "b", "c"}; }
};
FUNLIB_REGISTER("kummerm", new KummerM(c));
}
#endif
