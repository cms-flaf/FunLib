#ifndef FUNLIB_TRICOMIU_H
#define FUNLIB_TRICOMIU_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct TricomiU : IKernel {
    double a_lo, a_hi, b_lo, b_hi;
    TricomiU(const Constants& c) {
        a_lo = c.get("a_lo", 0); a_hi = c.get("a_hi", 0);
        b_lo = c.get("b_lo", 0); b_hi = c.get("b_hi", 0);
    }
    double eval(double x, const double* p, int) const override {
        double z = p[2] * x;
        if (z <= 0.0) return kNaN;
        double a = std::max(a_lo, std::min(p[0], a_hi));
        double b = std::max(b_lo, std::min(p[1], b_hi));
        return ROOT::Math::conf_hypergU(a, b, z);
    }
    std::vector<std::string> param_names() const override { return {"a", "b", "c"}; }
};
FUNLIB_REGISTER("tricomiu", new TricomiU(c));
}
#endif
