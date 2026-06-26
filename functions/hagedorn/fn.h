#ifndef FUNLIB_HAGEDORN_H
#define FUNLIB_HAGEDORN_H
#include "funlib_base.h"
namespace funlib {
struct Hagedorn : IKernel {
    double p0_min;
    Hagedorn(const Constants& c) { p0_min = c.get("p0_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double p0 = std::max(std::fabs(p[0]), p0_min);
        double n = p[1];
        double base = 1.0 + x / p0;
        if (base <= 0.0) return kNaN;
        return std::pow(base, -n);
    }
    std::vector<std::string> param_names() const override { return {"p0", "n"}; }
};
FUNLIB_REGISTER("hagedorn", new Hagedorn(c));
}
#endif
