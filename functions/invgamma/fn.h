#ifndef FUNLIB_INVGAMMA_H
#define FUNLIB_INVGAMMA_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct InvGamma : IKernel {
    double param_min;
    InvGamma(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double a = std::max(std::fabs(p[0]), param_min);
        double loc = p[1];
        double scale = std::max(std::fabs(p[2]), param_min);
        double v = fmath::invgamma_pdf(x, a, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"a", "loc", "scale"};
    }
};
FUNLIB_REGISTER("invgamma", new InvGamma(c));
}
#endif
