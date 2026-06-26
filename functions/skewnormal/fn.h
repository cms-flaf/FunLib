#ifndef FUNLIB_SKEWNORMAL_H
#define FUNLIB_SKEWNORMAL_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct SkewNormal : IKernel {
    SkewNormal(const Constants&) {}
    double eval(double x, const double* p, int) const override {
        double a = p[0];
        double loc = p[1];
        double scale = std::max(std::fabs(p[2]), 1e-9);
        double v = fmath::skewnorm_pdf(x, a, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"a", "loc", "scale"};
    }
};
FUNLIB_REGISTER("skewnormal", new SkewNormal(c));
}
#endif
