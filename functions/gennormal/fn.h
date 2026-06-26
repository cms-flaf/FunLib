#ifndef FUNLIB_GENNORMAL_H
#define FUNLIB_GENNORMAL_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct GenNormal : IKernel {
    double beta_min, scale_min;
    GenNormal(const Constants& c) {
        beta_min = c.get("beta_min", 1e-9);
        scale_min = c.get("scale_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double beta = std::max(std::fabs(p[0]), beta_min);
        double loc = p[1];
        double scale = std::max(std::fabs(p[2]), scale_min);
        double v = fmath::gennorm_pdf(x, beta, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"beta", "loc", "scale"};
    }
};
FUNLIB_REGISTER("gennormal", new GenNormal(c));
}
#endif
