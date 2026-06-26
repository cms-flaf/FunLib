#ifndef FUNLIB_INVERSEGAUSS_H
#define FUNLIB_INVERSEGAUSS_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct InverseGauss : IKernel {
    double param_min;
    InverseGauss(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double mu = std::max(std::fabs(p[0]), param_min);
        double loc = p[1];
        double scale = std::max(std::fabs(p[2]), param_min);
        double v = fmath::invgauss_pdf(x, mu, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"mu", "loc", "scale"};
    }
};
FUNLIB_REGISTER("inversegauss", new InverseGauss(c));
}
#endif
