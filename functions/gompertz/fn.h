#ifndef FUNLIB_GOMPERTZ_H
#define FUNLIB_GOMPERTZ_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct Gompertz : IKernel {
    double param_min;
    Gompertz(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double cc = std::max(std::fabs(p[0]), param_min);
        double loc = p[1];
        double scale = std::max(std::fabs(p[2]), param_min);
        double v = fmath::gompertz_pdf(x, cc, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"c", "loc", "scale"};
    }
};
FUNLIB_REGISTER("gompertz", new Gompertz(c));
}
#endif
