#ifndef FUNLIB_JOHNSONSU_H
#define FUNLIB_JOHNSONSU_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct JohnsonSU : IKernel {
    double param_min;
    JohnsonSU(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double a = p[0];
        double b = std::max(std::fabs(p[1]), param_min);
        double loc = p[2];
        double scale = std::max(std::fabs(p[3]), param_min);
        double v = fmath::johnsonsu_pdf(x, a, b, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"a", "b", "loc", "scale"};
    }
};
FUNLIB_REGISTER("johnsonsu", new JohnsonSU(c));
}
#endif
