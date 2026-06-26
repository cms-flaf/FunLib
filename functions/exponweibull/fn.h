#ifndef FUNLIB_EXPONWEIBULL_H
#define FUNLIB_EXPONWEIBULL_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct ExponWeibull : IKernel {
    double param_min;
    ExponWeibull(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double a = std::max(std::fabs(p[0]), param_min);
        double cc = std::max(std::fabs(p[1]), param_min);
        double loc = p[2];
        double scale = std::max(std::fabs(p[3]), param_min);
        double v = fmath::exponweib_pdf(x, a, cc, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"a", "c", "loc", "scale"};
    }
};
FUNLIB_REGISTER("exponweibull", new ExponWeibull(c));
}
#endif
