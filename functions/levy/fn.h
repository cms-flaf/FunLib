#ifndef FUNLIB_LEVY_H
#define FUNLIB_LEVY_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct Levy : IKernel {
    double scale_min;
    Levy(const Constants& c) { scale_min = c.get("scale_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double loc = p[0];
        double scale = std::max(std::fabs(p[1]), scale_min);
        double v = fmath::levy_pdf(x, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"loc", "scale"};
    }
};
FUNLIB_REGISTER("levy", new Levy(c));
}
#endif
