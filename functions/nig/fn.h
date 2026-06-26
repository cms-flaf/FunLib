#ifndef FUNLIB_NIG_H
#define FUNLIB_NIG_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct NIG : IKernel {
    double param_min;
    NIG(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double a = std::max(std::fabs(p[0]), param_min);
        double b = a * std::tanh(p[1]);
        double loc = p[2];
        double scale = std::max(std::fabs(p[3]), param_min);
        double v = fmath::norminvgauss_pdf(x, a, b, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"a", "b", "loc", "scale"};
    }
};
FUNLIB_REGISTER("nig", new NIG(c));
}
#endif
