#ifndef FUNLIB_FATIGUELIFE_H
#define FUNLIB_FATIGUELIFE_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct FatigueLife : IKernel {
    double param_min;
    FatigueLife(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double cc = std::max(std::fabs(p[0]), param_min);
        double loc = p[1];
        double scale = std::max(std::fabs(p[2]), param_min);
        double v = fmath::fatiguelife_pdf(x, cc, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"c", "loc", "scale"};
    }
};
FUNLIB_REGISTER("fatiguelife", new FatigueLife(c));
}
#endif
