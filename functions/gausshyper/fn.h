#ifndef FUNLIB_GAUSSHYPER_H
#define FUNLIB_GAUSSHYPER_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct GaussHyper : IKernel {
    double x0, ab_bound, c_min, c_max;
    GaussHyper(const Constants& c, double xmin, double xmax) {
        x0 = (xmin + xmax) / 2.0;
        ab_bound = c.get("ab_bound", 50);
        c_min = c.get("c_min", 1e-3);
        c_max = c.get("c_max", 100);
    }
    double eval(double x, const double* p, int) const override {
        double a = std::max(-ab_bound, std::min(p[0], ab_bound));
        double b = std::max(-ab_bound, std::min(p[1], ab_bound));
        double cc = std::max(c_min, std::min(p[2], c_max));
        return fmath::hyp2f1(a, b, cc, -x / x0);
    }
    std::vector<std::string> param_names() const override { return {"a", "b", "c"}; }
};
FUNLIB_REGISTER("gausshyper", new GaussHyper(c, xmin, xmax));
}
#endif
