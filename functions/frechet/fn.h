// Frechet (inverse Weibull). Mirror of frechet/fn.py.
#ifndef FUNLIB_FRECHET_H
#define FUNLIB_FRECHET_H
#include "funlib_base.h"
namespace funlib {
struct Frechet : IKernel {
    double s_min;
    Frechet(const Constants& c) { s_min = c.get("s_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double s = std::max(std::fabs(p[0]), s_min);
        double alpha = p[1];
        double u = x / s;
        if (u <= 0.0) return kNaN;
        return std::pow(u, -1.0 - alpha) * std::exp(-std::pow(u, -alpha));
    }
    std::vector<std::string> param_names() const override { return {"s", "alpha"}; }
};
FUNLIB_REGISTER("frechet", new Frechet(c));
}
#endif
