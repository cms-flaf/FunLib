// Beta-prime. Mirror of betaprime/fn.py.
#ifndef FUNLIB_BETAPRIME_H
#define FUNLIB_BETAPRIME_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct BetaPrime : IKernel {
    double param_min;
    BetaPrime(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double a = std::max(std::fabs(p[0]), param_min);
        double b = std::max(std::fabs(p[1]), param_min);
        double loc = p[2];
        double scale = std::max(std::fabs(p[3]), param_min);
        double val = fmath::betaprime_pdf(x, a, b, loc, scale);
        return std::isfinite(val) ? val : 0.0;
    }
    std::vector<std::string> param_names() const override {
        return {"a", "b", "loc", "scale"};
    }
};
FUNLIB_REGISTER("betaprime", new BetaPrime(c));
}
#endif
