#ifndef FUNLIB_GEV_H
#define FUNLIB_GEV_H
#include "funlib_base.h"
namespace funlib {
struct GEV : IKernel {
    double param_min, xi_eps;
    GEV(const Constants& c) {
        param_min = c.get("param_min", 1e-9);
        xi_eps = c.get("xi_eps", 1e-8);
    }
    double eval(double x, const double* p, int) const override {
        double mu = p[0];
        double sigma = std::max(std::fabs(p[1]), param_min);
        double xi = p[2];
        double z = (x - mu) / sigma;
        if (std::fabs(xi) < xi_eps) {
            double t = std::exp(-z);
            return t * std::exp(-t);
        }
        double base = 1.0 + xi * z;
        if (base <= 0.0) return kNaN;
        double tinv = std::pow(base, -1.0 / xi);
        return std::pow(base, -1.0) * tinv * std::exp(-tinv);
    }
    std::vector<std::string> param_names() const override {
        return {"mu", "sigma", "xi"};
    }
};
FUNLIB_REGISTER("gev", new GEV(c));
}
#endif
