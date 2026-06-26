#ifndef FUNLIB_GNFW_H
#define FUNLIB_GNFW_H
#include "funlib_base.h"
namespace funlib {
struct GNFW : IKernel {
    double rs_min;
    GNFW(const Constants& c) { rs_min = c.get("rs_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double rs = std::max(std::fabs(p[0]), rs_min);
        double gamma = p[1], beta = p[2];
        double u = x / rs;
        double denom = std::pow(u, gamma) * std::pow(1.0 + u, beta - gamma);
        if (denom == 0.0) return kInf;
        return 1.0 / denom;
    }
    std::vector<std::string> param_names() const override {
        return {"rs", "gamma", "beta"};
    }
};
FUNLIB_REGISTER("gnfw", new GNFW(c));
}
#endif
