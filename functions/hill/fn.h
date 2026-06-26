#ifndef FUNLIB_HILL_H
#define FUNLIB_HILL_H
#include "funlib_base.h"
namespace funlib {
struct Hill : IKernel {
    double K_min;
    Hill(const Constants& c) { K_min = c.get("K_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double K = std::max(std::fabs(p[0]), K_min);
        double h = p[1];
        double u = x / K;
        return 1.0 / (1.0 + std::pow(u, h));
    }
    std::vector<std::string> param_names() const override { return {"K", "h"}; }
};
FUNLIB_REGISTER("hill", new Hill(c));
}
#endif
