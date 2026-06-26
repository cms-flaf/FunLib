#ifndef FUNLIB_LOGNORMAL_H
#define FUNLIB_LOGNORMAL_H
#include "funlib_base.h"
namespace funlib {
struct LogNormal : IKernel {
    double sigma_min;
    LogNormal(const Constants& c) { sigma_min = c.get("sigma_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double mu = p[0];
        double sigma = std::max(std::fabs(p[1]), sigma_min);
        double z = (std::log(x) - mu) / sigma;
        return (mu / x) * std::exp(-0.5 * z * z);
    }
    std::vector<std::string> param_names() const override { return {"mu", "sigma"}; }
};
FUNLIB_REGISTER("lognormal", new LogNormal(c));
}
#endif
