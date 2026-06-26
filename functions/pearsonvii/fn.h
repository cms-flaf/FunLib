#ifndef FUNLIB_PEARSONVII_H
#define FUNLIB_PEARSONVII_H
#include "funlib_base.h"
namespace funlib {
struct PearsonVII : IKernel {
    double sigma_min;
    PearsonVII(const Constants& c) { sigma_min = c.get("sigma_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double mu = p[0];
        double sigma = std::max(std::fabs(p[1]), sigma_min);
        double m = p[2];
        double z = (x - mu) / sigma;
        return std::pow(1.0 + z * z, -m);
    }
    std::vector<std::string> param_names() const override {
        return {"mu", "sigma", "m"};
    }
};
FUNLIB_REGISTER("pearsonvii", new PearsonVII(c));
}
#endif
