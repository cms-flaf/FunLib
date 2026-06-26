#ifndef FUNLIB_VARIANCEGAMMA_H
#define FUNLIB_VARIANCEGAMMA_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct VarianceGamma : IKernel {
    double exp_max, alpha_min, z_min;
    VarianceGamma(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        alpha_min = c.get("alpha_min", 1e-9);
        z_min = c.get("z_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double lam = p[0];
        double alpha = std::max(std::fabs(p[1]), alpha_min);
        double beta = p[2], mu = p[3];
        double z = x - mu;
        double az = std::max(std::fabs(z), z_min);
        double arg = alpha * az;
        // K_nu is symmetric in order; GSL's cyl_bessel_k requires nu >= 0.
        double k = ROOT::Math::cyl_bessel_k(std::fabs(lam - 0.5), arg);
        double ex = beta * z;
        if (ex > exp_max) return kInf;
        return std::pow(az, lam - 0.5) * k * std::exp(ex);
    }
    std::vector<std::string> param_names() const override {
        return {"lam", "alpha", "beta", "mu"};
    }
};
FUNLIB_REGISTER("variancegamma", new VarianceGamma(c));
}
#endif
