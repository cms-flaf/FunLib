#ifndef FUNLIB_CRYSTALBALL_H
#define FUNLIB_CRYSTALBALL_H
#include "funlib_base.h"
namespace funlib {
inline double cb_tail(double t, double a, double n, int sign) {
    double A = std::pow(n / a, n) * std::exp(-0.5 * a * a);
    double B = n / a - a;
    double arg = B + sign * t;
    if (arg <= 0.0) return kNaN;
    return A * std::pow(arg, -n);
}
struct CrystalBall : IKernel {
    double param_min;
    std::string variant;
    CrystalBall(const Constants& c) {
        param_min = c.get("param_min", 1e-9);
        variant = c.gets("variant", "right_sided");
    }
    double eval(double x, const double* p, int) const override {
        if (variant == "right_sided") {
            double sigma = std::max(std::fabs(p[1]), param_min);
            double a = std::max(std::fabs(p[2]), param_min);
            double t = (x - p[0]) / sigma;
            if (t <= a) return std::exp(-0.5 * t * t);
            return cb_tail(t, a, p[3], +1);
        }
        if (variant == "left_sided") {
            double sigma = std::max(std::fabs(p[1]), param_min);
            double a = std::max(std::fabs(p[2]), param_min);
            double t = (x - p[0]) / sigma;
            if (t > -a) return std::exp(-0.5 * t * t);
            return cb_tail(t, a, p[3], -1);
        }
        // double_sided: mean,sigma,alpha_l,n_l,alpha_r,n_r
        double sigma = std::max(std::fabs(p[1]), param_min);
        double aL = std::max(std::fabs(p[2]), param_min);
        double aR = std::max(std::fabs(p[4]), param_min);
        double t = (x - p[0]) / sigma;
        if (t < -aL) return cb_tail(t, aL, p[3], -1);
        if (t > aR) return cb_tail(t, aR, p[5], +1);
        return std::exp(-0.5 * t * t);
    }
    std::vector<std::string> param_names() const override {
        if (variant == "double_sided")
            return {"mean", "sigma", "alpha_l", "n_l", "alpha_r", "n_r"};
        return {"mean", "sigma", "alpha", "n"};
    }
};
FUNLIB_REGISTER("crystalball", new CrystalBall(c));
}
#endif
