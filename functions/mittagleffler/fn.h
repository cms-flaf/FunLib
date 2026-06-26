#ifndef FUNLIB_MITTAGLEFFLER_H
#define FUNLIB_MITTAGLEFFLER_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
inline double mittag_leffler(double a, double z, double series_limit,
                             double lt_cutoff) {
    double az = std::fabs(z);
    if (az < 1e-12) return 1.0;
    if (az <= series_limit) {
        double s = 0.0;
        for (int k = 0; k < 130; k++) {
            double lt = k * std::log(az) - std::lgamma(a * k + 1.0);
            if (lt < -lt_cutoff && k > 5) continue;
            double t = std::exp(lt);
            s += (z > 0 || k % 2 == 0) ? t : -t;
        }
        return s;
    }
    double s = 0.0;
    for (int k = 1; k < 5; k++) {
        double g = 1.0 - a * k;
        double inv = fmath::rgamma(g);
        s -= inv * std::pow(z, -(double)k);
    }
    return s;
}
struct MittagLeffler : IKernel {
    double series_limit, lt_cutoff, alpha_min, tau_min;
    MittagLeffler(const Constants& c) {
        series_limit = c.get("series_limit", 0);
        lt_cutoff = c.get("lt_cutoff", 0);
        alpha_min = c.get("alpha_min", 1e-9);
        tau_min = c.get("tau_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double a = std::max(std::fabs(p[0]), alpha_min);
        double tau = std::max(std::fabs(p[1]), tau_min);
        return mittag_leffler(a, -x / tau, series_limit, lt_cutoff);
    }
    std::vector<std::string> param_names() const override {
        return {"alpha", "tau"};
    }
};
FUNLIB_REGISTER("mittagleffler", new MittagLeffler(c));
}
#endif
