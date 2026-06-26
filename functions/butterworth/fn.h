// Butterworth magnitude response. Mirror of butterworth/fn.py.
#ifndef FUNLIB_BUTTERWORTH_H
#define FUNLIB_BUTTERWORTH_H
#include "funlib_base.h"
namespace funlib {
struct Butterworth : IKernel {
    double xc_min;
    Butterworth(const Constants& c) { xc_min = c.get("xc_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double xc = std::max(std::fabs(p[0]), xc_min);
        double k = p[1];
        double u = x / xc;
        if (u <= 0.0) return kNaN;
        double log_u2k = 2.0 * k * std::log(u);
        double log1p_u2k = log_u2k > 30.0 ? log_u2k : std::log1p(std::exp(log_u2k));
        return std::exp(-0.5 * log1p_u2k);
    }
    std::vector<std::string> param_names() const override { return {"xc", "k"}; }
};
FUNLIB_REGISTER("butterworth", new Butterworth(c));
}
#endif
