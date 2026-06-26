// Burr XII. Mirror of burr/fn.py.
#ifndef FUNLIB_BURR_H
#define FUNLIB_BURR_H
#include "funlib_base.h"
namespace funlib {
struct Burr : IKernel {
    double exp_max, log1p_cutoff, param_min;
    Burr(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        log1p_cutoff = c.get("log1p_cutoff", 30);
        param_min = c.get("param_min", 1e-9);
    }
    double eval(double x, const double* p, int) const override {
        double s = std::max(std::fabs(p[0]), param_min);
        double cc = std::max(std::fabs(p[1]), param_min);
        double k = std::max(std::fabs(p[2]), param_min);
        double u = x / s;
        if (u <= 0.0) return kNaN;
        double log_u = std::log(u);
        double log_uc = cc * log_u;
        double log1p_uc =
            log_uc > log1p_cutoff ? log_uc : std::log1p(std::exp(log_uc));
        double ln_val = std::log(cc * k) + (cc - 1.0) * log_u - (k + 1.0) * log1p_uc;
        return ln_val > -exp_max ? std::exp(ln_val) : 0.0;
    }
    std::vector<std::string> param_names() const override { return {"s", "c", "k"}; }
};
FUNLIB_REGISTER("burr", new Burr(c));
}
#endif
