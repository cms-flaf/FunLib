#ifndef FUNLIB_WHITTAKERW_H
#define FUNLIB_WHITTAKERW_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct WhittakerW : IKernel {
    double z_hi, kappa_lo, kappa_hi, mu_lo, mu_hi;
    WhittakerW(const Constants& c) {
        z_hi = c.get("z_hi", 0);
        kappa_lo = c.get("kappa_lo", 0); kappa_hi = c.get("kappa_hi", 0);
        mu_lo = c.get("mu_lo", 0); mu_hi = c.get("mu_hi", 0);
    }
    double eval(double x, const double* p, int) const override {
        double z = p[2] * x;
        if (z <= 0.0 || z > z_hi) return kNaN;
        double kappa = std::max(kappa_lo, std::min(p[0], kappa_hi));
        double mu = std::max(mu_lo, std::min(p[1], mu_hi));
        // hyperu(a,b,z) = conf_hypergU; here a = mu-kappa+0.5, b = 1+2mu
        double u = ROOT::Math::conf_hypergU(mu - kappa + 0.5, 1.0 + 2.0 * mu, z);
        return std::exp(-0.5 * z) * std::pow(z, mu + 0.5) * u;
    }
    std::vector<std::string> param_names() const override {
        return {"kappa", "mu", "a"};
    }
};
FUNLIB_REGISTER("whittakerw", new WhittakerW(c));
}
#endif
