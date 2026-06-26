// Landau PDF (ROOT TMath::Landau, normalised). Mirror of landau/fn.py.
#ifndef FUNLIB_LANDAU_H
#define FUNLIB_LANDAU_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct Landau : IKernel {
    double param_min;
    Landau(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double call(double x, double mpv, double sigma) const {
        sigma = std::max(std::fabs(sigma), param_min);
        return TMath::Landau(x, mpv, sigma, true);
    }
    double eval(double x, const double* p, int) const override {
        return call(x, p[0], p[1]);
    }
    std::vector<std::string> param_names() const override { return {"mpv", "sigma"}; }
};
FUNLIB_REGISTER("landau", new Landau(c));
}  // namespace funlib
#endif
