// Complementary error function. Mirror of erfc/fn.py.
#ifndef FUNLIB_ERFC_H
#define FUNLIB_ERFC_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct Erfc : IKernel {
    double sigma_min;
    Erfc(const Constants& c) { sigma_min = c.get("sigma_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double mu = p[0];
        double sigma = std::max(std::fabs(p[1]), sigma_min);
        return TMath::Erfc((x - mu) / sigma);
    }
    std::vector<std::string> param_names() const override { return {"mu", "sigma"}; }
};
FUNLIB_REGISTER("erfc", new Erfc(c));
}
#endif
