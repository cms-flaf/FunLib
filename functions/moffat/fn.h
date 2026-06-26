#ifndef FUNLIB_MOFFAT_H
#define FUNLIB_MOFFAT_H
#include "funlib_base.h"
namespace funlib {
struct Moffat : IKernel {
    Moffat(const Constants&) {}
    double eval(double x, const double* p, int) const override {
        double al = std::max(std::fabs(p[0]), 1e-9);
        double beta = p[1];
        double u = x / al;
        return std::pow(1.0 + u * u, -beta);
    }
    std::vector<std::string> param_names() const override {
        return {"alpha", "beta"};
    }
};
FUNLIB_REGISTER("moffat", new Moffat(c));
}
#endif
