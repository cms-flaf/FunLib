#ifndef FUNLIB_SUMBESSELK0_H
#define FUNLIB_SUMBESSELK0_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct SumBesselK0 : IKernel {
    int n_terms;
    SumBesselK0(const Constants& c) { n_terms = (int)c.get("n_terms", 1); }
    double k0(double a, double x) const {
        double arg = a * x;
        if (arg <= 0.0) return kNaN;
        if (arg < 1e-6) return kInf;
        return TMath::BesselK0(arg);
    }
    double eval(double x, const double* p, int) const override {
        double total = k0(p[0], x);
        for (int k = 2; k <= n_terms; k++) {
            double R = p[2 * k - 3];
            double a = p[2 * k - 2];
            total += R * k0(a, x);
        }
        return total;
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v = {"a1"};
        for (int k = 2; k <= n_terms; k++) {
            v.push_back("R" + std::to_string(k));
            v.push_back("a" + std::to_string(k));
        }
        return v;
    }
};
FUNLIB_REGISTER("sumbesselk0", new SumBesselK0(c));
}
#endif
