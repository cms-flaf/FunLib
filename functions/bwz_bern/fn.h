#ifndef FUNLIB_BWZ_BERN_H
#define FUNLIB_BWZ_BERN_H
#include "bernstein.h"
#include "bwz.h"
namespace funlib {
struct BWZBern : IKernel {
    BWZ bwz;
    Bernstein bern;
    int n_terms;
    double exp_max;
    BWZBern(const Constants& c, double xmin, double xmax)
        : bwz(c), bern(xmin, xmax, (int)c.get("n_terms", 1)) {
        n_terms = (int)c.get("n_terms", 1);
        exp_max = c.get("exp_max", 700);
    }
    double eval(double x, const double* p, int) const override {
        double a = p[n_terms];
        double exp_arg = std::min(a * x, exp_max);
        return bwz.call(x) * bern.call(x, p) * std::exp(exp_arg);
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        for (int i = 1; i <= n_terms; i++) v.push_back("c" + std::to_string(i));
        v.push_back("a");
        return v;
    }
};
FUNLIB_REGISTER("bwz_bern", new BWZBern(c, xmin, xmax));
}
#endif
