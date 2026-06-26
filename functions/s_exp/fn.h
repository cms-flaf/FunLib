#ifndef FUNLIB_S_EXP_H
#define FUNLIB_S_EXP_H
#include "funlib_base.h"
namespace funlib {
struct SExp : IKernel {
    double exp_max;
    int n_terms;
    SExp(const Constants& c) {
        exp_max = c.get("exp_max", 700);
        n_terms = (int)c.get("n_terms", 1);
    }
    double eval(double x, const double* p, int) const override {
        double total = std::exp(std::min(p[0] * x, exp_max));
        for (int k = 2; k <= n_terms; k++) {
            double A = p[2 * k - 3];
            double b = p[2 * k - 2];
            total += A * std::exp(std::min(b * x, exp_max));
        }
        return total;
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v = {"b1"};
        for (int k = 2; k <= n_terms; k++) {
            v.push_back("A" + std::to_string(k));
            v.push_back("b" + std::to_string(k));
        }
        return v;
    }
};
FUNLIB_REGISTER("s_exp", new SExp(c));
}
#endif
