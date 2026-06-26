#ifndef FUNLIB_LAURENT_H
#define FUNLIB_LAURENT_H
#include "funlib_base.h"
namespace funlib {
struct Laurent : IKernel {
    double x_max;
    int n_terms;
    Laurent(const Constants& c, double xmax) : x_max(xmax) {
        n_terms = (int)c.get("n_terms", 1);
    }
    double eval(double x, const double* p, int) const override {
        double r = x_max / x;
        double pw = p[0];
        double poly = 1.0;
        for (int i = 1; i <= n_terms; i++) poly += p[i] * std::pow(r, i);
        return std::pow(r, pw) * poly;
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v = {"p"};
        for (int i = 1; i <= n_terms; i++) v.push_back("c" + std::to_string(i));
        return v;
    }
};
FUNLIB_REGISTER("laurent", new Laurent(c, xmax));
}
#endif
