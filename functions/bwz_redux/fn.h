#ifndef FUNLIB_BWZ_REDUX_H
#define FUNLIB_BWZ_REDUX_H
#include "bwz.h"
namespace funlib {
struct BWZRedux : IKernel {
    BWZ bwz;
    int n_terms;
    double exp_max;
    BWZRedux(const Constants& c) : bwz(c) {
        n_terms = (int)c.get("n_terms", 1);
        exp_max = c.get("exp_max", 700);
    }
    double eval(double x, const double* p, int) const override {
        double exp_arg = 0.0;
        for (int i = 1; i <= n_terms; i++) exp_arg += p[i - 1] * std::pow(x, i);
        if (exp_arg > exp_max) exp_arg = exp_max;
        return std::exp(exp_arg) * bwz.call(x, p[n_terms]);
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        for (int i = 1; i <= n_terms; i++) v.push_back("c" + std::to_string(i));
        v.push_back("p");
        return v;
    }
};
FUNLIB_REGISTER("bwz_redux", new BWZRedux(c));
}
#endif
