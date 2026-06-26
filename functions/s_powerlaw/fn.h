#ifndef FUNLIB_S_POWERLAW_H
#define FUNLIB_S_POWERLAW_H
#include "funlib_base.h"
namespace funlib {
struct SPowerlaw : IKernel {
    double x_mid;
    int n_terms;
    SPowerlaw(const Constants& c, double xmin, double xmax) {
        x_mid = (xmin + xmax) / 2.0;
        n_terms = (int)c.get("n_terms", 1);
    }
    double eval(double x, const double* p, int) const override {
        double xn = x / x_mid;
        double total = std::pow(xn, p[0]);
        for (int k = 2; k <= n_terms; k++) {
            double A = p[2 * k - 3];
            double b = p[2 * k - 2];
            total += A * std::pow(xn, b);
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
FUNLIB_REGISTER("s_powerlaw", new SPowerlaw(c, xmin, xmax));
}
#endif
