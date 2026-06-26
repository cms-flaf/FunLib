// Rational Pade [m/n] approximant. Mirror of rationalpade/fn.py.
#ifndef FUNLIB_RATIONALPADE_H
#define FUNLIB_RATIONALPADE_H
#include "funlib_base.h"
namespace funlib {
struct RationalPade : IKernel {
    double x0, xhalf;
    int num_deg, den_deg;
    RationalPade(const Constants& c, double xmin, double xmax) {
        x0 = (xmin + xmax) / 2.0;
        xhalf = (xmax - xmin) / 2.0;
        num_deg = (int)c.get("num_deg", 0);
        den_deg = (int)c.get("den_deg", 0);
    }
    double call(double x, const double* p) const {
        double t = (x - x0) / xhalf;
        double num = 1.0;
        for (int i = 1; i <= num_deg; i++) num += p[i - 1] * std::pow(t, i);
        double den = 1.0;
        for (int i = 1; i <= den_deg; i++) den += p[num_deg + i - 1] * std::pow(t, i);
        if (std::fabs(den) < 1e-12) return kInf;
        return num / den;
    }
    double eval(double x, const double* p, int) const override { return call(x, p); }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        for (int i = 1; i <= num_deg; i++) v.push_back("a" + std::to_string(i));
        for (int i = 1; i <= den_deg; i++) v.push_back("b" + std::to_string(i));
        return v;
    }
};
FUNLIB_REGISTER("rationalpade", new RationalPade(c, xmin, xmax));
}  // namespace funlib
#endif
