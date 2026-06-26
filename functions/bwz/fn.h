// Z-boson Breit-Wigner with fixed M_Z, Gamma_Z. Mirror of bwz/fn.py.
#ifndef FUNLIB_BWZ_H
#define FUNLIB_BWZ_H
#include "breit_wigner.h"
namespace funlib {
struct BWZ : IKernel {
    BreitWigner bw;
    double m, Gamma;
    bool free_power;
    BWZ(const Constants& c) : bw(c) {
        m = c.get("m", 91.1876);
        Gamma = c.get("Gamma", 2.4952);
        free_power = c.getb("free_power", false);
    }
    double call(double x, double p = 2.0) const { return bw.call(x, m, Gamma, p); }
    double eval(double x, const double* p, int) const override {
        double pw = free_power ? p[0] : 2.0;
        return call(x, pw);
    }
    std::vector<std::string> param_names() const override {
        if (free_power) return {"p"};
        return {};
    }
};
FUNLIB_REGISTER("bwz", new BWZ(c));
}  // namespace funlib
#endif
