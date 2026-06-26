// Generic Breit-Wigner propagator. Mirror of breit_wigner/fn.py.
#ifndef FUNLIB_BREIT_WIGNER_H
#define FUNLIB_BREIT_WIGNER_H
#include "funlib_base.h"
namespace funlib {
struct BreitWigner : IKernel {
    double Gamma_min;
    bool is_lorentzian, is_running, free_power;
    BreitWigner(const Constants& c) {
        Gamma_min = c.get("Gamma_min", 1e-9);
        is_lorentzian = c.getb("is_lorentzian", false);
        is_running = c.getb("is_running", false);
        free_power = c.getb("free_power", false);
    }
    double call(double x, double m, double Gamma, double p) const {
        m = std::fabs(m);
        Gamma = std::max(std::fabs(Gamma), Gamma_min);
        double d;
        if (is_lorentzian)
            d = std::pow(std::fabs(x - m), p) + std::pow(Gamma / 2.0, p);
        else {
            double Gm = Gamma * m;
            d = std::pow(std::fabs(x * x - m * m), p) + std::pow(Gm, p);
        }
        double num = is_running ? x * x : 1.0;
        if (d == 0.0) return kInf;
        return num / d;
    }
    double eval(double x, const double* p, int) const override {
        double pw = free_power ? p[2] : 2.0;
        return call(x, p[0], p[1], pw);
    }
    std::vector<std::string> param_names() const override {
        if (free_power) return {"m", "Gamma", "p"};
        return {"m", "Gamma"};
    }
};
FUNLIB_REGISTER("breit_wigner", new BreitWigner(c));
}  // namespace funlib
#endif
