#ifndef FUNLIB_VOIGTZ_H
#define FUNLIB_VOIGTZ_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct VoigtZ : IKernel {
    double m_z, g_z, exp_max, sigma_min, x0, xhalf;
    VoigtZ(const Constants& c, double xmin, double xmax) {
        m_z = c.get("M_Z", 91.1876);
        g_z = c.get("G_Z", 2.4952);
        exp_max = c.get("exp_max", 700);
        sigma_min = c.get("sigma_min", 1e-9);
        x0 = (xmin + xmax) / 2.0;
        xhalf = (xmax - xmin) / 2.0;
    }
    double eval(double x, const double* p, int) const override {
        double sigma = std::max(std::fabs(p[0]), sigma_min);
        double a = p[1], b = p[2];
        double voigt = TMath::Voigt(x - m_z, sigma, g_z);
        double xn = (x - x0) / xhalf;
        double arg = a * xn + b * xn * xn;
        if (arg > exp_max) arg = exp_max;
        return voigt * std::exp(arg);
    }
    std::vector<std::string> param_names() const override {
        return {"sigma", "a", "b"};
    }
};
FUNLIB_REGISTER("voigtz", new VoigtZ(c, xmin, xmax));
}
#endif
