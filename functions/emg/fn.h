#ifndef FUNLIB_EMG_H
#define FUNLIB_EMG_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct EMG : IKernel {
    double param_min;
    EMG(const Constants& c) { param_min = c.get("param_min", 1e-9); }
    double eval(double x, const double* p, int) const override {
        double K = std::max(std::fabs(p[0]), param_min);
        double loc = p[1];
        double scale = std::max(std::fabs(p[2]), param_min);
        double v = fmath::exponnorm_pdf(x, K, loc, scale);
        return std::isfinite(v) ? v : kNaN;
    }
    std::vector<std::string> param_names() const override {
        return {"K", "loc", "scale"};
    }
};
FUNLIB_REGISTER("emg", new EMG(c));
}
#endif
