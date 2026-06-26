// Airy Ai. Mirror of airyai/fn.py.
#ifndef FUNLIB_AIRYAI_H
#define FUNLIB_AIRYAI_H
#include "funlib_base.h"
#include "funlib_math.h"
namespace funlib {
struct AiryAi : IKernel {
    double arg_lo, arg_hi;
    AiryAi(const Constants& c) {
        arg_lo = c.get("arg_lo", 0);
        arg_hi = c.get("arg_hi", 0);
    }
    double eval(double x, const double* p, int) const override {
        double arg = p[0] * x + p[1];
        if (arg < arg_lo || arg > arg_hi) return kNaN;
        return ROOT::Math::airy_Ai(arg);
    }
    std::vector<std::string> param_names() const override { return {"a", "b"}; }
};
FUNLIB_REGISTER("airyai", new AiryAi(c));
}
#endif
