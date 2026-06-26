#ifndef FUNLIB_LANDAU_BERN_H
#define FUNLIB_LANDAU_BERN_H
#include "bernstein.h"
#include "landau.h"
namespace funlib {
struct LandauBern : IKernel {
    Bernstein bern;
    Landau landau;
    int n_terms;
    bool free_mpv;
    double landau_mpv;
    LandauBern(const Constants& c, double xmin, double xmax)
        : bern(xmin, xmax, (int)c.get("n_terms", 1)), landau(Constants{}) {
        n_terms = (int)c.get("n_terms", 1);
        free_mpv = (c.gets("variant", "fixed_mpv") == "free_mpv");
        landau_mpv = c.get("landau_mpv", 0);
    }
    double eval(double x, const double* p, int) const override {
        double sigma, mpv;
        const double* bern_c;
        if (free_mpv) {
            mpv = p[0];
            sigma = p[1];
            bern_c = p + 2;
        } else {
            sigma = p[n_terms];
            mpv = landau_mpv;
            bern_c = p;
        }
        return bern.call(x, bern_c) * landau.call(x, mpv, sigma);
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        if (free_mpv) {
            v.push_back("mpv");
            v.push_back("sigma");
            for (int i = 1; i <= n_terms; i++) v.push_back("c" + std::to_string(i));
        } else {
            for (int i = 1; i <= n_terms; i++) v.push_back("c" + std::to_string(i));
            v.push_back("sigma");
        }
        return v;
    }
};
FUNLIB_REGISTER("landau_bern", new LandauBern(c, xmin, xmax));
}
#endif
