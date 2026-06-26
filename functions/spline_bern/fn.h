#ifndef FUNLIB_SPLINE_BERN_H
#define FUNLIB_SPLINE_BERN_H
#include "bernstein.h"
#include <fstream>
#include <sstream>
namespace funlib {
// Natural cubic spline; matches scipy CubicSpline to machine precision at
// interior points (boundary-condition influence decays ~4^-k per knot and the
// CSV range extends several GeV beyond the fit window).
struct CubicSplineNat {
    std::vector<double> xs, ys, y2;
    void build(const std::vector<double>& X, const std::vector<double>& Y) {
        xs = X; ys = Y;
        int n = (int)X.size();
        y2.assign(n, 0.0);
        std::vector<double> u(n, 0.0);
        for (int i = 1; i < n - 1; i++) {
            double sig = (xs[i] - xs[i - 1]) / (xs[i + 1] - xs[i - 1]);
            double pp = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / pp;
            double dd = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) -
                        (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]);
            u[i] = (6.0 * dd / (xs[i + 1] - xs[i - 1]) - sig * u[i - 1]) / pp;
        }
        for (int i = n - 2; i >= 0; i--) y2[i] = y2[i] * y2[i + 1] + u[i];
    }
    double operator()(double x) const {
        if (x < xs.front() || x > xs.back()) return kNaN;
        int lo = 0, hi = (int)xs.size() - 1;
        while (hi - lo > 1) {
            int k = (hi + lo) >> 1;
            if (xs[k] > x) hi = k; else lo = k;
        }
        double h = xs[hi] - xs[lo];
        double a = (xs[hi] - x) / h, b = (x - xs[lo]) / h;
        return a * ys[lo] + b * ys[hi] +
               ((a * a * a - a) * y2[lo] + (b * b * b - b) * y2[hi]) * (h * h) / 6.0;
    }
};
struct SplineBern : IKernel {
    Bernstein bern;
    int n;
    CubicSplineNat spl;
    SplineBern(const Constants& c, double xmin, double xmax, int npar)
        : bern(xmin, xmax, npar), n(npar) {
        std::string dir = c.gets("_funcdir", ".");
        std::string variant = c.gets("variant", "fewz");
        std::string fname = (variant == "dyturbo") ? "dyturbo_smooth.csv"
                                                    : "fewz_smooth.csv";
        std::ifstream f(dir + "/" + fname);
        std::vector<double> X, Y;
        std::string line;
        std::getline(f, line);  // header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string a, b;
            std::getline(ss, a, ',');
            std::getline(ss, b, ',');
            X.push_back(std::stod(a));
            Y.push_back(std::stod(b));
        }
        spl.build(X, Y);
    }
    double eval(double x, const double* p, int) const override {
        double sv = spl(x);
        if (!std::isfinite(sv)) return sv;
        if (sv <= 0.0) return kNaN;
        return sv * bern.call(x, p);
    }
    std::vector<std::string> param_names() const override {
        std::vector<std::string> v;
        for (int i = 1; i <= n; i++) v.push_back("c" + std::to_string(i));
        return v;
    }
};
FUNLIB_REGISTER("spline_bern", new SplineBern(c, xmin, xmax, npar));
}
#endif
