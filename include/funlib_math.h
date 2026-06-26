/**
 * funlib_math.h -- special-function and scipy.stats PDF helpers used by the
 * C++ kernels.  Relies on ROOT (Math:: / TMath::) for special functions so
 * the results match the Python kernels (which call the same ROOT routines or
 * scipy, whose closed forms are reproduced here).
 */
#ifndef FUNLIB_MATH_H
#define FUNLIB_MATH_H

#include <cmath>
#include "Math/SpecFunc.h"
#include "TMath.h"

namespace funlib {
namespace fmath {

constexpr double kSqrt2 = 1.41421356237309515;
constexpr double kSqrt2Pi = 2.50662827463100050;
constexpr double kPi = 3.14159265358979312;

// Standard normal pdf / cdf.
inline double norm_pdf(double z) {
    return std::exp(-0.5 * z * z) / kSqrt2Pi;
}
inline double norm_cdf(double z) {
    return 0.5 * std::erfc(-z / kSqrt2);
}

// 1 / Gamma(z) -- entire, returns 0 at non-positive integer poles.
inline double rgamma(double z) {
    double g = std::tgamma(z);
    if (!std::isfinite(g)) return 0.0;
    return 1.0 / g;
}

// Beta function B(a,b) via log-gamma (a,b > 0).
inline double beta_fn(double a, double b) {
    return std::exp(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

// Modified Bessel K_nu(x) (ROOT, matches scipy.special.kv for x>0).
inline double bessel_k(double nu, double x) {
    return ROOT::Math::cyl_bessel_k(nu, x);
}

// Confluent hypergeometric 1F1(a,b,z) (Kummer M).
inline double hyp1f1(double a, double b, double z) {
    return ROOT::Math::conf_hyperg(a, b, z);
}

// Gauss hypergeometric 2F1(a,b;c;z), reproducing scipy.special.hyp2f1.
// ROOT::Math::hyperg (GSL) has a restricted domain; this series with a Pfaff
// transformation converges over z <= 0 (the H->mumu usage, z = -x/x0 < 0).
inline double hyp2f1_series(double a, double b, double c, double z) {
    double term = 1.0, sum = 1.0;
    for (int n = 0; n < 4000; n++) {
        term *= (a + n) * (b + n) / ((c + n) * (n + 1)) * z;
        sum += term;
        if (n > 3 && std::fabs(term) < 1e-17 * std::fabs(sum)) break;
    }
    return sum;
}
inline double hyp2f1(double a, double b, double c, double z) {
    if (z < 0.0) {
        // Pfaff: 2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1)), |z/(z-1)|<1.
        double w = z / (z - 1.0);
        return std::pow(1.0 - z, -a) * hyp2f1_series(a, c - b, c, w);
    }
    return hyp2f1_series(a, b, c, z);
}

// Tricomi confluent hypergeometric U(a,b,z).
inline double conf_hypergU(double a, double b, double z) {
    return ROOT::Math::conf_hypergU(a, b, z);
}

// Parabolic cylinder function D_nu(z), reproduces scipy.special.pbdv(nu,z)[0].
// Two mathematically equivalent forms; the choice avoids cancellation:
//   z >= 0 (decaying tail, the two 1F1 parts cancel) -> stable Tricomi-U form:
//     D_nu(z) = 2^{nu/2} e^{-z^2/4} U(-nu/2; 1/2; z^2/2)
//   z <  0 (growing tail, the two parts add) -> 1F1 (M) two-term form:
//     D_nu(z) = 2^{nu/2} e^{-z^2/4} [ sqrt(pi)/Gamma((1-nu)/2) M(-nu/2;1/2;z^2/2)
//                       - sqrt(2pi) z/Gamma(-nu/2) M((1-nu)/2;3/2;z^2/2) ].
inline double pbdv(double nu, double z) {
    double e = std::exp(-z * z / 4.0);
    double zz = z * z / 2.0;
    double pref = std::pow(2.0, nu / 2.0) * e;
    if (z >= 0.0) {
        return pref * conf_hypergU(-nu / 2.0, 0.5, zz);
    }
    double t1 = std::sqrt(kPi) * rgamma((1.0 - nu) / 2.0) * hyp1f1(-nu / 2.0, 0.5, zz);
    double t2 = kSqrt2Pi * z * rgamma(-nu / 2.0) * hyp1f1((1.0 - nu) / 2.0, 1.5, zz);
    return pref * (t1 - t2);
}

// ---- scipy.stats PDF closed forms (standardized then /scale) ---------------

// betaprime.pdf(x,a,b,loc,scale)
inline double betaprime_pdf(double x, double a, double b, double loc, double scale) {
    double z = (x - loc) / scale;
    if (z <= 0.0) return 0.0;
    double lp = (a - 1.0) * std::log(z) - (a + b) * std::log1p(z) -
                (std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
    return std::exp(lp) / scale;
}

// exponnorm.pdf(x,K,loc,scale) -- exponentially modified Gaussian (EMG).
inline double exponnorm_pdf(double x, double K, double loc, double scale) {
    double y = (x - loc) / scale;
    double arg = 0.5 / (K * K) - y / K;
    // 1/(2K) exp(arg) erfc(-(y - 1/K)/sqrt2)
    double v = 0.5 / K * std::exp(arg) * std::erfc(-(y - 1.0 / K) / kSqrt2);
    return v / scale;
}

// exponweib.pdf(x,a,c,loc,scale)
inline double exponweib_pdf(double x, double a, double c, double loc, double scale) {
    double y = (x - loc) / scale;
    if (y <= 0.0) return 0.0;
    double yc = std::pow(y, c);
    double v = a * c * std::pow(-std::expm1(-yc), a - 1.0) * std::exp(-yc) *
               std::pow(y, c - 1.0);
    return v / scale;
}

// fatiguelife.pdf(x,c,loc,scale) -- Birnbaum-Saunders.
inline double fatiguelife_pdf(double x, double c, double loc, double scale) {
    double y = (x - loc) / scale;
    if (y <= 0.0) return 0.0;
    double v = (y + 1.0) / (2.0 * c * std::sqrt(2.0 * kPi * y * y * y)) *
               std::exp(-(y - 1.0) * (y - 1.0) / (2.0 * c * c * y));
    return v / scale;
}

// gennorm.pdf(x,beta,loc,scale)
inline double gennorm_pdf(double x, double beta, double loc, double scale) {
    double y = (x - loc) / scale;
    double v = beta / (2.0 * std::tgamma(1.0 / beta)) *
               std::exp(-std::pow(std::fabs(y), beta));
    return v / scale;
}

// gompertz.pdf(x,c,loc,scale)
inline double gompertz_pdf(double x, double c, double loc, double scale) {
    double y = (x - loc) / scale;
    if (y < 0.0) return 0.0;
    double v = c * std::exp(y - c * std::expm1(y));
    return v / scale;
}

// invgauss.pdf(x,mu,loc,scale)
inline double invgauss_pdf(double x, double mu, double loc, double scale) {
    double y = (x - loc) / scale;
    if (y <= 0.0) return 0.0;
    double v = 1.0 / std::sqrt(2.0 * kPi * y * y * y) *
               std::exp(-(y - mu) * (y - mu) / (2.0 * mu * mu * y));
    return v / scale;
}

// invgamma.pdf(x,a,loc,scale)
inline double invgamma_pdf(double x, double a, double loc, double scale) {
    double y = (x - loc) / scale;
    if (y <= 0.0) return 0.0;
    double v = std::exp(-(a + 1.0) * std::log(y) - 1.0 / y - std::lgamma(a));
    return v / scale;
}

// johnsonsu.pdf(x,a,b,loc,scale)
inline double johnsonsu_pdf(double x, double a, double b, double loc, double scale) {
    double y = (x - loc) / scale;
    double v = b / std::sqrt(y * y + 1.0) * norm_pdf(a + b * std::asinh(y));
    return v / scale;
}

// levy.pdf(x,loc,scale)
inline double levy_pdf(double x, double loc, double scale) {
    double y = (x - loc) / scale;
    if (y <= 0.0) return 0.0;
    double v = 1.0 / kSqrt2Pi * std::exp(-1.0 / (2.0 * y)) / std::pow(y, 1.5);
    return v / scale;
}

// loggamma.pdf(x,c,loc,scale)
inline double loggamma_pdf(double x, double c, double loc, double scale) {
    double y = (x - loc) / scale;
    double v = std::exp(c * y - std::exp(y) - std::lgamma(c));
    return v / scale;
}

// norminvgauss.pdf(x,a,b,loc,scale)
inline double norminvgauss_pdf(double x, double a, double b, double loc,
                               double scale) {
    double y = (x - loc) / scale;
    double s = std::sqrt(1.0 + y * y);
    double v = a * std::exp(std::sqrt(a * a - b * b) + b * y) *
               bessel_k(1.0, a * s) / (kPi * s);
    return v / scale;
}

// skewnorm.pdf(x,a,loc,scale)
inline double skewnorm_pdf(double x, double a, double loc, double scale) {
    double y = (x - loc) / scale;
    double v = 2.0 * norm_pdf(y) * norm_cdf(a * y);
    return v / scale;
}

}  // namespace fmath
}  // namespace funlib

#endif  // FUNLIB_MATH_H
