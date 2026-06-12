"""
FunLib/symbolic/special_functions.py -- Custom SymPy Function subclasses.

Design principles
-----------------
- The parser returns a **pure SymPy symbolic expression** for every formula.
- Numerical evaluation uses SymPy's own infrastructure (`expr.evalf()` / `N(expr)`)
  which delegates to **mpmath** -- no scipy, no ROOT, no numpy required.
- For truly non-standard functions (TricomiU, LandauPDF, ...) custom SymPy
  `Function` subclasses implement `_eval_evalf` via mpmath.
- "Thin" functions that are expressible in terms of SymPy natives
  (NormalPDF, NormalCDF, GammaSp, RegUpperGamma, ...) define `eval()` class
  methods that fold them into native SymPy immediately during `sympify`.

Two categories
--------------
A.  Fold-through classes  -- define `eval()`, disappear into native SymPy:
      GammaSp  -> sp.gamma(x)
      BetaSp   -> sp.beta(a, b)
      Hyp1F1   -> sp.hyper([a], [b], z)
      Hyp2F1   -> sp.hyper([a, b], [c], z)
      NormalPDF  -> exp(-x^2/2) / sqrt(2pi)
      NormalCDF  -> (1 + erf(x/sqrt2)) / 2
      RegUpperGamma -> uppergamma(s, x) / gamma(s)
      RegLowerGamma -> lowergamma(s, x) / gamma(s)
      RegLowerGamma -> lowergamma(s, x) / gamma(s)

B.  Persistent classes  -- no analytic SymPy form; evaluate via mpmath:
      TricomiU(a, b, z)         -> mpmath.hyperu
      ParabolicCylD(nu, x)      -> mpmath.pcfd
      MittagLefflerE(alpha, z)  -> custom alternating/asymptotic series
      VoigtProfile(x, sigma, gamma) -> Faddeeva via mpmath.erfc(-iz)
      LandauPDF(x, mu, sigma)   -> CERNLIB C305 (8-region rational approx.)

LAMBDIFY_NS
-----------
Dict ready for use as the first element of lambdify's `modules` argument:

    from FunLib.symbolic.special_functions import LAMBDIFY_NS
    f = lambdify(args, expr, modules=[LAMBDIFY_NS, 'mpmath'])

After fold-through, only persistent classes need explicit entries.  All SymPy
native special functions (erfc, gamma, besselk, airyai, uppergamma, hyper, ...)
are handled automatically by `modules='mpmath'`.

Naming
------
GammaSp / BetaSp use collision-safe names to avoid shadowing shape parameters
called `gamma` / `beta` during lambdify variable binding.  Because no current
function has BOTH a `gamma` parameter AND uses Gamma(...) in its formula (verified),
the fold-through to sp.gamma is safe in practice.
"""

import math as _math
import mpmath as _mp
import sympy as sp
from sympy import Function

__all__ = [
    # Fold-through classes
    "GammaSp",
    "BetaSp",
    "Hyp1F1",
    "Hyp2F1",
    "NormalPDF",
    "NormalCDF",
    "RegUpperGamma",
    "RegLowerGamma",
    # Persistent classes
    "TricomiU",
    "ParabolicCylD",
    "MittagLefflerE",
    "VoigtProfile",
    "LandauPDF",
    # Lambdify evaluation namespace
    "LAMBDIFY_NS",
    # Backward-compat aliases used in latex_parser expression strings
    "_PARSER_ALIAS",
]


# -- A. Fold-through classes (eval -> native SymPy) -----------------------------


class GammaSp(Function):
    """Euler Gamma function Gamma(x).

    Named GammaSp to avoid collision with shape parameters called `gamma`.
    Folds immediately to sp.gamma(x) during sympify.
    """

    nargs = 1

    @classmethod
    def eval(cls, x):
        return sp.gamma(x)


class BetaSp(Function):
    """Euler Beta function B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b).

    Named BetaSp to avoid collision with shape parameters called `beta`.
    Folds immediately to sp.beta(a, b) during sympify.
    """

    nargs = 2

    @classmethod
    def eval(cls, a, b):
        return sp.beta(a, b)


class Hyp1F1(Function):
    """Confluent hypergeometric _1F_1(a; b; z) = Kummer M(a, b, z).

    Folds to sp.hyper([a], [b], z) during sympify.
    lambdify with 'mpmath' maps hyper -> mpmath.hyper automatically.
    """

    nargs = 3

    @classmethod
    def eval(cls, a, b, z):
        return sp.hyper([a], [b], z)


class Hyp2F1(Function):
    """Gauss hypergeometric _2F_1(a, b; c; z).

    Folds to sp.hyper([a, b], [c], z) during sympify.
    """

    nargs = 4

    @classmethod
    def eval(cls, a, b, c, z):
        return sp.hyper([a, b], [c], z)


class NormalPDF(Function):
    """Standard normal density phi(x) = exp(-x^2/2) / sqrt(2pi).

    Folds to a SymPy expression containing only exp, sqrt, pi.
    """

    nargs = 1

    @classmethod
    def eval(cls, x):
        return sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)


class NormalCDF(Function):
    """Standard normal CDF Phi(x) = (1 + erf(x/sqrt2)) / 2.

    Folds to a SymPy expression using sp.erf.
    """

    nargs = 1

    @classmethod
    def eval(cls, x):
        return (1 + sp.erf(x / sp.sqrt(2))) / 2


class RegUpperGamma(Function):
    """Regularised upper incomplete gamma Q(s, x) = Gamma(s, x) / Gamma(s).

    Folds to sp.uppergamma(s, x) / sp.gamma(s).
    With 'mpmath' lambdify: uppergamma -> mpmath.gammainc (unregularised),
    gamma -> mpmath.gamma, so the ratio gives the correct Q.
    """

    nargs = 2

    @classmethod
    def eval(cls, s, x):
        return sp.uppergamma(s, x) / sp.gamma(s)


class RegLowerGamma(Function):
    """Regularised lower incomplete gamma P(s, x) = gamma(s, x) / Gamma(s).

    Folds to sp.lowergamma(s, x) / sp.gamma(s).
    With 'mpmath' lambdify: lowergamma -> mpmath.gammainc (normalised=False),
    gamma -> mpmath.gamma, so the ratio gives the correct P.
    """

    nargs = 2

    @classmethod
    def eval(cls, s, x):
        return sp.lowergamma(s, x) / sp.gamma(s)


# -- B. Persistent classes (eval_evalf -> mpmath) -------------------------------


class TricomiU(Function):
    """Tricomi confluent hypergeometric function U(a, b, z).

    No SymPy native form.  Evaluates via mpmath.hyperu.
    Also used for Whittaker W decomposition (the W=e^...z^...U part in the LaTeX
    definition is parsed from the right-hand side of the W=...U equation).
    """

    nargs = 3

    def _eval_evalf(self, prec):
        try:
            a, b, z = [float(arg.evalf(n=max(15, prec // 3))) for arg in self.args]
            with _mp.workprec(prec):
                result = _mp.hyperu(_mp.mpf(a), _mp.mpf(b), _mp.mpf(z))
            return sp.Float(float(result), prec)
        except Exception:
            return None


class ParabolicCylD(Function):
    """Parabolic cylinder function D_nu(x).

    Evaluates via mpmath.pcfd(nu, x).
    """

    nargs = 2

    def _eval_evalf(self, prec):
        try:
            nu, x = [float(arg.evalf(n=max(15, prec // 3))) for arg in self.args]
            with _mp.workprec(prec):
                result = _mp.pcfd(_mp.mpf(nu), _mp.mpf(x))
            return sp.Float(float(result), prec)
        except Exception:
            return None


class MittagLefflerE(Function):
    """Mittag-Leffler function E_alpha(z) = Sigma_{k>=0} z^k / Gamma(alphak + 1).

    Mirrors the algorithm in mittagleffler/fn.py:
    - Convergent alternating series for |z| <= 8
    - Asymptotic expansion for |z| > 8
    """

    nargs = 2

    def _eval_evalf(self, prec):
        try:
            alpha, z = [float(arg.evalf(n=max(15, prec // 3))) for arg in self.args]
            result = _mittag_leffler_num(alpha, z)
            return sp.Float(result, prec)
        except Exception:
            return None


class VoigtProfile(Function):
    """Voigt profile V(x; sigma_G, gamma_L).

    V(x; sigma, gamma) = Re[w(z)] / (sigma sqrt(2pi))
    where w(z) = exp(-z^2) erfc(-iz) is the Faddeeva function and
    z = (x + igamma) / (sigma sqrt2).

    Evaluates entirely via mpmath -- no ROOT, no scipy required.
    """

    nargs = 3

    def _eval_evalf(self, prec):
        try:
            x, sigma, gamma = [
                float(arg.evalf(n=max(15, prec // 3))) for arg in self.args
            ]
            result = _voigt_profile_num(x, sigma, gamma)
            return sp.Float(result, prec)
        except Exception:
            return None


class LandauPDF(Function):
    """Landau probability density function f(x; mu, sigma) = (1/sigma) phi((x-mu)/sigma).

    phi(v) is computed via the CERNLIB G110 / Kolbig-Schorr (1984) eight-region
    rational approximation, exactly matching ROOT::Math::landau_pdf.

    Mathematical definition (Bromwich integral):
      phi(v) = (1/(2pii)) integral_{c-iinf}^{c+iinf} exp(sv + s ln s) ds,  c > 0

    The CERNLIB rational approximations are the standard numerical evaluation
    of this integral in HEP software.  They are NOT ROOT implementation details
    -- they are the accepted physics algorithm, published in Comp. Phys. Comm. 31
    (1984) 97-111 and used in CERNLIB, ROOT, and GSL.
    """

    nargs = 3  # (x, mu, sigma)

    def _eval_evalf(self, prec):
        try:
            x, mu, sigma = [float(arg.evalf(n=max(15, prec // 3))) for arg in self.args]
            result = _landau_pdf_num(x, mu, sigma)
            return sp.Float(result, prec)
        except Exception:
            return None


# -- Numerical implementations -------------------------------------------------


def _mittag_leffler_num(alpha, z, n_terms=130):
    """Mittag-Leffler E_alpha(z): series for |z|<=8, asymptotic for |z|>8."""
    a, z = float(alpha), float(z)
    az = abs(z)
    if az < 1e-12:
        return 1.0
    if az <= 8.0:
        s = 0.0
        for k in range(n_terms):
            try:
                lt = k * _math.log(az) - _math.lgamma(a * k + 1.0)
            except (ValueError, OverflowError):
                break
            if lt < -60.0 and k > 5:
                continue
            t = _math.exp(lt)
            s += t if (z > 0 or k % 2 == 0) else -t
        return s
    # Asymptotic expansion z -> -inf: E_alpha(z) ~ -Sigma_{k=1}^{4} z^{-k}/Gamma(1-alphak)
    s = 0.0
    for k in range(1, 5):
        try:
            inv = 1.0 / _math.gamma(1.0 - a * k)
        except (ValueError, OverflowError, ZeroDivisionError):
            inv = 0.0
        try:
            s -= inv * z ** (-k)
        except (OverflowError, ZeroDivisionError):
            pass
    return s


def _voigt_profile_num(x, sigma, gamma):
    """Voigt profile V(x; sigma, gamma) via Faddeeva function (mpmath)."""
    try:
        x, sigma, gamma = float(x), float(sigma), float(gamma)
        if sigma <= 0:
            return float("nan")
        z = _mp.mpc(x, gamma) / (sigma * _mp.sqrt(2))
        # w(z) = exp(-z^2) erfc(-iz) is the Faddeeva function
        w = _mp.exp(-(z**2)) * _mp.erfc(-1j * z)
        return float(w.real / (sigma * _mp.sqrt(2 * _mp.pi)))
    except Exception:
        return float("nan")


# -- CERNLIB G110 / Kolbig-Schorr (1984) Landau PDF ---------------------------
# Adapted from ROOT::Math::landau_pdf (PdfFuncMathCore.cxx) and verified to
# give bit-identical results to ROOT.TMath.Landau(x, mu, sigma, True).

_LANDAU_P1 = (
    0.4259894875,
    -0.1249762550,
    0.03984243700,
    -0.006298287635,
    0.001511162253,
)
_LANDAU_Q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)
_LANDAU_P2 = (
    0.1788541609,
    0.1173957403,
    0.01488850518,
    -0.001394989411,
    0.0001283617211,
)
_LANDAU_Q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)
_LANDAU_P3 = (
    0.1788544503,
    0.09359161662,
    0.006325387654,
    0.00006611667319,
    -0.000002031049101,
)
_LANDAU_Q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)
_LANDAU_P4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
_LANDAU_Q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)
_LANDAU_P5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
_LANDAU_Q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)
_LANDAU_P6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
_LANDAU_Q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)
_LANDAU_A1 = (0.04166666667, -0.01996527778, 0.02709538966)
_LANDAU_A2 = (-1.845568670, -4.284640743)


def _landau_horner(p, x):
    """Evaluate polynomial Sigmap[i]x^i via Horner's method."""
    return (((p[4] * x + p[3]) * x + p[2]) * x + p[1]) * x + p[0]


def _landau_std(v):
    """Standard Landau density phi(v) = ROOT::Math::landau_pdf(v, 1, 0)."""
    if v < -5.5:
        u = _math.exp(v + 1.0)
        if u < 1e-10:
            return 0.0
        ue = _math.exp(-1.0 / u)
        us = _math.sqrt(u)
        # Uses a1 coefficients for the correction factor
        return (
            0.3989422803
            * (ue / us)
            * (1.0 + (_LANDAU_A1[0] + (_LANDAU_A1[1] + _LANDAU_A1[2] * u) * u) * u)
        )
    elif v < -1.0:
        # KEY: exp factor uses u = exp(-v-1), but polynomial evaluated in v
        u = _math.exp(-v - 1.0)
        return (
            _math.exp(-u)
            * _math.sqrt(u)
            * _landau_horner(_LANDAU_P1, v)
            / _landau_horner(_LANDAU_Q1, v)
        )
    elif v < 1.0:
        return _landau_horner(_LANDAU_P2, v) / _landau_horner(_LANDAU_Q2, v)
    elif v < 5.0:
        return _landau_horner(_LANDAU_P3, v) / _landau_horner(_LANDAU_Q3, v)
    elif v < 12.0:
        u = 1.0 / v
        return u * u * _landau_horner(_LANDAU_P4, u) / _landau_horner(_LANDAU_Q4, u)
    elif v < 50.0:
        u = 1.0 / v
        return u * u * _landau_horner(_LANDAU_P5, u) / _landau_horner(_LANDAU_Q5, u)
    elif v < 300.0:
        u = 1.0 / v
        return u * u * _landau_horner(_LANDAU_P6, u) / _landau_horner(_LANDAU_Q6, u)
    else:
        u = 1.0 / (v - v * _math.log(v) / (v + 1.0))
        return u * u * (1.0 + (_LANDAU_A2[0] + _LANDAU_A2[1] * u) * u)


def _landau_pdf_num(x, mu, sigma):
    """Landau PDF f(x; mu, sigma) -- equivalent to ROOT.TMath.Landau(x,mu,sigma,True)."""
    try:
        x, mu, sigma = float(x), float(mu), float(sigma)
        if sigma <= 0.0:
            return 0.0
        return _landau_std((x - mu) / sigma) / sigma
    except (OverflowError, ValueError, ZeroDivisionError):
        return 0.0


# -- Lambdify evaluation namespace ---------------------------------------------
# Only persistent classes need explicit entries -- fold-through classes disappear
# during sympify and their native SymPy representations are handled by 'mpmath'.

LAMBDIFY_NS = {
    # Persistent custom Function classes -> mpmath-backed callables
    "TricomiU": lambda a, b, z: float(_mp.hyperu(a, b, z)),
    "ParabolicCylD": lambda nu, x: float(_mp.pcfd(nu, x)),
    "MittagLefflerE": _mittag_leffler_num,
    "VoigtProfile": _voigt_profile_num,
    "LandauPDF": _landau_pdf_num,
    # Piecewise is handled natively by lambdify + mpmath (generates if/else).
    # exp, log, sqrt etc. come from 'mpmath' module in the modules list.
}

# Backward-compatibility dict: expression-string names -> SymPy classes
# (used in latex_parser's SYMPY_NS and sympify locals).
_PARSER_ALIAS = {
    "gamma_sp": GammaSp,
    "beta_sp": BetaSp,
}
