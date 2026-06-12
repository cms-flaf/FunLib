#!/usr/bin/env python3
"""
FunLib/tools/latex_parser.py -- Convert a LaTeX formula string to a callable.

Parses a function LaTeX formula of the form ``f(x) = <expr>`` into a
Python callable using SymPy for symbolic parsing and lambdify for numerical
evaluation.

Architecture
------------
The parser returns a **pure SymPy symbolic expression**.  Numerical evaluation
uses SymPy's own infrastructure (lambdify -> mpmath under the hood).

Special functions fall into two categories:
  - SymPy native  (erfc, erf, gamma, loggamma, besselk, airyai, uppergamma,
    beta, hyper ...): handled automatically by ``modules='mpmath'`` in lambdify.
  - Custom classes (TricomiU, ParabolicCylD, MittagLefflerE, VoigtProfile,
    LandauPDF): defined in FunLib/symbolic/special_functions.py with mpmath
    _eval_evalf methods; listed explicitly in LAMBDIFY_NS.

"Thin" convenience classes (GammaSp, BetaSp, Hyp1F1, Hyp2F1, NormalPDF,
NormalCDF, RegUpperGamma) define eval() methods that fold them immediately
into native SymPy during sympify -- they do not appear in the final expression.

Dependencies
------------
- sympy        (parsing, lambdify, all special function symbolics)
- mpmath       (numerical evaluation backend -- via lambdify 'mpmath' module)
- FunLib.symbolic (custom Function subclasses + LAMBDIFY_NS)

No ROOT, no scipy, no numpy required.

Public API
----------
    from FunLib.tools.latex_parser import build_ref_fn

    ref_fn = build_ref_fn(
        key, latex_str, param_names, constants,
        latex_var_map=None, latex_aux=None
    )
    value = ref_fn(x, p1, p2, ...)   # x and params as positional floats
    # raises ValueError if the formula cannot be parsed
"""

from __future__ import annotations

import re
import sympy as sp
from sympy import lambdify, sympify, Symbol, pi as sp_pi

# Custom SymPy Function subclasses + lambdify evaluation namespace
from ..symbolic.special_functions import (
    GammaSp,
    BetaSp,
    TricomiU,
    Hyp1F1,
    Hyp2F1,
    ParabolicCylD,
    MittagLefflerE,
    NormalPDF,
    NormalCDF,
    RegUpperGamma,
    RegLowerGamma,
    VoigtProfile,
    LandauPDF,
    LAMBDIFY_NS,
    _PARSER_ALIAS,
)

__all__ = ["build_ref_fn", "SYMPY_NS", "FUNC_NAMES"]


# -- SymPy namespace for sympify -----------------------------------------------
# Maps string tokens in the converted expression to SymPy objects.
# Must contain every function name that appears in the expression strings
# produced by _conv_specials / _full_convert.

SYMPY_NS = {
    # Basic math
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "atan": sp.atan,
    "asinh": sp.asinh,
    "asin": sp.asin,
    "acos": sp.acos,
    "tanh": sp.tanh,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "sin": sp.sin,
    "cos": sp.cos,
    "pi": sp_pi,
    "e": sp.E,
    "Piecewise": sp.Piecewise,
    # SymPy native special functions
    # (lambdify maps these to mpmath automatically via modules='mpmath')
    "erfc": sp.erfc,
    "erf": sp.erf,
    "airyai": sp.airyai,
    "besselk": sp.besselk,
    "loggamma": sp.loggamma,
    "uppergamma": sp.uppergamma,
    # Collision-safe aliases (fold to sp.gamma / sp.beta via eval())
    "gamma_sp": GammaSp,
    "beta_sp": BetaSp,
    # Fold-through convenience classes (eval() -> native SymPy immediately)
    "Hyp1F1": Hyp1F1,
    "Hyp2F1": Hyp2F1,
    "NormalPDF": NormalPDF,
    "NormalCDF": NormalCDF,
    "RegUpperGamma": RegUpperGamma,
    "RegLowerGamma": RegLowerGamma,
    # Persistent custom classes (stay in expression, evaluated by LAMBDIFY_NS)
    "TricomiU": TricomiU,
    "ParabolicCylD": ParabolicCylD,
    "MittagLefflerE": MittagLefflerE,
    "VoigtProfile": VoigtProfile,
    "LandauPDF": LandauPDF,
    # landau_pdf (expression-level alias -> LandauPDF)
    "landau_pdf": LandauPDF,
}

# Lambdify evaluation namespace: persistent custom classes + any extras.
# SymPy native functions are handled by modules='mpmath' automatically.
_EVAL_NS = dict(LAMBDIFY_NS)
# Alias: landau_pdf in expression strings maps to the same LandauPDF callable
_EVAL_NS["landau_pdf"] = LAMBDIFY_NS.get("LandauPDF", lambda x, m, s: 0.0)

# True function names -- used by _add_mult to suppress * insertion before (
# Must include every function name that can appear in expression strings.
FUNC_NAMES = (set(SYMPY_NS) | set(_EVAL_NS) | {"float", "int", "pow", "comb"}) - {
    "pi",
    "inf",
    "e",
}


# -- LaTeX preprocessor --------------------------------------------------------


def _preprocess(s):
    """Strip LaTeX formatting that does not affect mathematical meaning."""
    # Negative space \! -> remove
    s = s.replace(r"\!", "")
    # Positive spacing commands -> single space (keeps adjacent symbols separated!)
    s = re.sub(
        r"\\(?:,|;|:|thinspace|medspace|thickspace|enspace|quad|qquad)\s*", " ", s
    )
    s = re.sub(r"\\ ", " ", s)
    # Size modifiers before brackets -- longest-first alternation
    s = re.sub(
        r"\\(?:Biggr|Biggl|Bigg|Bigr|Bigl|Big|biggr|biggl|bigg|bigr|bigl|big)\s*", "", s
    )
    # \left and \right
    s = re.sub(r"\\left\s*\.", "", s)
    s = re.sub(r"\\right\s*\.", "", s)
    s = re.sub(r"\\(?:left|right)\s*", "", s)
    # Font commands
    s = re.sub(r"\\(?:mathrm|mathit|mathbf|mathbb|text|mbox)\{([^}]*)\}", r"\1", s)
    # Decorations
    s = re.sub(r"\\(?:hat|tilde|bar|vec|dot)\{([^}]*)\}", r"\1", s)
    # dfrac / tfrac -> frac
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    # \frac12 (single digit, no braces)
    s = re.sub(r"\\frac([0-9])([0-9])", r"((\1)/(\2))", s)
    # \overline -> keep inner
    s = re.sub(r"\\overline\{([^}]*)\}", r"\1", s)
    # \cdot, \times -> *
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    # \min, \max
    s = s.replace(r"\min", "min").replace(r"\max", "max")
    # Known subscript patterns -> Python constant names
    s = re.sub(r"x_\{mid\}", "X0", s)
    s = re.sub(r"x_\{min\}", "XMIN", s)
    s = re.sub(r"x_\{max\}", "XMAX", s)
    # Semicolons in function argument lists -> commas
    s = s.replace(";", ",")
    # Math brackets -> parens
    s = s.replace("[", "(").replace("]", ")")
    # Remove trailing \ (line break)
    s = re.sub(r"\\\s*$", "", s.strip())
    return s


# -- Special pattern expansion -------------------------------------------------


def _conv_binom(s: str) -> str:
    """Replace \\binom{a}{b} where both a and b are integer literals with comb(a, b)."""
    from math import comb

    def _rep(m):
        return str(comb(int(m.group(1)), int(m.group(2))))

    for _ in range(10):
        ns = re.sub(r"\\binom\{(\d+)\}\{(\d+)\}", _rep, s)
        if ns == s:
            break
        s = ns
    return s


def _expand_fn_defs(s: str, fn_defs: dict | None) -> str:
    """
    Expand indexed function calls defined via ``latex_fn_defs`` in desc.yaml.

    *fn_defs* is a dict whose keys are patterns like ``"T_0(u)"``, ``"T_1(u)"``
    (base cases with integer subscripts) or ``"T_k(u)"`` (recurrence rule with
    a single-letter index variable).  Values are Python-expression strings; the
    recurrence RHS may reference the same function with lower concrete indices.

    Example (Chebyshev)::

        {"T_0(u)": "1", "T_1(u)": "u", "T_k(u)": "2*u*T_{k-1}(u)-T_{k-2}(u)"}

    Algorithm
    ---------
    1. Parse entries into base-case tables and at most one recurrence rule,
       grouped by ``(function_name, argument_string)``.
    2. For every ``FnName_N(arg)`` token in *s* (concrete integer N): memoised
       recursive evaluation applies the recurrence -- substituting arithmetic
       expressions (``k-1 -> N-1``, etc.) then the bare variable (``k -> N``) --
       until only base-case leaves remain.
    3. Repeat (guarded at 50 passes) until *s* contains no more expandable tokens.
    """
    if not fn_defs:
        return s

    # -- Parse fn_defs into groups keyed by (fn_name, arg_str) -----------------
    # base[(fn, arg)][n]  = rhs_str          integer-subscript entries
    # rec[(fn, arg)]       = (idx_var, tmpl)  letter-subscript recurrence rule
    base: dict = {}
    rec: dict = {}

    for lhs, rhs in fn_defs.items():
        lhs = lhs.strip()
        # Integer-subscript: FnName_N(arg) or FnName_{N}(arg)
        m = re.match(r"^(\w+)_\{?(\d+)\}?\(([^)]*)\)$", lhs)
        if m:
            fn, n, arg = m.group(1), int(m.group(2)), m.group(3).strip()
            base.setdefault((fn, arg), {})[n] = rhs
            continue
        # Letter-subscript: FnName_k(arg) or FnName_{k}(arg)
        m = re.match(r"^(\w+)_\{?([a-zA-Z])\}?\(([^)]*)\)$", lhs)
        if m and not m.group(2).isdigit():
            fn, var, arg = m.group(1), m.group(2), m.group(3).strip()
            rec[(fn, arg)] = (var, rhs)

    if not base and not rec:
        return s

    all_groups = set(base) | set(rec)
    memo: dict = {}  # (fn, arg, n) -> expanded Python expression

    def _eval(fn: str, arg: str, n: int) -> str | None:
        key = (fn, arg, n)
        if key in memo:
            return memo[key]
        # Base case?
        b = base.get((fn, arg), {})
        if n in b:
            memo[key] = b[n]
            return b[n]
        # Recurrence?
        if (fn, arg) not in rec:
            return None
        idx_var, tmpl = rec[(fn, arg)]
        # Substitute arithmetic in the template first (k-1 -> N-1, k-2 -> N-2, ...)
        # then replace the bare index variable (k -> N).
        t = re.sub(
            re.escape(idx_var) + r"\s*-\s*(\d+)",
            lambda mm, _n=n: str(_n - int(mm.group(1))),
            tmpl,
        )
        t = re.sub(
            r"(?<![a-zA-Z0-9_])" + re.escape(idx_var) + r"(?![a-zA-Z0-9_])",
            str(n),
            t,
        )
        # Recursively expand any FnName_M(arg) tokens remaining in t.
        for _g in range(n + 5):
            pat = re.compile(re.escape(fn) + r"_\{?(\d+)\}?\(" + re.escape(arg) + r"\)")
            mm = pat.search(t)
            if not mm:
                break
            sub_n = int(mm.group(1))
            sub_v = _eval(fn, arg, sub_n)
            if sub_v is None:
                break
            t = t[: mm.start()] + f"({sub_v})" + t[mm.end() :]
        memo[key] = t
        return t

    # -- Apply expansion to s until no more expandable tokens remain ------------
    for _guard in range(50):
        changed = False
        for fn, arg in all_groups:
            pat = re.compile(re.escape(fn) + r"_\{?(\d+)\}?\(" + re.escape(arg) + r"\)")
            mm = pat.search(s)
            if not mm:
                continue
            n = int(mm.group(1))
            result = _eval(fn, arg, n)
            if result is None:
                continue
            s = s[: mm.start()] + f"({result})" + s[mm.end() :]
            changed = True
            break  # restart outer loop after any substitution
        if not changed:
            break

    return s


def _expand_sum(s: str, constants: dict | None = None) -> str:
    """
    Generic \\sum_{var=lo}^{hi} body expansion.

    Finds every ``\\sum_{var=lo}^{hi}`` in *s* whose bounds *lo* and *hi*
    resolve to concrete integers (integer literals **or** names present in
    *constants*).  For each such sum:

    1. The body extends from immediately after the ``\\sum`` token to the
       first *unmatched* ``)`` (allowing sums inside ``\\exp(...)`` or
       similar), or to the end of the string if no unmatched ``)`` exists.
    2. Named integer constants from *constants* are substituted in the body
       so that expressions such as ``\\binom{n}{k}`` and ``(1-t)^{n-k}``
       contain only concrete values after index substitution.
    3. The summation variable is substituted with each integer from *lo* to
       *hi* inclusive, producing one term per value.
    4. The terms are joined with ``+`` and the ``\\sum`` token is replaced.

    After all expansions ``\\binom{a}{b}`` with integer-literal arguments is
    evaluated to its numeric value by :func:`_conv_binom`.
    """
    constants = dict(constants or {})

    def _resolve_int(val_str: str):
        val_str = val_str.strip()
        try:
            return int(val_str)
        except ValueError:
            pass
        val = constants.get(val_str)
        if val is not None:
            try:
                iv = int(val)
                if float(iv) == float(val):
                    return iv
            except (TypeError, ValueError):
                pass
        return None

    sum_pat = re.compile(r"\\sum_\{([a-zA-Z]+)=([^{}]+)\}\^\{([^{}]+)\}")

    for _guard in range(10):  # at most 10 sums per formula
        m = sum_pat.search(s)
        if not m:
            break
        var = m.group(1)
        lo = _resolve_int(m.group(2))
        hi = _resolve_int(m.group(3))
        if lo is None or hi is None:
            break  # unresolvable bounds -- leave the \sum as-is

        pre = s[: m.start()]
        rest = s[m.end() :]

        # Body ends at the first unmatched delimiter: ')' for sums inside
        # \exp(...) or similar, '}' for sums inside \frac{...}{...}.
        # Track paren and brace depth independently.
        paren_d = 0
        brace_d = 0
        body_end = len(rest)
        for idx, ch in enumerate(rest):
            if ch == "(":
                paren_d += 1
            elif ch == ")":
                if paren_d == 0:
                    body_end = idx
                    break
                paren_d -= 1
            elif ch == "{":
                brace_d += 1
            elif ch == "}":
                if brace_d == 0:
                    body_end = idx
                    break
                brace_d -= 1
        body = rest[:body_end]
        post = rest[body_end:]

        # Substitute named integer constants in the body first so that e.g.
        # \binom{n}{k} and (1-t)^{n-k} contain concrete numbers after index
        # substitution.  Only substitute names whose value is a pure integer.
        body_c = body
        for cname, cval in constants.items():
            try:
                iv = int(cval)
                if float(iv) == float(cval):
                    body_c = re.sub(
                        r"(?<![a-zA-Z_])" + re.escape(cname) + r"(?![a-zA-Z0-9_])",
                        str(iv),
                        body_c,
                    )
            except (TypeError, ValueError):
                pass

        terms = []
        for k_val in range(lo, hi + 1):
            term = re.sub(
                r"(?<![a-zA-Z])" + re.escape(var) + r"(?![a-zA-Z0-9_])",
                str(k_val),
                body_c,
            )
            terms.append(term)

        s = pre + "+".join(terms) + post

    # Evaluate all \binom{a}{b} with concrete integer arguments.
    return _conv_binom(s)


def _expand_cases(s, sym_map):
    """Convert \\begin{cases}...\\end{cases} to Piecewise(...) string."""
    m = re.search(r"\\begin\{cases\}(.*?)\\end\{cases\}", s, re.DOTALL)
    if not m:
        return s
    inner = m.group(1)
    rows = re.split(r"\\\\|\\(?=\s|$)", inner)
    branches = []
    for row in rows:
        if "&" not in row:
            continue
        expr_latex, cond_latex = row.split("&", 1)
        cond = cond_latex.strip()
        cond = re.sub(r"\\(?:le|leq)\b", "<=", cond)
        cond = re.sub(r"\\(?:ge|geq)\b", ">=", cond)
        cond = _full_convert(cond, sym_map)
        expr = _full_convert(expr_latex.strip(), sym_map)
        branches.append((expr, cond))

    if not branches:
        return s

    pairs = ", ".join(f"({e}, {c})" for e, c in branches[:-1])
    if pairs:
        pairs += f", ({branches[-1][0]}, True)"
    else:
        pairs = f"({branches[-1][0]}, True)"
    replacement = f"Piecewise({pairs})"
    sep = (
        " "
        if m.start() > 0 and (s[m.start() - 1].isalnum() or s[m.start() - 1] == "_")
        else ""
    )
    return s[: m.start()] + sep + replacement + s[m.end() :]


# -- Iterative LaTeX->Python expression converters -----------------------------


def _conv_fracs(s):
    """Replace innermost \\frac{A}{B} -> ((A)/(B))."""
    for _ in range(30):
        ns = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        if ns == s:
            break
        s = ns
    return s


def _conv_sqrt(s):
    for _ in range(10):
        ns = re.sub(r"\\sqrt\{([^{}]*)\}", r"((\1)**0.5)", s)
        if ns == s:
            break
        s = ns
    return s


def _conv_powers(s):
    """^{A} -> **(A), and e^{A} -> exp(A)."""
    for _ in range(10):
        ns = re.sub(r"\be\^\{([^{}]*)\}", r"exp(\1)", s)
        if ns == s:
            break
        s = ns
    for _ in range(10):
        ns = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
        if ns == s:
            break
        s = ns
    s = re.sub(r"\^([0-9a-zA-Z])", r"**\1", s)
    return s


def _conv_subscripts(s, sym_map):
    """_{A} -> remove braces: _A."""
    for _ in range(10):
        ns = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
        if ns == s:
            break
        s = ns
    return s


# -- Special function conversions ----------------------------------------------


def _conv_specials(s):
    """Convert LaTeX special function notation to Python call strings.

    Emits names that match:
    - SymPy native function names (erfc, airyai, besselk, loggamma, uppergamma)
    - Custom SymPy Function class names (TricomiU, Hyp1F1, etc.)
    - Collision-safe aliases (gamma_sp, beta_sp)
    - ROOT-backed alias (landau_pdf -- step-2 open problem)
    """
    # -- MUST run first: compound patterns that start with \ln/\log ------------
    # These must precede the bare \ln / \log -> log conversions below, otherwise
    # \ln\Gamma( gets split into log + \Gamma( before we can catch it.
    s = re.sub(r"\\ln\s*\\Gamma\s*\(", "loggamma(", s)
    s = re.sub(r"\\log\s*\\Gamma\s*\(", "loggamma(", s)

    # \exp(...) -- add * when preceded by identifier
    s = re.sub(r"([a-zA-Z0-9_)])\s*\\exp\s*\(", r"\1*exp(", s)
    s = re.sub(r"\\exp\s*\(", "exp(", s)
    s = re.sub(r"([a-zA-Z0-9_)])\s*\\exp\s*\{", r"\1*exp(", s)
    s = re.sub(r"\\exp\s*\{", "exp(", s)
    # e^x
    s = re.sub(r"\be\*\*\(([^()]*)\)", r"exp(\1)", s)
    # \ln, \log
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\ln\s*\(", r"\1*log(", s)
    s = re.sub(r"\\ln\s*\(", "log(", s)
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\ln\s+([a-zA-Z0-9_])", r"\1*log(\2)", s)
    s = re.sub(r"\\ln\s+([a-zA-Z0-9_])", r"log(\1)", s)
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\ln\b", r"\1*log", s)
    s = re.sub(r"\\ln\b", "log", s)
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\log\s*\(", r"\1*log(", s)
    s = re.sub(r"\\log\s*\(", "log(", s)
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\log\s+([a-zA-Z0-9_])", r"\1*log(\2)", s)
    s = re.sub(r"\\log\s+([a-zA-Z0-9_])", r"log(\1)", s)
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\log\b", r"\1*log", s)
    s = re.sub(r"\\log\b", "log", s)
    # \sqrt (after brace removal, bare \sqrt( form)
    s = re.sub(r"\\sqrt\s*\(", "(", s)

    # -- Bessel K -> SymPy native besselk(nu, x) -------------------------------
    s = re.sub(r"K_\{0\}\s*\(", "besselk(0,", s)
    s = re.sub(r"K_\{1\}\s*\(", "besselk(1,", s)
    s = re.sub(r"\bK_0\s*\(", "besselk(0,", s)
    s = re.sub(r"\bK_1\s*\(", "besselk(1,", s)
    s = re.sub(r"K_\{\\nu\}\s*\(", "besselk(nu,", s)
    s = re.sub(r"K_\{\\lambda-1/2\}\s*\(", "besselk(lam-0.5,", s)
    s = re.sub(r"K_\{lam-1/2\}\s*\(", "besselk(lam-0.5,", s)
    s = re.sub(r"K_\{nu\}\s*\(", "besselk(nu,", s)

    # -- Gamma function: keep gamma_sp alias (collision-safe) ------------------
    # Note: \ln\Gamma( and \log\Gamma( are handled at the TOP of _conv_specials
    # before any \ln/\log stripping can break them.
    # \Gamma( -> gamma_sp( (collision-safe alias)
    s = re.sub(r"\\Gamma\s*\(", "gamma_sp(", s)

    # -- erfc -> SymPy native ---------------------------------------------------
    s = re.sub(r"\\?erfc\s*\(", "erfc(", s)
    s = re.sub(r"erfc\s*\(", "erfc(", s)

    # -- Airy Ai -> SymPy native airyai -----------------------------------------
    s = re.sub(r"\bAi\s*\(", "airyai(", s)

    # -- Parabolic cylinder D_nu -> custom ParabolicCylD -------------------------
    s = re.sub(r"D_\{?\\?nu\}?\s*\(", "ParabolicCylD(nu,", s)

    # -- Tricomi U(a,b,z) -> custom TricomiU ------------------------------------
    s = re.sub(r"(?<![a-zA-Z])U\s*\(", "TricomiU(", s)

    # -- Kummer M(a,b,z) / _1F_1 -> custom Hyp1F1 --------------------------------
    s = re.sub(r"(?<![a-zA-Z])M\s*\(", "Hyp1F1(", s)
    s = re.sub(r"\{\}_\{1\}F_\{1\}\s*\(", "Hyp1F1(", s)
    s = re.sub(r"\{\}_\{2\}F_\{1\}\s*\(", "Hyp2F1(", s)
    s = re.sub(r"_1F_1\s*\(", "Hyp1F1(", s)
    s = re.sub(r"_2F_1\s*\(", "Hyp2F1(", s)

    # -- Whittaker W_{kappa,mu}(z) -> TricomiU (via W=e^{-z/2}z^{mu+1/2}U(...) defn) -
    s = re.sub(r"W_\{\\?kappa,\\?mu\}\s*\(", "TricomiU(mu-kappa+0.5,1+2*mu,", s)
    s = re.sub(r"W_\{kappa,mu\}\s*\(", "TricomiU(mu-kappa+0.5,1+2*mu,", s)

    # -- Mittag-Leffler E_alpha -> custom MittagLefflerE ----------------------------
    s = re.sub(r"E_\{?\\?alpha\}?\s*\(", "MittagLefflerE(alpha,", s)
    s = re.sub(r"E_\{?alpha\}?\s*\(", "MittagLefflerE(alpha,", s)

    # -- Beta B(a,b) -> beta_sp (collision-safe alias) --------------------------
    s = re.sub(r"(?<![a-zA-Z])B\s*\(", "beta_sp(", s)

    # -- IncUpperGamma(s, x) = regularised upper incomplete gamma -> RegUpperGamma
    #    Written as \mathrm{IncUpperGamma}(...) in LaTeX; \mathrm{} is stripped above.
    s = re.sub(r"(?<![a-zA-Z])IncUpperGamma\s*\(", "RegUpperGamma(", s)

    # -- IncLowerGamma(s, x) = regularised lower incomplete gamma -> RegLowerGamma
    s = re.sub(r"(?<![a-zA-Z])IncLowerGamma\s*\(", "RegLowerGamma(", s)

    # -- phi(x) = normal PDF -> NormalPDF -----------------------------------------
    s = re.sub(r"\\phi\s*\(", "NormalPDF(", s)

    # -- Phi(x) = normal CDF -> NormalCDF -----------------------------------------
    s = re.sub(r"\\Phi\s*\(", "NormalCDF(", s)

    # -- Hyperbolic trig -------------------------------------------------------
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\tanh\s*\(", r"\1*tanh(", s)
    s = re.sub(r"\\tanh\s*\(", "tanh(", s)
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\sinh\s*\(", r"\1*sinh(", s)
    s = re.sub(r"\\sinh\s*\(", "sinh(", s)
    s = re.sub(r"([a-zA-Z0-9_)}])\s*\\cosh\s*\(", r"\1*cosh(", s)
    s = re.sub(r"\\cosh\s*\(", "cosh(", s)

    # -- arcsinh, arctan etc. --------------------------------------------------
    s = re.sub(r"\\arcsinh\b", "asinh", s)
    s = re.sub(r"\barcsinh\b", "asinh", s)
    s = re.sub(r"\\arctan\b", "atan", s)
    s = re.sub(r"\barctan\b", "atan", s)
    s = re.sub(r"\\arccos\b", "acos", s)
    s = re.sub(r"\\arcsin\b", "asin", s)
    # Bare trig arguments (space-separated)
    s = re.sub(r"\\ln\s+([a-zA-Z0-9_])", r"log(\1)", s)
    s = re.sub(r"\blog\s+([a-zA-Z0-9_])", r"log(\1)", s)
    s = re.sub(r"\batan\s+([a-zA-Z0-9_])", r"atan(\1)", s)
    s = re.sub(r"\basinh\s+([a-zA-Z0-9_])", r"asinh(\1)", s)

    # -- |A| -> abs(A) ----------------------------------------------------------
    for _ in range(5):
        ns = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)
        if ns == s:
            break
        s = ns

    # -- x' prime notation -> xp ------------------------------------------------
    # x^{\prime n} with explicit integer power (works for any digit)
    s = re.sub(
        r"x\^\{\\prime\s*\\?[,\s]*(\d+)\}",
        lambda m: "xp" if m.group(1) == "1" else f"xp**{m.group(1)}",
        s,
    )
    s = re.sub(r"x\^\{\\prime\}", "xp", s)
    s = re.sub(r"x\^\\prime\b", "xp", s)
    s = re.sub(r"x\\prime\b", "xp", s)

    # -- Landau (ROOT-backed, step-2 open problem) ------------------------------
    s = re.sub(r"Landau\s*\(", "landau_pdf(", s)

    # -- Voigt V(...) -> VoigtProfile -------------------------------------------
    s = re.sub(r"V\s*\(", "VoigtProfile(", s)

    # Cleanup
    s = s.replace(r"\!", "")
    s = re.sub(r"\\[,;:]\s*", "", s)
    return s


# -- Greek letter conversion ---------------------------------------------------

_GREEK = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma_var",
    r"\Gamma": "Gamma_var",
    r"\delta": "delta",
    r"\Delta": "Delta",
    r"\epsilon": "eps",
    r"\varepsilon": "eps",
    r"\zeta": "zeta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\Theta": "Theta",
    r"\iota": "iota",
    r"\kappa": "kappa",
    r"\lambda": "lam",
    r"\Lambda": "Lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\Xi": "Xi",
    r"\pi": "pi",
    r"\Pi": "Pi",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\Sigma": "Sigma",
    r"\tau": "tau",
    r"\upsilon": "upsilon",
    r"\phi": "phi_var",
    r"\Phi": "Phi_var",
    r"\varphi": "phi_var",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\Psi": "Psi",
    r"\omega": "omega",
    r"\Omega": "Omega",
}


def _conv_greek(s):
    for latex, py in sorted(_GREEK.items(), key=lambda x: -len(x[0])):
        s = re.sub(re.escape(latex) + r"(?=[a-zA-Z\\])", py + " ", s)
        s = s.replace(latex, py)
    return s


# -- Implicit multiplication ---------------------------------------------------


def _add_mult(s, extra_func_names=None):
    """Add * between adjacent tokens that need implicit multiplication."""
    _func_names = (
        FUNC_NAMES if not extra_func_names else FUNC_NAMES | set(extra_func_names)
    )
    toks = re.findall(
        r"\d+\.?\d*(?:[eE][+-]?\d+)?|\w+|\*\*|<=|>=|!=|<|>|[+\-*/=(),^]|\s+", s
    )
    result = []
    prev = None
    for tok in toks:
        stripped = tok.strip()
        if not stripped:
            result.append(tok)
            continue
        if prev is not None:
            need = False
            p_close = (
                prev in (")",)
                or re.match(r"\d", prev)
                or (re.match(r"\w", prev) and prev not in {"**"})
            )
            t_open = (
                stripped == "("
                or re.match(r"[a-zA-Z]", stripped)
                or (re.match(r"\d", stripped) and re.match(r"\d", prev))
            )
            if p_close and t_open:
                if stripped == "(" and prev in _func_names:
                    need = False
                elif stripped != "(" and stripped in {
                    "+",
                    "-",
                    "*",
                    "/",
                    "**",
                    ",",
                    ")",
                }:
                    need = False
                else:
                    need = True
            if need:
                result.append("*")
        result.append(tok)
        if stripped:
            prev = stripped
    return "".join(result)


# -- Symbol map builder --------------------------------------------------------


def _build_sym_map(param_names, constants, latex_var_map=None):
    """Build a dict mapping LaTeX variable names -> Python expression strings."""
    sm = {
        "x": "x",
        "x_min": "XMIN",
        "x_max": "XMAX",
        "x_mid": "X0",
    }

    for cname, cval in (constants or {}).items():
        try:
            fv = repr(float(cval))
            sm[cname] = fv
            sm[cname.lower()] = fv
        except (TypeError, ValueError):
            pass  # skip non-numeric constants (e.g. variant="running")

    for cname, cval in (constants or {}).items():
        try:
            fv = repr(float(cval))
        except (TypeError, ValueError):
            continue
        canonical = cname.replace("_", "")
        if canonical != cname:
            sm[canonical] = fv
            sm[canonical.lower()] = fv

    for pn in param_names:
        sm[pn] = pn
        pn_c = re.sub(r"([a-zA-Z])_(?!var\b)([a-zA-Z0-9]+)", r"\1\2", pn)
        if pn_c != pn:
            sm[pn_c] = pn

    if "x0" in param_names:
        sm["X0"] = "x0"
        sm["x0"] = "x0"

    GREEK_TO_CANDIDATES = {
        "alpha": ["alpha", "a"],
        "beta": ["beta"],
        "gamma_var": ["gamma"],
        "Gamma_var": ["Gamma"],
        "delta": ["delta"],
        "eps": ["eps", "epsilon"],
        "zeta": ["zeta"],
        "eta": ["eta"],
        "theta": ["theta"],
        "kappa": ["kappa"],
        "lam": ["lam"],
        "mu": ["mu", "loc", "mean", "mpv"],
        "nu": ["nu"],
        "pi": [],
        "rho": ["rho"],
        "sigma": ["sigma", "scale"],
        "tau": ["tau"],
        "phi_var": ["phi"],
        "chi": ["chi"],
        "psi": ["psi"],
        "xi": ["xi", "loc"],
        "omega": ["omega", "scale"],
    }
    for greek_result, candidates in GREEK_TO_CANDIDATES.items():
        for candidate in candidates:
            if candidate in param_names:
                sm[greek_result] = candidate
                break

    for k, v in (latex_var_map or {}).items():
        v_str = str(v)
        if (
            re.match(r"^[a-zA-Z_]\w*$", v_str)
            and v_str in sm
            and not any(c in v_str for c in "+-*/(")
        ):
            resolved = sm[v_str]
        else:
            resolved = v_str
        sm[k] = resolved
        k_collapsed = re.sub(r"([a-zA-Z])_([a-zA-Z0-9]+)", r"\1\2", k)
        if k_collapsed != k:
            sm[k_collapsed] = resolved

    return sm


def _apply_sym_map(s, sm):
    """Apply sym_map substitutions using whole-word regex matching."""
    _placeholders = {}
    _ctr = [0]
    fn_pattern = "|".join(
        sorted((f for f in FUNC_NAMES if len(f) > 1), key=lambda f: -len(f))
    )

    def _protect(m):
        ph = f"__F{_ctr[0]}__"
        _placeholders[ph] = m.group(0)
        _ctr[0] += 1
        return ph

    if fn_pattern:
        s = re.sub(fn_pattern, _protect, s)

    s = re.sub(r"([a-zA-Z])_(?!var\b)([a-zA-Z0-9]+)", r"\1\2", s)
    s = re.sub(r"([a-zA-Z])([0-9]+)([a-zA-Z])(?!_)", r"\1\2*\3", s)

    for ph, fn in _placeholders.items():
        s = s.replace(ph, fn)

    _greek_ids = [
        v for v in sm.values() if re.match(r"^[a-z][a-z]+$", v) and v in sm.values()
    ]
    for _gi in sorted(set(_greek_ids), key=lambda x: -len(x)):
        for _gj in sorted(set(_greek_ids), key=lambda x: -len(x)):
            if _gi != _gj and len(_gi) + len(_gj) > 4:
                s = re.sub(
                    r"(?<![a-zA-Z])" + _gi + _gj + r"(?![a-zA-Z])", _gi + "*" + _gj, s
                )

    for k, v in sorted(sm.items(), key=lambda x: -len(x[0])):
        k_re = re.escape(k)
        if any(c in v for c in "+-*/") and not (v.startswith("(") and v.endswith(")")):
            v_sub = f"({v})"
        else:
            v_sub = v
        s = re.sub(r"(?<![a-zA-Z0-9_])" + k_re + r"(?![a-zA-Z0-9_])", v_sub, s)
    return s


# -- Auxiliary variable definition parser --------------------------------------


def _parse_aux_defs(parts, sm):
    """Parse 'var=expr' auxiliary definitions into a list of (name, pyexpr)."""
    defs = []
    for part in parts[1:]:
        part = part.strip()
        if not part or "=" not in part:
            continue
        lhs, rhs = part.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        rhs = re.sub(r"\s*\([^)]*:[^)]*\)\s*$", "", rhs).strip()
        lhs_clean = _preprocess(lhs).strip()
        lhs_clean = _conv_greek(lhs_clean).strip()
        lhs_clean = re.sub(r"^\\", "", lhs_clean)
        lhs_clean = re.sub(r"[\\\{\}]", "", lhs_clean)
        lhs_clean = lhs_clean.strip()
        var_name = None
        if "prime" in lhs or "x'" in lhs or "x'" in lhs:
            var_name = "xp"
        elif lhs_clean in (
            "t",
            "z",
            "u",
            "y",
            "w",
            "eps",
            "varepsilon",
            "A",
            "B",
            "C",
            "D",
            "E",
        ):
            var_name = lhs_clean
        elif lhs_clean.startswith("eps") or "varepsilon" in lhs:
            var_name = "eps"
        elif re.match(r"^[a-zA-Z_]\w*$", lhs_clean):
            var_name = lhs_clean
        if var_name is None:
            continue
        py_rhs = _full_convert(rhs, sm)
        defs.append((var_name, py_rhs))
    return defs


# -- Top-level formula splitter ------------------------------------------------


def _split_formula(s):
    """Split 'main_expr, aux_def1, aux_def2' into [main, def1, def2]."""
    parts = re.split(r",\s+(?=(?:[a-zA-Z\\])[^,]*=)", s)
    if len(parts) == 1:
        parts = re.split(r",\s{2,}", s)
    return parts


# -- Full LaTeX -> Python expression converter ----------------------------------


def _full_convert(s, sym_map, constants=None, fn_defs=None, extra_func_names=None):
    """Convert a LaTeX formula fragment to a Python expression string.

    ``constants`` is an optional dict of numeric constants (e.g. XMIN, n).
    ``fn_defs`` is an optional dict of indexed function definitions from
    ``latex_fn_defs`` in desc.yaml (e.g. Chebyshev recurrence).

    Any ``\\sum_{var=lo}^{hi}`` whose bounds resolve to concrete integers
    (literals or names present in *constants*) is expanded by
    :func:`_expand_sum` before any other processing.  Indexed function
    calls (e.g. ``T_3(u)``) are then expanded by :func:`_expand_fn_defs`.
    """
    s = _expand_sum(s, constants)
    s = _expand_fn_defs(s, fn_defs)

    s = _expand_cases(s, sym_map)
    s = _conv_specials(s)

    for _ in range(20):
        prev = s
        s = _conv_fracs(s)
        s = _conv_sqrt(s)
        s = _conv_powers(s)
        s = _conv_subscripts(s, sym_map)
        if s == prev:
            break

    for _ in range(5):
        ns = re.sub(r"(?<![a-zA-Z_])e\s*\*\*\s*\(([^()]*)\)", r"exp(\1)", s)
        if ns == s:
            break
        s = ns

    s = _conv_greek(s)

    for _bare_fn in ("atan", "asin", "acos", "asinh", "log", "exp", "sqrt"):
        s = re.sub(r"\b" + _bare_fn + r"\s+([a-zA-Z0-9_])", _bare_fn + r"(\1)", s)

    s = _add_mult(s, extra_func_names=extra_func_names)
    s = _apply_sym_map(s, sym_map)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -- Build reference function from LaTeX --------------------------------------


def build_ref_fn(
    key,
    latex_str,
    param_names,
    constants,
    latex_var_map=None,
    latex_aux=None,
    fn_defs=None,
    extra_fns=None,
):
    """
    Convert LaTeX formula to a callable f(x, *params) using SymPy.

    ``fn_defs`` is an optional dict of indexed function definitions from
    ``latex_fn_defs`` in desc.yaml (e.g. Chebyshev recurrence).  When
    provided, indexed calls such as ``T_3(u)`` in the formula are expanded
    recursively before SymPy sees the expression.

    ``extra_fns`` is an optional dict of {name: callable} for externally-defined
    functions that appear in the formula (e.g. a tabulated spline kernel).  Each
    name is registered as a SymPy Function so sympify treats it as a call rather
    than a symbol, and the callable is injected into the lambdify namespace.

    Returns ref_fn callable on success.  Raises ValueError if the formula
    cannot be parsed or compiled.
    """
    sym_map = _build_sym_map(param_names, constants, latex_var_map)

    # Augment constants with latex_var_map aliases so that _expand_sum can
    # resolve sum bounds written as LaTeX aliases (e.g. m/n for num_deg/den_deg).
    augmented_constants = dict(constants or {})
    for lat_name, py_name in (latex_var_map or {}).items():
        py_str = str(py_name)
        if py_str in augmented_constants:
            augmented_constants[lat_name] = augmented_constants[py_str]
        else:
            try:
                augmented_constants[lat_name] = float(py_str)
            except (TypeError, ValueError):
                pass

    s = latex_str.strip()
    s = re.sub(r"^f\s*\(x\)\s*=\s*", "", s)
    s = s.strip("'\"")

    depth_b = 0
    eq_positions = []
    first_top_comma = None
    for i, ch in enumerate(s):
        if ch in "{(":
            depth_b += 1
        elif ch in "})":
            depth_b -= 1
        elif depth_b == 0:
            if ch == "=":
                eq_positions.append(i)
            elif ch == "," and first_top_comma is None:
                first_top_comma = i
    if first_top_comma is not None:
        eq_before_comma = [p for p in eq_positions if p < first_top_comma]
    else:
        eq_before_comma = eq_positions
    if len(eq_before_comma) >= 2:
        s = s[: eq_before_comma[1]].rstrip(" ,")

    s_prep = _preprocess(s)
    parts = _split_formula(s_prep)

    main_latex = parts[0].strip()
    main_py = _full_convert(
        main_latex,
        sym_map,
        constants=augmented_constants,
        fn_defs=fn_defs,
        extra_func_names=set(extra_fns) if extra_fns else None,
    )

    aux_defs = _parse_aux_defs(parts, sym_map)

    for aux_latex in latex_aux or []:
        aux_parts = _split_formula(_preprocess(aux_latex))
        extra = _parse_aux_defs([None] + aux_parts, sym_map)
        aux_defs.extend(extra)

    _eq_parts = re.split(r"(?<![=<>!])=(?!=)", main_py)
    if len(_eq_parts) > 1:
        main_py = _eq_parts[-1].strip()

    for var_name, var_expr in aux_defs:
        main_py = re.sub(
            r"(?<![a-zA-Z0-9_])" + re.escape(var_name) + r"(?![a-zA-Z0-9_])",
            f"({var_expr})",
            main_py,
        )

    # Build SymPy symbols
    x_sym = Symbol("x")
    p_syms = {pn: Symbol(pn) for pn in param_names}
    c_syms = {}
    for cn, cv in (constants or {}).items():
        try:
            c_syms[cn] = sp.Float(float(cv))
        except (TypeError, ValueError):
            pass  # skip non-numeric constants (e.g. variant="running")

    all_local = {
        **SYMPY_NS,
        **{pn: v for pn, v in p_syms.items()},
        **c_syms,
        "x": x_sym,
    }
    for nm in ["t", "xp", "z", "u", "y", "eps"]:
        all_local[nm] = Symbol(nm)
    if extra_fns:
        for fn_name in extra_fns:
            all_local[fn_name] = sp.Function(fn_name)

    try:
        expr = sympify(main_py, locals=all_local)
    except Exception as e:
        raise ValueError(f"sympify failed: {e}\n  pyexpr={main_py!r}")

    args = [x_sym] + [p_syms[pn] for pn in param_names]

    modules = [_EVAL_NS, "mpmath"]
    if extra_fns:
        modules = [extra_fns, _EVAL_NS, "mpmath"]
    try:
        f_lam = lambdify(args, expr, modules=modules)
    except Exception as e:
        raise ValueError(f"lambdify failed: {e}")

    def ref_fn(x, *params):
        try:
            return float(f_lam(float(x), *[float(p) for p in params]))
        except Exception:
            return float("nan")

    return ref_fn
