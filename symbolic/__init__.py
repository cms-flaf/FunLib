"""
FunLib/symbolic -- Custom SymPy special functions for LaTeX formula validation.

Provides proper SymPy Function subclasses for special functions that either:
 - Are not available natively in SymPy (e.g. TricomiU, ParabolicCylD, MittagLefflerE), or
 - Have naming conflicts with common shape parameter names (e.g. gamma_sp, beta_sp), or
 - Require non-trivial evaluation logic (NormalPDF, NormalCDF, VoigtProfile).

Each class has a matching entry in LAMBDIFY_NS, the evaluation namespace for
lambdify.  Import both together:

    from FunLib.symbolic import special_functions as spf  # or: from . import special_functions as spf
    expr   = spf.TricomiU(a_sym, b_sym, z_sym)
    f_num  = lambdify(args, expr, modules=[spf.LAMBDIFY_NS, 'mpmath'])
"""

from .special_functions import (
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

__all__ = [
    "GammaSp",
    "BetaSp",
    "TricomiU",
    "Hyp1F1",
    "Hyp2F1",
    "ParabolicCylD",
    "MittagLefflerE",
    "NormalPDF",
    "NormalCDF",
    "RegUpperGamma",
    "RegLowerGamma",
    "VoigtProfile",
    "LandauPDF",
    "LAMBDIFY_NS",
    "_PARSER_ALIAS",
]
