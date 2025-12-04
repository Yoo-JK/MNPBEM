"""
Miscellaneous utilities for MNPBEM.
"""

from .units import eV2nm, nm2eV, HARTREE, TUNIT
from .options import BEMOptions, bemoptions, getbemoptions
from .helpers import inner, outer, matcross, matmul, spdiag, vecnorm, vecnormalize

__all__ = [
    "eV2nm",
    "nm2eV",
    "HARTREE",
    "TUNIT",
    "BEMOptions",
    "bemoptions",
    "getbemoptions",
    "inner",
    "outer",
    "matcross",
    "matmul",
    "spdiag",
    "vecnorm",
    "vecnormalize",
]
