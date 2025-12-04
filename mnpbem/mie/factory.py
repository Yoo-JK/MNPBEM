"""
Factory function for Mie solvers.
"""

from typing import Optional, Union, Any

from .mie_stat import MieStat
from ..misc.options import BEMOptions


def miesolver(
    epsin: Any,
    epsout: Any,
    diameter: float,
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> MieStat:
    """
    Factory function to create Mie solver.

    Parameters
    ----------
    epsin : dielectric function
        Dielectric function inside sphere.
    epsout : dielectric function
        Dielectric function outside sphere.
    diameter : float
        Sphere diameter in nm.
    options : BEMOptions or dict, optional
        Simulation options. Key option is 'sim':
        - 'stat': Quasistatic Mie theory
        - 'ret': Full Mie theory (not yet implemented)
    **kwargs : dict
        Additional options.

    Returns
    -------
    MieStat
        Mie solver instance.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, miesolver, bemoptions
    >>> op = bemoptions(sim='stat')
    >>> eps_gold = EpsTable('gold.dat')
    >>> eps_vac = EpsConst(1)
    >>> mie = miesolver(eps_gold, eps_vac, 10, op)
    """
    # Handle options
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = {
            'sim': options.sim,
            **options.extra
        }

    all_options = {**options, **kwargs}
    sim = all_options.pop('sim', 'stat')

    if sim == 'stat':
        return MieStat(epsin, epsout, diameter, **all_options)
    elif sim == 'ret':
        raise NotImplementedError(
            "Retarded Mie theory not yet implemented. Use sim='stat'."
        )
    else:
        raise ValueError(f"Unknown simulation type: {sim}")
