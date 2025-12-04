"""
Factory functions for BEM solvers.
"""

from typing import Optional, Union

from .bem_stat import BEMStat
from ..particles import ComParticle
from ..misc.options import BEMOptions


def bemsolver(
    p: ComParticle,
    options: Optional[Union[BEMOptions, dict]] = None,
    enei: Optional[float] = None,
    **kwargs
) -> BEMStat:
    """
    Factory function to create appropriate BEM solver.

    Creates a BEM solver based on the simulation type specified
    in the options.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    options : BEMOptions or dict, optional
        Simulation options. Key option is 'sim':
        - 'stat': Quasistatic solver (BEMStat)
        - 'ret': Retarded solver (not yet implemented)
    enei : float, optional
        Wavelength for precomputation.
    **kwargs : dict
        Additional options.

    Returns
    -------
    BEMStat
        BEM solver instance.

    Examples
    --------
    >>> from mnpbem import bemoptions, bemsolver, ComParticle
    >>> op = bemoptions(sim='stat')
    >>> bem = bemsolver(p, op)
    """
    # Handle options
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = {
            'sim': options.sim,
            'rel_cutoff': options.rel_cutoff,
            **options.extra
        }

    # Merge with kwargs
    all_options = {**options, **kwargs}

    # Get simulation type
    sim = all_options.pop('sim', 'stat')

    # Create appropriate solver
    if sim == 'stat':
        return BEMStat(p, enei=enei, **all_options)
    elif sim == 'ret':
        # TODO: Implement retarded solver
        raise NotImplementedError(
            "Retarded BEM solver not yet implemented. Use sim='stat'."
        )
    else:
        raise ValueError(f"Unknown simulation type: {sim}")
