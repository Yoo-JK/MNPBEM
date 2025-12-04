"""
Options handling for MNPBEM simulations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BEMOptions:
    """
    Options for BEM simulations.

    Attributes
    ----------
    sim : str
        Simulation type: 'stat' for quasistatic, 'ret' for retarded.
    waitbar : int
        Show progress bar (1) or not (0).
    rel_cutoff : float
        Cutoff parameter for face integration.
    order : int
        Order for exp(i*k*r) expansion.
    interp : str
        Particle surface interpolation: 'flat' or 'curv'.
    """
    sim: str = "ret"
    waitbar: int = 1
    rel_cutoff: float = 3.0
    order: int = 5
    interp: str = "flat"

    # Additional options stored as dict
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like setting."""
        if hasattr(self, key) and key != "extra":
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get option value with default."""
        if hasattr(self, key) and key != "extra":
            return getattr(self, key)
        return self.extra.get(key, default)

    def update(self, **kwargs) -> "BEMOptions":
        """Update options and return self."""
        for key, value in kwargs.items():
            self[key] = value
        return self


def bemoptions(**kwargs) -> BEMOptions:
    """
    Create BEM options structure.

    Parameters
    ----------
    **kwargs : dict
        Option name-value pairs.

    Returns
    -------
    BEMOptions
        Options structure with standard or user-defined options.

    Examples
    --------
    >>> op = bemoptions(sim='stat', waitbar=0, interp='curv')
    >>> op.sim
    'stat'
    """
    # Known attributes of BEMOptions
    known_attrs = {'sim', 'waitbar', 'rel_cutoff', 'order', 'interp'}

    # Separate known and extra options
    known_kwargs = {k: v for k, v in kwargs.items() if k in known_attrs}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in known_attrs}

    options = BEMOptions(**known_kwargs)
    options.extra = extra_kwargs

    return options


def getbemoptions(names: Optional[list] = None, *args, **kwargs) -> BEMOptions:
    """
    Get BEM options, optionally filtering by module names.

    Parameters
    ----------
    names : list, optional
        List of module names to filter options.
    *args : tuple
        Positional arguments (options structures).
    **kwargs : dict
        Keyword arguments for options.

    Returns
    -------
    BEMOptions
        Options structure.
    """
    # Start with default options
    options = BEMOptions()

    # Update from any positional argument that is already an options object
    for arg in args:
        if isinstance(arg, BEMOptions):
            for key in ['sim', 'waitbar', 'rel_cutoff', 'order', 'interp']:
                setattr(options, key, getattr(arg, key))
            options.extra.update(arg.extra)
        elif isinstance(arg, dict):
            options.update(**arg)

    # Update from keyword arguments
    options.update(**kwargs)

    return options
