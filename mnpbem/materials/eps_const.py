"""
Constant dielectric function.
"""

import numpy as np


class EpsConst(object):
    """
    Constant dielectric function.

    Represents a medium with a constant, wavelength-independent dielectric constant.

    Parameters
    ----------
    eps : float or complex
        Dielectric constant value

    Examples
    --------
    >>> # Vacuum
    >>> eps_vacuum = EpsConst(1.0)
    >>>
    >>> # Water
    >>> eps_water = EpsConst(1.33**2)
    >>>
    >>> # Get dielectric function at 500 nm
    >>> eps_val, k = eps_vacuum(500)
    """

    def __init__(self, eps):
        """
        Initialize constant dielectric function.

        Parameters
        ----------
        eps : float or complex
            Dielectric constant value
        """
        self.eps = eps

    def __call__(self, enei):
        """
        Get dielectric constant and wavenumber.

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        eps : float or complex or ndarray
            Dielectric constant (same shape as enei)
        k : float or complex or ndarray
            Wavenumber in medium (1/nm)
        """
        enei = np.asarray(enei)

        # Dielectric constant (broadcast to enei shape)
        eps = np.full_like(enei, self.eps, dtype=complex)

        # Wavenumber: k = 2π/λ × √ε
        k = 2 * np.pi / enei * np.sqrt(self.eps)

        return eps, k

    def wavenumber(self, enei):
        """
        Get wavenumber in medium.

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        k : float or complex or ndarray
            Wavenumber in medium (1/nm)
        """
        enei = np.asarray(enei)
        return 2 * np.pi / enei * np.sqrt(self.eps)

    def __repr__(self):
        return f"EpsConst(eps={self.eps})"

    def __str__(self):
        return f"Constant dielectric function: ε = {self.eps}"
