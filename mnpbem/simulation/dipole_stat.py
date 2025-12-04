"""
Dipole excitation for quasistatic BEM.
"""

import numpy as np
from typing import Optional, Union

from ..particles import ComParticle, CompStruct
from ..misc.options import BEMOptions


class DipoleStat:
    """
    Dipole excitation within quasistatic approximation.

    A point dipole creates a potential:
        phi(r) = (1/4*pi*eps) * p . (r - r0) / |r - r0|^3

    Parameters
    ----------
    pt : ndarray
        Dipole positions, shape (n_dip, 3) or (3,).
    dip : ndarray
        Dipole moments, shape (n_dip, 3) or (3,).
    medium : int
        Index of medium containing the dipole.

    Examples
    --------
    >>> # Single dipole at origin, pointing in z
    >>> exc = DipoleStat([0, 0, 0], [0, 0, 1])
    """

    def __init__(
        self,
        pt: np.ndarray,
        dip: np.ndarray,
        medium: int = 1,
        **kwargs
    ):
        """
        Initialize dipole excitation.

        Parameters
        ----------
        pt : ndarray
            Dipole positions.
        dip : ndarray
            Dipole moments.
        medium : int
            Index of excitation medium (1-based).
        """
        self.pt = np.atleast_2d(pt)
        self.dip = np.atleast_2d(dip)
        self.medium = medium
        self.options = kwargs

        if len(self.pt) != len(self.dip):
            raise ValueError("Number of positions and dipoles must match")

    @property
    def n_dip(self) -> int:
        """Number of dipoles."""
        return len(self.pt)

    def __call__(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute external potential for BEM solver.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        return self.potential(p, enei)

    def potential(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute potential derivative at particle boundary.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        pos = p.pos  # Face centroids (n_faces, 3)
        nvec = p.nvec  # Normal vectors (n_faces, 3)

        phip = np.zeros((p.n_faces, self.n_dip))

        for i, (r0, dip) in enumerate(zip(self.pt, self.dip)):
            # Distance vector: r - r0
            dr = pos - r0  # (n_faces, 3)
            r = np.linalg.norm(dr, axis=1)  # (n_faces,)

            # Avoid division by zero
            r = np.where(r == 0, np.inf, r)
            r3 = r ** 3
            r5 = r ** 5

            # Dipole potential gradient dotted with normal
            # phi = (1/4pi) * p . (r-r0) / r^3
            # grad phi = (1/4pi) * [p/r^3 - 3*(p.(r-r0))*(r-r0)/r^5]
            # phip = grad phi . n

            p_dot_dr = np.sum(dip * dr, axis=1)  # (n_faces,)
            term1 = np.sum(dip * nvec, axis=1) / r3
            term2 = 3 * p_dot_dr * np.sum(dr * nvec, axis=1) / r5

            phip[:, i] = (1 / (4 * np.pi)) * (term1 - term2)

        return CompStruct(p, enei, phip=phip)

    def decay_rate(self, sig: CompStruct) -> np.ndarray:
        """
        Compute radiative decay rate enhancement.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.

        Returns
        -------
        ndarray
            Decay rate enhancement for each dipole.
        """
        # Compute electric field at dipole positions from surface charges
        area = sig.p.area
        pos = sig.p.pos
        charges = sig.get('sig')

        gamma = np.zeros(self.n_dip)

        for i, (r0, dip) in enumerate(zip(self.pt, self.dip)):
            # Distance from surface elements to dipole
            dr = r0 - pos  # (n_faces, 3)
            r = np.linalg.norm(dr, axis=1)
            r = np.where(r == 0, np.inf, r)
            r3 = r ** 3

            # Electric field from surface charges
            # E = sum_j sigma_j * (r0 - pos_j) / (4*pi*r^3) * area_j
            sig_vals = charges[:, i] if charges.ndim > 1 else charges
            E = np.sum(sig_vals[:, np.newaxis] * dr / (4 * np.pi * r3[:, np.newaxis]) * area[:, np.newaxis], axis=0)

            # Decay rate proportional to Im(dip . E)
            gamma[i] = np.imag(np.sum(dip * E))

        return gamma

    def __repr__(self) -> str:
        return f"DipoleStat(n_dip={self.n_dip}, medium={self.medium})"


def dipole(
    pt: np.ndarray,
    dip: np.ndarray,
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> DipoleStat:
    """
    Factory function for dipole excitation.

    Parameters
    ----------
    pt : ndarray
        Dipole positions.
    dip : ndarray
        Dipole moments.
    options : BEMOptions or dict, optional
        Simulation options.
    **kwargs : dict
        Additional options.

    Returns
    -------
    DipoleStat
        Dipole excitation object.
    """
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = options.extra.copy()

    all_options = {**options, **kwargs}
    medium = all_options.pop('medium', 1)

    return DipoleStat(pt, dip, medium=medium, **all_options)
