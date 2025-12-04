"""
Layer structure for stratified media.
"""

import numpy as np
from typing import List, Optional, Any, Tuple

from ..misc.units import eV2nm


class LayerStructure:
    """
    Stratified media (layer structure) for substrate modeling.

    Handles reflection and transmission at planar interfaces.

    Parameters
    ----------
    eps : list
        List of dielectric function objects for each layer.
        eps[0] is the top semi-infinite layer, eps[-1] is the bottom.
    d : list or ndarray
        Thicknesses of intermediate layers (N-2 values for N layers).
    z_interface : list or ndarray, optional
        Z-positions of interfaces. If not given, computed from d.

    Attributes
    ----------
    eps : list
        Dielectric functions for each layer.
    d : ndarray
        Layer thicknesses.
    z : ndarray
        Z-positions of interfaces.
    """

    def __init__(
        self,
        eps: List[Any],
        d: Optional[np.ndarray] = None,
        z_interface: Optional[np.ndarray] = None
    ):
        """
        Initialize layer structure.

        Parameters
        ----------
        eps : list
            List of dielectric function objects.
        d : ndarray, optional
            Layer thicknesses for intermediate layers.
        z_interface : ndarray, optional
            Z-positions of interfaces.
        """
        self.eps = eps
        self.n_layers = len(eps)

        if d is None:
            d = np.array([])
        self.d = np.asarray(d, dtype=float)

        if z_interface is not None:
            self.z = np.asarray(z_interface, dtype=float)
        else:
            # Compute interface positions from thicknesses
            # Convention: first interface at z=0
            self.z = np.zeros(len(d) + 1)
            for i, thickness in enumerate(d):
                self.z[i + 1] = self.z[i] - thickness

    def indlayer(self, z: np.ndarray) -> np.ndarray:
        """
        Determine which layer each z-position is in.

        Parameters
        ----------
        z : ndarray
            Z-positions.

        Returns
        -------
        ndarray
            Layer indices (0-based).
        """
        z = np.asarray(z)
        indices = np.zeros(z.shape, dtype=int)

        for i, z_int in enumerate(self.z):
            indices[z < z_int] = i + 1

        return indices

    def fresnel(
        self,
        enei: float,
        kpar: np.ndarray,
        pol: str = 'p'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fresnel reflection and transmission coefficients.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        kpar : ndarray
            Parallel component of wavevector.
        pol : str
            Polarization: 'p' (TM) or 's' (TE).

        Returns
        -------
        r : ndarray
            Reflection coefficients.
        t : ndarray
            Transmission coefficients.
        """
        kpar = np.atleast_1d(kpar)
        k0 = 2 * np.pi / enei

        # Get dielectric values
        eps_vals = [eps(enei)[0] for eps in self.eps]

        # Compute perpendicular wavevector in each layer
        kz = []
        for eps in eps_vals:
            kz_sq = eps * k0**2 - kpar**2
            kz.append(np.sqrt(kz_sq.astype(complex)))

        # Fresnel coefficients at each interface
        r = np.zeros((len(self.z), len(kpar)), dtype=complex)
        t = np.ones((len(self.z), len(kpar)), dtype=complex)

        for i in range(len(self.z)):
            eps1 = eps_vals[i]
            eps2 = eps_vals[i + 1] if i + 1 < len(eps_vals) else eps_vals[i]
            kz1 = kz[i]
            kz2 = kz[i + 1] if i + 1 < len(kz) else kz[i]

            if pol == 'p':  # TM polarization
                r[i] = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)
                t[i] = 2 * eps2 * kz1 / (eps2 * kz1 + eps1 * kz2)
            else:  # TE polarization
                r[i] = (kz1 - kz2) / (kz1 + kz2)
                t[i] = 2 * kz1 / (kz1 + kz2)

        return r, t

    def reflection(
        self,
        enei: float,
        kpar: np.ndarray,
        pol: str = 'p',
        layer_from: int = 0
    ) -> np.ndarray:
        """
        Compute total reflection coefficient.

        Uses transfer matrix method for multilayer structures.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        kpar : ndarray
            Parallel wavevector component.
        pol : str
            Polarization: 'p' or 's'.
        layer_from : int
            Layer from which reflection is computed.

        Returns
        -------
        ndarray
            Total reflection coefficient.
        """
        if self.n_layers == 2:
            # Simple two-layer case
            r, _ = self.fresnel(enei, kpar, pol)
            return r[0]

        # Multi-layer: use transfer matrix method
        k0 = 2 * np.pi / enei
        kpar = np.atleast_1d(kpar)

        # Get dielectric values
        eps_vals = [eps(enei)[0] for eps in self.eps]

        # Compute kz in each layer
        kz = []
        for eps in eps_vals:
            kz_sq = eps * k0**2 - kpar**2
            kz.append(np.sqrt(kz_sq.astype(complex)))

        # Build up reflection from bottom
        r_total = np.zeros(len(kpar), dtype=complex)

        for i in range(len(self.z) - 1, layer_from - 1, -1):
            r12, _ = self._fresnel_interface(eps_vals[i], eps_vals[i + 1],
                                              kz[i], kz[i + 1], pol)

            if i < len(self.z) - 1:
                # Add phase from propagation in layer
                phase = np.exp(2j * kz[i + 1] * self.d[i])
                r_total = (r12 + r_total * phase) / (1 + r12 * r_total * phase)
            else:
                r_total = r12

        return r_total

    def _fresnel_interface(
        self,
        eps1: complex,
        eps2: complex,
        kz1: np.ndarray,
        kz2: np.ndarray,
        pol: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fresnel coefficients for single interface."""
        if pol == 'p':
            r = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)
            t = 2 * np.sqrt(eps1 * eps2) * kz1 / (eps2 * kz1 + eps1 * kz2)
        else:
            r = (kz1 - kz2) / (kz1 + kz2)
            t = 2 * kz1 / (kz1 + kz2)
        return r, t

    def mindist(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute minimum distance to any interface.

        Parameters
        ----------
        pos : ndarray
            Positions, shape (n, 3).

        Returns
        -------
        ndarray
            Minimum distances.
        """
        pos = np.atleast_2d(pos)
        z = pos[:, 2]

        min_dist = np.full(len(pos), np.inf)
        for z_int in self.z:
            dist = np.abs(z - z_int)
            min_dist = np.minimum(min_dist, dist)

        return min_dist

    def __repr__(self) -> str:
        return f"LayerStructure(n_layers={self.n_layers}, z={self.z})"
