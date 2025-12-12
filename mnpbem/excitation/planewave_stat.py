"""
Plane wave excitation for quasistatic approximation.

Matches MATLAB MNPBEM @planewavestat implementation.
"""

import numpy as np


class PlaneWaveStat:
    """
    Plane wave excitation within quasistatic approximation.

    In the quasistatic limit, the electric field is uniform and the
    potential derivative is simply the projection onto the surface normal.

    Parameters
    ----------
    pol : array_like, shape (3,) or (npol, 3)
        Light polarization direction(s), unit vectors
    medium : int, optional
        Index of medium for spectrum calculation (1-indexed, default: 1)

    Attributes
    ----------
    pol : ndarray, shape (npol, 3)
        Normalized polarization directions
    medium : int
        Medium index for spectrum calculations

    Examples
    --------
    >>> # Single polarization along x-axis
    >>> exc = PlaneWaveStat([1, 0, 0])
    >>>
    >>> # Multiple polarizations
    >>> exc = PlaneWaveStat([[1, 0, 0], [0, 1, 0]])

    Notes
    -----
    MATLAB equivalent: @planewavestat class

    In quasistatic approximation:
    - phip = -nvec · pol (surface derivative of scalar potential)
    - Scattering cross section: sca = 8π/3 × k⁴ × |dip|²
    - Absorption cross section: abs = 4πk × Im(pol · dip)
    """

    name = 'planewave'

    def __init__(self, pol, medium=1):
        """
        Initialize plane wave excitation.

        Parameters
        ----------
        pol : array_like
            Light polarization direction(s)
        medium : int
            Medium index (1-indexed, MATLAB convention)
        """
        # Convert to numpy array
        pol = np.atleast_2d(pol).astype(float)

        # Normalize polarization vectors
        norms = np.linalg.norm(pol, axis=1, keepdims=True)
        self.pol = pol / norms

        # Medium index (1-indexed like MATLAB)
        self.medium = medium

    def potential(self, p, enei):
        """
        Compute potential of plane wave excitation.

        MATLAB: exc = compstruct(p, enei, 'phip', -p.nvec * transpose(obj.pol))

        Parameters
        ----------
        p : ComParticle
            Particle object with surface mesh
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : dict
            Dictionary containing:
            - 'phip': surface derivative of scalar potential, shape (nfaces, npol)
            - 'p': particle reference
            - 'enei': wavelength
        """
        # Get normal vectors from particle
        nvec = p.nvec  # (nfaces, 3)

        # Compute phip = -nvec · pol
        # nvec: (nfaces, 3), pol: (npol, 3)
        # Result: (nfaces, npol)
        phip = -np.dot(nvec, self.pol.T)

        return {
            'phip': phip,
            'p': p,
            'enei': enei
        }

    def scattering(self, sig):
        """
        Compute scattering cross section.

        MATLAB:
            dip = matmul((repmat(sig.p.area, [1,3]) .* sig.p.pos)', sig.sig)
            sca = 8*pi/3 * k^4 * sum(abs(dip)^2, 1)

        Parameters
        ----------
        sig : dict
            Solution containing:
            - 'sig': surface charge, shape (nfaces,) or (nfaces, npol)
            - 'p': particle reference
            - 'enei': wavelength

        Returns
        -------
        sca : ndarray
            Scattering cross section for each polarization
        """
        # Extract solution components
        surface_charge = sig['sig']  # (nfaces,) or (nfaces, npol)
        p = sig['p']
        enei = sig['enei']

        # Ensure 2D array
        if surface_charge.ndim == 1:
            surface_charge = surface_charge[:, np.newaxis]

        # Get area and position
        area = p.area  # (nfaces,)
        pos = p.pos    # (nfaces, 3)

        # Compute dipole moment: dip = (area * pos)' * sig
        # weighted_pos: (nfaces, 3), each column weighted by area
        weighted_pos = area[:, np.newaxis] * pos  # (nfaces, 3)

        # dip = weighted_pos.T @ surface_charge
        # Result: (3, npol)
        dip = weighted_pos.T @ surface_charge

        # Get wavenumber from medium
        # MATLAB: eps = sig.p.eps{obj.medium}; [~, k] = eps(sig.enei)
        eps_func = p.eps[self.medium - 1]  # Convert to 0-indexed
        _, k = eps_func(enei)

        # Scattering cross section: sca = 8*pi/3 * k^4 * sum(|dip|^2)
        # dip is (3, npol), sum over first dimension (xyz components)
        sca = 8 * np.pi / 3 * k**4 * np.sum(np.abs(dip)**2, axis=0)

        return sca

    def absorption(self, sig):
        """
        Compute absorption cross section.

        MATLAB:
            dip = matmul((repmat(sig.p.area, [1,3]) .* sig.p.pos)', sig.sig)
            abs = 4*pi*k * imag(dot(transpose(obj.pol), dip, 1))

        Parameters
        ----------
        sig : dict
            Solution containing surface charge

        Returns
        -------
        abs : ndarray
            Absorption cross section for each polarization
        """
        # Extract solution components
        surface_charge = sig['sig']
        p = sig['p']
        enei = sig['enei']

        # Ensure 2D array
        if surface_charge.ndim == 1:
            surface_charge = surface_charge[:, np.newaxis]

        # Get area and position
        area = p.area
        pos = p.pos

        # Compute dipole moment
        weighted_pos = area[:, np.newaxis] * pos
        dip = weighted_pos.T @ surface_charge  # (3, npol)

        # Get wavenumber
        eps_func = p.eps[self.medium - 1]
        _, k = eps_func(enei)

        # Absorption cross section: abs = 4*pi*k * Im(pol · dip)
        # pol: (npol, 3), dip: (3, npol)
        # We need dot product for each polarization: pol[i] · dip[:, i]
        npol = self.pol.shape[0]
        abs_cs = np.zeros(npol)

        for i in range(npol):
            # Dot product of polarization with dipole moment
            dot_prod = np.dot(self.pol[i], dip[:, i])
            abs_cs[i] = 4 * np.pi * k * np.imag(dot_prod)

        return abs_cs

    def extinction(self, sig):
        """
        Compute extinction cross section.

        MATLAB: ext = scattering(obj, sig) + absorption(obj, sig)

        Parameters
        ----------
        sig : dict
            Solution containing surface charge

        Returns
        -------
        ext : ndarray
            Extinction cross section for each polarization
        """
        return self.scattering(sig) + self.absorption(sig)

    def __repr__(self):
        return f"PlaneWaveStat(pol={self.pol.tolist()}, medium={self.medium})"
