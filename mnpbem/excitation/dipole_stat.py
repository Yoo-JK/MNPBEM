"""
Dipole excitation for quasistatic approximation.

Matches MATLAB MNPBEM @dipolestat implementation.
"""

import numpy as np


class DipoleStat:
    """
    Oscillating dipole excitation within quasistatic approximation.

    Computes the external potential and fields from an oscillating dipole
    for use in BEM simulations. Also calculates total and radiative
    decay rates for the dipole near nanoparticles.

    Parameters
    ----------
    pt_pos : array_like, shape (npt, 3)
        Position(s) of the dipole(s)
    dip : array_like, shape (3,) or (ndip, 3), optional
        Dipole moment direction(s). Default: eye(3) for x, y, z orientations
    full : bool, optional
        If True, dip has shape (npt, 3, ndip) with different moments at each position

    Attributes
    ----------
    pt_pos : ndarray, shape (npt, 3)
        Dipole positions
    dip : ndarray, shape (npt, 3, ndip)
        Dipole moment directions for each position
    npt : int
        Number of dipole positions
    ndip : int
        Number of dipole orientations

    Notes
    -----
    MATLAB equivalent: @dipolestat class

    Electric field from dipole [Jackson Eq. (4.13)]:
        E = (3(p·r̂)r̂ - p) / (ε·r³)

    Potential derivative:
        phip = -nvec · E

    Examples
    --------
    >>> # Dipole at origin with x-polarization
    >>> exc = DipoleStat([[0, 0, 20]], [[1, 0, 0]])
    >>>
    >>> # Dipole with all three orientations (default)
    >>> exc = DipoleStat([[0, 0, 20]])
    """

    name = 'dipole'

    def __init__(self, pt_pos, dip=None, full=False):
        """
        Initialize dipole excitation.

        Parameters
        ----------
        pt_pos : array_like
            Dipole position(s)
        dip : array_like, optional
            Dipole orientation(s)
        full : bool
            If True, different dipole at each position
        """
        # Dipole positions
        self.pt_pos = np.atleast_2d(pt_pos).astype(float)
        self.npt = self.pt_pos.shape[0]

        # Dipole orientations
        if dip is None:
            # Default: x, y, z orientations
            dip = np.eye(3)

        dip = np.asarray(dip, dtype=float)

        if full:
            # Different dipole at each position: (npt, 3, ndip)
            if dip.ndim == 2:
                self.dip = dip[:, :, np.newaxis]
            else:
                self.dip = dip
            self.ndip = self.dip.shape[2]
        else:
            # Same dipole orientations for all positions
            if dip.ndim == 1:
                dip = dip[np.newaxis, :]  # (1, 3)

            # Replicate for all positions: (npt, 3, ndip)
            ndip = dip.shape[0]
            self.dip = np.zeros((self.npt, 3, ndip))
            for i in range(self.npt):
                self.dip[i, :, :] = dip.T  # (3, ndip)
            self.ndip = ndip

    def field(self, p, enei):
        """
        Compute electric field from dipole excitation.

        MATLAB: field.m using Jackson Eq. (4.13)
            E = (3(p·r̂)r̂ - p) / (ε·r³)

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
            - 'e': electric field, shape (nfaces, 3, npt, ndip)
            - 'p': particle reference
            - 'enei': wavelength
        """
        pos1 = p.pos   # Field points: (nfaces, 3)
        pos2 = self.pt_pos  # Dipole positions: (npt, 3)

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Get dielectric constant at dipole positions
        # For simplicity, use the outside medium (eps[0])
        eps_val, _ = p.eps[0](enei)
        eps = np.full(n2, complex(eps_val))

        # Allocate output: (n1, 3, n2, ndip)
        e = np.zeros((n1, 3, n2, self.ndip), dtype=complex)

        # Distance vectors: pos1 - pos2
        # x[i,j] = pos1[i,0] - pos2[j,0]
        x = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        y = pos1[:, 1:2] - pos2[:, 1:2].T  # (n1, n2)
        z = pos1[:, 2:3] - pos2[:, 2:3].T  # (n1, n2)

        # Distance
        r = np.sqrt(x**2 + y**2 + z**2)

        # Avoid division by zero
        r = np.maximum(r, 1e-30)

        # Unit vectors
        x_hat = x / r
        y_hat = y / r
        z_hat = z / r

        # Loop over dipole orientations
        for idip in range(self.ndip):
            # Dipole moment at each dipole position: (n2, 3)
            dip_vec = self.dip[:, :, idip]  # (npt, 3)

            # Broadcast dipole: dx[i,j] = dip[j, 0]
            dx = np.broadcast_to(dip_vec[:, 0], (n1, n2))
            dy = np.broadcast_to(dip_vec[:, 1], (n1, n2))
            dz = np.broadcast_to(dip_vec[:, 2], (n1, n2))

            # Inner product: p · r̂
            p_dot_r = x_hat * dx + y_hat * dy + z_hat * dz

            # Electric field [Jackson Eq. (4.13)]:
            # E = (3(p·r̂)r̂ - p) / (ε·r³)
            factor = 1.0 / (eps * r**3)

            e[:, 0, :, idip] = factor * (3 * p_dot_r * x_hat - dx)
            e[:, 1, :, idip] = factor * (3 * p_dot_r * y_hat - dy)
            e[:, 2, :, idip] = factor * (3 * p_dot_r * z_hat - dz)

        return {
            'e': e,
            'p': p,
            'enei': enei
        }

    def potential(self, p, enei):
        """
        Compute potential of dipole excitation for BEM.

        MATLAB: phip = -inner(p.nvec, exc.e)
            phip = -nvec · E

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
            - 'phip': surface derivative of scalar potential, shape (nfaces, npt, ndip)
            - 'p': particle reference
            - 'enei': wavelength
        """
        # Get electric field
        field_result = self.field(p, enei)
        e = field_result['e']  # (nfaces, 3, npt, ndip)

        # Normal vectors
        nvec = p.nvec  # (nfaces, 3)

        # Compute phip = -nvec · E
        # nvec: (nfaces, 3), e: (nfaces, 3, npt, ndip)
        # Result: (nfaces, npt, ndip)
        phip = -np.einsum('ij,ijkl->ikl', nvec, e)

        return {
            'phip': phip,
            'p': p,
            'enei': enei
        }

    def decayrate(self, sig):
        """
        Compute total and radiative decay rates.

        MATLAB: decayrate.m
        Total decay rate enhancement from imaginary part of induced field.
        Radiative decay rate from induced dipole moment.

        Parameters
        ----------
        sig : dict
            Solution containing:
            - 'sig': surface charge, shape (nfaces, npt, ndip)
            - 'p': particle reference
            - 'enei': wavelength

        Returns
        -------
        tot : ndarray, shape (npt, ndip)
            Total decay rate (normalized to free space)
        rad : ndarray, shape (npt, ndip)
            Radiative decay rate (normalized to free space)
        rad0 : ndarray, shape (npt, ndip)
            Free-space decay rate (for reference)
        """
        p = sig['p']
        enei = sig['enei']
        surface_charge = sig['sig']  # (nfaces, npt, ndip)

        # Free-space decay rate (Wigner-Weisskopf)
        # gamma = 4/3 * (2*pi/lambda)^3
        gamma = 4 / 3 * (2 * np.pi / enei) ** 3

        # Get refractive index at dipole position
        eps_val, _ = p.eps[0](enei)
        nb = np.sqrt(np.real(eps_val))

        # Compute induced dipole moment
        # dip_ind = sum(area * pos * sig)
        area = p.area  # (nfaces,)
        pos = p.pos    # (nfaces, 3)

        # Weighted positions
        weighted_pos = area[:, np.newaxis] * pos  # (nfaces, 3)

        # Initialize output arrays
        tot = np.zeros((self.npt, self.ndip))
        rad = np.zeros((self.npt, self.ndip))
        rad0 = np.full((self.npt, self.ndip), nb * gamma)

        # Compute induced field at dipole positions
        # We need the field induced by the surface charges at the dipole positions
        # This requires a separate Green function evaluation
        # For now, compute using direct summation

        for ipt in range(self.npt):
            pt_pos = self.pt_pos[ipt]  # Dipole position

            for idip in range(self.ndip):
                # Dipole moment
                dip_vec = self.dip[ipt, :, idip]

                # Surface charge for this position and orientation
                if surface_charge.ndim == 3:
                    sig_vals = surface_charge[:, ipt, idip]
                elif surface_charge.ndim == 2:
                    sig_vals = surface_charge[:, ipt]
                else:
                    sig_vals = surface_charge

                # Induced dipole moment
                dip_ind = weighted_pos.T @ sig_vals  # (3,)

                # Compute induced field at dipole position
                # E_ind = sum over faces of (sigma * area * r_hat / r^2 / eps0)
                r_vec = pt_pos - pos  # (nfaces, 3)
                r_mag = np.linalg.norm(r_vec, axis=1)
                r_mag = np.maximum(r_mag, 1e-30)

                # Field from surface charge distribution
                e_ind = np.sum(
                    sig_vals[:, np.newaxis] * area[:, np.newaxis] *
                    r_vec / (r_mag[:, np.newaxis]**3),
                    axis=0
                ) / eps_val

                # Total decay rate: 1 + Im(E · dip) / (0.5 * nb * gamma)
                tot[ipt, idip] = 1 + np.imag(np.dot(e_ind, dip_vec)) / (0.5 * nb * gamma)

                # Radiative decay rate: |nb^2 * dip_ind + dip|^2
                rad[ipt, idip] = np.linalg.norm(nb**2 * dip_ind + dip_vec)**2

        return tot, rad, rad0

    def __repr__(self):
        return f"DipoleStat(npt={self.npt}, ndip={self.ndip})"
