"""
Dipole excitation for full Maxwell equations (retarded case).

Matches MATLAB MNPBEM @dipoleret implementation.
"""

import numpy as np


class DipoleRet:
    """
    Oscillating dipole excitation for full Maxwell equations.

    Computes the scalar and vector potentials from an oscillating dipole
    including retardation effects for use in BEM simulations.

    Parameters
    ----------
    pt_pos : array_like, shape (npt, 3)
        Position(s) of the dipole(s)
    dip : array_like, shape (3,) or (ndip, 3), optional
        Dipole moment direction(s). Default: eye(3) for x, y, z orientations
    full : bool, optional
        If True, dip has shape (npt, 3, ndip) with different moments at each position
    medium : int, optional
        Embedding medium index (1-indexed, default: 1)

    Attributes
    ----------
    pt_pos : ndarray, shape (npt, 3)
        Dipole positions
    dip : ndarray, shape (npt, 3, ndip)
        Dipole moment directions for each position
    medium : int
        Medium index

    Notes
    -----
    MATLAB equivalent: @dipoleret class

    Scalar potential and derivative:
        phi = -p·r̂ * F / ε
        F = (ik - 1/r) * G
        G = exp(ikr) / r

    Vector potential [Jackson Eq. (9.16)]:
        a = -ik₀ * p * G

    Electric field [Jackson Eq. (9.18)]:
        E = k² * G * (p - (p·r̂)r̂) / ε + G * (1/r² - ik/r) * (3(p·r̂)r̂ - p) / ε

    Magnetic field [Jackson Eq. (9.18)]:
        H = k² * G * (1 - 1/(ikr)) * (r̂ × p) / sqrt(ε)
    """

    name = 'dipole'

    def __init__(self, pt_pos, dip=None, full=False, medium=1, spec=None):
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
        medium : int
            Embedding medium index (1-indexed)
        spec : SpectrumRet, optional
            Spectrum object for scattering/radiative decay rate
        """
        # Dipole positions
        self.pt_pos = np.atleast_2d(pt_pos).astype(float)
        self.npt = self.pt_pos.shape[0]
        self.medium = medium
        self.spec = spec

        # Dipole orientations
        if dip is None:
            dip = np.eye(3)

        dip = np.asarray(dip, dtype=float)

        if full:
            if dip.ndim == 2:
                self.dip = dip[:, :, np.newaxis]
            else:
                self.dip = dip
            self.ndip = self.dip.shape[2]
        else:
            if dip.ndim == 1:
                dip = dip[np.newaxis, :]

            ndip = dip.shape[0]
            self.dip = np.zeros((self.npt, 3, ndip))
            for i in range(self.npt):
                self.dip[i, :, :] = dip.T
            self.ndip = ndip

    def potential(self, p, enei):
        """
        Compute potentials for dipole excitation.

        MATLAB: potential.m
            phi = -p·r̂ * F / ε
            a = -ik₀ * p * G

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
            - 'phi1', 'phi2': scalar potentials, shape (nfaces, npt, ndip)
            - 'phi1p', 'phi2p': scalar potential derivatives
            - 'a1', 'a2': vector potentials, shape (nfaces, 3, npt, ndip)
            - 'a1p', 'a2p': vector potential derivatives
            - 'p': particle reference
            - 'enei': wavelength
        """
        pos1 = p.pos   # Field points: (nfaces, 3)
        nvec = p.nvec  # Normal vectors: (nfaces, 3)
        pos2 = self.pt_pos  # Dipole positions: (npt, 3)

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Get dielectric function and wavenumber
        eps_val, k = p.eps[self.medium - 1](enei)
        k0 = 2 * np.pi / enei  # Wavenumber in vacuum

        # Initialize arrays
        phi1 = np.zeros((n1, n2, self.ndip), dtype=complex)
        phi1p = np.zeros((n1, n2, self.ndip), dtype=complex)
        phi2 = np.zeros((n1, n2, self.ndip), dtype=complex)
        phi2p = np.zeros((n1, n2, self.ndip), dtype=complex)

        a1 = np.zeros((n1, 3, n2, self.ndip), dtype=complex)
        a1p = np.zeros((n1, 3, n2, self.ndip), dtype=complex)
        a2 = np.zeros((n1, 3, n2, self.ndip), dtype=complex)
        a2p = np.zeros((n1, 3, n2, self.ndip), dtype=complex)

        # Position difference
        x = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        y = pos1[:, 1:2] - pos2[:, 1:2].T
        z = pos1[:, 2:3] - pos2[:, 2:3].T

        # Distance
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 1e-30)

        # Unit vectors
        x_hat = x / r
        y_hat = y / r
        z_hat = z / r

        # Green function and derivative
        G = np.exp(1j * k * r) / r
        F = (1j * k - 1.0 / r) * G

        # Normal vector components (broadcast to n1, n2)
        nx = nvec[:, 0:1] * np.ones((1, n2))
        ny = nvec[:, 1:2] * np.ones((1, n2))
        nz = nvec[:, 2:3] * np.ones((1, n2))

        # Inner products
        en = nx * x_hat + ny * y_hat + nz * z_hat  # nvec · r̂

        # Loop over dipole orientations
        for idip in range(self.ndip):
            dip_vec = self.dip[:, :, idip]  # (npt, 3)

            dx = np.broadcast_to(dip_vec[:, 0], (n1, n2))
            dy = np.broadcast_to(dip_vec[:, 1], (n1, n2))
            dz = np.broadcast_to(dip_vec[:, 2], (n1, n2))

            # Inner products
            ep = x_hat * dx + y_hat * dy + z_hat * dz  # p · r̂
            np_dot = nx * dx + ny * dy + nz * dz       # nvec · p

            # Scalar potential: phi = -p·r̂ * F / ε
            phi = -ep * F / eps_val

            # Scalar potential derivative [MATLAB potential.m]
            phip = ((np_dot - 3 * en * ep) / r**2 * (1 - 1j * k * r) * G / eps_val +
                    k**2 * ep * en * G / eps_val)

            # Vector potential [Jackson Eq. (9.16)]: a = -ik₀ * p * G
            a = np.zeros((n1, 3, n2), dtype=complex)
            a[:, 0, :] = -1j * k0 * dx * G
            a[:, 1, :] = -1j * k0 * dy * G
            a[:, 2, :] = -1j * k0 * dz * G

            # Surface derivative of vector potential
            ap = np.zeros((n1, 3, n2), dtype=complex)
            ap[:, 0, :] = -1j * k0 * dx * en * F
            ap[:, 1, :] = -1j * k0 * dy * en * F
            ap[:, 2, :] = -1j * k0 * dz * en * F

            # Store (simple case: both sides same)
            phi1[:, :, idip] = phi
            phi1p[:, :, idip] = phip
            phi2[:, :, idip] = phi
            phi2p[:, :, idip] = phip

            a1[:, :, :, idip] = a
            a1p[:, :, :, idip] = ap
            a2[:, :, :, idip] = a
            a2p[:, :, :, idip] = ap

        return {
            'phi1': phi1, 'phi1p': phi1p,
            'phi2': phi2, 'phi2p': phi2p,
            'a1': a1, 'a1p': a1p,
            'a2': a2, 'a2p': a2p,
            'p': p,
            'enei': enei
        }

    def field(self, p, enei):
        """
        Compute electromagnetic fields from dipole.

        MATLAB: field.m using Jackson Eq. (9.18)

        Parameters
        ----------
        p : ComParticle
            Particle object
        enei : float
            Wavelength in nm

        Returns
        -------
        exc : dict
            Dictionary containing:
            - 'e': electric field, shape (nfaces, 3, npt, ndip)
            - 'h': magnetic field, shape (nfaces, 3, npt, ndip)
        """
        pos1 = p.pos
        pos2 = self.pt_pos

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Dielectric function and wavenumber
        eps_val, k = p.eps[self.medium - 1](enei)

        # Initialize
        e = np.zeros((n1, 3, n2, self.ndip), dtype=complex)
        h = np.zeros((n1, 3, n2, self.ndip), dtype=complex)

        # Position difference
        x = pos1[:, 0:1] - pos2[:, 0:1].T
        y = pos1[:, 1:2] - pos2[:, 1:2].T
        z = pos1[:, 2:3] - pos2[:, 2:3].T

        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 1e-30)

        x_hat = x / r
        y_hat = y / r
        z_hat = z / r

        # Green function
        G = np.exp(1j * k * r) / r

        for idip in range(self.ndip):
            dip_vec = self.dip[:, :, idip]

            dx = np.broadcast_to(dip_vec[:, 0], (n1, n2))
            dy = np.broadcast_to(dip_vec[:, 1], (n1, n2))
            dz = np.broadcast_to(dip_vec[:, 2], (n1, n2))

            # Inner product p · r̂
            p_dot_r = x_hat * dx + y_hat * dy + z_hat * dz

            # Magnetic field [Jackson (9.18)]
            # H = k² * G * (1 - 1/(ikr)) * (r̂ × p) / sqrt(ε)
            fac_h = k**2 * G * (1 - 1 / (1j * k * r)) / np.sqrt(eps_val)
            h[:, 0, :, idip] = fac_h * (y_hat * dz - z_hat * dy)
            h[:, 1, :, idip] = fac_h * (z_hat * dx - x_hat * dz)
            h[:, 2, :, idip] = fac_h * (x_hat * dy - y_hat * dx)

            # Electric field [Jackson (9.18)]
            # E = k²G(p - (p·r̂)r̂)/ε + G(1/r² - ik/r)(3(p·r̂)r̂ - p)/ε
            fac1 = k**2 * G / eps_val
            fac2 = G * (1 / r**2 - 1j * k / r) / eps_val

            e[:, 0, :, idip] = fac1 * (dx - p_dot_r * x_hat) + fac2 * (3 * p_dot_r * x_hat - dx)
            e[:, 1, :, idip] = fac1 * (dy - p_dot_r * y_hat) + fac2 * (3 * p_dot_r * y_hat - dy)
            e[:, 2, :, idip] = fac1 * (dz - p_dot_r * z_hat) + fac2 * (3 * p_dot_r * z_hat - dz)

        return {
            'e': e,
            'h': h,
            'p': p,
            'enei': enei
        }

    def decayrate(self, sig, spec=None):
        """
        Compute total and radiative decay rates for dipole near particle.

        MATLAB: decayrate.m
        Total decay rate from imaginary part of induced field.
        Radiative decay rate from scattering cross section.

        Parameters
        ----------
        sig : dict
            Solution from BEM solver containing:
            - 'sig1', 'sig2': surface charges
            - 'h1', 'h2': surface currents
            - 'p': particle
            - 'enei': wavelength
        spec : SpectrumRet, optional
            Spectrum object for scattering. Uses self.spec if not provided.

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

        # Wavenumber in vacuum
        k0 = 2 * np.pi / enei

        # Wigner-Weisskopf rate: gamma = 4/3 * k0^3
        gamma = 4 / 3 * k0**3

        # Refractive index of embedding medium
        eps_val, k = p.eps[self.medium - 1](enei)
        nb = np.sqrt(np.real(eps_val))

        # Initialize output
        tot = np.zeros((self.npt, self.ndip))
        rad = np.zeros((self.npt, self.ndip))
        rad0 = np.full((self.npt, self.ndip), nb * gamma)

        # Use provided spec or self.spec
        if spec is None:
            spec = self.spec

        # Compute scattering cross section for radiative decay rate
        if spec is not None:
            sca, _ = spec.scattering(sig)
            if np.isscalar(sca):
                sca = np.full((self.npt, self.ndip), sca)
            else:
                sca = np.broadcast_to(sca, (self.npt, self.ndip))
            # rad = sca / (2*pi*k0) normalized to free-space rate
            rad = sca / (2 * np.pi * k0) / (0.5 * nb * gamma)

        # Compute induced field at dipole positions for total decay rate
        # This requires Green function evaluation from surface to dipole
        pos = p.pos
        area = p.area

        # Get surface charges
        sig1 = sig.get('sig1', np.zeros(p.nfaces))
        sig2 = sig.get('sig2', np.zeros(p.nfaces))

        if sig1.ndim == 1:
            sig_total = sig1 + sig2
        else:
            sig_total = sig1 + sig2  # Sum inside and outside contributions

        for ipt in range(self.npt):
            pt_pos = self.pt_pos[ipt]

            for idip in range(self.ndip):
                dip_vec = self.dip[ipt, :, idip]

                # Get surface charge for this configuration
                if sig_total.ndim == 1:
                    sig_vals = sig_total
                else:
                    # Need to map (npt, ndip) to proper index
                    idx = ipt * self.ndip + idip
                    if idx < sig_total.shape[1]:
                        sig_vals = sig_total[:, idx]
                    else:
                        sig_vals = sig_total[:, 0]

                # Compute induced field at dipole position
                # Using retarded Green function: G = exp(ikr)/r
                r_vec = pt_pos - pos  # (nfaces, 3)
                r_mag = np.linalg.norm(r_vec, axis=1)
                r_mag = np.maximum(r_mag, 1e-30)
                r_hat = r_vec / r_mag[:, np.newaxis]

                # Green function
                G = np.exp(1j * k * r_mag) / r_mag

                # Electric field from surface charge (simplified)
                # E = sum(sig * area * (-grad G))
                # For retarded case: -grad G = (ik - 1/r) * G * r_hat
                grad_G_factor = (1j * k - 1 / r_mag) * G
                e_ind = np.sum(
                    sig_vals[:, np.newaxis] * area[:, np.newaxis] *
                    grad_G_factor[:, np.newaxis] * r_hat,
                    axis=0
                ) / eps_val

                # Total decay rate: 1 + Im(E_ind · dip) / (0.5 * nb * gamma)
                tot[ipt, idip] = 1 + np.imag(np.dot(e_ind, dip_vec)) / (0.5 * nb * gamma)

        return tot, rad, rad0

    def scattering(self, sig, spec=None):
        """
        Compute scattering cross section for dipole excitation.

        Parameters
        ----------
        sig : dict
            Solution from BEM solver
        spec : SpectrumRet, optional
            Spectrum object

        Returns
        -------
        sca : ndarray
            Scattering cross section
        """
        if spec is None:
            spec = self.spec
        if spec is None:
            raise ValueError("SpectrumRet object required for scattering calculation")

        return spec.scattering(sig)

    def __repr__(self):
        return f"DipoleRet(npt={self.npt}, ndip={self.ndip}, medium={self.medium})"
