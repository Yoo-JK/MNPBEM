"""
Plane wave excitation for full Maxwell equations (retarded case).

Matches MATLAB MNPBEM @planewaveret implementation.
"""

import numpy as np
from ..spectrum import SpectrumRet
from ..geometry import trisphere


class PlaneWaveRet:
    """
    Plane wave excitation for solution of full Maxwell equations.

    Implements electromagnetic plane wave excitation with vector potential
    formulation, including phase factors for retarded Green functions.

    Parameters
    ----------
    pol : array_like, shape (3,) or (npol, 3)
        Light polarization direction(s), unit vectors
    dir : array_like, shape (3,) or (npol, 3)
        Light propagation direction(s), unit vectors
    medium : int, optional
        Index of medium for excitation (1-indexed, default: 1)

    Attributes
    ----------
    pol : ndarray, shape (npol, 3)
        Normalized polarization directions
    dir : ndarray, shape (npol, 3)
        Normalized propagation directions
    medium : int
        Medium index for excitation

    Notes
    -----
    MATLAB equivalent: @planewaveret class

    The vector potential is given by:
        a = exp(i*k*pos·dir) / (i*k0) * pol

    The surface derivative is:
        ap = i*k * (nvec·dir) * a

    Polarization and propagation directions must be orthogonal.
    """

    name = 'planewave'

    def __init__(self, pol, dir, medium=1, pinfty=None):
        """
        Initialize plane wave excitation.

        MATLAB: planewaveret(pol, dir, op, PropertyPair)

        Parameters
        ----------
        pol : array_like
            Light polarization direction(s)
        dir : array_like
            Light propagation direction(s)
        medium : int
            Medium index (1-indexed, MATLAB convention)
        pinfty : Particle, optional
            Discretized unit sphere for far-field integration.
            If None, creates trisphere(256, 2).
        """
        # Convert to numpy arrays
        pol = np.atleast_2d(pol).astype(float)
        dir = np.atleast_2d(dir).astype(float)

        # Check dimensions match
        if pol.shape[0] != dir.shape[0]:
            raise ValueError(
                f"Polarization ({pol.shape[0]}) and direction ({dir.shape[0]}) "
                "must have same number of rows"
            )

        # Normalize
        pol_norms = np.linalg.norm(pol, axis=1, keepdims=True)
        dir_norms = np.linalg.norm(dir, axis=1, keepdims=True)
        self.pol = pol / pol_norms
        self.dir = dir / dir_norms

        # Check orthogonality
        dot_products = np.sum(self.pol * self.dir, axis=1)
        if np.any(np.abs(dot_products) > 1e-10):
            raise ValueError("Polarization and direction must be orthogonal")

        self.medium = medium

        # Initialize spectrum for far-field calculations
        # MATLAB: obj.spec = spectrumret(trisphere(256, 2), 'medium', obj.medium)
        if pinfty is None:
            pinfty = trisphere(256, 2.0)
        self.spec = SpectrumRet(pinfty, medium=medium)

    def potential(self, p, enei):
        """
        Compute vector potential of plane wave excitation.

        MATLAB:
            a = exp(1i*k*pos*dir')/(1i*k0) * pol
            ap = (1i*k * nvec*dir') .* a * pol

        The potential is only set on faces where the specified medium
        is on the inside (a1) or outside (a2) of the boundary.

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
            - 'a1', 'a2': vector potentials for inside/outside, shape (nfaces, 3, npol)
            - 'a1p', 'a2p': surface derivatives, shape (nfaces, 3, npol)
            - 'p': particle reference
            - 'enei': wavelength
        """
        # Get refractive index of medium
        eps_val, _ = p.eps[self.medium - 1](enei)
        nb = np.sqrt(eps_val)

        # Wavenumber in vacuum and medium
        k0 = 2 * np.pi / enei
        k = k0 * nb

        # Get particle properties
        nfaces = p.nfaces
        pos = p.pos      # (nfaces, 3)
        nvec = p.nvec    # (nfaces, 3)
        npol = self.pol.shape[0]

        # Initialize potentials (zero by default)
        a1 = np.zeros((nfaces, 3, npol), dtype=complex)
        a1p = np.zeros((nfaces, 3, npol), dtype=complex)
        a2 = np.zeros((nfaces, 3, npol), dtype=complex)
        a2p = np.zeros((nfaces, 3, npol), dtype=complex)

        # Get inout array for each face
        inout_faces = p.inout_faces  # (nfaces, 2): column 0=inside, 1=outside

        # Loop over inside (inout=0) and outside (inout=1)
        for inout_col in range(2):
            # Find faces where this medium is on this side
            # MATLAB: ind = find(p.inout(:, inout) == obj.medium)
            ind = np.where(inout_faces[:, inout_col] == self.medium)[0]

            if len(ind) > 0:
                # Compute potentials only for these faces
                for i in range(npol):
                    # Phase factor: exp(i*k*pos·dir) / (i*k0)
                    phase = np.exp(1j * k * np.dot(pos[ind], self.dir[i])) / (1j * k0)

                    # Vector potential: a = phase * pol
                    a_vals = phase[:, np.newaxis] * self.pol[i]

                    # Surface derivative: ap = (i*k * nvec·dir) * phase * pol
                    nvec_dot_dir = np.dot(nvec[ind], self.dir[i])
                    ap_vals = (1j * k * nvec_dot_dir)[:, np.newaxis] * phase[:, np.newaxis] * self.pol[i]

                    # Store in appropriate array based on inside/outside
                    if inout_col == 0:  # Inside
                        a1[ind, :, i] = a_vals
                        a1p[ind, :, i] = ap_vals
                    else:  # Outside
                        a2[ind, :, i] = a_vals
                        a2p[ind, :, i] = ap_vals

        return {
            'a1': a1,
            'a1p': a1p,
            'a2': a2,
            'a2p': a2p,
            'p': p,
            'enei': enei
        }

    def extinction(self, sig):
        """
        Compute extinction cross section using optical theorem.

        MATLAB:
            [field, k] = farfield(obj.spec, sig, obj.dir)
            ext = 4*pi/k * diag(imag(inner(obj.pol, field.e)))'

        Parameters
        ----------
        sig : dict
            Solution containing surface charges and currents

        Returns
        -------
        ext : ndarray
            Extinction cross section for each polarization
        """
        # Get farfield and wavenumber
        field_e, k = self._farfield(sig, self.dir)

        # Extinction from optical theorem: ext = 4*pi/k * Im(pol · field_e)
        # pol: (npol, 3), field_e: (npol, 3, npol) or (npol, 3)
        npol = self.pol.shape[0]
        ext = np.zeros(npol)

        for i in range(npol):
            # Dot product of polarization with electric field
            if field_e.ndim == 3:
                dot_prod = np.dot(np.conj(self.pol[i]), field_e[i, :, i])
            else:
                dot_prod = np.dot(np.conj(self.pol[i]), field_e[i, :])
            ext[i] = 4 * np.pi / np.real(k) * np.imag(dot_prod)

        return ext

    def scattering(self, sig):
        """
        Compute scattering cross section.

        MATLAB:
            [sca, dsca] = scattering(obj.spec, sig)
            nb = sqrt(sig.p.eps{1}(sig.enei))
            sca = sca / (0.5 * nb)

        Parameters
        ----------
        sig : dict
            Solution containing surface charges and currents

        Returns
        -------
        sca : ndarray
            Scattering cross section for each polarization
        dsca : ndarray
            Differential scattering cross section
        """
        # Get scattering from spectrum
        sca, dsca = self.spec.scattering(sig)

        # Refractive index of embedding medium
        p = sig['p']
        enei = sig['enei']
        eps_val, _ = p.eps[0](enei)
        nb = np.sqrt(np.real(eps_val))

        # Normalize: scattering cross section = radiated power / incoming power
        # Incoming power proportional to 0.5 * epsb * (clight / nb) = 0.5 * nb
        sca = sca / (0.5 * nb)
        dsca = dsca / (0.5 * nb)

        return sca, dsca

    def absorption(self, sig):
        """
        Compute absorption cross section.

        MATLAB: abs = extinction - scattering

        Parameters
        ----------
        sig : dict
            Solution containing surface charges and currents

        Returns
        -------
        abs : ndarray
            Absorption cross section for each polarization
        """
        sca, _ = self.scattering(sig)
        return self.extinction(sig) - sca

    def _farfield(self, sig, dirs):
        """
        Compute far-field amplitude in given directions.

        MATLAB: Based on Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010), Eq. (50).

        Parameters
        ----------
        sig : dict
            Solution with 'sig1', 'sig2' (charges) and 'h1', 'h2' (currents)
        dirs : ndarray, shape (ndir, 3)
            Directions to compute far-field

        Returns
        -------
        e : ndarray
            Electric far-field amplitude, shape (ndir, 3) or (ndir, 3, npol)
        k : complex
            Wavenumber in medium
        """
        p = sig['p']
        enei = sig['enei']

        # Wavenumber in medium
        _, k = p.eps[self.medium - 1](enei)
        k0 = 2 * np.pi / enei

        pos = p.pos    # (nfaces, 3)
        area = p.area  # (nfaces,)

        dirs = np.atleast_2d(dirs)
        ndir = dirs.shape[0]

        # Check for surface currents in solution
        if 'h1' not in sig and 'h2' not in sig:
            # No surface currents - return zero field
            if 'sig' in sig:
                # Use scalar charges for simplified far-field
                surface_charge = sig['sig']
                if surface_charge.ndim == 1:
                    npol = 1
                else:
                    npol = surface_charge.shape[1]

                e = np.zeros((ndir, 3, npol), dtype=complex)
                return e, k
            else:
                return np.zeros((ndir, 3), dtype=complex), k

        # Get surface currents and charges
        h1 = sig.get('h1', np.zeros((p.nfaces, 3)))
        h2 = sig.get('h2', np.zeros((p.nfaces, 3)))
        sig1 = sig.get('sig1', np.zeros(p.nfaces))
        sig2 = sig.get('sig2', np.zeros(p.nfaces))

        # Ensure proper dimensions
        if h1.ndim == 2:
            h1 = h1[:, :, np.newaxis]
            h2 = h2[:, :, np.newaxis]
        if sig1.ndim == 1:
            sig1 = sig1[:, np.newaxis]
            sig2 = sig2[:, np.newaxis]

        npol = h1.shape[2]

        # Get inout array for each face
        inout_faces = p.inout_faces  # (nfaces, 2): column 0=inside, 1=outside

        # Phase factor: exp(-i*k*dir·pos) * area
        # MATLAB: phase = exp(-1i * k * dir * p.pos') * spdiag(p.area)
        phase = np.exp(-1j * k * np.dot(dirs, pos.T)) * area  # (ndir, nfaces)

        # Electric far-field
        e = np.zeros((ndir, 3, npol), dtype=complex)

        # MATLAB: separates contributions by inout
        # ind = p.index(find(p.inout(:, 1) == obj.medium)') for inner
        # ind = p.index(find(p.inout(:, 2) == obj.medium)') for outer

        # Inner surface: faces where inside medium == self.medium
        ind1 = np.where(inout_faces[:, 0] == self.medium)[0]
        # Outer surface: faces where outside medium == self.medium
        ind2 = np.where(inout_faces[:, 1] == self.medium)[0]

        for ipol in range(npol):
            # Contribution from inner surface (h1, sig1)
            if len(ind1) > 0:
                phase1 = phase[:, ind1]  # (ndir, nind1)
                # Current term: i*k0 * phase @ h1
                current_term1 = 1j * k0 * np.dot(phase1, h1[ind1, :, ipol])
                # Charge term: -i*k * dir * (phase @ sig1)
                sig_sum1 = np.dot(phase1, sig1[ind1, ipol])
                charge_term1 = -1j * k * dirs * sig_sum1[:, np.newaxis]
                e[:, :, ipol] += current_term1 + charge_term1

            # Contribution from outer surface (h2, sig2)
            if len(ind2) > 0:
                phase2 = phase[:, ind2]  # (ndir, nind2)
                # Current term: i*k0 * phase @ h2
                current_term2 = 1j * k0 * np.dot(phase2, h2[ind2, :, ipol])
                # Charge term: -i*k * dir * (phase @ sig2)
                sig_sum2 = np.dot(phase2, sig2[ind2, ipol])
                charge_term2 = -1j * k * dirs * sig_sum2[:, np.newaxis]
                e[:, :, ipol] += current_term2 + charge_term2

        # Squeeze if single polarization
        if npol == 1:
            e = e.squeeze(axis=2)

        return e, k

    def __repr__(self):
        return (
            f"PlaneWaveRet(pol={self.pol.tolist()}, "
            f"dir={self.dir.tolist()}, medium={self.medium})"
        )
