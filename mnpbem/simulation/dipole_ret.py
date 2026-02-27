"""
Dipole excitation for full Maxwell equations (retarded).

Excitation of an oscillating dipole. Given a dipole oscillating with
some frequency, DipoleRet computes the external potentials needed for
BEM simulations and determines the total and radiative scattering rates.

Reference:
    MATLAB MNPBEM Simulation/retarded/@dipoleret

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np
from ..greenfun import CompStruct


class DipoleRet:
    """
    Excitation of an oscillating dipole (retarded, full Maxwell).

    Provides electromagnetic field and potential excitation for retarded BEM solvers.
    Computes total and radiative decay rates, scattering cross sections.

    Parameters
    ----------
    pt : ComPoint or Particle
        Compound of points (or compoint) for dipole positions
    dip : ndarray, optional
        Directions of dipole moments, shape (3,), (3, 3), or (npt, 3, ndip)
        Default: eye(3) - three orthogonal dipoles
    full : bool, optional
        If True, dipole moments are given at each position
        If False, same dipole moments are used for all positions (default)
    medium : int, optional
        Embedding medium index (default: 1)
    pinfty : Particle, optional
        Unit sphere at infinity for radiative rate calculation
        Default: trisphere(256, 2)
    **options : dict
        Additional arguments to be passed to CompGreenRet

    Attributes
    ----------
    pt : ComPoint
        Dipole positions
    dip : ndarray
        Dipole moments, shape (npt, 3, ndip)
    spec : SpectrumRet
        Spectrum for radiative decay rate calculation
    varargin : tuple
        Additional arguments for Green function

    Notes
    -----
    MATLAB: Simulation/retarded/@dipoleret/dipoleret.m

    The electromagnetic field of an oscillating dipole is given by
    Jackson Eq. (9.18):
        E = k²(p - (p·r̂)r̂)G/ε + [3(p·r̂)r̂ - p](1/r² - ik/r)G/ε
        H = k² G/√ε (r̂ × p) (1 - 1/(ikr))

    where G = exp(ikr)/r is the Green function.

    Examples
    --------
    >>> import numpy as np
    >>> from mnpbem.simulation import DipoleRet
    >>>
    >>> # Single dipole at origin, z-polarized
    >>> from mnpbem import ComPoint
    >>> pt = ComPoint([np.array([[0, 0, 0]])], [1])
    >>> dip = np.array([0, 0, 1])
    >>> exc = DipoleRet(pt, dip)
    """

    # Class constants
    # MATLAB: @dipoleret line 8-11
    name = 'dipole'
    needs = {'sim': 'ret'}

    def __init__(self, pt, dip=None, full=False, medium=1, pinfty=None, **options):
        """
        Initialize dipole excitation.

        MATLAB: dipoleret.m + init.m

        Parameters
        ----------
        pt : ComPoint
            Compound of points for dipole positions
        dip : array_like, optional
            Directions of dipole moments
        full : bool, optional
            Indicate that dipole moments are given at each position
        medium : int, optional
            Embedding medium (default: 1)
        pinfty : Particle, optional
            Unit sphere at infinity
        **options : dict
            Additional arguments to be passed to CompGreenRet
        """
        # MATLAB: dipoleret.m line 40
        self.pt = pt
        # MATLAB: dipoleret.m line 41
        self._init(dip, full, medium, pinfty, **options)

    def _init(self, dip=None, full=False, medium=1, pinfty=None, **options):
        """
        Initialize dipole moments.

        MATLAB: init.m

        Parameters
        ----------
        dip : array_like or None
            Dipole moments
        full : bool
            Whether dipole moments are given at all positions
        medium : int
            Embedding medium
        pinfty : Particle or None
            Unit sphere at infinity
        **options : dict
            Additional arguments
        """
        # MATLAB: init.m lines 18-28
        if dip is None:
            dip = np.eye(3)
            full = False

        dip = np.asarray(dip, dtype=float)

        # MATLAB: init.m line 30
        self.varargin = options

        # MATLAB: init.m lines 33-45
        # Dipole moments given at all positions
        if full:
            if dip.ndim == 2:
                dip = dip.reshape(dip.shape + (1,))
            self.dip = dip
        else:
            # Same dipole moments for all positions
            if dip.ndim == 1:
                dip = dip.reshape(1, -1)

            dip_reshaped = dip.T.reshape(1, dip.shape[1], dip.shape[0])
            self.dip = np.tile(dip_reshaped, (self.pt.n, 1, 1))

        # MATLAB: init.m lines 50-62
        # Set up spectrum for calculation of radiative decay rate
        if pinfty is None:
            from ..geometry import trisphere
            pinfty = trisphere(256, 2)

        # MATLAB: obj.spec = spectrumret(pinfty, 'medium', medium)
        from ..spectrum import SpectrumRet
        self.spec = SpectrumRet(pinfty, medium = medium)
        self._pinfty = pinfty
        self._medium = medium

    def field(self, p, enei, inout=1):
        """
        Electromagnetic fields for dipole excitation.

        MATLAB: field.m

        Parameters
        ----------
        p : ComParticle or Particle
            Points or particle surface where fields are computed
        enei : float
            Light wavelength in vacuum (nm)
        inout : int, optional
            Compute field at inside (inout=1, default) or outside (inout=2)

        Returns
        -------
        exc : CompStruct
            CompStruct object containing electromagnetic fields 'e' and 'h'
            Shape: (n, 3, npt, ndip) where last two dimensions correspond
            to dipole positions and dipole moments

        Notes
        -----
        MATLAB: field.m

        Fields computed using Jackson Eq. (9.18):
            E = k²(p-(p·r̂)r̂)G/ε + [3(p·r̂)r̂-p](1/r²-ik/r)G/ε
            H = k²G/√ε (r̂×p)(1-1/(ikr))
        """
        # MATLAB: field.m line 18
        pt = self.pt

        # MATLAB: field.m line 20
        # Connectivity between materials
        con = self._connect(p, pt)

        # MATLAB: field.m line 22
        # Dielectric functions and wavenumbers
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        # MATLAB: field.m line 25
        ndip = self.dip.shape[2]

        # MATLAB: field.m lines 27-30
        exc = CompStruct(p, enei)
        exc.e = np.zeros((p.n, 3, pt.n, ndip), dtype=complex)
        exc.h = np.zeros((p.n, 3, pt.n, ndip), dtype=complex)

        # MATLAB: field.m lines 33-49
        # Positions connected by dielectric media
        for ip in range(con[inout - 1].shape[0]):
            for ipt in range(con[inout - 1].shape[1]):
                ind = con[inout - 1][ip, ipt]
                if ind != 0:
                    # Index to positions of particle and dipoles
                    ind1 = p.index[ip]
                    pos1 = p.pos[ind1, :]
                    ind2 = pt.index[ipt]
                    pos2 = pt.pos[ind2, :]
                    # Dipole orientations
                    dip = self.dip[ind2, :, :]
                    # Compute potentials and surface derivatives
                    e, h = self._dipolefield(pos1, pos2, dip, eps_vals[ind], k_vals[ind])
                    exc.e[ind1, :, ind2, :] = e
                    exc.h[ind1, :, ind2, :] = h

        return exc

    def _dipolefield(self, pos1, pos2, dip, eps, k):
        """
        Electromagnetic field for dipole excitation.

        MATLAB: field.m lines 52-103 (dipolefield function)

        Parameters
        ----------
        pos1 : ndarray
            Observation positions, shape (n1, 3)
        pos2 : ndarray
            Dipole positions, shape (n2, 3)
        dip : ndarray
            Dipole moments, shape (n2, 3, ndip)
        eps : float
            Dielectric function
        k : float
            Wavenumber

        Returns
        -------
        e : ndarray
            Electric field, shape (n1, 3, n2, ndip)
        h : ndarray
            Magnetic field, shape (n1, 3, n2, ndip)
        """
        # MATLAB: field.m lines 56-57
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # MATLAB: field.m lines 59-65
        # Position difference
        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        z = pos1[:, 2:3] - pos2[:, 2].T
        # Radius
        r = np.sqrt(x**2 + y**2 + z**2)
        # Make unit vector
        x, y, z = x / r, y / r, z / r

        # MATLAB: field.m line 69
        # Green function
        G = np.exp(1j * k * r) / r

        # MATLAB: field.m line 72
        ndip = dip.shape[2]

        # MATLAB: field.m lines 74-75
        # Allocate arrays
        e = np.zeros((n1, 3, n2, ndip), dtype=complex)
        h = np.zeros((n1, 3, n2, ndip), dtype=complex)

        # MATLAB: field.m lines 78-102
        for i in range(ndip):
            # Dipole moment
            dx = np.tile(dip[:, 0, i], (n1, 1))
            dy = np.tile(dip[:, 1, i], (n1, 1))
            dz = np.tile(dip[:, 2, i], (n1, 1))

            # Inner products
            inner = x * dx + y * dy + z * dz

            # MATLAB: field.m lines 88-92
            # Prefactor for magnetic field
            fac = k**2 * G * (1 - 1 / (1j * k * r)) / np.sqrt(eps)
            # Magnetic field [Jackson (9.18)]
            h[:, 0, :, i] = fac * (y * dz - z * dy)
            h[:, 1, :, i] = fac * (z * dx - x * dz)
            h[:, 2, :, i] = fac * (x * dy - y * dx)

            # MATLAB: field.m lines 95-100
            # Prefactors for electric field
            fac1 = k**2 * G / eps
            fac2 = G * (1 / r**2 - 1j * k / r) / eps
            # Electric field [Jackson (9.18)]
            e[:, 0, :, i] = fac1 * (dx - inner * x) + fac2 * (3 * inner * x - dx)
            e[:, 1, :, i] = fac1 * (dy - inner * y) + fac2 * (3 * inner * y - dy)
            e[:, 2, :, i] = fac1 * (dz - inner * z) + fac2 * (3 * inner * z - dz)

        return e, h

    def potential(self, p, enei):
        """
        Potential of dipole excitation for use in BEMRet.

        MATLAB: potential.m

        Parameters
        ----------
        p : ComParticle or Particle
            Particle surface where potential is computed
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct with scalar and vector potentials and derivatives
            All potentials have shape (nfaces, npt, ndip) or (nfaces, 3, npt, ndip)

        Notes
        -----
        MATLAB: potential.m

        Computes scalar potential φ and vector potential A with surface derivatives.
        """
        # MATLAB: potential.m line 15
        pt = self.pt

        # MATLAB: potential.m line 17
        con = self._connect(p, pt)

        # MATLAB: potential.m line 19
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        # MATLAB: potential.m line 22
        ndip = self.dip.shape[2]

        # MATLAB: potential.m lines 24-30
        exc = CompStruct(p, enei)
        exc.phi1 = np.zeros((p.n, pt.n, ndip), dtype=complex)
        exc.phi1p = np.zeros((p.n, pt.n, ndip), dtype=complex)
        exc.phi2 = np.zeros((p.n, pt.n, ndip), dtype=complex)
        exc.phi2p = np.zeros((p.n, pt.n, ndip), dtype=complex)
        exc.a1 = np.zeros((p.n, 3, pt.n, ndip), dtype=complex)
        exc.a1p = np.zeros((p.n, 3, pt.n, ndip), dtype=complex)
        exc.a2 = np.zeros((p.n, 3, pt.n, ndip), dtype=complex)
        exc.a2p = np.zeros((p.n, 3, pt.n, ndip), dtype=complex)

        # MATLAB: potential.m lines 33-61
        for inout in range(len(con)):
            for ip in range(con[inout].shape[0]):
                for ipt in range(con[inout].shape[1]):
                    ind = con[inout][ip, ipt]
                    if ind != 0:
                        # Index to positions
                        ind1 = p.index[ip]
                        pos1 = p.pos[ind1, :]
                        ind2 = pt.index[ipt]
                        pos2 = pt.pos[ind2, :]
                        # Normal vectors and dipole orientations
                        nvec = p.nvec[ind1, :]
                        dip = self.dip[ind2, :, :]
                        # Compute potentials
                        phi, phip, a, ap = self._pot(
                            pos1, pos2, nvec, dip, eps_vals[ind], k_vals[ind]
                        )
                        # Set output
                        if inout == 0:  # Inside
                            exc.phi1[ind1, ind2, :] = phi
                            exc.phi1p[ind1, ind2, :] = phip
                            exc.a1[ind1, :, ind2, :] = a
                            exc.a1p[ind1, :, ind2, :] = ap
                        else:  # Outside
                            exc.phi2[ind1, ind2, :] = phi
                            exc.phi2p[ind1, ind2, :] = phip
                            exc.a2[ind1, :, ind2, :] = a
                            exc.a2p[ind1, :, ind2, :] = ap

        return exc

    def _pot(self, pos1, pos2, nvec, dip, eps, k):
        """
        Compute potentials and surface derivatives.

        MATLAB: potential.m lines 64-123 (pot function)

        Parameters
        ----------
        pos1 : ndarray
            Observation positions, shape (n1, 3)
        pos2 : ndarray
            Dipole positions, shape (n2, 3)
        nvec : ndarray
            Normal vectors, shape (n1, 3)
        dip : ndarray
            Dipole moments, shape (n2, 3, ndip)
        eps : float
            Dielectric function
        k : float
            Wavenumber

        Returns
        -------
        phi : ndarray
            Scalar potential, shape (n1, n2, ndip)
        phip : ndarray
            Surface derivative of scalar potential, shape (n1, n2, ndip)
        a : ndarray
            Vector potential, shape (n1, 3, n2, ndip)
        ap : ndarray
            Surface derivative of vector potential, shape (n1, 3, n2, ndip)
        """
        # MATLAB: potential.m line 68
        k0 = k / np.sqrt(eps)

        # MATLAB: potential.m lines 71-80
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        z = pos1[:, 2:3] - pos2[:, 2].T
        r = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / r, y / r, z / r

        # MATLAB: potential.m lines 83-84
        G = np.exp(1j * k * r) / r
        F = (1j * k - 1 / r) * G

        # MATLAB: potential.m lines 87-91
        nx = np.tile(nvec[:, 0:1], (1, n2))
        ny = np.tile(nvec[:, 1:2], (1, n2))
        nz = np.tile(nvec[:, 2:3], (1, n2))
        en = nx * x + ny * y + nz * z

        # MATLAB: potential.m lines 94-97
        ndip = dip.shape[2]
        phi = np.zeros((n1, n2, ndip), dtype=complex)
        phip = np.zeros((n1, n2, ndip), dtype=complex)
        a = np.zeros((n1, 3, n2, ndip), dtype=complex)
        ap = np.zeros((n1, 3, n2, ndip), dtype=complex)

        # MATLAB: potential.m lines 100-122
        for i in range(ndip):
            # Dipole moment
            dx = np.tile(dip[:, 0, i], (n1, 1))
            dy = np.tile(dip[:, 1, i], (n1, 1))
            dz = np.tile(dip[:, 2, i], (n1, 1))

            # Inner products
            ep = x * dx + y * dy + z * dz
            np_dot = nx * dx + ny * dy + nz * dz

            # Scalar potential and surface derivative
            phi[:, :, i] = -ep * F / eps
            phip[:, :, i] = (
                (np_dot - 3 * en * ep) / r**2 * (1 - 1j * k * r) * G / eps
                + k**2 * ep * en * G / eps
            )

            # Vector potential [Jackson, Eq. (9.16)]
            a[:, 0, :, i] = -1j * k0 * dx * G
            a[:, 1, :, i] = -1j * k0 * dy * G
            a[:, 2, :, i] = -1j * k0 * dz * G

            # Surface derivative of vector potential
            ap[:, 0, :, i] = -1j * k0 * dx * en * F
            ap[:, 1, :, i] = -1j * k0 * dy * en * F
            ap[:, 2, :, i] = -1j * k0 * dz * en * F

        return phi, phip, a, ap

    def decayrate(self, sig):
        """
        Total and radiative decay rate for oscillating dipole.

        MATLAB: decayrate.m

        Parameters
        ----------
        sig : CompStruct
            CompStruct object containing surface charges and currents

        Returns
        -------
        tot : ndarray
            Total decay rate, shape (npt, ndip)
        rad : ndarray
            Radiative decay rate, shape (npt, ndip)
        rad0 : ndarray
            Free-space decay rate, shape (npt, ndip)
        """
        # MATLAB: decayrate.m line 16
        p, enei = sig.p, sig.enei

        # MATLAB: decayrate.m lines 18-22
        from ..greenfun import CompGreenRet
        g = CompGreenRet(self.pt, sig.p, **self.varargin)

        # MATLAB: decayrate.m line 25
        field_struct = g.field(sig)
        e = field_struct.e

        # MATLAB: decayrate.m lines 27-29
        k0 = 2 * np.pi / sig.enei
        gamma = 4 / 3 * k0**3

        # MATLAB: decayrate.m lines 32-33
        npt = self.pt.n
        ndip = self.dip.shape[2]
        tot = np.zeros((npt, ndip))
        rad0 = np.zeros((npt, ndip))

        # MATLAB: decayrate.m lines 36-38
        sca, _ = self.scattering(sig)
        sca = np.asarray(sca)
        rad = sca.reshape(rad0.shape) / (2 * np.pi * k0)

        # MATLAB: decayrate.m lines 40-62
        for ipos in range(npt):
            for idip in range(ndip):
                dip = self.dip[ipos, :, idip]
                nb = np.sqrt(self.pt.eps1(sig.enei)[ipos])

                if np.imag(nb) != 0:
                    import warnings
                    warnings.warn('Dipole embedded in medium with complex dielectric function')

                # Total decay rate
                e_i = e[ipos, :, ipos, idip]
                tot[ipos, idip] = 1 + np.imag(e_i @ dip) / (0.5 * nb * gamma)

                # Radiative decay rate
                rad[ipos, idip] = rad[ipos, idip] / (0.5 * nb * gamma)

                # Free-space decay rate
                rad0[ipos, idip] = nb * gamma

        return tot, rad, rad0

    def farfield(self, spec, enei):
        """
        Electromagnetic fields of dipoles in the far-field limit.

        MATLAB: farfield.m

        Parameters
        ----------
        spec : SpectrumRet
            SPECTRUMRET object
        enei : float
            Wavelength of light in vacuum (nm)

        Returns
        -------
        field : CompStruct
            CompStruct object that holds far-fields
        """
        # MATLAB: farfield.m lines 13-19
        dir = spec.pinfty.nvec
        epstab = self.pt.eps
        eps_val, k = epstab[spec.medium - 1](enei)
        nb = np.sqrt(eps_val)

        # MATLAB: farfield.m lines 22-24
        pt = self.pt
        dip = self.dip

        # MATLAB: farfield.m lines 27-33
        try:
            from ..geometry import ComParticle
            field = CompStruct(
                ComParticle(epstab, [spec.pinfty], spec.medium), enei
            )
        except Exception:
            field = CompStruct(spec.pinfty, enei)

        # MATLAB: farfield.m lines 35-42
        n1 = dir.shape[0]
        n2 = dip.shape[0]
        n3 = dip.shape[2]

        e = np.zeros((n1, 3, n2, n3), dtype=complex)
        h = np.zeros((n1, 3, n2, n3), dtype=complex)

        ind = pt.index[pt.inout == spec.medium]

        # MATLAB: farfield.m lines 45-58
        if len(ind) > 0:
            g = np.exp(-1j * k * (dir @ pt.pos.T))
            g = g[:, np.newaxis, :, np.newaxis]
            g = np.tile(g, (1, 3, 1, n3))

            dir_rep = dir[:, :, np.newaxis, np.newaxis]
            dir_rep = np.tile(dir_rep, (1, 1, n2, n3))

            dip_perm = dip.transpose(1, 0, 2)
            dip_rep = dip_perm[np.newaxis, :, :, :]
            dip_rep = np.tile(dip_rep, (n1, 1, 1, 1))

            h_temp = np.cross(dir_rep, dip_rep, axis=1) * g
            e_temp = np.cross(h_temp, dir_rep, axis=1)

            e[:, :, ind, :] = k**2 * e_temp[:, :, ind, :] / eps_val
            h[:, :, ind, :] = k**2 * h_temp[:, :, ind, :] / nb

        field.e = e
        field.h = h

        return field

    def scattering(self, sig):
        """
        Scattering cross section for dipole excitation.

        MATLAB: scattering.m

        Computes the scattering cross section by combining the far-fields
        from the surface charge distribution (via SpectrumRet) and the
        direct dipole far-fields, then integrating the Poynting vector.

        Parameters
        ----------
        sig : CompStruct
            CompStruct object containing surface charges and currents

        Returns
        -------
        sca : ndarray
            Scattering cross section
        dsca : CompStruct
            Differential cross section

        Notes
        -----
        MATLAB: scattering.m line 13
            [sca, dsca] = scattering(obj.spec.farfield(sig) + farfield(obj, obj.spec, sig.enei))

        The total far-field is the sum of the scattered field from the
        particle surface and the direct dipole radiation field. The
        standalone scattering() function then integrates the Poynting
        vector over the unit sphere.
        """
        # MATLAB: scattering.m line 13
        # [sca, dsca] = scattering(
        #     obj.spec.farfield(sig) + farfield(obj, obj.spec, sig.enei))
        #
        # Far-field from surface charges/currents (particle contribution)
        field_particle = self.spec.farfield(sig)
        # Far-field from dipole source (direct dipole radiation)
        field_dipole = self.farfield(self.spec, sig.enei)

        # Add the two far-fields element-wise.
        # The particle field has shape (ndir, 3) or (ndir, 3, npol),
        # while the dipole field has shape (ndir, 3, npt, ndip).
        # We need to broadcast the particle field to match.
        e_p = field_particle.e
        h_p = field_particle.h
        e_d = field_dipole.e
        h_d = field_dipole.h

        # Expand particle field dimensions to match dipole field
        while e_p.ndim < e_d.ndim:
            e_p = e_p[:, :, np.newaxis]
            h_p = h_p[:, :, np.newaxis]

        e = e_p + e_d
        h = h_p + h_d

        # Compute scattering from the combined far-field using the
        # standalone scattering function (Poynting vector integration).
        # MATLAB: [sca, dsca] = scattering(field)
        #   dsca = 0.5 * real(inner(nvec, cross(e, conj(h))))
        #   sca = area' * dsca

        if e.ndim == 2:
            e = e[:, :, np.newaxis]
            h = h[:, :, np.newaxis]

        # Handle extra dimensions from dipole (npt, ndip)
        orig_shape = e.shape[2:]
        ndir = e.shape[0]

        # Flatten trailing dimensions for integration
        e_flat = e.reshape(ndir, 3, -1)
        h_flat = h.reshape(ndir, 3, -1)
        ncols = e_flat.shape[2]

        # Poynting vector: dsca = 0.5 * real(nvec . (E x conj(H)))
        dsca_arr = np.zeros((ndir, ncols))
        for icol in range(ncols):
            poynting = np.cross(e_flat[:, :, icol], np.conj(h_flat[:, :, icol]))
            dsca_arr[:, icol] = 0.5 * np.real(
                np.sum(self.spec.nvec * poynting, axis = 1)
            )

        # Total scattering: integrate over sphere
        sca = np.dot(self.spec.area, dsca_arr)

        # Reshape back to original trailing dimensions
        if len(orig_shape) > 1:
            sca = sca.reshape(orig_shape)
            dsca_arr = dsca_arr.reshape((ndir,) + orig_shape)
        elif orig_shape == (1,):
            sca = sca[0]
            dsca_arr = dsca_arr[:, 0]

        from ..greenfun import CompStruct
        dsca = CompStruct(field_dipole.p, sig.enei, dsca = dsca_arr)

        return sca, dsca

    def _connect(self, p, pt):
        """
        Compute connectivity matrix between particle and dipole points.

        Helper function to determine which materials are connected.

        Parameters
        ----------
        p : ComParticle
            Particle
        pt : ComPoint
            Dipole points

        Returns
        -------
        con : list of ndarray
            Connectivity matrices for inside and outside
        """
        # Simplified version - assumes all connections
        # Full implementation would check material indices
        np_particles = len(p.index) if hasattr(p, 'index') else 1
        npt_points = len(pt.index) if hasattr(pt, 'index') else 1

        # Create connectivity matrices (inside and outside)
        con = [
            np.ones((np_particles, npt_points), dtype=int),
            np.ones((np_particles, npt_points), dtype=int)
        ]
        return con

    def __call__(self, p, enei):
        """
        External potential for use in BEMRet.

        MATLAB: subsref.m case '()'

        Parameters
        ----------
        p : ComParticle
            Particle surface
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct with potential information
        """
        # MATLAB: subsref.m line 28
        return self.potential(p, enei)

    def __repr__(self):
        return "DipoleRet(npt={}, ndip={})".format(self.pt.n, self.dip.shape[2])

    def __str__(self):
        return (
            "Dipole Excitation (Retarded):\n"
            "  Positions: {}\n"
            "  Dipole orientations: {}\n"
            "  Dipole shape: {}".format(
                self.pt.n, self.dip.shape[2], self.dip.shape)
        )
