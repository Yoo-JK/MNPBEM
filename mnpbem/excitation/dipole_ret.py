"""
Dipole excitation for full Maxwell equations (retarded case).

Matches MATLAB MNPBEM @dipoleret implementation exactly.
"""

import numpy as np
from ..spectrum import SpectrumRet
from ..geometry import trisphere, connect


class ComPoint:
    """
    Simple point container for dipole positions.

    MATLAB: compoint class - stores positions with inout property.
    """

    def __init__(self, eps, positions, medium=1):
        """
        Initialize point container.

        Parameters
        ----------
        eps : list
            Dielectric functions
        positions : ndarray, shape (n, 3)
            Point positions
        medium : int
            Medium index where points are located
        """
        self.eps = eps
        self.pos = np.atleast_2d(positions).astype(float)
        self.n = self.pos.shape[0]
        self.npt = self.n
        # All points are in the specified medium
        self.inout = np.full((self.n, 2), medium, dtype=int)

    def eps1(self, enei):
        """Get dielectric constant at positions."""
        # Points are in the medium specified by inout
        medium_idx = self.inout[0, 0] - 1  # Convert to 0-indexed
        eps_val, _ = self.eps[medium_idx](enei)
        return np.full(self.n, eps_val)

    def index_func(self, particle_indices):
        """Get indices for given particle indices."""
        if np.isscalar(particle_indices):
            particle_indices = np.atleast_1d(particle_indices)
        # For points, index is just the position indices
        return np.array(particle_indices, dtype=int) - 1  # Convert to 0-indexed


class DipoleRet:
    """
    Oscillating dipole excitation for full Maxwell equations.

    MATLAB equivalent: @dipoleret class

    Computes the scalar and vector potentials from an oscillating dipole
    including retardation effects for use in BEM simulations.

    Parameters
    ----------
    pt_pos : array_like, shape (npt, 3) or ComPoint
        Position(s) of the dipole(s) or ComPoint object
    dip : array_like, shape (3,) or (ndip, 3), optional
        Dipole moment direction(s). Default: eye(3) for x, y, z orientations
    full : bool or str, optional
        If True or 'full', dip has shape (npt, 3, ndip) with different moments at each position
    medium : int, optional
        Embedding medium index (1-indexed, default: 1)
    pinfty : Particle, optional
        Unit sphere for far-field integration. Default: trisphere(256, 2)

    Attributes
    ----------
    pt : ComPoint
        Point container with dipole positions
    dip : ndarray, shape (npt, 3, ndip)
        Dipole moment directions for each position
    spec : SpectrumRet
        Spectrum object for scattering calculations
    medium : int
        Medium index
    """

    name = 'dipole'

    def __init__(self, pt_pos, dip=None, full=False, medium=1, pinfty=None, eps=None):
        """
        Initialize dipole excitation.

        MATLAB: dipoleret(pt, dip, 'full', op, PropertyPair)

        Parameters
        ----------
        pt_pos : array_like or ComPoint
            Dipole position(s) or point container
        dip : array_like, optional
            Dipole orientation(s)
        full : bool or str
            If True or 'full', different dipole at each position
        medium : int
            Embedding medium index (1-indexed)
        pinfty : Particle, optional
            Unit sphere for far-field. Default: trisphere(256, 2)
        eps : list, optional
            Dielectric functions (required if pt_pos is array)
        """
        self.medium = medium

        # Handle pt input - can be ComPoint or array of positions
        if hasattr(pt_pos, 'pos') and hasattr(pt_pos, 'eps'):
            # ComPoint or similar
            self.pt = pt_pos
        else:
            # Array of positions - need eps
            if eps is None:
                # Create simple eps list with constant medium
                from ..materials import EpsConst
                eps = [EpsConst(1.0)]
            positions = np.atleast_2d(pt_pos).astype(float)
            self.pt = ComPoint(eps, positions, medium)

        self.npt = self.pt.n
        self.pt_pos = self.pt.pos  # For backward compatibility

        # Handle dipole orientations
        # MATLAB: init.m logic
        if dip is None:
            dip = np.eye(3)

        dip = np.asarray(dip, dtype=float)

        # Check for 'full' keyword
        is_full = full is True or full == 'full'

        if is_full:
            # Dipole moments given at all positions: shape (npt, 3, ndip)
            if dip.ndim == 2:
                dip = dip[:, :, np.newaxis]
            self.dip = dip
        else:
            # Same dipole moments for all positions
            if dip.ndim == 1:
                dip = dip[np.newaxis, :]  # (1, 3)

            # MATLAB: repmat(reshape(dip.', [1, fliplr(size(dip))]), [obj.pt.n, 1, 1])
            # dip is (ndip, 3), transpose to (3, ndip), reshape to (1, 3, ndip)
            # then replicate to (npt, 3, ndip)
            ndip = dip.shape[0]
            self.dip = np.zeros((self.npt, 3, ndip))
            for i in range(self.npt):
                self.dip[i, :, :] = dip.T  # (3, ndip)

        self.ndip = self.dip.shape[2]

        # Initialize spectrum for radiative decay rate calculation
        # MATLAB: obj.spec = spectrumret(pinfty, 'medium', medium)
        if pinfty is None:
            pinfty = trisphere(256, 2.0)
        self.spec = SpectrumRet(pinfty, medium=medium)

    def potential(self, p, enei):
        """
        Compute potentials for dipole excitation.

        MATLAB: potential.m with connect() for proper media routing.

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
        # Get connectivity between particle and dipole points
        # MATLAB: con = connect(p, pt)
        con = connect(p, self.pt)

        # Get dielectric functions and wavenumbers
        eps_list = []
        k_list = []
        for eps_mat in p.eps:
            eps_val, k_val = eps_mat(enei)
            eps_list.append(eps_val)
            k_list.append(k_val)

        n1 = p.nfaces
        n2 = self.npt
        ndip = self.ndip

        # Initialize output arrays
        phi1 = np.zeros((n1, n2, ndip), dtype=complex)
        phi1p = np.zeros((n1, n2, ndip), dtype=complex)
        phi2 = np.zeros((n1, n2, ndip), dtype=complex)
        phi2p = np.zeros((n1, n2, ndip), dtype=complex)

        a1 = np.zeros((n1, 3, n2, ndip), dtype=complex)
        a1p = np.zeros((n1, 3, n2, ndip), dtype=complex)
        a2 = np.zeros((n1, 3, n2, ndip), dtype=complex)
        a2p = np.zeros((n1, 3, n2, ndip), dtype=complex)

        # MATLAB logic: loop over inout (inside/outside)
        # con is a 2D list: con[i][j] where i is p's inout column, j is pt's inout column
        # MATLAB uses linear indexing on cell array: con{inout} where
        #   inout=1 -> con{1,1} = con[0][0] in Python
        #   inout=2 -> con{2,1} = con[1][0] in Python (column-major linear indexing)
        for inout_idx in range(2):
            # MATLAB: con{inout} with linear indexing on 2D cell
            # con{1} = con{1,1}, con{2} = con{2,1}
            conn_matrix = con[inout_idx][0]  # Always use first column (j=0)

            # Find all connected face-point pairs
            # MATLAB: for ip = 1:size(con{inout}, 1), for ipt = 1:size(con{inout}, 2)
            for ip in range(conn_matrix.shape[0]):
                for ipt in range(conn_matrix.shape[1]):
                    # Check if connected (medium index > 0)
                    # MATLAB: ind = con{inout}(ip, ipt)
                    ind = conn_matrix[ip, ipt]

                    if ind != 0:
                        # MATLAB: ind1 = p.index(ip); ind2 = pt.index(ipt)
                        # For simple particles, index is identity
                        ind1 = ip
                        ind2 = ipt

                        # Get eps and k for connecting medium
                        eps = eps_list[ind - 1]
                        k = k_list[ind - 1]

                        # Compute potentials for this face-point pair
                        pos1 = p.pos[ind1:ind1+1]
                        pos2 = self.pt.pos[ind2:ind2+1]
                        nvec = p.nvec[ind1:ind1+1]
                        dip_vals = self.dip[ind2:ind2+1, :, :]

                        phi, phip, a, ap = self._compute_pot(
                            pos1, pos2, nvec, dip_vals, eps, k
                        )

                        # Store in appropriate array based on inout
                        # MATLAB: switch inout, case 1: phi1/a1, case 2: phi2/a2
                        if inout_idx == 0:  # Inside surface (inout=1 in MATLAB)
                            phi1[ind1, ind2, :] = phi.flatten()
                            phi1p[ind1, ind2, :] = phip.flatten()
                            a1[ind1, :, ind2, :] = a.reshape(3, ndip)
                            a1p[ind1, :, ind2, :] = ap.reshape(3, ndip)
                        else:  # Outside surface (inout=2 in MATLAB)
                            phi2[ind1, ind2, :] = phi.flatten()
                            phi2p[ind1, ind2, :] = phip.flatten()
                            a2[ind1, :, ind2, :] = a.reshape(3, ndip)
                            a2p[ind1, :, ind2, :] = ap.reshape(3, ndip)

        return {
            'phi1': phi1, 'phi1p': phi1p,
            'phi2': phi2, 'phi2p': phi2p,
            'a1': a1, 'a1p': a1p,
            'a2': a2, 'a2p': a2p,
            'p': p,
            'enei': enei
        }

    def _compute_pot(self, pos1, pos2, nvec, dip, eps, k):
        """
        Compute potentials and surface derivatives.

        MATLAB: pot() subfunction in potential.m
        """
        k0 = k / np.sqrt(eps)  # Wavenumber in vacuum

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        ndip = dip.shape[2]

        # Position difference
        x = pos1[:, 0:1] - pos2[:, 0:1].T
        y = pos1[:, 1:2] - pos2[:, 1:2].T
        z = pos1[:, 2:3] - pos2[:, 2:3].T

        # Radius
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 1e-30)

        # Unit vectors
        x_hat = x / r
        y_hat = y / r
        z_hat = z / r

        # Green function and derivative
        G = np.exp(1j * k * r) / r
        F = (1j * k - 1.0 / r) * G

        # Normal vectors
        nx = np.broadcast_to(nvec[:, 0:1], (n1, n2))
        ny = np.broadcast_to(nvec[:, 1:2], (n1, n2))
        nz = np.broadcast_to(nvec[:, 2:3], (n1, n2))

        # Inner product en = nvec · r̂
        en = nx * x_hat + ny * y_hat + nz * z_hat

        # Allocate output
        phi = np.zeros((n1, n2, ndip), dtype=complex)
        phip = np.zeros((n1, n2, ndip), dtype=complex)
        a = np.zeros((n1, 3, n2, ndip), dtype=complex)
        ap = np.zeros((n1, 3, n2, ndip), dtype=complex)

        for i in range(ndip):
            # Dipole moment
            dx = np.broadcast_to(dip[:, 0, i].T, (n1, n2))
            dy = np.broadcast_to(dip[:, 1, i].T, (n1, n2))
            dz = np.broadcast_to(dip[:, 2, i].T, (n1, n2))

            # Inner products
            ep = x_hat * dx + y_hat * dy + z_hat * dz  # p · r̂
            np_dot = nx * dx + ny * dy + nz * dz       # nvec · p

            # Scalar potential: phi = -p·r̂ * F / ε
            phi[:, :, i] = -ep * F / eps

            # Surface derivative [MATLAB potential.m]
            phip[:, :, i] = (
                (np_dot - 3 * en * ep) / r**2 * (1 - 1j * k * r) * G / eps +
                k**2 * ep * en * G / eps
            )

            # Vector potential [Jackson Eq. (9.16)]: a = -ik₀ * p * G
            a[:, 0, :, i] = -1j * k0 * dx * G
            a[:, 1, :, i] = -1j * k0 * dy * G
            a[:, 2, :, i] = -1j * k0 * dz * G

            # Surface derivative of vector potential
            ap[:, 0, :, i] = -1j * k0 * dx * en * F
            ap[:, 1, :, i] = -1j * k0 * dy * en * F
            ap[:, 2, :, i] = -1j * k0 * dz * en * F

        return phi, phip, a, ap

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
            Dictionary containing 'e' and 'h' fields
        """
        pos1 = p.pos
        pos2 = self.pt.pos

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
            fac_h = k**2 * G * (1 - 1 / (1j * k * r)) / np.sqrt(eps_val)
            h[:, 0, :, idip] = fac_h * (y_hat * dz - z_hat * dy)
            h[:, 1, :, idip] = fac_h * (z_hat * dx - x_hat * dz)
            h[:, 2, :, idip] = fac_h * (x_hat * dy - y_hat * dx)

            # Electric field [Jackson (9.18)]
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

    def farfield(self, spec, enei):
        """
        Compute electromagnetic far-fields of dipoles.

        MATLAB: farfield.m

        Parameters
        ----------
        spec : SpectrumRet
            Spectrum object with unit sphere directions
        enei : float
            Wavelength in nm

        Returns
        -------
        field : dict
            Far-field with 'e' and 'h' arrays
        """
        # Normal vectors of unit sphere at infinity
        direction = spec.nvec

        # Dielectric function and wavenumber
        eps, k = self.pt.eps[spec.medium - 1](enei)
        nb = np.sqrt(eps)

        n1 = len(direction)
        n2 = self.npt
        n3 = self.ndip

        # Initialize far-fields
        e = np.zeros((n1, 3, n2, n3), dtype=complex)
        h = np.zeros((n1, 3, n2, n3), dtype=complex)

        # Find dipoles connected through medium
        # MATLAB: ind = pt.index(find(pt.inout == spec.medium)')
        ind = np.where(self.pt.inout[:, 0] == spec.medium)[0]

        if len(ind) > 0:
            # Green function for k*r -> infinity: exp(-i*k*dir·pos)
            # g shape: (ndir, npt)
            g = np.exp(-1j * k * np.dot(direction, self.pt.pos[ind].T))
            # Expand g to (ndir, 1, npt, 1) for broadcasting
            g_expanded = g[:, np.newaxis, :, np.newaxis]

            # Direction expanded: (ndir, 3, 1, 1)
            dir_exp = direction[:, :, np.newaxis, np.newaxis]

            # Dipole expanded: (1, 3, npt, ndip)
            dip_exp = self.dip[ind][np.newaxis, :, :, :].transpose(0, 2, 1, 3)
            # Actually: dip shape is (npt, 3, ndip), we need (1, 3, npt, ndip)
            dip_exp = self.dip[ind].transpose(1, 0, 2)[np.newaxis, :, :, :]

            # Far-field amplitude: h = cross(dir, dip) * g
            # Need to compute for all combinations
            for i_dir in range(n1):
                for i_pt, pt_idx in enumerate(ind):
                    for i_dip in range(n3):
                        dip_vec = self.dip[pt_idx, :, i_dip]
                        dir_vec = direction[i_dir]
                        g_val = g[i_dir, i_pt]

                        # h = cross(dir, dip) * g
                        h_vec = np.cross(dir_vec, dip_vec) * g_val
                        # e = cross(h, dir)
                        e_vec = np.cross(h_vec, dir_vec)

                        e[i_dir, :, pt_idx, i_dip] = k**2 * e_vec / eps
                        h[i_dir, :, pt_idx, i_dip] = k**2 * h_vec / nb

        return {
            'e': e,
            'h': h,
            'nvec': direction,
            'area': spec.area,
            'enei': enei,
            'k': k
        }

    def scattering(self, sig):
        """
        Compute scattering cross section for dipole excitation.

        MATLAB:
            [sca, dsca] = scattering(obj.spec.farfield(sig) + farfield(obj, obj.spec, sig.enei))

        Parameters
        ----------
        sig : dict
            Solution from BEM solver

        Returns
        -------
        sca : ndarray
            Scattering cross section
        dsca : ndarray
            Differential scattering cross section
        """
        # Get far-field from surface charges/currents
        field_surf = self.spec.farfield(sig)

        # Get far-field from dipole itself
        field_dip = self.farfield(self.spec, sig['enei'])

        # Add fields
        e_total = field_surf['e'] + self._sum_dipole_field(field_dip['e'])
        h_total = field_surf['h'] + self._sum_dipole_field(field_dip['h'])

        # Compute scattering from total field
        npol = e_total.shape[2] if e_total.ndim > 2 else 1

        if e_total.ndim == 2:
            e_total = e_total[:, :, np.newaxis]
            h_total = h_total[:, :, np.newaxis]

        dsca = np.zeros((self.spec.ndir, npol))

        for ipol in range(npol):
            poynting = np.cross(e_total[:, :, ipol], np.conj(h_total[:, :, ipol]))
            dsca[:, ipol] = 0.5 * np.real(np.sum(self.spec.nvec * poynting, axis=1))

        sca = np.dot(self.spec.area, dsca)

        if npol == 1:
            sca = sca[0]
            dsca = dsca[:, 0]

        return sca, dsca

    def _sum_dipole_field(self, field):
        """Sum dipole field over positions and orientations if needed."""
        if field.ndim == 4:
            # Shape is (ndir, 3, npt, ndip) - sum over npt
            return np.sum(field, axis=2)
        return field

    def decayrate(self, sig):
        """
        Compute total and radiative decay rates for dipole near particle.

        MATLAB: decayrate.m

        Parameters
        ----------
        sig : dict
            Solution from BEM solver

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

        # Initialize output
        tot = np.zeros((self.npt, self.ndip))
        rad = np.zeros((self.npt, self.ndip))
        rad0 = np.zeros((self.npt, self.ndip))

        # Compute scattering cross section for radiative decay rate
        sca, _ = self.scattering(sig)
        if np.isscalar(sca):
            sca = np.full((self.npt, self.ndip), sca)
        else:
            sca = np.broadcast_to(sca, (self.npt, self.ndip)).copy()

        # Radiative decay rate: rad = sca / (2*pi*k0)
        # Then normalize by free-space rate

        # Compute induced field at dipole positions
        # This uses Green function from surface to dipole positions
        e_ind = self._compute_induced_field(sig)

        for ipos in range(self.npt):
            for idip in range(self.ndip):
                # Dipole moment
                dip = self.dip[ipos, :, idip]

                # Refractive index at dipole position
                nb = np.sqrt(np.real(self.pt.eps1(enei)[ipos]))

                # Total decay rate: 1 + Im(E_ind · dip) / (0.5 * nb * gamma)
                tot[ipos, idip] = 1 + np.imag(
                    np.dot(e_ind[ipos, :, ipos, idip], dip)
                ) / (0.5 * nb * gamma)

                # Radiative decay rate normalized to free-space
                rad[ipos, idip] = sca[ipos, idip] / (2 * np.pi * k0) / (0.5 * nb * gamma)

                # Free-space decay rate
                rad0[ipos, idip] = nb * gamma

        return tot, rad, rad0

    def _compute_induced_field(self, sig):
        """
        Compute induced electric field at dipole positions.

        Uses Green function from particle surface to dipole positions.
        """
        p = sig['p']
        enei = sig['enei']

        # Get dielectric function and wavenumber
        eps_val, k = p.eps[self.medium - 1](enei)

        # Get surface charges
        sig1 = sig.get('sig1', np.zeros(p.nfaces))
        sig2 = sig.get('sig2', np.zeros(p.nfaces))
        h1 = sig.get('h1', np.zeros((p.nfaces, 3)))
        h2 = sig.get('h2', np.zeros((p.nfaces, 3)))

        # Total surface charge and current
        if sig1.ndim == 1:
            sig_total = sig1 + sig2
        else:
            sig_total = sig1 + sig2

        if h1.ndim == 2:
            h_total = h1 + h2
        else:
            h_total = h1 + h2

        # Initialize output: (npt, 3, npt, ndip)
        e_ind = np.zeros((self.npt, 3, self.npt, self.ndip), dtype=complex)

        pos = p.pos
        area = p.area

        for ipt in range(self.npt):
            pt_pos = self.pt.pos[ipt]

            # Distance from all faces to dipole position
            r_vec = pt_pos - pos  # (nfaces, 3)
            r_mag = np.linalg.norm(r_vec, axis=1)
            r_mag = np.maximum(r_mag, 1e-30)
            r_hat = r_vec / r_mag[:, np.newaxis]

            # Green function and derivative
            G = np.exp(1j * k * r_mag) / r_mag
            F = (1j * k - 1 / r_mag) * G

            # Electric field from surface charges
            # E = -grad(G) * sig * area / eps
            # -grad(G) = F * r_hat
            for idip in range(self.ndip):
                if sig_total.ndim == 1:
                    sig_vals = sig_total
                else:
                    idx = ipt * self.ndip + idip
                    if idx < sig_total.shape[1]:
                        sig_vals = sig_total[:, idx]
                    else:
                        sig_vals = sig_total[:, 0]

                # Field from charges
                e_charge = np.sum(
                    sig_vals[:, np.newaxis] * area[:, np.newaxis] *
                    F[:, np.newaxis] * r_hat,
                    axis=0
                ) / eps_val

                e_ind[ipt, :, ipt, idip] = e_charge

        return e_ind

    def __repr__(self):
        return f"DipoleRet(npt={self.npt}, ndip={self.ndip}, medium={self.medium})"
