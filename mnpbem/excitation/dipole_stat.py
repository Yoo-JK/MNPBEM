"""
Dipole excitation for quasistatic approximation.

Matches MATLAB MNPBEM @dipolestat implementation exactly.
"""

import numpy as np


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
        medium_idx = self.inout[0, 0] - 1  # Convert to 0-indexed
        eps_val, _ = self.eps[medium_idx](enei)
        return np.full(self.n, eps_val)


class DipoleStat:
    """
    Oscillating dipole excitation within quasistatic approximation.

    MATLAB equivalent: @dipolestat class

    Computes the external potential and fields from an oscillating dipole
    for use in BEM simulations. Also calculates total and radiative
    decay rates for the dipole near nanoparticles.

    Parameters
    ----------
    pt_pos : array_like, shape (npt, 3) or ComPoint
        Position(s) of the dipole(s) or ComPoint object
    dip : array_like, shape (3,) or (ndip, 3), optional
        Dipole moment direction(s). Default: eye(3) for x, y, z orientations
    full : bool or str, optional
        If True or 'full', dip has shape (npt, 3, ndip) with different moments at each position
    eps : list, optional
        Dielectric functions (required if pt_pos is array)

    Attributes
    ----------
    pt : ComPoint
        Point container with dipole positions
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
    """

    name = 'dipole'

    def __init__(self, pt_pos, dip=None, full=False, eps=None):
        """
        Initialize dipole excitation.

        MATLAB: dipolestat(pt, dip, op, PropertyPair)

        Parameters
        ----------
        pt_pos : array_like or ComPoint
            Dipole position(s) or point container
        dip : array_like, optional
            Dipole orientation(s)
        full : bool or str
            If True or 'full', different dipole at each position
        eps : list, optional
            Dielectric functions (required if pt_pos is array)
        """
        # Handle pt input - can be ComPoint or array of positions
        if hasattr(pt_pos, 'pos') and hasattr(pt_pos, 'eps'):
            # ComPoint or similar
            self.pt = pt_pos
        else:
            # Array of positions - need eps
            if eps is None:
                from ..materials import EpsConst
                eps = [EpsConst(1.0)]
            positions = np.atleast_2d(pt_pos).astype(float)
            self.pt = ComPoint(eps, positions, medium=1)

        self.npt = self.pt.n
        self.pt_pos = self.pt.pos  # For backward compatibility

        # Handle dipole orientations - same logic as MATLAB init.m
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
            ndip = dip.shape[0]
            self.dip = np.zeros((self.npt, 3, ndip))
            for i in range(self.npt):
                self.dip[i, :, :] = dip.T  # (3, ndip)

        self.ndip = self.dip.shape[2]

    def field(self, p, enei):
        """
        Compute electric field from dipole excitation.

        MATLAB: field.m
            e = efield( p.pos, pt.pos, obj.dip, pt.eps1( enei ) );

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
        pos2 = self.pt.pos  # Dipole positions: (npt, 3)

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # MATLAB: pt.eps1(enei) - dielectric constant at dipole positions
        # Returns (npt,) array - one eps per dipole position
        eps = self.pt.eps1(enei)  # (npt,)

        # Allocate output: (n1, 3, n2, ndip)
        e = np.zeros((n1, 3, n2, self.ndip), dtype=complex)

        # Distance vectors: pos1 - pos2
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

        # Loop over dipole orientations
        for idip in range(self.ndip):
            dip_vec = self.dip[:, :, idip]  # (npt, 3)

            # Broadcast dipole
            dx = np.broadcast_to(dip_vec[:, 0], (n1, n2))
            dy = np.broadcast_to(dip_vec[:, 1], (n1, n2))
            dz = np.broadcast_to(dip_vec[:, 2], (n1, n2))

            # Inner product: p · r̂
            p_dot_r = x_hat * dx + y_hat * dy + z_hat * dz

            # Electric field [Jackson Eq. (4.13)]
            # MATLAB: bsxfun(@times, r.^3, eps.') - eps broadcast along columns (dipole positions)
            # factor(i,j) = r(i,j)^3 * eps(j)
            factor = r**3 * eps[np.newaxis, :]  # (n1, n2)

            e[:, 0, :, idip] = (3 * p_dot_r * x_hat - dx) / factor
            e[:, 1, :, idip] = (3 * p_dot_r * y_hat - dy) / factor
            e[:, 2, :, idip] = (3 * p_dot_r * z_hat - dz) / factor

        return {
            'e': e,
            'p': p,
            'enei': enei
        }

    def potential(self, p, enei):
        """
        Compute potential of dipole excitation for BEM.

        MATLAB: phip = -inner(p.nvec, exc.e)

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

        # Wigner-Weisskopf decay rate in free space
        gamma = 4 / 3 * (2 * np.pi / enei) ** 3

        # Note: nb is computed per position inside the loop below
        # MATLAB: nb = sqrt(subsref(obj.pt.eps1(sig.enei), substruct('()', {ipos})))

        # Compute induced dipole moment
        # MATLAB: indip = matmul(bsxfun(@times, sig.p.pos, sig.p.area)', sig.sig)
        area = p.area  # (nfaces,)
        pos = p.pos    # (nfaces, 3)
        weighted_pos = area[:, np.newaxis] * pos  # (nfaces, 3)

        # Initialize output arrays
        tot = np.zeros((self.npt, self.ndip))
        rad = np.zeros((self.npt, self.ndip))
        rad0 = np.zeros((self.npt, self.ndip))

        # Get eps at all dipole positions
        # MATLAB: obj.pt.eps1(sig.enei)
        pt_eps = self.pt.eps1(enei)  # (npt,)

        # Compute induced field at dipole positions using Green function
        # For static case, need to compute from surface charges
        for ipt in range(self.npt):
            pt_pos = self.pt.pos[ipt]

            # Refractive index at this dipole position
            # MATLAB: nb = sqrt(subsref(obj.pt.eps1(sig.enei), substruct('()', {ipos})))
            nb = np.sqrt(np.real(pt_eps[ipt]))
            if np.imag(pt_eps[ipt]) != 0:
                import warnings
                warnings.warn('Dipole embedded in medium with complex dielectric function')

            for idip in range(self.ndip):
                dip_vec = self.dip[ipt, :, idip]

                # Get surface charge for this position and orientation
                if surface_charge.ndim == 3:
                    sig_vals = surface_charge[:, ipt, idip]
                elif surface_charge.ndim == 2:
                    sig_vals = surface_charge[:, ipt]
                else:
                    sig_vals = surface_charge

                # Induced dipole moment
                # MATLAB: indip(:, ipos, idip) = weighted_pos.T @ sig_vals
                dip_ind = weighted_pos.T @ sig_vals  # (3,)

                # Compute induced field at dipole position
                r_vec = pt_pos - pos  # (nfaces, 3)
                r_mag = np.linalg.norm(r_vec, axis=1)
                r_mag = np.maximum(r_mag, 1e-30)

                # Field from surface charge distribution (static Green function)
                eps_at_pt = pt_eps[ipt]
                e_ind = np.sum(
                    sig_vals[:, np.newaxis] * area[:, np.newaxis] *
                    r_vec / (r_mag[:, np.newaxis]**3),
                    axis=0
                ) / eps_at_pt

                # Total decay rate: 1 + Im(E · dip) / (0.5 * nb * gamma)
                tot[ipt, idip] = 1 + np.imag(np.dot(e_ind, dip_vec)) / (0.5 * nb * gamma)

                # Radiative decay rate: |nb^2 * dip_ind + dip|^2
                # MATLAB: rad(ipos, idip) = norm(nb^2 * indip(:, ipos, idip).' + dip)^2
                rad[ipt, idip] = np.linalg.norm(nb**2 * dip_ind + dip_vec)**2

                # Free-space decay rate
                rad0[ipt, idip] = nb * gamma

        return tot, rad, rad0

    def __repr__(self):
        return f"DipoleStat(npt={self.npt}, ndip={self.ndip})"
