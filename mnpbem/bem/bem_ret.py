"""
BEM solver for full Maxwell equations (retarded).

Given an external excitation, BEMRet computes the surface charges
and currents such that the boundary conditions of Maxwell's equations
are fulfilled.

Reference:
    Garcia de Abajo and Howie, PRB 65, 115418 (2002)

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np
from ..greenfun import CompGreenRet


class BEMRet:
    """
    BEM solver for full Maxwell equations (retarded).

    Solves the boundary element method equations in the retarded regime
    to find surface charges and currents given external fields.

    Parameters
    ----------
    p : ComParticle
        Composite particle with geometry and material properties
    enei : float, optional
        Photon energy (eV) or wavelength (nm) for pre-initialization

    Attributes
    ----------
    p : ComParticle
        The particle object
    enei : float or None
        Current wavelength/energy (None if not initialized)
    k : float or None
        Wavenumber in vacuum (2π/λ)

    Notes
    -----
    The BEM equations for retarded case are more complex than quasistatic,
    involving coupled equations for surface charges (σ) and currents (h).

    The implementation follows Garcia de Abajo and Howie, PRB 65, 115418 (2002),
    Equations (19-22).

    MATLAB convention (bemret/private/initmat.m):
        G1 = g{1,1}.G(enei) - g{2,1}.G(enei)  # inside Green function
        G2 = g{2,2}.G(enei) - g{1,2}.G(enei)  # outside Green function
        H1 = g{1,1}.H1(enei) - g{2,1}.H1(enei)
        H2 = g{2,2}.H2(enei) - g{1,2}.H2(enei)

    For single particle, g{1,2} and g{2,1} are typically 0.

    Key matrices:
        G1, G2 : Green functions (inside/outside) with different wavenumbers
        H1, H2 : Surface derivatives with ±2π terms
        L1, L2 : G * ε * G^(-1) matrices (or just ε for single particle)
        Sigma1, Sigma2 : H * G^(-1) matrices
        Deltai : inv(Sigma1 - Sigma2)
        Sigmai : Inverse of combined Sigma matrix

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.bem import BEMRet
    >>>
    >>> # Create gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Create BEM solver
    >>> bem = BEMRet(p)
    >>>
    >>> # Initialize at specific wavelength
    >>> bem.init(600.0)
    """

    def __init__(self, p, enei=None):
        """
        Initialize BEM solver for retarded approximation.

        Parameters
        ----------
        p : ComParticle
            Composite particle
        enei : float, optional
            Photon energy (eV) or wavelength (nm) for pre-initialization
        """
        self.p = p
        self.enei = None

        # BEM matrices (initialized on demand)
        self.k = None
        self.nvec = None
        self.eps1 = None
        self.eps2 = None
        self.G1i = None
        self.G2i = None
        self.L1 = None
        self.L2 = None
        self.Sigma1 = None
        self.Deltai = None
        self.Sigmai = None

        # Initialize at specific energy if provided
        if enei is not None:
            self.init(enei)

    def init(self, enei):
        """
        Initialize BEM solver for specific wavelength/energy.

        Computes all necessary matrices for solving the BEM equations.
        Follows MATLAB bemret/private/initmat.m exactly.

        Parameters
        ----------
        enei : float
            Photon energy (eV) or wavelength (nm)

        Returns
        -------
        self : BEMRet
            Returns self for chaining
        """
        # Skip if already initialized at this energy
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        self.enei = enei

        # Outer surface normals
        self.nvec = self.p.nvec

        # Wavenumber in vacuum
        self.k = 2 * np.pi / enei

        # Get dielectric functions and wavenumbers for each material
        # MATLAB: [~, k] = cellfun(@(eps)(eps(enei)), obj.p1.eps)
        eps_vals = []
        k_vals = []
        for eps_func in self.p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        # For single particle: eps[0] = outside, eps[1] = inside
        # k_out = k_vals[0], k_in = k_vals[1]
        k_out = k_vals[0]  # outside (vacuum)
        k_in = k_vals[1]   # inside (metal)

        # Dielectric function values
        eps1_vals = self.p.eps1(enei)  # inside (nfaces,)
        eps2_vals = self.p.eps2(enei)  # outside (nfaces,)

        # Check if all values are the same (can use scalar)
        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(eps2_vals, eps2_vals[0]):
            self.eps1 = eps1_vals[0]
            self.eps2 = eps2_vals[0]
        else:
            self.eps1 = np.diag(eps1_vals)
            self.eps2 = np.diag(eps2_vals)

        # Create Green functions with correct wavenumbers
        # MATLAB: G1 = g{1,1}.G - g{2,1}.G (inside medium, k_in)
        #         G2 = g{2,2}.G - g{1,2}.G (outside medium, k_out)
        # For single particle, cross terms are 0
        g_in = CompGreenRet(self.p, self.p, enei, k=k_in)   # inside Green function
        g_out = CompGreenRet(self.p, self.p, enei, k=k_out) # outside Green function

        G1 = g_in.G   # inside Green function
        G2 = g_out.G  # outside Green function

        H1_mat = g_in.H1()   # H1 with inside k
        H2_mat = g_out.H2()  # H2 with outside k

        # Compute inverses
        self.G1i = np.linalg.inv(G1)
        self.G2i = np.linalg.inv(G2)

        # L matrices [Eq. (22)]
        # MATLAB: if all(obj.g.con{1,2} == 0), L1 = eps1; L2 = eps2
        # For single particle with simple connectivity: L1 = eps1, L2 = eps2
        if np.isscalar(self.eps1):
            self.L1 = self.eps1
            self.L2 = self.eps2
        else:
            # Full case: L1 = G1 * eps1 * G1^(-1)
            self.L1 = G1 @ self.eps1 @ self.G1i
            self.L2 = G2 @ self.eps2 @ self.G2i

        # Sigma matrices [Eq. (21)]
        # Sigma1 = H1 * G1^(-1)
        # Sigma2 = H2 * G2^(-1)
        self.Sigma1 = H1_mat @ self.G1i
        Sigma2 = H2_mat @ self.G2i

        # Inverse Delta matrix
        self.Deltai = np.linalg.inv(self.Sigma1 - Sigma2)

        # Combined Sigma matrix [Eq. (21,22)]
        # Sigma = Sigma1*L1 - Sigma2*L2 + k²*(L*Deltai)*(nvec*nvec')*L
        L = self.L1 - self.L2

        if np.isscalar(L):
            # Simplified case for uniform materials
            Sigma = self.Sigma1 * self.L1 - Sigma2 * self.L2
            # Add magnetic coupling term
            nvec_outer = self.nvec @ self.nvec.T  # (nfaces, nfaces)
            Sigma = Sigma + self.k**2 * L * (self.Deltai * nvec_outer) * L
        else:
            # Full matrix case
            nvec_outer = self.nvec @ self.nvec.T
            Sigma = (self.Sigma1 @ self.L1 - Sigma2 @ self.L2 +
                     self.k**2 * ((L @ self.Deltai) * nvec_outer) @ L)

        self.Sigmai = np.linalg.inv(Sigma)

        return self

    def _excitation(self, exc):
        """
        Process excitation to get phi, a, alpha, De.

        MATLAB: bemret/private/excitation.m

        Parameters
        ----------
        exc : dict
            Excitation with fields phi1, phi2, a1, a2, phi1p, phi2p, a1p, a2p

        Returns
        -------
        phi, a, alpha, De : ndarray
            Processed excitation variables for BEM equations
        """
        enei = exc['enei']

        # Default values for potentials
        nfaces = self.p.nfaces

        # Helper to get field with default of 0
        def get_field(name, default_shape=None):
            val = exc.get(name, 0)
            if isinstance(val, np.ndarray):
                return val
            elif val == 0 and default_shape is not None:
                return np.zeros(default_shape, dtype=complex)
            return val

        # Get potential values with defaults of 0
        phi1 = get_field('phi1')
        phi1p = get_field('phi1p')
        a1 = get_field('a1')
        a1p = get_field('a1p')
        phi2 = get_field('phi2')
        phi2p = get_field('phi2p')
        a2 = get_field('a2')
        a2p = get_field('a2p')

        # Wavenumber of light in vacuum
        k = 2 * np.pi / enei

        # Dielectric functions
        eps1 = self.p.eps1(enei)  # (nfaces,)
        eps2 = self.p.eps2(enei)  # (nfaces,)

        # Outer surface normal
        nvec = self.nvec

        # External excitation - Garcia de Abajo and Howie, PRB 65, 115418 (2002)

        # Eqs. (10,11): potential jumps
        phi = self._subtract(phi2, phi1)
        a = self._subtract(a2, a1)

        # Eq. (15): alpha = a2p - a1p - 1i*k*(outer(nvec, phi2)*eps2 - outer(nvec, phi1)*eps1)
        outer_term2 = self._outer_eps(nvec, phi2, eps2)
        outer_term1 = self._outer_eps(nvec, phi1, eps1)
        alpha = self._subtract(a2p, a1p) - 1j * k * self._subtract(outer_term2, outer_term1)

        # Eq. (18): De = eps2*phi2p - eps1*phi1p - 1i*k*(inner(nvec,a2)*eps2 - inner(nvec,a1)*eps1)
        matmul_term2 = self._matmul_eps(eps2, phi2p)
        matmul_term1 = self._matmul_eps(eps1, phi1p)
        inner_term2 = self._inner_eps(nvec, a2, eps2)
        inner_term1 = self._inner_eps(nvec, a1, eps1)

        De = self._subtract(matmul_term2, matmul_term1) - 1j * k * self._subtract(inner_term2, inner_term1)

        return phi, a, alpha, De

    def _subtract(self, a, b):
        """Subtract b from a, handling scalars and arrays."""
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a - b
        elif isinstance(a, np.ndarray):
            if b == 0:
                return a
            return a - b
        elif isinstance(b, np.ndarray):
            if a == 0:
                return -b
            return a - b
        else:
            return a - b

    def _outer_eps(self, nvec, phi, eps):
        """Compute outer(nvec, phi) * eps. Returns (nfaces, 3) or (nfaces, 3, npol)."""
        if isinstance(phi, np.ndarray):
            if phi.ndim == 1:
                # phi is (nfaces,), eps is (nfaces,)
                return nvec * (phi * eps)[:, np.newaxis]  # (nfaces, 3)
            else:
                # phi is (nfaces, npol)
                npol = phi.shape[1]
                result = np.zeros((len(nvec), 3, npol), dtype=complex)
                for ipol in range(npol):
                    result[:, :, ipol] = nvec * (phi[:, ipol] * eps)[:, np.newaxis]
                return result
        elif phi == 0:
            return 0
        else:
            return nvec * (phi * eps)

    def _inner_eps(self, nvec, a, eps):
        """Compute inner(nvec, a) * eps. Returns (nfaces,) or (nfaces, npol)."""
        if isinstance(a, np.ndarray) and a.ndim >= 2:
            if a.ndim == 2:
                # a is (nfaces, 3), nvec is (nfaces, 3)
                dot = np.sum(nvec * a, axis=1)  # (nfaces,)
                return dot * eps
            else:
                # a is (nfaces, 3, npol)
                npol = a.shape[2]
                result = np.zeros((len(nvec), npol), dtype=complex)
                for ipol in range(npol):
                    dot = np.sum(nvec * a[:, :, ipol], axis=1)
                    result[:, ipol] = dot * eps
                return result
        elif isinstance(a, np.ndarray) and a.size == 0:
            return 0
        elif not isinstance(a, np.ndarray) and a == 0:
            return 0
        else:
            return 0

    def _matmul_eps(self, eps, phi_p):
        """Compute eps * phi_p (element-wise for diagonal eps)."""
        if isinstance(phi_p, np.ndarray):
            if phi_p.ndim == 1:
                return eps * phi_p
            else:
                # (nfaces, npol)
                return eps[:, np.newaxis] * phi_p
        elif phi_p == 0:
            return 0
        else:
            return eps * phi_p

    def solve(self, exc):
        """
        Solve BEM equations for given excitation.

        Computes surface charges and currents from external fields.
        MATLAB: bemret/mldivide.m

        Parameters
        ----------
        exc : dict
            Excitation dictionary with fields:
                - enei : wavelength/energy
                - phi1, phi2, a1, a2 : potentials (optional, default 0)
                - phi1p, phi2p, a1p, a2p : potential derivatives

        Returns
        -------
        sig : dict
            Dictionary containing:
                - sig1, sig2 : surface charge distributions (inside/outside)
                - h1, h2 : surface current distributions (inside/outside)
                - enei : wavelength/energy
                - p : particle object

        Notes
        -----
        This implements Equations (19-20) from Garcia de Abajo & Howie (2002).
        """
        enei = exc['enei']

        # Initialize at excitation energy if needed
        self.init(enei)

        # Compute excitation variables from raw inputs
        # MATLAB: [phi, a, alpha, De] = excitation(obj, exc)
        phi, a, alpha, De = self._excitation(exc)

        # Get stored variables
        k = self.k
        nvec = self.nvec
        G1i = self.G1i
        G2i = self.G2i
        L1 = self.L1
        L2 = self.L2
        Sigma1 = self.Sigma1
        Deltai = self.Deltai
        Sigmai = self.Sigmai
        nfaces = self.p.nfaces

        # Ensure phi, a have proper shapes
        if not isinstance(phi, np.ndarray) or phi.ndim == 0 or (isinstance(phi, np.ndarray) and phi.size == 1 and phi == 0):
            phi = np.zeros(nfaces, dtype=complex)
        if not isinstance(a, np.ndarray) or a.ndim == 0 or (isinstance(a, np.ndarray) and a.size == 1 and a == 0):
            a = np.zeros((nfaces, 3), dtype=complex)
        if not isinstance(alpha, np.ndarray):
            alpha = np.zeros((nfaces, 3), dtype=complex)
        if not isinstance(De, np.ndarray):
            De = np.zeros(nfaces, dtype=complex)

        # Determine number of polarizations from array with most dimensions
        npol = 1
        if isinstance(a, np.ndarray) and a.ndim == 3:
            npol = a.shape[2]
        elif isinstance(alpha, np.ndarray) and alpha.ndim == 3:
            npol = alpha.shape[2]
        elif isinstance(phi, np.ndarray) and phi.ndim == 2:
            npol = phi.shape[1]
        elif isinstance(De, np.ndarray) and De.ndim == 2:
            npol = De.shape[1]

        # Squeeze arrays if npol == 1 (single polarization case)
        if npol == 1:
            if isinstance(a, np.ndarray) and a.ndim == 3:
                a = a[:, :, 0]
            if isinstance(alpha, np.ndarray) and alpha.ndim == 3:
                alpha = alpha[:, :, 0]
            if isinstance(phi, np.ndarray) and phi.ndim == 2:
                phi = phi[:, 0]
            if isinstance(De, np.ndarray) and De.ndim == 2:
                De = De[:, 0]

        # Allocate output arrays
        if npol == 1:
            sig1_all = np.zeros(nfaces, dtype=complex)
            sig2_all = np.zeros(nfaces, dtype=complex)
            h1_all = np.zeros((nfaces, 3), dtype=complex)
            h2_all = np.zeros((nfaces, 3), dtype=complex)

            # Modify alpha and De [Eqs. before (19)]
            # MATLAB: L1_phi = matmul(L1, phi)
            if np.isscalar(L1):
                L1_phi = L1 * phi
                L1_a = L1 * a
            else:
                L1_phi = L1 @ phi
                L1_a = L1 @ a

            # alpha = alpha - matmul(Sigma1, a) + 1i*k*outer(nvec, L1*phi)
            alpha_mod = alpha - (Sigma1 @ a) + 1j * k * (nvec * L1_phi[:, np.newaxis])
            # De = De - matmul(Sigma1, matmul(L1, phi)) + 1i*k*inner(nvec, L1*a)
            if np.isscalar(L1):
                De_mod = De - Sigma1 @ (L1 * phi) + 1j * k * np.sum(nvec * L1_a, axis=1)
            else:
                De_mod = De - Sigma1 @ L1 @ phi + 1j * k * np.sum(nvec * L1_a, axis=1)

            # Eq. (19): surface charge
            L_diff = L1 - L2
            if np.isscalar(L_diff):
                inner_term = np.sum(nvec * (L_diff * (Deltai @ alpha_mod)), axis=1)
            else:
                inner_term = np.sum(nvec * (L_diff @ (Deltai @ alpha_mod)), axis=1)

            sig2 = Sigmai @ (De_mod + 1j * k * inner_term)

            # Eq. (20): surface current
            if np.isscalar(L_diff):
                outer_term = nvec * (L_diff * sig2)[:, np.newaxis]
            else:
                outer_term = nvec * (L_diff @ sig2)[:, np.newaxis]
            h2 = Deltai @ (1j * k * outer_term + alpha_mod)

            # Surface charges and currents [from Eqs. (10-11)]
            sig1_all = G1i @ (sig2 + phi)
            h1_all = G1i @ (h2 + a)
            sig2_all = G2i @ sig2
            h2_all = G2i @ h2

        else:
            # Multiple polarizations
            sig1_all = np.zeros((nfaces, npol), dtype=complex)
            sig2_all = np.zeros((nfaces, npol), dtype=complex)
            h1_all = np.zeros((nfaces, 3, npol), dtype=complex)
            h2_all = np.zeros((nfaces, 3, npol), dtype=complex)

            for ipol in range(npol):
                phi_i = phi[:, ipol] if phi.ndim > 1 else phi
                a_i = a[:, :, ipol] if a.ndim > 2 else a
                alpha_i = alpha[:, :, ipol] if alpha.ndim > 2 else alpha
                De_i = De[:, ipol] if De.ndim > 1 else De

                # Same computation as single polarization
                if np.isscalar(L1):
                    L1_phi = L1 * phi_i
                    L1_a = L1 * a_i
                else:
                    L1_phi = L1 @ phi_i
                    L1_a = L1 @ a_i

                alpha_mod = alpha_i - (Sigma1 @ a_i) + 1j * k * (nvec * L1_phi[:, np.newaxis])
                if np.isscalar(L1):
                    De_mod = De_i - Sigma1 @ (L1 * phi_i) + 1j * k * np.sum(nvec * L1_a, axis=1)
                else:
                    De_mod = De_i - Sigma1 @ L1 @ phi_i + 1j * k * np.sum(nvec * L1_a, axis=1)

                L_diff = L1 - L2
                if np.isscalar(L_diff):
                    inner_term = np.sum(nvec * (L_diff * (Deltai @ alpha_mod)), axis=1)
                else:
                    inner_term = np.sum(nvec * (L_diff @ (Deltai @ alpha_mod)), axis=1)

                sig2 = Sigmai @ (De_mod + 1j * k * inner_term)

                if np.isscalar(L_diff):
                    outer_term = nvec * (L_diff * sig2)[:, np.newaxis]
                else:
                    outer_term = nvec * (L_diff @ sig2)[:, np.newaxis]
                h2 = Deltai @ (1j * k * outer_term + alpha_mod)

                sig1_all[:, ipol] = G1i @ (sig2 + phi_i)
                h1_all[:, :, ipol] = G1i @ (h2 + a_i)
                sig2_all[:, ipol] = G2i @ sig2
                h2_all[:, :, ipol] = G2i @ h2

        return {
            'sig1': sig1_all,
            'sig2': sig2_all,
            'h1': h1_all,
            'h2': h2_all,
            'enei': enei,
            'p': self.p
        }

    def __call__(self, enei):
        """
        Initialize solver at specific energy (alternative syntax).

        Parameters
        ----------
        enei : float
            Photon energy or wavelength

        Returns
        -------
        self : BEMRet
            Returns self for chaining
        """
        return self.init(enei)

    def __repr__(self):
        status = f"λ={self.enei:.1f}nm" if self.enei is not None else "not initialized"
        return f"BEMRet(p: {self.p.nfaces} faces, {status})"

    def __str__(self):
        status = f"Initialized at λ={self.enei:.2f} nm" if self.enei is not None else "Not initialized"
        mat_info = f"  Sigmai matrix: {self.Sigmai.shape}" if self.Sigmai is not None else "  Sigmai matrix: Not computed"

        return (
            f"BEM Solver (Retarded/Full Maxwell):\n"
            f"  Particle: {self.p.nfaces} faces\n"
            f"  Status: {status}\n"
            f"{mat_info}\n"
            f"  Wavenumber k: {self.k:.6f}" if self.k is not None else "  Wavenumber k: Not computed"
        )
