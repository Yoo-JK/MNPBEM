"""
Refined retarded Green function with polar integration.

This module implements proper Green function refinement using:
1. Multi-order Taylor expansion: G = Σ g_n × (ik)^n / factorial(n)
2. Polar integration for diagonal elements
3. Boundary element integration for near-field off-diagonal elements

MATLAB reference: Greenfun/@greenret/
"""

import numpy as np
from typing import Optional, Tuple
from scipy.sparse import csr_matrix, find as sparse_find
import math

try:
    from .refine_utils import refinematrix
    from ..geometry.particle import Particle
except ImportError:
    from refine_utils import refinematrix
    # For standalone testing, we'll handle the import later
    Particle = None


class GreenRetRefined(object):
    """
    Retarded Green function with proper polar integration refinement.

    Attributes
    ----------
    p1, p2 : Particle
        Source and target particles
    deriv : str
        'norm' for normal derivative, 'cart' for Cartesian derivative
    order : int
        Order of multi-order expansion (default: 2)
    ind : ndarray
        Linear indices of refined elements
    row, col : ndarray
        Row and column indices of refined elements
    g : ndarray, shape (n_refined, order+1)
        Refined Green function expansion coefficients
    f : ndarray, shape (n_refined, order+1)
        Refined derivative expansion coefficients
    """

    def __init__(self, p1, p2, deriv='norm', order=2, **options):
        """
        Initialize refined retarded Green function.

        Parameters
        ----------
        p1, p2 : Particle
            Source and target particles
        deriv : str, optional
            'norm' (surface derivative) or 'cart' (Cartesian derivative)
            Default: 'norm'
        order : int, optional
            Order of Taylor expansion (default: 2)
            Higher order = better accuracy but more computation
        **options : dict
            AbsCutoff : float
                Absolute distance cutoff for refinement (nm)
            RelCutoff : float
                Relative distance cutoff (multiples of element radius)
                Default: 3
        """
        self.p1 = p1
        self.p2 = p2
        self.deriv = deriv
        self.order = order
        self._d_cache = None

        # Initialize refinement
        self._init_refinement(**options)

    def _init_refinement(self, **options):
        """
        Compute refined Green function elements.

        MATLAB reference: Greenfun/@greenret/private/init.m
        """
        # Refinement matrix: 0=far, 1=near, 2=diagonal
        AbsCutoff = options.get('AbsCutoff', 0)
        RelCutoff = options.get('RelCutoff', 3)

        ir = refinematrix(self.p1, self.p2, AbsCutoff=AbsCutoff, RelCutoff=RelCutoff)

        # Linear indices of refined elements (MATLAB line 24)
        self.ind = np.array(ir.nonzero()).T  # (n_refined, 2) array of (row, col)
        n_refined = len(self.ind)

        # If no refinement needed, return empty arrays
        if n_refined == 0 or self.order == 0:
            self.g = np.array([])
            self.f = np.array([])
            self.row = np.array([], dtype=int)
            self.col = np.array([], dtype=int)
            return

        # Store row and column indices separately
        self.row = self.ind[:, 0]
        self.col = self.ind[:, 1]

        # Allocate arrays for multi-order expansion (MATLAB lines 35-40)
        order = self.order
        self.g = np.zeros((n_refined, order + 1), dtype=complex)
        if self.deriv == 'cart':
            self.f = np.zeros((n_refined, 3, order + 1), dtype=complex)
        else:
            self.f = np.zeros((n_refined, order + 1), dtype=complex)

        # Refine diagonal elements (ir == 2)
        # MATLAB lines 42-95
        self._refine_diagonal(ir)

        # Refine off-diagonal elements (ir == 1)
        # MATLAB lines 98-177
        self._refine_offdiagonal(ir)

    def _refine_diagonal(self, ir):
        """
        Refine diagonal elements using polar integration.

        MATLAB reference: Greenfun/@greenret/private/init.m lines 42-95

        Algorithm:
        1. Find diagonal elements where ir == 2
        2. Use quadpol() for polar integration
        3. For each order n: g_n = ∫ r^(n-1) / n! dA
        4. For derivatives: f_n = ∫ (n-1) × (n·r) × r^(n-3) / n! dA
        """
        # Find diagonal elements (MATLAB line 47)
        diag_mask = ir.toarray() == 2
        if not np.any(diag_mask):
            return

        face_idx, face2_idx = np.where(diag_mask)

        # Find corresponding indices in refinement array
        # Create mapping from (row, col) to refinement index
        refine_map = {(r, c): i for i, (r, c) in enumerate(zip(self.row, self.col))}
        iface = np.array([refine_map[(r, c)] for r, c in zip(face_idx, face2_idx)])

        # Polar integration points and weights (MATLAB line 51)
        # Returns: pos (n_total, 3), weight (n_total,), row (n_total,)
        # where row[i] indicates which face point i belongs to
        if Particle is None:
            # Standalone testing - import here
            import sys
            sys.path.insert(0, '/home/user/MNPBEM')
            from mnpbem.geometry.particle import Particle as Part
            pos, weight, row_indices = Part.quadpol(self.p2, face2_idx)
        else:
            pos, weight, row_indices = Particle.quadpol(self.p2, face2_idx)

        # Get positions of source faces
        pos1 = self.p1.pos[face_idx]  # (n_diag, 3)
        nvec1 = self.p1.nvec[face_idx]  # (n_diag, 3)

        # Process each diagonal face
        for i, (face, face2, iref) in enumerate(zip(face_idx, face2_idx, iface)):
            # Get integration points for this face
            # row_indices contains the index into face2_idx, not face2_idx itself
            face_points = (row_indices == i)
            pos_face = pos[face_points]  # (n_points, 3)
            w_face = weight[face_points]  # (n_points,)

            # Vector from integration points to face centroid (MATLAB line 56)
            vec = pos1[i] - pos_face  # (n_points, 3)

            # Distance (MATLAB line 58)
            r = np.sqrt(np.sum(vec**2, axis=1))  # (n_points,)

            # Green function: g_n = Σ w × r^(n-1) / n! (MATLAB lines 61-63)
            for n in range(self.order + 1):
                self.g[iref, n] = np.sum(w_face * r**(n - 1)) / math.factorial(n)

            # Surface derivative (MATLAB lines 65-94)
            n_dot_r = np.sum(vec * nvec1[i], axis=1)

            if self.deriv == 'norm':
                for n in range(self.order + 1):
                    self.f[iref, n] = np.sum(w_face * (n - 1) * n_dot_r * r**(n - 3)) / math.factorial(n)
            else:
                # deriv='cart': MATLAB init.m lines 74-93
                rr = np.maximum(r, 1e-4 * np.max(r))
                in1 = np.sum(vec * self.p1.tvec1[face], axis=1)
                in2 = np.sum(vec * self.p1.tvec2[face], axis=1)
                for n in range(self.order + 1):
                    f1 = np.sum(w_face * (n - 1) * n_dot_r * r**(n - 3)) / math.factorial(n)
                    f2 = np.sum(w_face * (n - 1) * in1 * r**n / rr**3) / math.factorial(n)
                    f3 = np.sum(w_face * (n - 1) * in2 * r**n / rr**3) / math.factorial(n)
                    self.f[iref, :, n] = (nvec1[i] * f1 +
                                          self.p1.tvec1[face] * f2 +
                                          self.p1.tvec2[face] * f3)

    def _refine_offdiagonal(self, ir):
        """
        Refine off-diagonal near-field elements using boundary element integration.

        MATLAB reference: Greenfun/@greenret/private/init.m lines 98-177

        Algorithm:
        1. Find faces that have near-field neighbors (ir == 1)
        2. Use quad() for boundary element integration
        3. For each order n: g_n = ∫ (r-r0)^n / (r × n!) dA
           where r0 is distance to face centroid
        4. For derivatives: Similar but with directional factors
        """
        # Faces to be refined (columns with any ir==1) (MATLAB line 100)
        ir_array = ir.toarray()
        reface = np.where(np.any(ir_array == 1, axis=0))[0]

        if len(reface) == 0:
            return

        # Boundary element integration (MATLAB line 102)
        # Returns: pos (n_total, 3), w_sparse (n_faces, n_points), iface (n_points,)
        # Note: self.p2.quad is an attribute, so we need to call the method via the class
        if Particle is None:
            # Standalone testing - import here
            import sys
            sys.path.insert(0, '/home/user/MNPBEM')
            from mnpbem.geometry.particle import Particle as Part
            pos_all, w_sparse, row_indices = Part.quad(self.p2, reface)
        else:
            pos_all, w_sparse, row_indices = Particle.quad(self.p2, reface)

        # Get source positions
        pos1 = self.p1.pos
        nvec1 = self.p1.nvec

        # Create mapping from (row, col) to refinement index
        refine_map = {(r, c): i for i, (r, c) in enumerate(zip(self.row, self.col))}

        # Process each face to be refined (MATLAB line 116)
        for face_idx, face in enumerate(reface):
            # Find neighbor faces that need refinement for this face
            nb = np.where(ir_array[:, face] == 1)[0]

            if len(nb) == 0:
                continue

            # Indices in refinement array
            iface = np.array([refine_map[(n, face)] for n in nb])

            # Get integration points for this face
            face_mask = (row_indices == face_idx)
            pos_face = pos_all[face_mask]  # (n_points, 3)

            # Extract weights from sparse matrix for this face
            # w_sparse is (n_faces, n_points)
            w_row = w_sparse[face_idx, :].toarray().ravel()
            w_face = w_row[w_row != 0]  # Get non-zero weights

            # Difference vectors (MATLAB lines 133-137)
            # Broadcasting: (n_nb, 1, 3) - (1, n_points, 3) = (n_nb, n_points, 3)
            x = pos1[nb, 0:1] - pos_face[:, 0]  # (n_nb, n_points)
            y = pos1[nb, 1:2] - pos_face[:, 1]
            z = pos1[nb, 2:3] - pos_face[:, 2]

            # Distance from integration points to centroids (MATLAB line 137)
            r = np.sqrt(x**2 + y**2 + z**2)  # (n_nb, n_points)

            # Distance from face centroids (MATLAB lines 140-142)
            vec0 = self.p2.pos[face] - pos1[nb]  # (n_nb, 3)
            r0 = np.sqrt(np.sum(vec0**2, axis=1))  # (n_nb,)

            # Green function expansion (MATLAB lines 145-148)
            # g_n = ∫ (r-r0)^n / (r × n!) dA
            for n in range(self.order + 1):
                integrand = (r - r0[:, np.newaxis])**n / r / math.factorial(n)
                self.g[iface, n] = integrand @ w_face  # Matrix-vector product

            # Surface derivative expansion (MATLAB lines 151-164)
            # Inner product: n·(x,y,z)
            n_dot_r = (nvec1[nb, 0:1] * x +
                      nvec1[nb, 1:2] * y +
                      nvec1[nb, 2:3] * z)  # (n_nb, n_points)

            if self.deriv == 'norm':
                self.f[iface, 0] = -(n_dot_r / r**3) @ w_face
                for n in range(1, self.order + 1):
                    term1 = -(r - r0[:, np.newaxis])**n / (r**3 * math.factorial(n))
                    term2 = (r - r0[:, np.newaxis])**(n-1) / (r**2 * math.factorial(n-1))
                    self.f[iface, n] = (n_dot_r * (term1 + term2)) @ w_face
            else:
                # deriv='cart': MATLAB init.m lines 165-175
                f_scalar = -1.0 / r**3
                self.f[iface, 0, 0] = (x * f_scalar) @ w_face
                self.f[iface, 1, 0] = (y * f_scalar) @ w_face
                self.f[iface, 2, 0] = (z * f_scalar) @ w_face
                for n in range(1, self.order + 1):
                    term1 = -(r - r0[:, np.newaxis])**n / (r**3 * math.factorial(n))
                    term2 = (r - r0[:, np.newaxis])**(n-1) / (r**2 * math.factorial(n-1))
                    f_scalar = term1 + term2
                    self.f[iface, 0, n] = (x * f_scalar) @ w_face
                    self.f[iface, 1, n] = (y * f_scalar) @ w_face
                    self.f[iface, 2, n] = (z * f_scalar) @ w_face

    def _ensure_cache(self):
        """Build and cache wavelength-independent distance quantities."""
        if self._d_cache is not None:
            return
        from ._numba_ret_kernels import green_ret_distances, numba_enabled

        pos1 = self.p1.pos
        pos2 = self.p2.pos
        area2 = self.p2.area
        nvec1 = self.p1.nvec
        same = self.p1 is self.p2

        if numba_enabled():
            d, inv_d, n_dot_r, x, y, z = green_ret_distances(
                pos1, pos2, nvec1, area2, same = same, want_r = True
            )
            inv_d2 = inv_d * inv_d
        else:
            x = pos1[:, 0:1] - pos2[:, 0]  # (n1, n2)
            y = pos1[:, 1:2] - pos2[:, 1]
            z = pos1[:, 2:3] - pos2[:, 2]
            d = np.sqrt(x**2 + y**2 + z**2)
            d = np.maximum(d, np.finfo(float).eps)
            inv_d = 1.0 / d
            inv_d2 = inv_d * inv_d
            n_dot_r = (nvec1[:, 0:1] * x + nvec1[:, 1:2] * y + nvec1[:, 2:3] * z)

        self._d_cache = {
            'x': x, 'y': y, 'z': z, 'd': d,
            'inv_d': inv_d, 'inv_d2': inv_d2,
            'area2': area2, 'n_dot_r': n_dot_r,
        }

    def eval(self, k, key):
        """
        Evaluate Green function with proper refinement.

        MATLAB reference: Greenfun/@greenret/private/eval1.m

        Parameters
        ----------
        k : float
            Wavenumber (2π/λ where λ is wavelength in medium)
        key : str
            'G' - Green function
            'F' - Surface derivative
            'H1' - F + 2π (inside)
            'H2' - F - 2π (outside)

        Returns
        -------
        g : ndarray, shape (n1, n2)
            Green function matrix
        """
        self._ensure_cache()
        c = self._d_cache
        d = c['d']
        inv_d = c['inv_d']
        inv_d2 = c['inv_d2']
        area2 = c['area2']

        # Numba fast path for distinct-particle (meshfield / observer) case.
        # Refinement overrides act on the *pre-phase* matrix, so the kernel
        # produces the pre-phase result, the caller overwrites refined
        # entries via numpy fancy indexing, and a final numba phase apply
        # multiplies in exp(i k d).
        use_numba = self.p1 is not self.p2
        nb = None
        if use_numba:
            from ..simulation import _meshfield_numba as _nb
            if _nb.numba_enabled():
                nb = _nb

        # Evaluate based on key
        if key == 'G':
            if nb is not None:
                G = nb.ret_G_pre(inv_d, area2)
                if len(self.ind) > 0:
                    ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                    G_refined = self.g @ ik_powers
                    G[self.row, self.col] = G_refined
                phase = nb.ret_phase(d, k)
                nb.apply_phase_2d(G, phase)
                return G

            G = inv_d * area2[np.newaxis, :] + 0j

            if len(self.ind) > 0:
                ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                G_refined = self.g @ ik_powers
                G[self.row, self.col] = G_refined

            G = G * np.exp(1j * k * d)
            return G

        elif key == 'F':
            if self.deriv == 'cart':
                x, y, z = c['x'], c['y'], c['z']
                nvec = self.p1.nvec
                if nb is not None:
                    F = nb.ret_F_cart_pre(inv_d, inv_d2, x, y, z, nvec, area2, k)
                    if len(self.ind) > 0:
                        ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                        nvec_ref = nvec[self.row]
                        F_refined = np.einsum('ij,ijk,k->i', nvec_ref, self.f, ik_powers)
                        F[self.row, self.col] = F_refined
                    phase = nb.ret_phase(d, k)
                    nb.apply_phase_2d(F, phase)
                    return F

                # MATLAB eval1.m lines 110-124: F via Gp inner product
                f_aux = (1j * k - inv_d) * inv_d2
                F = (nvec[:, 0:1] * (f_aux * x) +
                     nvec[:, 1:2] * (f_aux * y) +
                     nvec[:, 2:3] * (f_aux * z)) * area2[np.newaxis, :]

                if len(self.ind) > 0:
                    ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                    # MATLAB line 120: F(ind) = inner(nvec(i,:), f) * ik_powers
                    nvec_ref = nvec[self.row]
                    F_refined = np.einsum('ij,ijk,k->i', nvec_ref, self.f, ik_powers)
                    F[self.row, self.col] = F_refined

                F = F * np.exp(1j * k * d)
                return F
            else:
                n_dot_r = c['n_dot_r']
                if nb is not None:
                    F = nb.ret_F_norm_pre(inv_d, inv_d2, n_dot_r, area2, k)
                    if len(self.ind) > 0:
                        ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                        F_refined = self.f @ ik_powers
                        F[self.row, self.col] = F_refined
                    phase = nb.ret_phase(d, k)
                    nb.apply_phase_2d(F, phase)
                    return F

                F = n_dot_r * (1j * k - inv_d) * inv_d2 * area2[np.newaxis, :]

                if len(self.ind) > 0:
                    ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                    F_refined = self.f @ ik_powers
                    F[self.row, self.col] = F_refined

                F = F * np.exp(1j * k * d)
                return F

        elif key == 'H1':
            H1 = self.eval(k, 'F')
            if self.p1 is self.p2:
                np.fill_diagonal(H1, np.diag(H1) + 2.0 * np.pi)
            return H1

        elif key == 'H2':
            H2 = self.eval(k, 'F')
            if self.p1 is self.p2:
                np.fill_diagonal(H2, np.diag(H2) - 2.0 * np.pi)
            return H2

        elif key == 'Gp':
            x, y, z = c['x'], c['y'], c['z']
            if nb is not None:
                Gp = nb.ret_Gp_pre(inv_d, inv_d2, x, y, z, area2, k)
                if len(self.ind) > 0 and self.deriv == 'cart':
                    ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                    Gp_refined = np.einsum('ijk,k->ij', self.f, ik_powers)
                    Gp[self.row, 0, self.col] = Gp_refined[:, 0]
                    Gp[self.row, 1, self.col] = Gp_refined[:, 1]
                    Gp[self.row, 2, self.col] = Gp_refined[:, 2]
                phase = nb.ret_phase(d, k)
                nb.apply_phase_3d_axis02(Gp, phase)
                return Gp

            phase = np.exp(1j * k * d)
            f_aux = (1j * k - inv_d) * inv_d2
            # Gp as (n1, n2, 3) — then transpose to (n1, 3, n2)
            Gp_x = f_aux * x * area2[np.newaxis, :]
            Gp_y = f_aux * y * area2[np.newaxis, :]
            Gp_z = f_aux * z * area2[np.newaxis, :]

            # Apply refinement (MATLAB eval1.m lines 96-98)
            if len(self.ind) > 0 and self.deriv == 'cart':
                ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                # f is (n_ref, 3, order+1), Gp_refined = f @ ik_powers → (n_ref, 3)
                Gp_refined = np.einsum('ijk,k->ij', self.f, ik_powers)
                Gp_x[self.row, self.col] = Gp_refined[:, 0]
                Gp_y[self.row, self.col] = Gp_refined[:, 1]
                Gp_z[self.row, self.col] = Gp_refined[:, 2]

            Gp_x *= phase; Gp_y *= phase; Gp_z *= phase
            Gp = np.stack([Gp_x, Gp_y, Gp_z], axis=1)  # (n1, 3, n2)
            return Gp

        elif key == 'H1p':
            Gp = self.eval(k, 'Gp')
            if self.p1 is self.p2:
                H1p = Gp.copy()
                nvec = self.p1.nvec
                idx = np.arange(len(nvec))
                H1p[idx, :, idx] += 2.0 * np.pi * nvec.T
                return H1p
            return Gp

        elif key == 'H2p':
            Gp = self.eval(k, 'Gp')
            if self.p1 is self.p2:
                H2p = Gp.copy()
                nvec = self.p1.nvec
                idx = np.arange(len(nvec))
                H2p[idx, :, idx] -= 2.0 * np.pi * nvec.T
                return H2p
            return Gp

        else:
            raise ValueError("Unknown key: {}".format(key))

    def __repr__(self):
        n_refined = len(self.ind) if hasattr(self, 'ind') else 0
        return ("GreenRetRefined(n1 = {}, n2 = {}, "
                "order = {}, n_refined = {})".format(self.p1.n, self.p2.n, self.order, n_refined))


# Test code
if __name__ == "__main__":
    print("Testing GreenRetRefined:")
    print("=" * 70)

    import sys
    sys.path.insert(0, '/home/user/MNPBEM')
    from mnpbem.geometry.particle import Particle

    # Create simple test particle
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]) * 10.0

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5],
        [2, 7, 3], [2, 6, 7],
        [0, 7, 4], [0, 3, 7],
        [1, 6, 2], [1, 5, 6],
    ])

    p = Particle(verts, faces)

    print("\nParticle: {} faces".format(p.n))

    # Create refined Green function
    print("\nCreating refined Green function...")
    g = GreenRetRefined(p, p, order=2, RelCutoff=3)

    print("{}".format(g))
    print("Refined elements: {}".format(len(g.ind)))
    print("  g shape: {}".format(g.g.shape))
    print("  f shape: {}".format(g.f.shape))

    # Test evaluation at 600 nm wavelength
    wavelength = 600.0  # nm
    k = 2 * np.pi / wavelength

    print("\nEvaluating at lambda={} nm (k={:.6f} nm^-1):".format(wavelength, k))

    G = g.eval(k, 'G')
    print("  G shape: {}".format(G.shape))
    print("  G diagonal range: [{:.2e}, {:.2e}]".format(np.min(np.abs(np.diag(G))), np.max(np.abs(np.diag(G)))))
    print("  G off-diag range: [{:.2e}, "
          "{:.2e}]".format(np.min(np.abs(G[~np.eye(p.n, dtype=bool)])),
                           np.max(np.abs(G[~np.eye(p.n, dtype=bool)]))))

    F = g.eval(k, 'F')
    print("  F shape: {}".format(F.shape))
    print("  F diagonal range: [{:.2e}, {:.2e}]".format(np.min(np.abs(np.diag(F))), np.max(np.abs(np.diag(F)))))

    H1 = g.eval(k, 'H1')
    H2 = g.eval(k, 'H2')
    print("  H1 diagonal: {} ...".format(np.diag(H1)[:3]))
    print("  H2 diagonal: {} ...".format(np.diag(H2)[:3]))
    print("  H1-H2 diagonal diff: {:.6f} (should be 4pi={:.6f})".format(np.mean(np.abs(np.diag(H1) - np.diag(H2))), 4 * np.pi))

    print("\n" + "=" * 70)
    print("✓ GreenRetRefined tests passed!")
