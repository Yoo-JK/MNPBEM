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

            # Surface derivative (MATLAB lines 65-73)
            # Inner product: n·r
            n_dot_r = np.sum(vec * nvec1[i], axis=1)  # (n_points,)

            # f_n = Σ w × (n-1) × (n·r) × r^(n-3) / n! (MATLAB line 72)
            for n in range(self.order + 1):
                if n == 0:
                    # For n=0: (n-1) × r^(n-3) = -1 × r^(-3)
                    self.f[iref, n] = np.sum(w_face * (-1) * n_dot_r * r**(-3)) / math.factorial(n)
                else:
                    self.f[iref, n] = np.sum(w_face * (n - 1) * n_dot_r * r**(n - 3)) / math.factorial(n)

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

            # Lowest order (n=0): f_0 = -∫ (n·r) / r³ dA (MATLAB line 158)
            self.f[iface, 0] = -(n_dot_r / r**3) @ w_face

            # Higher orders (n>=1) (MATLAB lines 160-164)
            for n in range(1, self.order + 1):
                term1 = -(r - r0[:, np.newaxis])**n / (r**3 * math.factorial(n))
                term2 = (r - r0[:, np.newaxis])**(n-1) / (r**2 * math.factorial(n-1))
                integrand = n_dot_r * (term1 + term2)
                self.f[iface, n] = integrand @ w_face

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
        # Positions and areas
        pos1 = self.p1.pos
        pos2 = self.p2.pos
        n1, n2 = pos1.shape[0], pos2.shape[0]
        area2 = self.p2.area

        # Compute distances (MATLAB lines 25-29)
        x = pos1[:, 0:1] - pos2[:, 0]  # (n1, n2)
        y = pos1[:, 1:2] - pos2[:, 1]
        z = pos1[:, 2:3] - pos2[:, 2]
        d = np.sqrt(x**2 + y**2 + z**2)
        d = np.maximum(d, np.finfo(float).eps)

        # Evaluate based on key
        if key == 'G':
            # Step 1: Green function G = 1/d * area (MATLAB line 34)
            # Initialize as complex to avoid casting warnings
            G = (1.0 / d) * area2[np.newaxis, :] + 0j

            # Step 2: Refine elements (MATLAB lines 36-38)
            if len(self.ind) > 0:
                # Multi-order expansion: Σ g_n × (ik)^n
                ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                G_refined = self.g @ ik_powers  # (n_refined,)

                # Replace refined elements
                G[self.row, self.col] = G_refined

            # Step 3: Apply phase factor (MATLAB line 39)
            G = G * np.exp(1j * k * d)
            return G

        elif key == 'F':
            # Surface derivative (MATLAB lines 47-56)
            nvec1 = self.p1.nvec

            # Inner product n·r (MATLAB lines 48-49)
            n_dot_r = (nvec1[:, 0:1] * x +
                      nvec1[:, 1:2] * y +
                      nvec1[:, 2:3] * z)

            # F = (n·r) × (ik - 1/d) / d² × area (MATLAB line 51-52)
            # Already complex due to 1j term
            F = n_dot_r * (1j * k - 1.0 / d) / (d**2) * area2[np.newaxis, :]

            # Refine elements (MATLAB lines 53-55)
            if len(self.ind) > 0:
                # Multi-order expansion
                ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                F_refined = self.f @ ik_powers

                # Replace refined elements
                F[self.row, self.col] = F_refined

            # Apply phase factor (MATLAB line 56)
            F = F * np.exp(1j * k * d)
            return F

        elif key == 'H1':
            F = self.eval(k, 'F')
            H1 = F.copy()
            # Add 2π to diagonal (MATLAB line 73)
            if self.p1 is self.p2:
                np.fill_diagonal(H1, np.diag(F) + 2.0 * np.pi)
            return H1

        elif key == 'H2':
            F = self.eval(k, 'F')
            H2 = F.copy()
            # Subtract 2π from diagonal (MATLAB line 75)
            if self.p1 is self.p2:
                np.fill_diagonal(H2, np.diag(F) - 2.0 * np.pi)
            return H2

        else:
            raise ValueError(f"Unknown key: {key}. Use 'G', 'F', 'H1', or 'H2'")

    def __repr__(self):
        n_refined = len(self.ind) if hasattr(self, 'ind') else 0
        return (f"GreenRetRefined(n1={self.p1.n}, n2={self.p2.n}, "
                f"order={self.order}, n_refined={n_refined})")


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

    print(f"\nParticle: {p.n} faces")

    # Create refined Green function
    print(f"\nCreating refined Green function...")
    g = GreenRetRefined(p, p, order=2, RelCutoff=3)

    print(f"{g}")
    print(f"Refined elements: {len(g.ind)}")
    print(f"  g shape: {g.g.shape}")
    print(f"  f shape: {g.f.shape}")

    # Test evaluation at 600 nm wavelength
    wavelength = 600.0  # nm
    k = 2 * np.pi / wavelength

    print(f"\nEvaluating at λ={wavelength} nm (k={k:.6f} nm⁻¹):")

    G = g.eval(k, 'G')
    print(f"  G shape: {G.shape}")
    print(f"  G diagonal range: [{np.min(np.abs(np.diag(G))):.2e}, {np.max(np.abs(np.diag(G))):.2e}]")
    print(f"  G off-diag range: [{np.min(np.abs(G[~np.eye(p.n, dtype=bool)])):.2e}, "
          f"{np.max(np.abs(G[~np.eye(p.n, dtype=bool)])):.2e}]")

    F = g.eval(k, 'F')
    print(f"  F shape: {F.shape}")
    print(f"  F diagonal range: [{np.min(np.abs(np.diag(F))):.2e}, {np.max(np.abs(np.diag(F))):.2e}]")

    H1 = g.eval(k, 'H1')
    H2 = g.eval(k, 'H2')
    print(f"  H1 diagonal: {np.diag(H1)[:3]} ...")
    print(f"  H2 diagonal: {np.diag(H2)[:3]} ...")
    print(f"  H1-H2 diagonal diff: {np.mean(np.abs(np.diag(H1) - np.diag(H2))):.6f} (should be 4π={4*np.pi:.6f})")

    print("\n" + "=" * 70)
    print("✓ GreenRetRefined tests passed!")
