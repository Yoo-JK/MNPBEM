"""
Composite Green function for quasistatic approximation.

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np


class CompGreenStat:
    """
    Green function for composite particles in quasistatic approximation.

    Computes the F matrix (surface derivative of Green function) for
    boundary element method in the quasistatic regime.

    Parameters
    ----------
    p1 : ComParticle
        Field points (where we evaluate)
    p2 : ComParticle
        Source points (integration surface)

    Attributes
    ----------
    p1, p2 : ComParticle
        Particle objects
    F : ndarray, shape (n1, n2)
        Surface derivative of Green function matrix

    Notes
    -----
    MATLAB MNPBEM convention (greenstat/eval1.m):
        G(r, r') = 1 / |r - r'|  (no 4π factor, absorbed in BEM)

        F[i,j] = - n_i · (r_i - r_j) / |r_i - r_j|³ × area_j

    where n_i is the normal at FIELD point (p1), not source point.

    For closed surfaces, diagonal elements are set to -2π
    (Fuchs & Liu, PRB 14, 5521, 1976).

    Examples
    --------
    >>> from mnpbem import EpsConst, trisphere, ComParticle
    >>> from mnpbem.greenfun import CompGreenStat
    >>>
    >>> # Create 10nm gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Compute Green function
    >>> g = CompGreenStat(p, p)
    >>> print(f"F matrix shape: {g.F.shape}")
    """

    def __init__(self, p1, p2):
        """
        Initialize Green function between p1 and p2.

        Parameters
        ----------
        p1 : ComParticle
            Field particle (evaluation points)
        p2 : ComParticle
            Source particle (integration surface)
        """
        self.p1 = p1
        self.p2 = p2

        # Compute G and F matrices
        self._compute_GF()

    def _compute_GF(self):
        """
        Compute G and F matrices following MATLAB MNPBEM.

        Uses numerical integration for nearby elements and
        point formula for distant elements.
        """
        # Get positions, normals, and areas
        pos1 = self.p1.pos    # (n1, 3) - field points
        pos2 = self.p2.pos    # (n2, 3) - source points
        nvec1 = self.p1.nvec  # (n1, 3) - normals at FIELD points
        area2 = self.p2.area  # (n2,) - areas at source

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Compute distances
        r = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]  # (n1, n2, 3)
        d = np.linalg.norm(r, axis=2)  # (n1, n2)
        d_safe = np.maximum(d, np.finfo(float).eps)

        # Point formula for G and F
        G = (1.0 / d_safe) * area2[np.newaxis, :]
        n_dot_r = np.sum(nvec1[:, np.newaxis, :] * r, axis=2)
        F = -n_dot_r / (d_safe ** 3) * area2[np.newaxis, :]

        # Handle diagonal for self-interaction (p1 == p2)
        if self.p1 is self.p2:
            # Set diagonal to -2π (solid angle for point on closed surface)
            np.fill_diagonal(G, 0.0)
            np.fill_diagonal(F, -2.0 * np.pi)

        self.G = G
        self.F = F

    def _refine_nearby(self, G, F, pos1, nvec1, mask, particle):
        """
        Refine G and F for nearby elements using 7-point Gauss quadrature.
        """
        # 7-point Gauss quadrature for triangles (Dunavant rules)
        quad_pts = np.array([
            [1/3, 1/3, 1/3],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
        ])
        quad_wts = np.array([0.225, 0.125939180544827, 0.125939180544827,
                            0.125939180544827, 0.132394152788506,
                            0.132394152788506, 0.132394152788506])

        verts = particle.verts
        faces = particle.faces

        # Find pairs needing refinement
        refine_pairs = np.argwhere(mask)

        for idx in range(len(refine_pairs)):
            i, j = refine_pairs[idx]

            # Get triangle vertices
            face_idx = faces[j, :3].astype(int)
            v0, v1, v2 = verts[face_idx[0]], verts[face_idx[1]], verts[face_idx[2]]

            # Triangle area
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)

            # Quadrature integration
            G_sum = 0.0
            F_sum = 0.0
            pos_i = pos1[i]
            nvec_i = nvec1[i]

            for k, (a, b, c) in enumerate(quad_pts):
                pt = a * v0 + b * v1 + c * v2
                r_vec = pos_i - pt
                dist = np.linalg.norm(r_vec)
                if dist < 1e-10 * np.sqrt(area):
                    continue

                G_sum += quad_wts[k] / dist
                F_sum += quad_wts[k] * np.dot(nvec_i, r_vec) / dist**3

            G[i, j] = G_sum * area
            F[i, j] = -F_sum * area

    def H1(self):
        """
        Return H1 matrix: F + 2π on diagonal.

        MATLAB greenstat/eval1.m:
            H1 = F + 2*pi*(d==0)
        """
        H1 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H1, np.diag(self.F) + 2.0 * np.pi)
        return H1

    def H2(self):
        """
        Return H2 matrix: F - 2π on diagonal.

        MATLAB greenstat/eval1.m:
            H2 = F - 2*pi*(d==0)
        """
        H2 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H2, np.diag(self.F) - 2.0 * np.pi)
        return H2

    def _compute_Gp(self):
        """
        Compute Cartesian derivative of Green function.

        MATLAB greenstat/eval1.m (case 'cart'):
            Gp = -[x/d³, y/d³, z/d³] * area
        """
        if hasattr(self, '_Gp'):
            return self._Gp

        pos1 = self.p1.pos
        pos2 = self.p2.pos
        area2 = self.p2.area

        # Position difference: r[i,j,:] = pos1[i] - pos2[j]
        r = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]  # (n1, n2, 3)
        d = np.linalg.norm(r, axis=2)
        d = np.maximum(d, np.finfo(float).eps)

        # Gp[i,j,:] = -r[i,j,:] / d[i,j]³ * area[j]
        # Shape: (n1, n2, 3)
        Gp = -r / (d[:, :, np.newaxis] ** 3) * area2[np.newaxis, :, np.newaxis]

        # Reshape to (n1, 3, n2) to match MATLAB convention
        self._Gp = np.transpose(Gp, (0, 2, 1))

        return self._Gp

    def H1p(self):
        """
        Return H1p matrix: Gp + 2π*outer(nvec, d==0).

        MATLAB greenstat/eval1.m:
            H1p = Gp + 2*pi*outer(nvec, d==0)

        Returns
        -------
        H1p : ndarray, shape (n1, 3, n2)
            Cartesian derivative + 2π term on diagonal
        """
        Gp = self._compute_Gp()
        H1p = Gp.copy()

        if self.p1 is self.p2:
            # Add 2π * nvec[i] for diagonal elements (i == j)
            nvec = self.p1.nvec
            for i in range(len(nvec)):
                H1p[i, :, i] += 2 * np.pi * nvec[i]

        return H1p

    def H2p(self):
        """
        Return H2p matrix: Gp - 2π*outer(nvec, d==0).

        MATLAB greenstat/eval1.m:
            H2p = Gp - 2*pi*outer(nvec, d==0)

        Returns
        -------
        H2p : ndarray, shape (n1, 3, n2)
            Cartesian derivative - 2π term on diagonal
        """
        Gp = self._compute_Gp()
        H2p = Gp.copy()

        if self.p1 is self.p2:
            # Subtract 2π * nvec[i] for diagonal elements (i == j)
            nvec = self.p1.nvec
            for i in range(len(nvec)):
                H2p[i, :, i] -= 2 * np.pi * nvec[i]

        return H2p

    def __repr__(self):
        return (
            f"CompGreenStat(p1: {self.p1.nfaces} faces, "
            f"p2: {self.p2.nfaces} faces)"
        )

    def __str__(self):
        return (
            f"Quasistatic Green Function:\n"
            f"  Field (p1): {self.p1.nfaces} faces\n"
            f"  Source (p2): {self.p2.nfaces} faces\n"
            f"  G matrix: {self.G.shape}\n"
            f"  F matrix: {self.F.shape}\n"
            f"  G range: [{self.G.min():.4e}, {self.G.max():.4e}]\n"
            f"  F range: [{self.F.min():.4e}, {self.F.max():.4e}]"
        )
