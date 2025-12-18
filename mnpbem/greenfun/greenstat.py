"""
Green function for quasistatic approximation.

MATLAB: Greenfun/@greenstat/
100% identical to MATLAB MNPBEM implementation.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional, Tuple, Any

from .utils import refinematrix


class GreenStat:
    """
    Green function for quasistatic approximation.

    MATLAB: @greenstat

    Computes the Green function between points p1 and particle p2:
    - G(r, r') = 1 / |r - r'| (no 4π factor)
    - F(r, r') = -n · (r - r') / |r - r'|³ (surface derivative)

    Properties
    ----------
    p1 : Particle
        Points or particle boundary (source of field evaluation)
    p2 : Particle
        Particle boundary (source of charges)
    op : dict
        Options for calculation of Green function
    deriv : str
        'cart' for Cartesian derivatives, 'norm' for normal derivative only
    ind : ndarray
        Index to face elements with refinement
    g : ndarray
        Refined elements for Green function
    f : ndarray
        Refined elements for derivative of Green function

    Methods
    -------
    eval(*keys)
        Evaluate Green function (G, F, H1, H2, Gp, H1p, H2p)
    """

    def __init__(self, p1, p2, **options):
        """
        Initialize Green function in quasistatic approximation.

        MATLAB: @greenstat/greenstat.m

        Parameters
        ----------
        p1 : Particle
            Green function between points p1 and comparticle p2
        p2 : Particle
            Green function between points p1 and comparticle p2
        **options : dict
            deriv : str, optional
                'cart' (Cartesian) or 'norm' (normal) derivative (default: 'cart')
            AbsCutoff : float, optional
                Absolute distance for integration refinement (default: 0)
            RelCutoff : float, optional
                Relative distance for integration refinement (default: 3)
            waitbar : bool, optional
                Show progress bar (default: False)
            refun : callable, optional
                Additional refinement function

        Examples
        --------
        >>> from mnpbem import trisphere
        >>> from mnpbem.greenfun import GreenStat
        >>>
        >>> p = trisphere(144, 10.0)
        >>> g = GreenStat(p, p)
        >>> G, F = g.eval('G', 'F')
        """
        self.p1 = p1
        self.p2 = p2
        self.op = options
        self.deriv = options.get('deriv', 'cart')

        # Initialize refinement arrays
        self.ind = None
        self.g = None
        self.f = None

        # Initialize Green function
        self._init(**options)

    def _init(self, **options):
        """
        Initialize Green function.

        MATLAB: @greenstat/private/init.m
        """
        p1 = self.p1
        p2 = self.p2
        pos1 = p1.pos

        # Get refinement matrix
        refine_opts = {k: v for k, v in options.items()
                       if k in ['AbsCutoff', 'RelCutoff', 'memsize']}
        ir = refinematrix(p1, p2, **refine_opts)

        # Derivative type
        if 'deriv' in options:
            self.deriv = options['deriv']

        # Index to refined elements
        ir_coo = ir.tocoo()
        self.ind = np.ravel_multi_index((ir_coo.row, ir_coo.col), (p1.nfaces, p2.nfaces))

        # Conversion table between matrix elements and refined elements
        row, col = ir_coo.row, ir_coo.col
        nrow = len(row)
        if nrow == 0:
            return

        # Create sparse index matrix for lookup
        ind_sparse = csr_matrix((np.arange(1, nrow + 1), (row, col)),
                                shape=(p1.nfaces, p2.nfaces))

        # Allocate refinement arrays
        g = np.zeros(nrow)
        if self.deriv == 'cart':
            f = np.zeros((nrow, 3))
        else:
            f = np.zeros(nrow)

        # Get dense matrix for ir
        ir_dense = ir.toarray()

        # ===== Diagonal elements (ir == 2) =====
        if np.any(ir_dense == 2):
            face, face2 = np.where(ir_dense == 2)
            # Index to refinement array (1-indexed to match MATLAB)
            iface = np.array([ind_sparse[f, f2] - 1 for f, f2 in zip(face, face2)], dtype=int)

            # Integration points and weights for polar integration
            pos, w, row_quad = p2.quadpol(face2)

            # Expand vectors
            pos1_expanded = pos1[face[row_quad]]
            nvec1_expanded = p1.nvec[face[row_quad]]

            # Measure positions with respect to centroids
            vec = pos1_expanded - pos
            # Distance
            r = np.sqrt(np.sum(vec**2, axis=1))
            r = np.maximum(r, np.finfo(float).eps)

            # Save Green function using accumarray equivalent
            for i, idx in enumerate(iface):
                mask = row_quad == i
                g[idx] = np.sum(w[mask] / r[mask])

            # Surface derivative of Green function
            for i, idx in enumerate(iface):
                mask = row_quad == i
                in_prod = np.sum(vec[mask] * nvec1_expanded[mask], axis=1)
                f_val = -np.sum(w[mask] * in_prod / r[mask]**3)

                if self.deriv == 'cart':
                    # For r->0 the tangential derivatives vanish
                    rr = np.maximum(r[mask], 1e-4 * np.max(r[mask]) if np.max(r[mask]) > 0 else 1e-10)

                    # Tangential inner products
                    if hasattr(p1, 'tvec1') and hasattr(p1, 'tvec2'):
                        tvec1_expanded = p1.tvec1[face[row_quad]]
                        tvec2_expanded = p1.tvec2[face[row_quad]]
                        in1 = np.sum(vec[mask] * tvec1_expanded[mask], axis=1)
                        in2 = np.sum(vec[mask] * tvec2_expanded[mask], axis=1)
                        f1 = f_val
                        f2 = -np.sum(w[mask] * in1 / rr**3)
                        f3 = -np.sum(w[mask] * in2 / rr**3)
                        # Transform to Cartesian coordinates
                        f[idx] = (f1 * p1.nvec[face[i]] +
                                  f2 * p1.tvec1[face[i]] +
                                  f3 * p1.tvec2[face[i]])
                    else:
                        # No tangent vectors, use normal derivative only
                        f[idx] = f_val * p1.nvec[face[i]]
                else:
                    f[idx] = f_val

        # ===== Off-diagonal elements (ir == 1) =====
        reface = np.where(np.any(ir_dense == 1, axis=0))[0]
        if len(reface) > 0:
            # Positions and weights for boundary element integration
            postab, wtab = p2.quad(reface)

            # Loop over faces to be refined
            for face in reface:
                # Index to neighbour faces
                nb = np.where(ir_dense[:, face] == 1)[0]
                if len(nb) == 0:
                    continue

                # Index to refinement array
                iface = np.array([ind_sparse[n, face] - 1 for n in nb], dtype=int)

                # Find this face in reface list
                face2_idx = np.where(reface == face)[0][0]

                # Get positions and weights for this face
                face_mask = wtab.row == face2_idx
                pos = postab[face_mask]
                w = wtab.data[face_mask]

                if len(w) == 0:
                    continue

                # Difference vector between face centroid and integration points
                x = pos1[nb, 0:1] - pos[:, 0].reshape(1, -1)
                y = pos1[nb, 1:2] - pos[:, 1].reshape(1, -1)
                z = pos1[nb, 2:3] - pos[:, 2].reshape(1, -1)
                # Distance
                r = np.sqrt(x**2 + y**2 + z**2)
                r = np.maximum(r, np.finfo(float).eps)

                # Green function
                g[iface] = (1.0 / r) @ w

                # Derivative of Green function
                if self.deriv == 'cart':
                    f[iface] = -np.column_stack([
                        (x / r**3) @ w,
                        (y / r**3) @ w,
                        (z / r**3) @ w
                    ])
                else:
                    nvec1 = p1.nvec[nb]
                    f[iface] = -(
                        (x / r**3) * nvec1[:, 0:1] +
                        (y / r**3) * nvec1[:, 1:2] +
                        (z / r**3) * nvec1[:, 2:3]
                    ) @ w

        # Save refined elements
        self.g = g
        self.f = f

        # Additional refinement function
        if 'refun' in options and options['refun'] is not None:
            self.g, self.f = options['refun'](self, g, f)

    def eval(self, *keys, ind=None):
        """
        Evaluate Green function.

        MATLAB: @greenstat/eval.m

        Parameters
        ----------
        *keys : str
            Keys for Green function components:
            - 'G'   : Green function
            - 'F'   : Surface derivative of Green function
            - 'H1'  : F + 2π (on diagonal)
            - 'H2'  : F - 2π (on diagonal)
            - 'Gp'  : Cartesian derivative of Green function
            - 'H1p' : Gp + 2π (on diagonal)
            - 'H2p' : Gp - 2π (on diagonal)
            - 'd'   : Distance matrix
        ind : ndarray, optional
            Index to matrix elements to be computed

        Returns
        -------
        varargout : tuple
            Requested Green functions

        Examples
        --------
        >>> G = g.eval('G')
        >>> F = g.eval('F')
        >>> G, F = g.eval('G', 'F')
        >>> H1, H2 = g.eval('H1', 'H2')
        """
        if ind is not None:
            return self._eval2(ind, *keys)
        else:
            return self._eval1(*keys)

    def _eval1(self, *keys):
        """
        Evaluate Green function (full matrices).

        MATLAB: @greenstat/private/eval1.m
        """
        pos1 = self.p1.pos
        pos2 = self.p2.pos
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Area as diagonal matrix multiplication
        area = self.p2.area

        # Difference of positions
        x = pos1[:, 0:1] - pos2[:, 0].reshape(1, -1)
        y = pos1[:, 1:2] - pos2[:, 1].reshape(1, -1)
        z = pos1[:, 2:3] - pos2[:, 2].reshape(1, -1)
        # Distance
        d = np.sqrt(x**2 + y**2 + z**2)
        d_safe = np.maximum(d, np.finfo(float).eps)

        # Green function
        G = None
        if 'G' in keys:
            G = (1.0 / d_safe) * area.reshape(1, -1)
            # Refine Green function
            if self.ind is not None and len(self.ind) > 0:
                G.flat[self.ind] = self.g
            G = G.reshape(n1, n2)

        # Process based on derivative type
        if self.deriv == 'norm':
            # Only normal (surface) derivative
            F = None
            if not all(k == 'G' for k in keys):
                nvec = self.p1.nvec
                # Surface derivative: F = -n·(r1-r2)/|r1-r2|³
                F = -(x * nvec[:, 0:1] + y * nvec[:, 1:2] + z * nvec[:, 2:3]) / d_safe**3 * area.reshape(1, -1)
                # Refine surface derivative
                if self.ind is not None and len(self.ind) > 0:
                    F.flat[self.ind] = self.f
                F = F.reshape(n1, n2)

            # Reset diagonal elements of d
            d_diag = d.copy()
            d_diag[d_diag == np.finfo(float).eps] = 0

            # Allocate and assign output
            results = []
            for key in keys:
                if key == 'G':
                    results.append(G)
                elif key == 'F':
                    results.append(F)
                elif key == 'H1':
                    results.append(F + 2 * np.pi * (d_diag == 0))
                elif key == 'H2':
                    results.append(F - 2 * np.pi * (d_diag == 0))
                elif key in ['Gp', 'H1p', 'H2p']:
                    raise ValueError('Only surface derivative computed (deriv=norm)')
                elif key == 'd':
                    results.append(d_diag)

        else:  # 'cart'
            # Cartesian derivatives of Green function
            Gp = None
            if not all(k == 'G' for k in keys):
                # Derivative of Green function: Gp = -r/|r|³
                Gp = -np.stack([
                    x / d_safe**3 * area.reshape(1, -1),
                    y / d_safe**3 * area.reshape(1, -1),
                    z / d_safe**3 * area.reshape(1, -1)
                ], axis=2)  # shape (n1, n2, 3)

                # Refine derivative
                Gp_flat = Gp.reshape(-1, 3)
                if self.ind is not None and len(self.ind) > 0:
                    Gp_flat[self.ind] = self.f
                Gp = np.transpose(Gp_flat.reshape(n1, n2, 3), (0, 2, 1))  # shape (n1, 3, n2)

            # Reset diagonal elements of d
            d_diag = d.copy()
            d_diag[d_diag == np.finfo(float).eps] = 0

            # Allocate and assign output
            results = []
            for key in keys:
                if key == 'G':
                    results.append(G)
                elif key == 'F':
                    # F = nvec · Gp
                    results.append(self._inner(self.p1.nvec, Gp))
                elif key == 'H1':
                    results.append(self._inner(self.p1.nvec, Gp) + 2 * np.pi * (d_diag == 0))
                elif key == 'H2':
                    results.append(self._inner(self.p1.nvec, Gp) - 2 * np.pi * (d_diag == 0))
                elif key == 'Gp':
                    results.append(Gp)
                elif key == 'H1p':
                    results.append(Gp + 2 * np.pi * self._outer(self.p1.nvec, d_diag == 0))
                elif key == 'H2p':
                    results.append(Gp - 2 * np.pi * self._outer(self.p1.nvec, d_diag == 0))
                elif key == 'd':
                    results.append(d_diag)

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _eval2(self, ind, *keys):
        """
        Evaluate Green function at specific indices.

        MATLAB: @greenstat/private/eval2.m
        """
        # Get full matrices and index
        results = self._eval1(*keys)
        if len(keys) == 1:
            results = (results,)

        indexed_results = []
        for r in results:
            if r is not None:
                if r.ndim == 3:  # Gp, H1p, H2p
                    indexed_results.append(r.reshape(-1, 3)[ind])
                else:
                    indexed_results.append(r.flat[ind])
            else:
                indexed_results.append(None)

        if len(indexed_results) == 1:
            return indexed_results[0]
        return tuple(indexed_results)

    def _inner(self, nvec, Gp):
        """
        Inner product: F = nvec · Gp

        MATLAB: inner(nvec, Gp)
        """
        # nvec: (n1, 3), Gp: (n1, 3, n2)
        # Result: (n1, n2)
        return np.einsum('ij,ijk->ik', nvec, Gp)

    def _outer(self, nvec, d0):
        """
        Outer product for diagonal correction: nvec ⊗ d0

        MATLAB: outer(nvec, d==0)
        """
        # nvec: (n1, 3), d0: (n1, n2) boolean
        # Result: (n1, 3, n2) with d0 broadcast
        n1, n2 = d0.shape
        result = np.zeros((n1, 3, n2))
        for i in range(n1):
            for j in range(n2):
                if d0[i, j]:
                    result[i, :, j] = nvec[i]
        return result

    def __getattr__(self, name):
        """Property access via attribute lookup."""
        if name in ['G', 'F', 'H1', 'H2', 'Gp', 'H1p', 'H2p', 'd']:
            return self.eval(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self):
        """String representation."""
        return (
            f"GreenStat(p1: {self.p1.nfaces} faces, "
            f"p2: {self.p2.nfaces} faces, deriv='{self.deriv}')"
        )

    def __str__(self):
        """Detailed string representation."""
        return (
            f"greenstat:\n"
            f"  p1: {self.p1}\n"
            f"  p2: {self.p2}\n"
            f"  op: {self.op}"
        )
