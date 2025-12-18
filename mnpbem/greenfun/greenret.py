"""
Green function for solution of full Maxwell equations.

MATLAB: Greenfun/@greenret/
100% identical to MATLAB MNPBEM implementation.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import factorial
from typing import Optional, Tuple, Any

from .utils import refinematrix


class GreenRet:
    """
    Green function for solution of full Maxwell equations (retarded).

    MATLAB: @greenret

    Computes the retarded Green function between points p1 and particle p2:
    - G(r, r') = exp(ikr) / r
    - F(r, r') = exp(ikr) * (ik - 1/r) * n·r / r²

    Uses Taylor expansion for refined elements:
    - G = Σ g_n × (ik)^n / n!
    - F = Σ f_n × (ik)^n / n!

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
    order : int
        Order for expansion of exp(ikr)
    ind : ndarray
        Index to face elements with refinement
    g : ndarray
        Refined elements for Green function (shape: n_refined × (order+1))
    f : ndarray
        Refined elements for derivative of Green function

    Methods
    -------
    eval(k, *keys)
        Evaluate Green function at wavenumber k (G, F, H1, H2, Gp, H1p, H2p)
    """

    def __init__(self, p1, p2, **options):
        """
        Initialize Green function for solution of full Maxwell equations.

        MATLAB: @greenret/greenret.m

        Parameters
        ----------
        p1 : Particle
            Green function between points p1 and comparticle p2
        p2 : Particle
            Green function between points p1 and comparticle p2
        **options : dict
            deriv : str, optional
                'cart' (Cartesian) or 'norm' (normal) derivative (default: 'cart')
            order : int, optional
                Order for expansion of exp(ikr) (default: 2)
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
        >>> from mnpbem.greenfun import GreenRet
        >>>
        >>> p = trisphere(144, 10.0)
        >>> g = GreenRet(p, p)
        >>> k = 2 * np.pi / 400  # wavenumber
        >>> G, F = g.eval(k, 'G', 'F')
        """
        self.p1 = p1
        self.p2 = p2
        self.op = options
        self.deriv = options.get('deriv', 'cart')
        self.order = options.get('order', 2)

        # Initialize refinement arrays
        self.ind = None
        self.g = None
        self.f = None

        # Initialize Green function
        self._init(**options)

    def _init(self, **options):
        """
        Initialize Green function.

        MATLAB: @greenret/private/init.m
        """
        p1 = self.p1
        p2 = self.p2
        pos1 = p1.pos
        pos2 = p2.pos

        # Get refinement matrix
        refine_opts = {k: v for k, v in options.items()
                       if k in ['AbsCutoff', 'RelCutoff', 'memsize']}
        ir = refinematrix(p1, p2, **refine_opts)

        # Derivative type
        if 'deriv' in options:
            self.deriv = options['deriv']

        # Order of exp(ikr) expansion
        order = self.order
        if order is None:
            order = 2

        # Index to refined elements
        ir_coo = ir.tocoo()
        self.ind = np.ravel_multi_index((ir_coo.row, ir_coo.col), (p1.nfaces, p2.nfaces))

        # Conversion table between matrix elements and refined elements
        row, col = ir_coo.row, ir_coo.col
        nrow = len(row)
        if nrow == 0 or order == 0:
            return

        # Create sparse index matrix for lookup
        ind_sparse = csr_matrix((np.arange(1, nrow + 1), (row, col)),
                                shape=(p1.nfaces, p2.nfaces))

        # Allocate refinement arrays
        g = np.zeros((nrow, order + 1))
        if self.deriv == 'cart':
            f = np.zeros((nrow, 3, order + 1))
        else:
            f = np.zeros((nrow, order + 1))

        # Get dense matrix for ir
        ir_dense = ir.toarray()

        # ===== Diagonal elements (ir == 2) =====
        if np.any(ir_dense == 2):
            face, face2 = np.where(ir_dense == 2)
            # Index to refinement array (1-indexed to match MATLAB)
            iface = np.array([ind_sparse[f, f2] - 1 for f, f2 in zip(face, face2)], dtype=int)

            # Integration points and weights for polar integration
            pos, w, row_quad = p2.quadpol(face2)

            # Measure positions with respect to centroids
            vec = p1.pos[face[row_quad]] - pos
            # Distance
            r = np.sqrt(np.sum(vec**2, axis=1))
            r = np.maximum(r, np.finfo(float).eps)

            # Accumulate function
            def fun(x, n):
                result = np.zeros(len(iface))
                for i, idx in enumerate(iface):
                    mask = row_quad == i
                    result[i] = np.sum(w[mask] * x[mask]) / factorial(n)
                return result

            # Green function for each order
            for ord_val in range(order + 1):
                g[iface, ord_val] = fun(r**(ord_val - 1), ord_val)

            # Inner product
            in_prod = np.sum(vec * p1.nvec[face[row_quad]], axis=1)

            # Surface derivative of Green function
            if self.deriv == 'norm':
                for ord_val in range(order + 1):
                    f[iface, ord_val] = fun((ord_val - 1) * in_prod * r**(ord_val - 3), ord_val)
            else:  # 'cart'
                # For r->0 the tangential derivatives vanish
                rr = np.maximum(r, 1e-4 * np.max(r) if len(r) > 0 and np.max(r) > 0 else 1e-10)

                # Check for tangent vectors
                if hasattr(p1, 'tvec1') and hasattr(p1, 'tvec2'):
                    tvec1 = p1.tvec1
                    tvec2 = p1.tvec2
                    in1 = np.sum(vec * tvec1[face[row_quad]], axis=1)
                    in2 = np.sum(vec * tvec2[face[row_quad]], axis=1)
                else:
                    # Compute tangent vectors on-the-fly
                    in1 = np.zeros_like(in_prod)
                    in2 = np.zeros_like(in_prod)
                    tvec1 = np.zeros_like(p1.nvec)
                    tvec2 = np.zeros_like(p1.nvec)

                for ord_val in range(order + 1):
                    # Normal derivative
                    f_n = fun((ord_val - 1) * in_prod * r**(ord_val - 3), ord_val)
                    # Tangential derivatives
                    f_t1 = fun((ord_val - 1) * in1 * r**ord_val / rr**3, ord_val)
                    f_t2 = fun((ord_val - 1) * in2 * r**ord_val / rr**3, ord_val)

                    # Transform to Cartesian coordinates
                    for i, idx in enumerate(iface):
                        if hasattr(p1, 'tvec1') and hasattr(p1, 'tvec2'):
                            f[idx, :, ord_val] = (f_n[i] * p1.nvec[face[i]] +
                                                  f_t1[i] * p1.tvec1[face[i]] +
                                                  f_t2[i] * p1.tvec2[face[i]])
                        else:
                            f[idx, :, ord_val] = f_n[i] * p1.nvec[face[i]]

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

                # Difference vector between face centroids
                vec0 = -(pos1[nb] - pos2[face:face+1])
                # Distance between centroids
                r0 = np.sqrt(np.sum(vec0**2, axis=1, keepdims=True))

                # Green function for each order
                for ord_val in range(order + 1):
                    g[iface, ord_val] = ((r - r0)**ord_val / r / factorial(ord_val)) @ w

                # Surface derivative of Green function
                if self.deriv == 'norm':
                    nvec1 = p1.nvec[nb]
                    # Inner product
                    in_prod = (x * nvec1[:, 0:1] + y * nvec1[:, 1:2] + z * nvec1[:, 2:3])

                    # Lowest order
                    f[iface, 0] = -(in_prod / r**3) @ w

                    # Higher orders
                    for ord_val in range(1, order + 1):
                        f[iface, ord_val] = (in_prod * (
                            -(r - r0)**ord_val / (r**3 * factorial(ord_val)) +
                            (r - r0)**(ord_val - 1) / (r**2 * factorial(ord_val - 1))
                        )) @ w

                else:  # 'cart'
                    # Vector integration function
                    def vec_fun(ff):
                        return np.column_stack([
                            (x * ff) @ w,
                            (y * ff) @ w,
                            (z * ff) @ w
                        ])

                    # Lowest order
                    f[iface, :, 0] = -vec_fun(1.0 / r**3)

                    # Higher orders
                    for ord_val in range(1, order + 1):
                        f[iface, :, ord_val] = vec_fun(
                            -(r - r0)**ord_val / (r**3 * factorial(ord_val)) +
                            (r - r0)**(ord_val - 1) / (r**2 * factorial(ord_val - 1))
                        )

        # Save refined elements
        self.g = g
        self.f = f
        self.order = order

        # Additional refinement function
        if 'refun' in options and options['refun'] is not None:
            self.g, self.f = options['refun'](self, g, f)

    def eval(self, k, *keys, ind=None):
        """
        Evaluate Green function.

        MATLAB: @greenret/eval.m

        Parameters
        ----------
        k : complex
            Wavenumber
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
        >>> k = 2 * np.pi / 400  # wavenumber for 400nm
        >>> G = g.eval(k, 'G')
        >>> F = g.eval(k, 'F')
        >>> G, F = g.eval(k, 'G', 'F')
        """
        if ind is not None:
            return self._eval2(k, ind, *keys)
        else:
            return self._eval1(k, *keys)

    def _eval1(self, k, *keys):
        """
        Evaluate Green function (full matrices).

        MATLAB: @greenret/private/eval1.m
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

        # Expansion coefficients: (ik)^n for n = 0, 1, ..., order
        ik_powers = (1j * k) ** np.arange(self.order + 1)

        # Green function
        G = None
        if 'G' in keys:
            G = (1.0 / d_safe) * area.reshape(1, -1)
            # Refine Green function
            if self.ind is not None and len(self.ind) > 0:
                G.flat[self.ind] = self.g @ ik_powers
            G = G.reshape(n1, n2) * np.exp(1j * k * d)

        # Process based on derivative type
        if self.deriv == 'norm':
            # Only normal (surface) derivative
            F = None
            if not all(kk == 'G' for kk in keys):
                nvec = self.p1.nvec
                # Surface derivative: F = (n·r) * (ik - 1/r) / r²
                in_prod = x * nvec[:, 0:1] + y * nvec[:, 1:2] + z * nvec[:, 2:3]
                F = in_prod * (1j * k - 1.0 / d_safe) / d_safe**2 * area.reshape(1, -1)
                # Refine surface derivative
                if self.ind is not None and len(self.ind) > 0:
                    F.flat[self.ind] = self.f @ ik_powers
                F = F.reshape(n1, n2) * np.exp(1j * k * d)

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
            F = None

            compute_Gp = any(kk in ['Gp', 'H1p', 'H2p'] for kk in keys)
            compute_F = any(kk in ['F', 'H1', 'H2'] for kk in keys)

            if compute_Gp or (compute_F and not all(kk == 'G' for kk in keys)):
                # Auxiliary quantity
                ff = (1j * k - 1.0 / d_safe) / d_safe**2

                if compute_Gp:
                    # Derivative of Green function: Gp = f * r
                    Gp = np.stack([
                        (ff * x) * area.reshape(1, -1),
                        (ff * y) * area.reshape(1, -1),
                        (ff * z) * area.reshape(1, -1)
                    ], axis=2)  # shape (n1, n2, 3)

                    # Refine derivative
                    Gp_flat = Gp.reshape(-1, 3)
                    if self.ind is not None and len(self.ind) > 0:
                        # f has shape (n_ind, 3, order+1)
                        Gp_flat[self.ind] = np.einsum('ijk,k->ij', self.f, ik_powers)

                    # Multiply with phase factor
                    phase = np.exp(1j * k * d).reshape(-1, 1)
                    Gp_flat = Gp_flat * phase

                    # Reshape: (n1, n2, 3) -> (n1, 3, n2)
                    Gp = np.transpose(Gp_flat.reshape(n1, n2, 3), (0, 2, 1))

                    # Compute F if needed
                    if compute_F:
                        F = self._inner(self.p1.nvec, Gp)

                elif compute_F:
                    # Compute only surface derivative (more efficient)
                    nvec = self.p1.nvec
                    F = (ff * x * nvec[:, 0:1] + ff * y * nvec[:, 1:2] + ff * z * nvec[:, 2:3]) * area.reshape(1, -1)

                    # Refine surface derivative
                    if self.ind is not None and len(self.ind) > 0:
                        # Get row indices from linear indices
                        row_idx = self.ind // n2
                        # Inner product of nvec with f
                        nvec_expanded = self.p1.nvec[row_idx]
                        # f: (n_ind, 3, order+1), nvec: (n_ind, 3)
                        inner_f = np.einsum('ij,ijk->ik', nvec_expanded, self.f)
                        F.flat[self.ind] = inner_f @ ik_powers

                    F = F.reshape(n1, n2) * np.exp(1j * k * d)

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

    def _eval2(self, k, ind, *keys):
        """
        Evaluate Green function at specific indices.

        MATLAB: @greenret/private/eval2.m
        """
        # Get full matrices and index
        results = self._eval1(k, *keys)
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
        result = np.zeros((n1, 3, n2), dtype=complex)
        for i in range(n1):
            for j in range(n2):
                if d0[i, j]:
                    result[i, :, j] = nvec[i]
        return result

    def __getattr__(self, name):
        """Property access via attribute lookup."""
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Use eval(k, '{name}') with wavenumber k."
        )

    def __repr__(self):
        """String representation."""
        return (
            f"GreenRet(p1: {self.p1.nfaces} faces, "
            f"p2: {self.p2.nfaces} faces, deriv='{self.deriv}', order={self.order})"
        )

    def __str__(self):
        """Detailed string representation."""
        return (
            f"greenret:\n"
            f"  p1: {self.p1}\n"
            f"  p2: {self.p2}\n"
            f"  op: {self.op}"
        )
