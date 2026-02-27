import os
import sys
import numpy as np
from typing import Optional, Tuple, Any, List, Dict

from .clustertree import ClusterTree
from .hmatrix import HMatrix
from .compgreen_ret import CompGreenRet


class ACACompGreenRet(object):

    # MATLAB: +aca/@compgreenret
    # ACA-accelerated retarded Green function.
    # Wraps CompGreenRet with H-matrix acceleration.
    # Uses ClusterTree to partition the particle and HMatrix with ACA
    # for off-diagonal (low-rank) blocks, falling back to direct computation
    # for near-field (dense) blocks.

    def __init__(self,
            p: Any,
            htol: float = 1e-6,
            kmax: int = 100,
            cleaf: int = 32,
            **options: Any):

        # MATLAB: +aca/@compgreenret/compgreenret.m -> init.m

        self.p = p

        # Create underlying dense CompGreenRet for evaluation
        # MATLAB: obj.g = compgreenret(p, p, varargin{:})
        self.g = CompGreenRet(p, p, **options)

        # Build cluster tree from particle positions
        # MATLAB: tree = clustertree(p, varargin{:})
        pos = p.pos
        ipart_arr = self._build_ipart_arr(p)
        self.tree = ClusterTree(pos, cleaf = cleaf, ipart_arr = ipart_arr)

        # Create template H-matrix
        # MATLAB: obj.hmat = hmatrix(tree, varargin{:})
        self.hmat_template = HMatrix(tree = self.tree, htol = htol, kmax = kmax)

        # Cache for evaluated H-matrices keyed by (i, j, key, enei)
        self._cache = {}  # type: Dict[Tuple, HMatrix]

        # Store options
        self.htol = htol
        self.kmax = kmax
        self.cleaf = cleaf
        self.options = options

    def _build_ipart_arr(self, p: Any) -> Optional[np.ndarray]:

        # Build particle index array for cluster tree.
        # Maps each face to its particle index (0-based).
        if not hasattr(p, 'p') or len(p.p) <= 1:
            return None

        total_n = p.n
        ipart_arr = np.empty(total_n, dtype = np.int64)
        offset = 0
        for idx, part in enumerate(p.p):
            n_part = part.n
            ipart_arr[offset:offset + n_part] = idx
            offset += n_part

        return ipart_arr

    def eval(self,
            i: int,
            j: int,
            key: str,
            enei: float) -> HMatrix:

        # MATLAB: +aca/@compgreenret/eval.m
        # Evaluate retarded Green function and return H-matrix.
        #
        # Parameters
        # ----------
        # i, j : int
        #     Region indices (0-based). i indexes in/outside of p1, j indexes p2.
        # key : str
        #     'G' (Green function), 'F' (surface derivative),
        #     'H1' (F + 2*pi), 'H2' (F - 2*pi)
        # enei : float
        #     Light wavelength in vacuum (nm)
        #
        # Returns
        # -------
        # hmat : HMatrix
        #     H-matrix representation of the Green function

        cache_key = (i, j, key, enei)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build evaluation function for the underlying Green function.
        # This function takes particle-ordered row/col indices and returns
        # the corresponding matrix values.
        # MATLAB: fun = @(row, col) eval(obj.g, i, j, key, enei, sub2ind(...))
        def eval_func(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            return self._eval_elements(i, j, key, enei, row, col)

        # Create a fresh H-matrix from the template
        hmat = HMatrix(tree = self.tree, htol = self.htol, kmax = self.kmax)
        hmat.row1 = self.hmat_template.row1.copy()
        hmat.col1 = self.hmat_template.col1.copy()
        hmat.row2 = self.hmat_template.row2.copy()
        hmat.col2 = self.hmat_template.col2.copy()
        hmat.val = [None] * len(hmat.row1)
        hmat.lhs = [None] * len(hmat.row2)
        hmat.rhs = [None] * len(hmat.row2)

        # Fill the H-matrix using ACA
        # MATLAB: hmat = fillval(hmat, fun) then ACA for low-rank blocks
        hmat.aca(eval_func)

        self._cache[cache_key] = hmat
        return hmat

    def _eval_elements(self,
            i: int,
            j: int,
            key: str,
            enei: float,
            row: np.ndarray,
            col: np.ndarray) -> np.ndarray:

        # Evaluate Green function elements for given row/col pairs.
        # row, col are arrays of face indices in particle ordering.
        # Returns a 1D array of values at those (row, col) positions.

        # Get the full dense matrix from CompGreenRet
        # We cache the dense matrix per (i, j, key, enei) to avoid recomputation
        dense_key = ('dense', i, j, key, enei)
        if dense_key not in self._cache:
            dense_mat = self.g.eval(i, j, key, enei)
            self._cache[dense_key] = dense_mat

        dense_mat = self._cache[dense_key]

        if np.isscalar(dense_mat):
            return np.full(len(row), complex(dense_mat))

        return dense_mat[row, col]

    def eval_full(self,
            i: int,
            j: int,
            key: str,
            enei: float) -> np.ndarray:

        # Evaluate retarded Green function as full dense matrix.
        # Equivalent to converting the H-matrix back to dense form.

        hmat = self.eval(i, j, key, enei)
        return hmat.full()

    def potential(self,
            sig: Any,
            inout: int = 1) -> Any:

        # MATLAB: +aca/@compgreenret/potential.m
        # Potentials and surface derivatives inside/outside of particle.
        # Uses H-matrix multiply instead of dense matrix-vector product.
        #
        # Parameters
        # ----------
        # sig : CompStruct
        #     Surface charges and currents
        # inout : int
        #     1 for inside, 2 for outside
        #
        # Returns
        # -------
        # pot : CompStruct
        #     Potentials and surface derivatives

        enei = sig.enei

        # Set parameters that depend on inside/outside
        # MATLAB: H = subsref({'H1', 'H2'}, substruct('{}', {inout}))
        H_key = 'H1' if inout == 1 else 'H2'

        # Evaluate H-matrices for Green functions
        G1 = self.eval(inout - 1, 0, 'G', enei)
        G2 = self.eval(inout - 1, 1, 'G', enei)

        # Surface derivatives as H-matrices
        H1 = self.eval(inout - 1, 0, H_key, enei)
        H2 = self.eval(inout - 1, 1, H_key, enei)

        # Potential and surface derivative using H-matrix multiply
        # MATLAB: matmul = @(x, y) reshape(x * reshape(y, size(y,1), []), size(y))
        # Scalar potential
        phi = self._hmat_matmul(G1, sig.sig1) + self._hmat_matmul(G2, sig.sig2)
        phip = self._hmat_matmul(H1, sig.sig1) + self._hmat_matmul(H2, sig.sig2)

        # Vector potential
        a = self._hmat_matmul(G1, sig.h1) + self._hmat_matmul(G2, sig.h2)
        ap = self._hmat_matmul(H1, sig.h1) + self._hmat_matmul(H2, sig.h2)

        # Build output CompStruct
        from .compgreen_stat import CompStruct
        if inout == 1:
            pot = CompStruct(self.p, enei, phi1 = phi, phi1p = phip, a1 = a, a1p = ap)
        else:
            pot = CompStruct(self.p, enei, phi2 = phi, phi2p = phip, a2 = a, a2p = ap)

        return pot

    def _hmat_matmul(self,
            hmat: HMatrix,
            x: Any) -> Any:

        # Matrix-vector (or matrix-matrix) multiply using H-matrix.
        # Handles scalar zero, 1D vectors, and multi-dimensional arrays.

        if np.isscalar(x):
            if x == 0:
                return 0
            # H-matrix times scalar: need to build full vector
            return hmat.full() * x

        if not isinstance(x, np.ndarray):
            return 0

        if x.ndim == 1:
            return hmat.mtimes_vec(x)
        elif x.ndim == 2:
            return hmat.mtimes_vec(x)
        else:
            # For higher-dimensional arrays (n, 3, ...),
            # reshape to 2D, multiply, reshape back
            original_shape = x.shape
            n = original_shape[0]
            x_2d = x.reshape(n, -1)
            result_2d = hmat.mtimes_vec(x_2d)
            return result_2d.reshape(original_shape)

    def field(self,
            sig: Any,
            inout: int = 1) -> Any:

        # Electric and magnetic field using H-matrix acceleration.
        # Falls back to dense CompGreenRet for derivative (Gp) terms
        # which are 3D tensors and not easily H-matrix compressed.

        return self.g.field(sig, inout = inout)

    def compression(self,
            i: int = 0,
            j: int = 0,
            key: str = 'G',
            enei: float = 600.0) -> float:

        # Compute compression ratio for a given Green function evaluation.
        hmat = self.eval(i, j, key, enei)
        return hmat.compression()

    def clear_cache(self) -> None:

        # Clear the H-matrix cache.
        self._cache = {}
