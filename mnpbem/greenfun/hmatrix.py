import os
import sys
import numpy as np
from typing import Optional, Tuple, Any, List, Callable, Dict

from .clustertree import ClusterTree


class HMatrix(object):

    # MATLAB: @hmatrix
    # Hierarchical matrix using cluster tree and admissibility.
    # Stores dense blocks and low-rank (UV) blocks.
    # See S. Boerm et al., Eng. Analysis with Bound. Elem. 27, 405 (2003).

    def __init__(self,
            tree: Optional[ClusterTree] = None,
            htol: float = 1e-6,
            kmax: int = 100,
            fadmiss: Optional[Callable] = None):

        self.tree = tree
        self.htol = htol
        self.kmax = kmax

        # Block indices: dense blocks (row1, col1) and low-rank blocks (row2, col2)
        self.row1 = np.array([], dtype = np.int64)  # tree node indices for dense blocks
        self.col1 = np.array([], dtype = np.int64)
        self.row2 = np.array([], dtype = np.int64)  # tree node indices for low-rank blocks
        self.col2 = np.array([], dtype = np.int64)

        # Storage: lists of arrays
        self.val = []   # dense blocks: list of 2D arrays
        self.lhs = []   # low-rank left factors: list of 2D arrays (m x k)
        self.rhs = []   # low-rank right factors: list of 2D arrays (n x k)

        if tree is not None:
            self._init(tree, fadmiss = fadmiss)

    def _init(self,
            tree: ClusterTree,
            fadmiss: Optional[Callable] = None) -> None:

        # MATLAB: @hmatrix/private/init.m
        self.tree = tree

        # Compute admissibility
        admiss = tree.admissibility(tree, fadmiss = fadmiss)

        # Separate dense (==2) and low-rank (==1) blocks
        row1_list = []
        col1_list = []
        row2_list = []
        col2_list = []

        for (i1, i2), val in admiss.items():
            if val == 2:
                row1_list.append(i1)
                col1_list.append(i2)
            elif val == 1:
                row2_list.append(i1)
                col2_list.append(i2)

        self.row1 = np.array(row1_list, dtype = np.int64)
        self.col1 = np.array(col1_list, dtype = np.int64)
        self.row2 = np.array(row2_list, dtype = np.int64)
        self.col2 = np.array(col2_list, dtype = np.int64)

        # Initialize empty storage
        self.val = [None] * len(self.row1)
        self.lhs = [None] * len(self.row2)
        self.rhs = [None] * len(self.row2)

    def aca(self, fun: Callable) -> 'HMatrix':

        # MATLAB: @hmatrix/aca.m
        # Fills the H-matrix using Adaptive Cross Approximation.
        # fun(row, col) returns matrix values for given row/col indices (0-based, particle ordering)

        tree = self.tree
        # Map from cluster to particle indices
        ind_c2p = tree.ind[:, 0]

        # Wrapped function: takes cluster indices, returns values
        def fun2(row_c: np.ndarray, col_c: np.ndarray) -> np.ndarray:
            return fun(ind_c2p[row_c], ind_c2p[col_c])

        # Compute dense blocks
        for i in range(len(self.row1)):
            indr = tree.cind[self.row1[i]]
            indc = tree.cind[self.col1[i]]
            rows = np.arange(indr[0], indr[1] + 1, dtype = np.int64)
            cols = np.arange(indc[0], indc[1] + 1, dtype = np.int64)
            row_grid, col_grid = np.meshgrid(rows, cols, indexing = 'ij')
            self.val[i] = fun2(row_grid.ravel(), col_grid.ravel()).reshape(row_grid.shape)

        # Compute low-rank blocks using ACA
        for i in range(len(self.row2)):
            indr = tree.cind[self.row2[i]]
            indc = tree.cind[self.col2[i]]
            rows = np.arange(indr[0], indr[1] + 1, dtype = np.int64)
            cols = np.arange(indc[0], indc[1] + 1, dtype = np.int64)

            lhs, rhs = self._aca_block(fun2, rows, cols, self.htol, self.kmax)
            self.lhs[i] = lhs
            self.rhs[i] = rhs

        return self

    def _aca_block(self,
            fun: Callable,
            rows: np.ndarray,
            cols: np.ndarray,
            htol: float,
            kmax: int) -> Tuple[np.ndarray, np.ndarray]:

        # Partially-pivoted Adaptive Cross Approximation for a single block
        # Returns (U, V) such that A ~ U @ V.T
        m = len(rows)
        n = len(cols)
        max_rank = min(m, n, kmax)

        # Probe dtype from function output
        probe = fun(rows[:1], cols[:1])
        out_dtype = np.complex128 if np.iscomplexobj(probe) else np.float64

        U_cols = []
        V_cols = []

        # Track used rows and columns
        used_rows = set()
        used_cols = set()

        # Residual tracking: we'll keep the approximation built so far
        # Start with row 0
        pivot_row_local = 0
        frobenius_sq = 0.0

        for k in range(max_rank):
            # Compute row of residual at pivot_row_local
            row_global = rows[pivot_row_local]
            row_c = np.full(n, row_global, dtype = np.int64)
            col_c = cols.copy()
            row_vals = fun(row_c, col_c)

            # Subtract contributions from previous approximants
            for j in range(len(U_cols)):
                row_vals = row_vals - U_cols[j][pivot_row_local] * V_cols[j]

            # Find pivot column (maximum absolute value in unused columns)
            abs_row = np.abs(row_vals)
            # Mask used columns
            for uc in used_cols:
                abs_row[uc] = 0.0
            pivot_col_local = np.argmax(abs_row)
            pivot_val = row_vals[pivot_col_local]

            if np.abs(pivot_val) < 1e-15:
                break

            # Compute column of residual at pivot_col_local
            col_global = cols[pivot_col_local]
            row_c2 = rows.copy()
            col_c2 = np.full(m, col_global, dtype = np.int64)
            col_vals = fun(row_c2, col_c2)

            # Subtract contributions from previous approximants
            for j in range(len(U_cols)):
                col_vals = col_vals - V_cols[j][pivot_col_local] * U_cols[j]

            # New rank-1 term: u = col_vals / pivot_val, v = row_vals
            u_new = col_vals / pivot_val
            v_new = row_vals.copy()

            U_cols.append(u_new)
            V_cols.append(v_new)

            used_rows.add(pivot_row_local)
            used_cols.add(pivot_col_local)

            # Convergence check
            u_norm_sq = np.sum(u_new ** 2)
            v_norm_sq = np.sum(v_new ** 2)
            new_term_sq = u_norm_sq * v_norm_sq

            # Update Frobenius norm estimate of approximation
            cross_terms = 0.0
            for j in range(len(U_cols) - 1):
                cross_terms += 2.0 * np.dot(U_cols[j], u_new) * np.dot(V_cols[j], v_new)
            frobenius_sq += new_term_sq + cross_terms

            if frobenius_sq > 0 and np.sqrt(new_term_sq) < htol * np.sqrt(abs(frobenius_sq)):
                break

            # Choose next pivot row: row with max |u_new| among unused rows
            abs_u = np.abs(u_new)
            for ur in used_rows:
                abs_u[ur] = 0.0
            pivot_row_local = np.argmax(abs_u)

        if len(U_cols) == 0:
            return np.zeros((m, 1), dtype = out_dtype), np.zeros((n, 1), dtype = out_dtype)

        # Stack into matrices
        rank = len(U_cols)
        U = np.empty((m, rank), dtype = out_dtype)
        V = np.empty((n, rank), dtype = out_dtype)
        for j in range(rank):
            U[:, j] = U_cols[j]
            V[:, j] = V_cols[j]

        return U, V

    def full(self) -> np.ndarray:

        # MATLAB: @hmatrix/full.m
        # Convert H-matrix to full dense matrix
        tree = self.tree
        n = tree.n
        mat = np.zeros((n, n), dtype = np.float64)

        # Check if any block is complex
        is_complex = False
        for v in self.val:
            if v is not None and np.iscomplexobj(v):
                is_complex = True
                break
        if not is_complex:
            for l in self.lhs:
                if l is not None and np.iscomplexobj(l):
                    is_complex = True
                    break
        if not is_complex:
            for r in self.rhs:
                if r is not None and np.iscomplexobj(r):
                    is_complex = True
                    break

        if is_complex:
            mat = np.zeros((n, n), dtype = np.complex128)

        # Fill dense blocks
        for i in range(len(self.row1)):
            if self.val[i] is None:
                continue
            r_start = tree.cind[self.row1[i], 0]
            r_end = tree.cind[self.row1[i], 1] + 1
            c_start = tree.cind[self.col1[i], 0]
            c_end = tree.cind[self.col1[i], 1] + 1
            mat[r_start:r_end, c_start:c_end] = self.val[i]

        # Fill low-rank blocks
        for i in range(len(self.row2)):
            if self.lhs[i] is None or self.rhs[i] is None:
                continue
            r_start = tree.cind[self.row2[i], 0]
            r_end = tree.cind[self.row2[i], 1] + 1
            c_start = tree.cind[self.col2[i], 0]
            c_end = tree.cind[self.col2[i], 1] + 1
            # lhs @ rhs.T
            mat[r_start:r_end, c_start:c_end] = self.lhs[i] @ self.rhs[i].T

        # Transform from cluster ordering to particle ordering
        ind_c2p = tree.ind[:, 0]
        # mat is in cluster ordering, need to permute to particle ordering
        # mat_particle[p_i, p_j] = mat_cluster[c_i, c_j]
        # where c_i = part_to_cluster[p_i], but we need inverse:
        # particle row i should come from cluster row ind[:,1][i]
        result = mat[np.ix_(tree.ind[:, 1], tree.ind[:, 1])]
        return result

    def mtimes_vec(self, v: np.ndarray) -> np.ndarray:

        # MATLAB: mtimes2 - H-matrix times dense vector/matrix
        tree = self.tree
        n = tree.n

        # Convert to cluster ordering
        if v.ndim == 1:
            v_cluster = tree.part2cluster(v)
            result = np.zeros(n, dtype = v.dtype)
        else:
            v_cluster = tree.part2cluster(v)
            result = np.zeros((n, v.shape[1]), dtype = v.dtype)

        # Dense blocks
        for i in range(len(self.row1)):
            if self.val[i] is None:
                continue
            r_start = tree.cind[self.row1[i], 0]
            r_end = tree.cind[self.row1[i], 1] + 1
            c_start = tree.cind[self.col1[i], 0]
            c_end = tree.cind[self.col1[i], 1] + 1

            if v.ndim == 1:
                result[r_start:r_end] += self.val[i] @ v_cluster[c_start:c_end]
            else:
                result[r_start:r_end] += self.val[i] @ v_cluster[c_start:c_end]

        # Low-rank blocks
        for i in range(len(self.row2)):
            if self.lhs[i] is None or self.rhs[i] is None:
                continue
            r_start = tree.cind[self.row2[i], 0]
            r_end = tree.cind[self.row2[i], 1] + 1
            c_start = tree.cind[self.col2[i], 0]
            c_end = tree.cind[self.col2[i], 1] + 1

            # lhs @ (rhs.T @ v)
            if v.ndim == 1:
                tmp = self.rhs[i].T @ v_cluster[c_start:c_end]
                result[r_start:r_end] += self.lhs[i] @ tmp
            else:
                tmp = self.rhs[i].T @ v_cluster[c_start:c_end]
                result[r_start:r_end] += self.lhs[i] @ tmp

        # Convert back to particle ordering
        return tree.cluster2part(result)

    def __matmul__(self, other: Any) -> Any:

        if isinstance(other, np.ndarray):
            return self.mtimes_vec(other)
        elif isinstance(other, HMatrix):
            return self._mtimes_hmat(other)
        else:
            raise TypeError('[error] Unsupported type for H-matrix multiplication')

    def __rmul__(self, scalar: float) -> 'HMatrix':

        # MATLAB: mtimes with scalar * hmatrix
        result = self._copy()
        result.val = [scalar * v if v is not None else None for v in result.val]
        result.lhs = [scalar * l if l is not None else None for l in result.lhs]
        return result

    def __mul__(self, other: Any) -> Any:

        if isinstance(other, (int, float, complex)):
            return self.__rmul__(other)
        elif isinstance(other, np.ndarray):
            return self.mtimes_vec(other)
        elif isinstance(other, HMatrix):
            return self._mtimes_hmat(other)
        else:
            raise TypeError('[error] Unsupported type for H-matrix multiplication')

    def __neg__(self) -> 'HMatrix':

        # MATLAB: uminus
        result = self._copy()
        result.val = [-v if v is not None else None for v in result.val]
        result.lhs = [-l if l is not None else None for l in result.lhs]
        return result

    def __add__(self, other: 'HMatrix') -> 'HMatrix':

        # MATLAB: plus
        if not isinstance(other, HMatrix):
            raise TypeError('[error] Unsupported type for H-matrix addition')
        return self._plus_hmat(other)

    def __sub__(self, other: 'HMatrix') -> 'HMatrix':

        # MATLAB: minus
        return self.__add__(-other)

    def _plus_hmat(self, other: 'HMatrix') -> 'HMatrix':

        # MATLAB: plus2 - Add two H-matrices with same structure
        result = self._copy()

        # Add dense blocks
        for i in range(len(result.row1)):
            if result.val[i] is not None and other.val[i] is not None:
                result.val[i] = result.val[i] + other.val[i]
            elif other.val[i] is not None:
                result.val[i] = other.val[i].copy()

        # Add low-rank blocks: combine and recompress
        for i in range(len(result.row2)):
            lhs1 = result.lhs[i]
            rhs1 = result.rhs[i]
            lhs2 = other.lhs[i]
            rhs2 = other.rhs[i]

            if lhs1 is None and lhs2 is None:
                continue
            elif lhs1 is None:
                result.lhs[i] = lhs2.copy()
                result.rhs[i] = rhs2.copy()
            elif lhs2 is None:
                pass  # keep result as is
            else:
                # Combine: [lhs1, lhs2] and [rhs1, rhs2]
                m = lhs1.shape[0]
                n = rhs1.shape[0]
                k1 = lhs1.shape[1]
                k2 = lhs2.shape[1]
                new_lhs = np.empty((m, k1 + k2), dtype = lhs1.dtype)
                new_lhs[:, :k1] = lhs1
                new_lhs[:, k1:] = lhs2
                new_rhs = np.empty((n, k1 + k2), dtype = rhs1.dtype)
                new_rhs[:, :k1] = rhs1
                new_rhs[:, k1:] = rhs2
                result.lhs[i] = new_lhs
                result.rhs[i] = new_rhs

        # Recompress
        result.truncate()
        return result

    def truncate(self, htol: Optional[float] = None) -> 'HMatrix':

        # MATLAB: @hmatrix/truncate.m
        # Truncate low-rank blocks via SVD
        if htol is None:
            htol = self.htol

        for i in range(len(self.lhs)):
            if self.lhs[i] is None or self.rhs[i] is None:
                continue
            self.lhs[i], self.rhs[i] = self._truncate_block(
                self.lhs[i], self.rhs[i], htol)

        self.htol = htol
        return self

    def _truncate_block(self,
            lhs: np.ndarray,
            rhs: np.ndarray,
            htol: float) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB: truncate/fun
        if np.linalg.norm(lhs.ravel()) < np.finfo(float).eps:
            return lhs, rhs
        if np.linalg.norm(rhs.ravel()) < np.finfo(float).eps:
            return lhs, rhs

        q1, r1 = np.linalg.qr(lhs, mode = 'reduced')
        q2, r2 = np.linalg.qr(rhs, mode = 'reduced')

        # SVD of r1 @ r2.T
        u_svd, s_svd, vt_svd = np.linalg.svd(r1 @ r2.T, full_matrices = False)

        # Find largest singular values: keep k such that cumsum(s) < (1-htol)*sum(s)
        total = np.sum(s_svd)
        if total < np.finfo(float).eps:
            return lhs[:, :1] * 0, rhs[:, :1] * 0

        cum = np.cumsum(s_svd)
        threshold = (1.0 - htol) * total
        k_idx = np.where(cum < threshold)[0]

        if len(k_idx) == 0:
            # Keep at least rank 1
            k = 1
        else:
            k = len(k_idx)

        # Truncated decomposition
        new_lhs = q1 @ (u_svd[:, :k] * s_svd[:k][np.newaxis, :])
        new_rhs = q2 @ vt_svd[:k, :].T  # q2 @ conj(v[:, :k])

        return new_lhs, new_rhs

    def compression(self) -> float:

        # MATLAB: @hmatrix/compression.m
        # Ratio of H-matrix elements to full matrix elements
        n_elements = 0
        for v in self.val:
            if v is not None:
                n_elements += v.size
        for i in range(len(self.lhs)):
            if self.lhs[i] is not None:
                n_elements += self.lhs[i].size
            if self.rhs[i] is not None:
                n_elements += self.rhs[i].size

        total = self.tree.n * self.tree.n
        if total == 0:
            return 0.0
        return n_elements / total

    def diag(self) -> np.ndarray:

        # MATLAB: @hmatrix/diag.m
        tree = self.tree
        n = tree.n
        diag_dtype = np.float64
        for v in self.val:
            if v is not None and np.iscomplexobj(v):
                diag_dtype = np.complex128
                break
        d = np.zeros(n, dtype = diag_dtype)

        # Find diagonal dense blocks
        for i in range(len(self.row1)):
            if self.row1[i] == self.col1[i] and self.val[i] is not None:
                r_start = tree.cind[self.row1[i], 0]
                r_end = tree.cind[self.row1[i], 1] + 1
                d[r_start:r_end] = np.diag(self.val[i])

        # Convert to particle indices
        return d[tree.ind[:, 1]]

    def eye_hmat(self) -> 'HMatrix':

        # MATLAB: @hmatrix/eye.m
        result = self._copy()

        # Clear all blocks
        result.val = [None] * len(result.row1)
        result.lhs = [None] * len(result.row2)
        result.rhs = [None] * len(result.row2)

        # Pad with zeros
        result.pad()

        # Set diagonal dense blocks to identity
        for i in range(len(result.row1)):
            if result.row1[i] == result.col1[i]:
                result.val[i] = np.eye(result.val[i].shape[0], result.val[i].shape[1])

        return result

    def pad(self) -> 'HMatrix':

        # MATLAB: @hmatrix/pad.m
        tree = self.tree
        siz = tree.cind[:, 1] - tree.cind[:, 0] + 1

        # Detect dtype from existing blocks
        pad_dtype = np.float64
        for v in self.val:
            if v is not None and np.iscomplexobj(v):
                pad_dtype = np.complex128
                break
        if pad_dtype == np.float64:
            for l in self.lhs:
                if l is not None and np.iscomplexobj(l):
                    pad_dtype = np.complex128
                    break

        for i in range(len(self.val)):
            if self.val[i] is None:
                m = siz[self.row1[i]]
                n_col = siz[self.col1[i]]
                self.val[i] = np.zeros((m, n_col), dtype = pad_dtype)

        for i in range(len(self.lhs)):
            if self.lhs[i] is None:
                m = siz[self.row2[i]]
                self.lhs[i] = np.zeros((m, 1), dtype = pad_dtype)
            if self.rhs[i] is None:
                n_col = siz[self.col2[i]]
                self.rhs[i] = np.zeros((n_col, 1), dtype = pad_dtype)

        return self

    def fillval(self, fun: Callable) -> 'HMatrix':

        # MATLAB: @hmatrix/fillval.m
        # Fill dense blocks with function values
        tree = self.tree
        ind_c2p = tree.ind[:, 0]

        def fun2(row_c: np.ndarray, col_c: np.ndarray) -> np.ndarray:
            return fun(ind_c2p[row_c], ind_c2p[col_c])

        for i in range(len(self.row1)):
            indr = tree.cind[self.row1[i]]
            indc = tree.cind[self.col1[i]]
            rows = np.arange(indr[0], indr[1] + 1, dtype = np.int64)
            cols = np.arange(indc[0], indc[1] + 1, dtype = np.int64)
            row_grid, col_grid = np.meshgrid(rows, cols, indexing = 'ij')
            self.val[i] = fun2(row_grid.ravel(), col_grid.ravel()).reshape(row_grid.shape)

        return self

    def lu(self) -> 'HMatrix':

        # MATLAB: @hmatrix/lu.m
        # Approximate LU decomposition for H-matrix
        # Pure Python implementation using recursive block LU
        # For simplicity, convert to dense, factorize, then store back
        mat = self.full()
        from scipy.linalg import lu as scipy_lu
        P, L, U = scipy_lu(mat)

        # Store LU as a combined matrix (L has unit diagonal, store L-I+U)
        # We store the factored result so solve() can use it
        self._lu_P = P
        self._lu_L = L
        self._lu_U = U
        self._lu_done = True

        return self

    def solve(self, b: np.ndarray) -> np.ndarray:

        # MATLAB: @hmatrix/solve.m
        # Solve A*x = b using LU factored H-matrix
        if hasattr(self, '_lu_done') and self._lu_done:
            # Use stored LU factors
            from scipy.linalg import solve_triangular
            # P @ L @ U @ x = b
            # L @ U @ x = P.T @ b
            pb = self._lu_P.T @ b
            y = solve_triangular(self._lu_L, pb, lower = True)
            x = solve_triangular(self._lu_U, y, lower = False)
            return x
        else:
            # Direct solve using dense conversion
            mat = self.full()
            return np.linalg.solve(mat, b)

    def _mtimes_hmat(self, other: 'HMatrix') -> 'HMatrix':

        # Simplified H-matrix * H-matrix multiplication via dense conversion
        # Full H-matrix arithmetic is extremely complex; this provides correctness
        mat1 = self.full()
        mat2 = other.full()
        result_mat = mat1 @ mat2

        # Build a new H-matrix from the result
        result = HMatrix(tree = self.tree, htol = self.htol, kmax = self.kmax)
        result.row1 = self.row1.copy()
        result.col1 = self.col1.copy()
        result.row2 = self.row2.copy()
        result.col2 = self.col2.copy()
        result.val = [None] * len(result.row1)
        result.lhs = [None] * len(result.row2)
        result.rhs = [None] * len(result.row2)

        # Fill from the dense result using ACA-like approach
        def mat_fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            return result_mat[row, col]

        result.aca(mat_fun)
        return result

    def _copy(self) -> 'HMatrix':

        result = HMatrix.__new__(HMatrix)
        result.tree = self.tree
        result.htol = self.htol
        result.kmax = self.kmax
        result.row1 = self.row1.copy()
        result.col1 = self.col1.copy()
        result.row2 = self.row2.copy()
        result.col2 = self.col2.copy()
        result.val = [v.copy() if v is not None else None for v in self.val]
        result.lhs = [l.copy() if l is not None else None for l in self.lhs]
        result.rhs = [r.copy() if r is not None else None for r in self.rhs]
        return result

    @staticmethod
    def from_func(tree: ClusterTree,
            fun: Callable,
            htol: float = 1e-6,
            kmax: int = 100,
            fadmiss: Optional[Callable] = None) -> 'HMatrix':

        hmat = HMatrix(tree = tree, htol = htol, kmax = kmax, fadmiss = fadmiss)
        hmat.aca(fun)
        return hmat
