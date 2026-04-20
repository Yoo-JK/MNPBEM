import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .greentab_layer import GreenTabLayer
from .refine_utils import refinematrixlayer


class GreenRetLayer(object):

    name = 'greenretlayer'

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            tab: Optional[Dict[str, Any]] = None,
            deriv: str = 'cart',
            **options: Any) -> None:

        self.p1 = p1
        self.p2 = p2
        self.layer = layer
        self.deriv = deriv
        self.enei = None

        self.G = None
        self.F = None
        self.Gp = None

        # Tabulated Green function
        if tab is not None:
            self.tab = GreenTabLayer(layer, tab = tab)
        else:
            self.tab = GreenTabLayer(layer)

        # Compute positions and distances for reflected Green function
        self._init_positions()

        # Compute refinement indices
        # MATLAB: init.m -> refinematrixlayer, then init1/init2 for off-diagonal
        self._init_refinement(**options)

    def _init_positions(self) -> None:

        pos1 = self.p1.pos
        pos2 = self.p2.pos

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Radial distance between face pairs
        dx = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        dy = pos1[:, 1:2] - pos2[:, 1:2].T  # (n1, n2)

        self._r = np.sqrt(dx ** 2 + dy ** 2)  # (n1, n2)
        self._z1 = pos1[:, 2]  # (n1,)
        self._z2 = pos2[:, 2]  # (n2,)
        self._dx = dx
        self._dy = dy

    def _init_refinement(self, **options) -> None:
        """Identify diagonal and off-diagonal elements needing refinement.

        MATLAB: @greenretlayer/private/init.m
        """
        RelCutoff = options.get('RelCutoff', 0)
        AbsCutoff = options.get('AbsCutoff', 0)

        # MATLAB: always call refinematrixlayer; defaults (0,0) still select
        # near-surface elements (d2<0 or id<0 in effective comparison).
        # DO NOT short-circuit — was causing 752% error for d<1nm substrates.

        # Compute refinement matrix
        ir = refinematrixlayer(self.p1, self.p2, self.layer,
                AbsCutoff = AbsCutoff, RelCutoff = RelCutoff)

        ir_array = ir.toarray()

        # Linear indices (row-major) of diagonal elements (ir==2)
        diag_rows, diag_cols = np.where(ir_array == 2)
        if len(diag_rows) > 0:
            self._diag_id = np.ravel_multi_index(
                (diag_rows, diag_cols),
                (self.p1.n, self.p2.n))
            self._diag_faces = diag_rows  # Face indices for diagonal elements
        else:
            self._diag_id = np.array([], dtype = int)
            self._diag_faces = np.array([], dtype = int)

        # Linear indices of off-diagonal refinement elements (ir==1)
        offdiag_rows, offdiag_cols = np.where(ir_array == 1)
        if len(offdiag_rows) > 0:
            self._offdiag_ind = np.ravel_multi_index(
                (offdiag_rows, offdiag_cols),
                (self.p1.n, self.p2.n))
        else:
            self._offdiag_ind = np.array([], dtype = int)

    # -----------------------------------------------------------------
    #  Helper: interpolate tab + return r_rounded and zmin
    # -----------------------------------------------------------------
    def _interp_components_with_pos(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray
            ) -> Tuple[Dict, Dict, Dict, np.ndarray, np.ndarray]:
        """Per-component interpolation returning also r_rounded and zmin.

        Mirrors MATLAB greentablayer/interp.m 5-output mode:
        [G, Fr, Fz, r, zmin].
        """
        layer = self.layer

        r = np.maximum(r, layer.rmin)
        z1_r, z2_r = layer.round_z(z1, z2)

        zmin1, _ = layer.mindist(z1_r)
        zmin2, _ = layer.mindist(z2_r)
        zmin = zmin1 + zmin2

        G_dict, Fr_dict, Fz_dict = self.tab.eval_components(
            self.enei, r, z1_r, z2_r)

        return G_dict, Fr_dict, Fz_dict, r, zmin

    # -----------------------------------------------------------------
    #  Divergent term coefficient (Waxenegger et al. Eq. 17)
    # -----------------------------------------------------------------
    def _compute_divergent_coeff(self,
            face_z: np.ndarray
            ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Compute the divergent term coefficient f for faces in the layer.

        MATLAB initrefl1.m lines 89-98:
          [~, ind] = mindist(layer, p.pos(id(lin), 3));
          z0 = layer.z(ind);
          dir = sign(p.pos(id(lin), 3) - z0);
          [~, ~, f0, r0, z0] = interp(tab, 0*z0, z0+1e-10*dir, z0+1e-10*dir);
          f = f0 .* (r0.^2 + z0.^2).^1.5 ./ z0;

        Parameters
        ----------
        face_z : ndarray
            z-coordinates of faces located in the layer.

        Returns
        -------
        f_dict : dict of ndarray
            Per-component divergent coefficient.
        zmin0 : ndarray
            zmin values at the probe points.
        """
        layer = self.layer

        _, ind = layer.mindist(face_z)
        z0_layer = layer.z[ind - 1]  # 1-based ind -> 0-based

        direction = np.sign(face_z - z0_layer)
        direction[direction == 0] = 1.0

        z_probe = z0_layer + 1e-10 * direction
        r_zero = np.zeros_like(z_probe)

        # Get per-component values near the singularity
        _, _, Fz0_dict, r0, zmin0 = self._interp_components_with_pos(
            r_zero, z_probe, z_probe)

        d0 = np.sqrt(r0 ** 2 + zmin0 ** 2)

        f_dict = {}
        for name in Fz0_dict:
            # f = f0 * d^3 / zmin  (extract normalized coefficient)
            f_dict[name] = Fz0_dict[name] * d0 ** 3 / np.maximum(np.abs(zmin0), 1e-30)

        return f_dict, zmin0

    # -----------------------------------------------------------------
    #  Diagonal refinement (polar integration)
    # -----------------------------------------------------------------
    def _refine_diagonal_norm(self,
            G_dict: Dict[str, np.ndarray],
            F_dict: Dict[str, np.ndarray]
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Refine diagonal elements for deriv='norm'.

        MATLAB: initrefl1.m lines 44-131
        """
        if len(self._diag_id) == 0:
            return G_dict, F_dict

        n1 = self.p1.n
        n2 = self.p2.n
        p = self.p1
        id_linear = self._diag_id
        id_faces = self._diag_faces

        # Polar integration points and weights
        pos_quad, weight_quad, row_quad = p.quadpol(id_faces)

        # Centroids expanded to integration point size
        pos_centroid = p.pos[id_faces[row_quad]]

        # Difference vectors
        dx = pos_quad[:, 0] - pos_centroid[:, 0]
        dy = pos_quad[:, 1] - pos_centroid[:, 1]
        r_quad = np.sqrt(dx ** 2 + dy ** 2)
        z1_quad = pos_quad[:, 2]
        z2_quad = pos_centroid[:, 2]

        # Normal vectors
        nvec_quad = p.nvec[id_faces[row_quad]]
        r_safe = np.maximum(r_quad, 1e-10)
        in_product = (dx * nvec_quad[:, 0] + dy * nvec_quad[:, 1]) / r_safe

        # Interpolate Green function at quadrature points
        g_dict, fr_dict, fz_dict, r_rounded, zmin_quad = \
            self._interp_components_with_pos(r_quad, z1_quad, z2_quad)
        rr_quad = np.sqrt(r_rounded ** 2 + zmin_quad ** 2)

        # Identify faces located in the layer (close to interface)
        _, lin = self.layer.indlayer(p.pos[id_faces, 2])

        # Expand to integration point size
        lin_row = lin[row_quad]

        # Compute divergent coefficient for in-layer faces
        if np.any(lin):
            lin_faces_z = p.pos[id_faces[lin], 2]
            f_coeff, _ = self._compute_divergent_coeff(lin_faces_z)

            lin_map = np.full(len(id_faces), -1, dtype = int)
            lin_map[lin] = np.arange(np.sum(lin))
            irow = lin_map[row_quad[lin_row]]

            # Linear index for in-layer diagonal matrix entries
            lin2_row = id_faces[lin]
            lin2_col = id_faces[lin]
            lin2_linear = np.ravel_multi_index((lin2_row, lin2_col), (n1, n2))
        else:
            f_coeff = None

        names = list(g_dict.keys())
        n_faces_diag = len(id_faces)

        for name in names:
            g_vals = g_dict[name]
            fr_vals = fr_dict[name]
            fz_vals = fz_dict[name].copy()

            # Integrate G using polar quadrature
            G_refined = np.zeros(n_faces_diag, dtype = complex)
            np.add.at(G_refined, row_quad, weight_quad * g_vals)
            G_flat = G_dict[name].ravel()
            G_flat[id_linear] = G_refined
            G_dict[name] = G_flat.reshape(n1, n2)

            # Subtract divergent term from fz for in-layer faces
            if np.any(lin) and f_coeff is not None:
                f = f_coeff[name]
                rr_safe = np.maximum(rr_quad[lin_row], 1e-30)
                fz_vals[lin_row] -= f[irow] * zmin_quad[lin_row] / rr_safe ** 3

            # Integrate F using polar quadrature
            f_integrand = fr_vals * in_product + fz_vals * nvec_quad[:, 2]
            F_refined = np.zeros(n_faces_diag, dtype = complex)
            np.add.at(F_refined, row_quad, weight_quad * f_integrand)

            F_flat = F_dict[name].ravel()
            F_flat[id_linear] = F_refined

            # Add back divergent term
            if np.any(lin) and f_coeff is not None:
                f = f_coeff[name]
                F_flat[lin2_linear] += 2 * np.pi * f * p.nvec[id_faces[lin], 2]

            F_dict[name] = F_flat.reshape(n1, n2)

        return G_dict, F_dict

    def _refine_diagonal_cart_gp_only(self,
            Gp_dict: Dict[str, np.ndarray]) -> None:
        """Refine only the Gp (Cartesian derivative) diagonal elements.

        MATLAB: initrefl2.m lines 43-129, Gp parts only.
        """
        if len(self._diag_id) == 0:
            return

        n1 = self.p1.n
        n2 = self.p2.n
        p = self.p1
        id_faces = self._diag_faces

        pos_quad, weight_quad, row_quad = p.quadpol(id_faces)
        pos_centroid = p.pos[id_faces[row_quad]]

        dx = pos_quad[:, 0] - pos_centroid[:, 0]
        dy = pos_quad[:, 1] - pos_centroid[:, 1]
        r_quad = np.sqrt(dx ** 2 + dy ** 2)
        z1_quad = pos_quad[:, 2]
        z2_quad = pos_centroid[:, 2]

        _, fr_dict, fz_dict, r_rounded, zmin_quad = \
            self._interp_components_with_pos(r_quad, z1_quad, z2_quad)
        rr_quad = np.sqrt(r_rounded ** 2 + zmin_quad ** 2)
        r_safe = np.maximum(r_rounded, 1e-10)

        _, lin = self.layer.indlayer(p.pos[id_faces, 2])
        lin_row = lin[row_quad]

        if np.any(lin):
            lin_faces_z = p.pos[id_faces[lin], 2]
            f_coeff, _ = self._compute_divergent_coeff(lin_faces_z)
            lin_map = np.full(len(id_faces), -1, dtype = int)
            lin_map[lin] = np.arange(np.sum(lin))
            irow = lin_map[row_quad[lin_row]]
        else:
            f_coeff = None

        names = list(Gp_dict.keys())
        n_faces_diag = len(id_faces)

        for name in names:
            fr_vals = fr_dict[name]
            fz_vals = fz_dict[name].copy()

            # Subtract divergent term
            if np.any(lin) and f_coeff is not None:
                f = f_coeff[name]
                rr_safe = np.maximum(rr_quad[lin_row], 1e-30)
                fz_vals[lin_row] -= f[irow] * zmin_quad[lin_row] / rr_safe ** 3

            Gp_x_refined = np.zeros(n_faces_diag, dtype = complex)
            Gp_y_refined = np.zeros(n_faces_diag, dtype = complex)
            Gp_z_refined = np.zeros(n_faces_diag, dtype = complex)
            np.add.at(Gp_x_refined, row_quad, weight_quad * fr_vals * dx / r_safe)
            np.add.at(Gp_y_refined, row_quad, weight_quad * fr_vals * dy / r_safe)
            np.add.at(Gp_z_refined, row_quad, weight_quad * fz_vals)

            Gp = Gp_dict[name]
            Gp[id_faces, 0, id_faces] = Gp_x_refined
            Gp[id_faces, 1, id_faces] = Gp_y_refined
            Gp[id_faces, 2, id_faces] = Gp_z_refined

            # Add back divergent term for z-component
            if np.any(lin) and f_coeff is not None:
                f = f_coeff[name]
                lin_faces_idx = id_faces[lin]
                Gp[lin_faces_idx, 2, lin_faces_idx] += 2 * np.pi * f

            Gp_dict[name] = Gp

    # -----------------------------------------------------------------
    #  Off-diagonal refinement (boundary element integration)
    # -----------------------------------------------------------------
    def _refine_offdiagonal_components(self) -> None:
        """Refine off-diagonal elements for all components (G, F, Gp).

        MATLAB: initrefl1.m lines 134-186 + initrefl2.m lines 132-190
        Uses full boundary element integration over source faces.
        """
        if len(self._offdiag_ind) == 0:
            return

        n1 = self.p1.n
        n2 = self.p2.n
        names = list(self.G_comp.keys())

        ind_matrix = np.zeros((n1, n2), dtype = int)
        offdiag_rows, offdiag_cols = np.unravel_index(
            self._offdiag_ind, (n1, n2))
        ind_matrix[offdiag_rows, offdiag_cols] = 1

        columns_with_refine = np.unique(offdiag_cols)

        for col in columns_with_refine:
            rows = np.where(ind_matrix[:, col])[0]
            if len(rows) == 0:
                continue

            pos1 = self.p1.pos[rows]
            pos_quad, w_sparse, _ = _particle_quad(self.p2, np.array([col]))

            _, nz_cols, nz_vals = _sparse_find(w_sparse)
            pos2 = pos_quad[nz_cols]
            w = nz_vals

            if len(w) == 0:
                continue

            x = pos1[:, 0:1] - pos2[:, 0].T
            y = pos1[:, 1:2] - pos2[:, 1].T
            r = np.sqrt(x ** 2 + y ** 2)

            z1 = np.tile(pos1[:, 2:3], (1, len(w)))
            z2 = np.tile(pos2[:, 2].T, (len(rows), 1))

            nvec = self.p1.nvec[rows]
            r_safe = np.maximum(r, 1e-10)
            in_product = (nvec[:, 0:1] * x + nvec[:, 1:2] * y) / r_safe

            g_dict_q, fr_dict_q, fz_dict_q = self.tab.eval_components(
                self.enei,
                r.ravel(),
                z1.ravel(),
                z2.ravel())

            for name in names:
                g_q = g_dict_q[name].reshape(r.shape)
                fr_q = fr_dict_q[name].reshape(r.shape)
                fz_q = fz_dict_q[name].reshape(r.shape)

                # Refine G
                self.G_comp[name][rows, col] = g_q @ w

                # Refine F (normal derivative)
                f_integrand = fr_q * in_product + fz_q * nvec[:, 2:3]
                self.F_comp[name][rows, col] = f_integrand @ w

                # Refine Gp (Cartesian derivative)
                self.Gp_comp[name][rows, 0, col] = (fr_q * x / r_safe) @ w
                self.Gp_comp[name][rows, 1, col] = (fr_q * y / r_safe) @ w
                self.Gp_comp[name][rows, 2, col] = fz_q @ w

    # -----------------------------------------------------------------
    #  Apply refinement to per-component Green functions
    # -----------------------------------------------------------------
    def _apply_refinement_components(self) -> None:
        """Apply diagonal and off-diagonal refinement to per-component Green functions.

        Handles both norm and cart derivatives simultaneously since
        eval_components always needs both F_comp and Gp_comp.
        """
        # Diagonal refinement
        if len(self._diag_id) > 0:
            # Refine G_comp and F_comp (norm-style: initrefl1.m)
            self.G_comp, self.F_comp = self._refine_diagonal_norm(
                self.G_comp, self.F_comp)
            # Refine Gp_comp (cart-style: initrefl2.m)
            self._refine_diagonal_cart_gp_only(self.Gp_comp)

        # Off-diagonal refinement
        if len(self._offdiag_ind) > 0:
            self._refine_offdiagonal_components()

    # -----------------------------------------------------------------
    #  Main evaluation methods
    # -----------------------------------------------------------------
    def eval(self,
            enei: float) -> None:

        if self.enei is not None and np.isclose(self.enei, enei):
            return

        self.enei = enei

        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        # Round z-values to avoid being too close to layer interface
        z1, z2 = self.layer.round_z(self._z1, self._z2)

        r_flat = self._r.ravel()
        z1_flat = np.repeat(z1, n2)
        z2_flat = np.tile(z2, n1)

        # Enforce minimum radial distance
        r_flat = np.maximum(r_flat, self.layer.rmin)

        # Compute reflected Green function
        G, Fr, Fz = self.tab.eval(enei, r_flat, z1_flat, z2_flat)

        G = G.reshape(n1, n2)
        Fr = Fr.reshape(n1, n2)
        Fz = Fz.reshape(n1, n2)

        # Multiply with p2.area (MATLAB initrefl1.m line 36, initrefl2.m line 35)
        area2 = self.p2.area  # (n2,)
        G = G * area2[np.newaxis, :]
        Fr = Fr * area2[np.newaxis, :]
        Fz = Fz * area2[np.newaxis, :]

        # Store Green function
        self.G = G

        # Compute surface derivative
        if self.deriv == 'norm':
            self._compute_F_norm(G, Fr, Fz)
        else:
            self._compute_F_cart(G, Fr, Fz)

    def eval_components(self,
            enei: float) -> None:
        """Compute per-component reflected Green function (G, F, Gp).

        Always computes both normal and Cartesian derivatives regardless
        of self.deriv, since field() needs Gp and potential() needs F.

        After calling, results are stored in:
          self.G_comp  : dict of (n1, n2) arrays
          self.F_comp  : dict of (n1, n2) arrays (normal derivative)
          self.Gp_comp : dict of (n1, n2, 3) arrays (Cartesian derivative)
        """
        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        z1, z2 = self.layer.round_z(self._z1, self._z2)

        r_flat = self._r.ravel()
        z1_flat = np.repeat(z1, n2)
        z2_flat = np.tile(z2, n1)
        r_flat = np.maximum(r_flat, self.layer.rmin)

        G_dict, Fr_dict, Fz_dict = self.tab.eval_components(
            enei, r_flat, z1_flat, z2_flat)

        self.enei = enei
        self.G_comp = {}
        self.F_comp = {}
        self.Gp_comp = {}

        nvec1 = self.p1.nvec
        r_safe = np.maximum(self._r, np.finfo(float).eps)
        area2 = self.p2.area  # (n2,)

        for name in G_dict:
            G = G_dict[name].reshape(n1, n2)
            Fr = Fr_dict[name].reshape(n1, n2)
            Fz = Fz_dict[name].reshape(n1, n2)

            # Multiply with p2.area (MATLAB initrefl1.m line 36, initrefl2.m line 35)
            G = G * area2[np.newaxis, :]
            Fr = Fr * area2[np.newaxis, :]
            Fz = Fz * area2[np.newaxis, :]

            self.G_comp[name] = G

            # Cartesian derivative: Gp (n1, 3, n2) -- matches MATLAB shape
            Gp = np.zeros((n1, 3, n2), dtype = complex)
            Gp[:, 0, :] = Fr * self._dx / r_safe
            Gp[:, 1, :] = Fr * self._dy / r_safe
            Gp[:, 2, :] = Fz
            self.Gp_comp[name] = Gp

            # Normal derivative: F = nvec . Gp  (inner product)
            F = np.zeros((n1, n2), dtype = complex)
            F += nvec1[:, 0:1] * Gp[:, 0, :]
            F += nvec1[:, 1:2] * Gp[:, 1, :]
            F += nvec1[:, 2:3] * Gp[:, 2, :]
            self.F_comp[name] = F

        # Apply refinement if configured
        has_refinement = (len(self._diag_id) > 0 or len(self._offdiag_ind) > 0)
        if has_refinement:
            self._apply_refinement_components()

    # -----------------------------------------------------------------
    #  Surface derivative computation (unrefined, for eval())
    # -----------------------------------------------------------------
    def _compute_F_norm(self,
            G: np.ndarray,
            Fr: np.ndarray,
            Fz: np.ndarray) -> None:

        nvec1 = self.p1.nvec
        n1 = nvec1.shape[0]
        n2 = self.p2.pos.shape[0]

        r_safe = np.maximum(self._r, np.finfo(float).eps)

        # Normal derivative: F = nvec_x * Fr * dx/r + nvec_y * Fr * dy/r + nvec_z * Fz
        F = np.zeros((n1, n2), dtype = complex)
        F += nvec1[:, 0:1] * Fr * self._dx / r_safe
        F += nvec1[:, 1:2] * Fr * self._dy / r_safe
        F += nvec1[:, 2:3] * Fz

        self.F = F

    def _compute_F_cart(self,
            G: np.ndarray,
            Fr: np.ndarray,
            Fz: np.ndarray) -> None:
        """Compute Cartesian derivative Gp and normal derivative F.

        MATLAB: initrefl2.m
        - Gp is the 3D Cartesian derivative, shape (n1, 3, n2)
        - F is the 2D normal derivative: F[i,j] = nvec[i] . Gp[i,:,j]
        """

        nvec1 = self.p1.nvec
        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        r_safe = np.maximum(self._r, np.finfo(float).eps)

        # Cartesian derivative: Gp[:, 0, :] = Fr * dx/r, etc.
        # Shape (n1, 3, n2) to match MATLAB convention
        Gp = np.zeros((n1, 3, n2), dtype = complex)
        Gp[:, 0, :] = Fr * self._dx / r_safe
        Gp[:, 1, :] = Fr * self._dy / r_safe
        Gp[:, 2, :] = Fz

        # Normal derivative: F = inner(nvec, Gp)
        # F[i,j] = nvec[i,0]*Gp[i,0,j] + nvec[i,1]*Gp[i,1,j] + nvec[i,2]*Gp[i,2,j]
        F = np.einsum('ik,ikj->ij', nvec1, Gp)

        self.Gp = Gp
        self.F = F

    def setup_tabulation(self, nr = 30, nz = 20):

        z1, z2 = self.layer.round_z(self._z1, self._z2)

        # r: logarithmic (rmin -> max radial distance)
        r_max = max(self._r.max(), self.layer.rmin * 10)
        r_grid = np.geomspace(self.layer.rmin, r_max, nr)

        # z1, z2: linear (face z-coordinate range)
        z_all = np.concatenate([z1, z2])
        z_min, z_max = z_all.min(), z_all.max()
        if np.isclose(z_min, z_max):
            z_max = z_min + 1.0
        z1_grid = np.linspace(z_min, z_max, nz)
        z2_grid = np.linspace(z_min, z_max, nz)

        self.tab.setup_grid(r_grid, z1_grid, z2_grid)

    def __repr__(self) -> str:
        n_diag = len(self._diag_id) if hasattr(self, '_diag_id') else 0
        n_offdiag = len(self._offdiag_ind) if hasattr(self, '_offdiag_ind') else 0
        return 'GreenRetLayer(n1={}, n2={}, deriv={}, diag={}, offdiag={})'.format(
            self.p1.pos.shape[0], self.p2.pos.shape[0], self.deriv,
            n_diag, n_offdiag)


# =====================================================================
#  Module-level helpers
# =====================================================================

def _sparse_find(sparse_matrix):
    """Extract non-zero entries from a sparse matrix.

    Returns (rows, cols, values) like MATLAB's find().
    """
    from scipy.sparse import find as sp_find
    rows, cols, vals = sp_find(sparse_matrix)
    return rows, cols, vals


def _particle_quad(p, ind):
    """Call particle.quad() method avoiding name conflict with quad attribute.

    The Particle class has both a `quad` attribute (QuadFace data) and a
    `quad()` method.  Because instance attributes shadow methods, we call
    the private implementations directly.

    For ComParticle: use the concatenated `pc` (Particle) to match MATLAB
    @compound/quad() delegation behavior.

    Returns (pos, w_sparse, iface) matching MATLAB quad().
    """
    # ComParticle handling: delegate to concatenated Particle (pc)
    if hasattr(p, 'pc') and p.pc is not None and not hasattr(p, 'interp'):
        p = p.pc
    if p.interp == 'flat':
        return p._quad_flat(ind)
    else:
        return p._quad_curv(ind)
