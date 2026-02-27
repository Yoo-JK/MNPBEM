import numpy as np
from typing import Tuple, Optional, Union, List, Any
from .polygon import Polygon
from .edgeprofile import EdgeProfile
from .particle import Particle
from .mesh_generators import fvgrid, _add_midpoints_flat


class Polygon3(object):

    # MATLAB: @polygon3/polygon3.m + @polygon3/init.m

    def __init__(self,
            poly: Polygon,
            z: float,
            edge: Optional[EdgeProfile] = None,
            refun: Optional[Any] = None):

        self.poly = poly.copy()
        self.z = z
        self.edge = edge if edge is not None else EdgeProfile()
        self._refun = refun

    def __repr__(self) -> str:
        return 'Polygon3(z={}, poly={}, edge={})'.format(self.z, self.poly, self.edge)

    def set(self, **kwargs) -> 'Polygon3':
        # MATLAB: @polygon3/set.m
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            elif hasattr(self.poly, key):
                setattr(self.poly, key, val)
        return self

    def flip(self, axis: int) -> 'Polygon3':
        # MATLAB: @polygon3/flip.m
        self.poly = self.poly.flip(axis)
        return self

    def shift(self, vec: np.ndarray) -> 'Polygon3':
        # MATLAB: @polygon3/shift.m
        vec = np.asarray(vec, dtype = float)
        self.poly = self.poly.shift(vec[:2])
        self.z = self.z + vec[2]
        return self

    def shiftbnd(self, dist: float) -> 'Polygon3':
        # MATLAB: @polygon3/shiftbnd.m
        self.poly = self.poly.shiftbnd(dist)
        self.edge = EdgeProfile()
        return self

    def copy(self) -> 'Polygon3':
        import copy
        return copy.deepcopy(self)

    def plate(self,
            dir: int = 1,
            edge: Optional[EdgeProfile] = None,
            hdata: Optional[dict] = None,
            options: Optional[dict] = None,
            refun: Optional[Any] = None) -> Tuple[Particle, 'Polygon3']:
        # MATLAB: @polygon3/plate.m
        # Simplified plate: triangulate polygon and create flat plate at z-level
        if edge is not None:
            self.edge = edge

        if options is None:
            options = {'output': False}
        if hdata is None:
            hdata = {}

        # triangulate the 2D polygon
        poly_for_mesh = self.poly.copy()

        # get full polygon for meshing
        full_pos = poly_for_mesh.get_full_polygon()
        mesh_poly = Polygon(full_pos)

        verts_2d, faces_2d = mesh_poly.polymesh2d(hdata = hdata, options = options)

        # create 3D vertices at the plate z-level
        z_col = np.full((verts_2d.shape[0], 1), self.z)
        verts_3d = np.empty((verts_2d.shape[0], 3))
        verts_3d[:, :2] = verts_2d
        verts_3d[:, 2] = self.z

        # create particle
        p = Particle(verts_3d, faces_2d)

        # add midpoints (flat)
        p = _add_midpoints_flat(p)

        # apply edge profile vertical shift to boundary vertices
        if self.edge is not None and self.edge.pos is not None:
            d_vals, _ = self.poly.dist(p.verts2[:, :2])
            vshift_vals = self.edge.vshift(self.z, d_vals)
            if not np.isscalar(vshift_vals) or vshift_vals != 0.0:
                p.verts2[:, 2] = p.verts2[:, 2] + vshift_vals

        # create final particle with midpoints
        p = Particle(p.verts2, p.faces2)

        # check normal direction and flip if needed
        nvec_sum = np.sum(p.nvec[:, 2])
        if np.sign(nvec_sum) != dir:
            p = p.flipfaces()

        # update polygon z-value
        result_poly3 = self.copy()

        return p, result_poly3

    def vribbon(self,
            z: Optional[np.ndarray] = None,
            edge: Optional[EdgeProfile] = None) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        # MATLAB: @polygon3/vribbon.m
        if edge is not None:
            self.edge = edge

        if z is None:
            if self.edge is not None and self.edge.z is not None:
                z = self.edge.z.copy()
            else:
                assert False, '[error] z-values required for vribbon'

        # edge profile horizontal shift function
        def hshift_fun(z_vals: np.ndarray) -> np.ndarray:
            return self.edge.hshift(z_vals)

        p, up, lo = self._ribbon_v(z, hshift_fun)
        return p, up, lo

    def _ribbon_v(self,
            z: np.ndarray,
            hshift_fun: Any) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        # MATLAB: ribbon() subfunction in vribbon.m

        # smoothened polygon with midpoints
        poly_smooth = self.poly.copy().midpoints()

        pos = poly_smooth.pos
        nvec = poly_smooth.compute_normals()

        # close contour: append first point
        pos_closed = np.empty((pos.shape[0] + 1, 2))
        pos_closed[:pos.shape[0]] = pos
        pos_closed[pos.shape[0]] = pos[0]

        nvec_closed = np.empty((nvec.shape[0] + 1, 2))
        nvec_closed[:nvec.shape[0]] = nvec
        nvec_closed[nvec.shape[0]] = nvec[0]

        # extend z-values for midpoints (interleave with averages)
        n_z = len(z)
        z_ext = np.empty(2 * n_z - 1)
        z_ext[0::2] = z
        z_ext[1::2] = 0.5 * (z[:-1] + z[1:])

        # create grid indices: odd positions along polygon, odd along z
        poly_indices = np.arange(0, pos_closed.shape[0], 2)  # 0, 2, 4, ...
        z_indices = np.arange(0, len(z_ext), 2)  # 0, 2, 4, ...

        u, faces = fvgrid(poly_indices.astype(float), z_indices.astype(float))

        # u[:, 0] -> polygon position index, u[:, 1] -> z index
        u_int = u.astype(int)

        # build 3D vertices
        n_verts = u_int.shape[0]
        verts = np.empty((n_verts, 3))
        verts[:, 0] = pos_closed[u_int[:, 0], 0]
        verts[:, 1] = pos_closed[u_int[:, 0], 1]
        verts[:, 2] = z_ext[u_int[:, 1]]

        # apply horizontal shift from edge profile
        shift_vals = hshift_fun(verts[:, 2])
        verts[:, 0] = verts[:, 0] + shift_vals * nvec_closed[u_int[:, 0], 0]
        verts[:, 1] = verts[:, 1] + shift_vals * nvec_closed[u_int[:, 0], 1]

        # create particle
        p = Particle(verts, faces)

        # check normal direction: should point outward
        # find point closest to first polygon point
        dx = p.pos[:, 0] - pos[0, 0]
        dy = p.pos[:, 1] - pos[0, 1]
        ind = np.argmin(dx ** 2 + dy ** 2)

        # reference normal direction
        ref_vec = np.array([nvec[0, 0], nvec[0, 1], 0.0])
        if np.dot(ref_vec, p.nvec[ind]) < 0:
            p = p.flipfaces()

        # boundary polygons for upper and lower
        def _shifted_poly(z_val: float) -> Polygon:
            shift_val = hshift_fun(np.array([z_val]))[0]
            shifted = self.poly.copy().shiftbnd(shift_val)
            return shifted

        up = self.copy()
        up.poly = _shifted_poly(np.max(z))
        up.z = np.max(z)

        lo = self.copy()
        lo.poly = _shifted_poly(np.min(z))
        lo.z = np.min(z)

        return p, up, lo

    def hribbon(self,
            d: np.ndarray,
            dir: int = 1) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        # MATLAB: @polygon3/hribbon.m
        p, inner, outer = self._ribbon_h(d, dir)
        return p, inner, outer

    def _ribbon_h(self,
            d: np.ndarray,
            dir: int) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        # MATLAB: ribbon() subfunction in hribbon.m

        # smoothened polygon with midpoints
        poly_smooth = self.poly.copy().midpoints()

        pos = poly_smooth.pos
        nvec = poly_smooth.compute_normals()

        # close contour: append first point
        pos_closed = np.empty((pos.shape[0] + 1, 2))
        pos_closed[:pos.shape[0]] = pos
        pos_closed[pos.shape[0]] = pos[0]

        nvec_closed = np.empty((nvec.shape[0] + 1, 2))
        nvec_closed[:nvec.shape[0]] = nvec
        nvec_closed[nvec.shape[0]] = nvec[0]

        # extend d-values for midpoints
        d = np.asarray(d, dtype = float)
        n_d = len(d)
        d_ext = np.empty(2 * n_d - 1)
        d_ext[0::2] = d
        d_ext[1::2] = 0.5 * (d[:-1] + d[1:])

        # grid indices
        poly_indices = np.arange(0, pos_closed.shape[0], 2)
        d_indices = np.arange(0, len(d_ext), 2)

        u, faces = fvgrid(poly_indices.astype(float), d_indices.astype(float))
        u_int = u.astype(int)

        # compute displaced positions for each d-value
        n_pos = pos_closed.shape[0]
        n_d_ext = len(d_ext)
        x = np.zeros((n_pos, n_d_ext))
        y = np.zeros((n_pos, n_d_ext))

        for i in range(n_d_ext):
            # shift boundary by d_ext[i]
            poly_temp = poly_smooth.copy()
            _, distp = poly_temp.shiftbnd(d_ext[i], return_dist = True)

            # handle closed polygons: need (n_pos) values including closing point
            if len(distp) != n_pos:
                dist_full = np.empty(n_pos)
                dist_full[:len(distp)] = distp
                dist_full[len(distp):] = distp[0]
            else:
                dist_full = distp

            x[:, i] = pos_closed[:, 0] + dist_full * nvec_closed[:, 0]
            y[:, i] = pos_closed[:, 1] + dist_full * nvec_closed[:, 1]

        # assemble vertices from grid
        n_verts = u_int.shape[0]
        verts_x = np.empty(n_verts)
        verts_y = np.empty(n_verts)
        for k in range(n_verts):
            verts_x[k] = x[u_int[k, 0], u_int[k, 1]]
            verts_y[k] = y[u_int[k, 0], u_int[k, 1]]

        verts = np.empty((n_verts, 3))
        verts[:, 0] = verts_x
        verts[:, 1] = verts_y
        verts[:, 2] = self.z

        # create particle
        p = Particle(verts, faces)

        # check normal direction
        nvec_sum = np.sum(p.nvec[:, 2])
        if np.sign(nvec_sum) != dir:
            p = p.flipfaces()

        # inner/outer boundary polygons
        inner = self.copy()
        inner.poly = self.poly.copy().shiftbnd(np.min(d))
        inner.z = self.z

        outer = self.copy()
        outer.poly = self.poly.copy().shiftbnd(np.max(d))
        outer.z = self.z

        return p, inner, outer
