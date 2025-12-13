"""
Particle class for discretized surfaces.

Matches MATLAB MNPBEM @particle implementation exactly.
"""

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags


class QuadFace:
    """
    Integration rules for triangular/quadrilateral boundary elements.

    Matches MATLAB @quadface class.

    Parameters
    ----------
    rule : int
        Integration rule (default: 3, 7-point Dunavant rule)
    npol : int
        Number of points for polar integration (default: 5)
    """

    def __init__(self, rule=3, npol=5):
        """Initialize quadrature rules."""
        self.npol = npol

        # Standard triangle integration (Dunavant rules)
        # Rule 3: 7-point rule (degree 5)
        if rule == 1:
            # 1-point rule (centroid)
            self.x = np.array([1/3])
            self.y = np.array([1/3])
            self.w = np.array([1.0])
        elif rule == 2:
            # 3-point rule
            self.x = np.array([1/6, 2/3, 1/6])
            self.y = np.array([1/6, 1/6, 2/3])
            self.w = np.array([1/3, 1/3, 1/3])
        else:  # rule == 3 (default)
            # 7-point rule (Dunavant)
            a = 0.470142064105115
            b = 0.059715871789770
            c = 0.101286507323456
            self.x = np.array([1/3, a, 1-2*a, a, b, 1-2*b, b])
            self.y = np.array([1/3, a, a, 1-2*a, b, b, 1-2*b])
            w1 = 0.225
            w2 = 0.132394152788506
            w3 = 0.125939180544827
            self.w = np.array([w1, w2, w2, w2, w3, w3, w3])

        # Polar integration for triangles (radial from centroid)
        self._init_polar_tri(npol)
        # Polar integration for quadrilaterals
        self._init_polar_quad(npol)

    def _init_polar_tri(self, npol):
        """Initialize polar integration for triangles."""
        # Gauss-Legendre quadrature points and weights
        from numpy.polynomial.legendre import leggauss
        r_pts, r_wts = leggauss(npol)
        # Transform from [-1,1] to [0,1]
        r_pts = 0.5 * (r_pts + 1)
        r_wts = 0.5 * r_wts

        # Angular divisions (3 corners of triangle)
        n_theta = 3 * npol
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

        # Create polar grid
        x_list, y_list, w_list = [], [], []
        centroid = np.array([1/3, 1/3])

        for i_theta in range(n_theta):
            th = theta[i_theta]
            # Direction from centroid
            direction = np.array([np.cos(th), np.sin(th)])

            # Find max radius to triangle edge
            r_max = self._max_radius_triangle(centroid, direction)

            for i_r, (r, wr) in enumerate(zip(r_pts, r_wts)):
                r_actual = r * r_max
                pt = centroid + r_actual * direction
                # Jacobian: r * dr * dtheta
                w = r_actual * r_max * wr * (2*np.pi / n_theta)
                x_list.append(pt[0])
                y_list.append(pt[1])
                w_list.append(w)

        self.x3 = np.array(x_list)
        self.y3 = np.array(y_list)
        self.w3 = np.array(w_list)

    def _init_polar_quad(self, npol):
        """Initialize polar integration for quadrilaterals."""
        from numpy.polynomial.legendre import leggauss
        r_pts, r_wts = leggauss(npol)
        r_pts = 0.5 * (r_pts + 1)
        r_wts = 0.5 * r_wts

        n_theta = 4 * npol
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

        x_list, y_list, w_list = [], [], []
        centroid = np.array([0, 0])  # Center of [-1,1]x[-1,1]

        for i_theta in range(n_theta):
            th = theta[i_theta]
            direction = np.array([np.cos(th), np.sin(th)])
            r_max = self._max_radius_quad(centroid, direction)

            for i_r, (r, wr) in enumerate(zip(r_pts, r_wts)):
                r_actual = r * r_max
                pt = centroid + r_actual * direction
                w = r_actual * r_max * wr * (2*np.pi / n_theta)
                x_list.append(pt[0])
                y_list.append(pt[1])
                w_list.append(w)

        self.x4 = np.array(x_list)
        self.y4 = np.array(y_list)
        self.w4 = np.array(w_list)

    def _max_radius_triangle(self, center, direction):
        """Find max radius from center to triangle edge."""
        # Triangle vertices: (0,0), (1,0), (0,1)
        vertices = np.array([[0, 0], [1, 0], [0, 1]])

        r_max = 10.0  # Large initial value
        for i in range(3):
            v1 = vertices[i]
            v2 = vertices[(i+1) % 3]
            edge = v2 - v1

            # Ray-edge intersection
            denom = direction[0] * edge[1] - direction[1] * edge[0]
            if abs(denom) > 1e-10:
                t = ((v1[0] - center[0]) * edge[1] - (v1[1] - center[1]) * edge[0]) / denom
                s = ((v1[0] - center[0]) * direction[1] - (v1[1] - center[1]) * direction[0]) / denom
                if t > 0 and 0 <= s <= 1:
                    r_max = min(r_max, t)

        return r_max

    def _max_radius_quad(self, center, direction):
        """Find max radius from center to quad edge ([-1,1]x[-1,1])."""
        r_max = 10.0
        edges = [
            (np.array([-1, -1]), np.array([1, -1])),
            (np.array([1, -1]), np.array([1, 1])),
            (np.array([1, 1]), np.array([-1, 1])),
            (np.array([-1, 1]), np.array([-1, -1])),
        ]

        for v1, v2 in edges:
            edge = v2 - v1
            denom = direction[0] * edge[1] - direction[1] * edge[0]
            if abs(denom) > 1e-10:
                t = ((v1[0] - center[0]) * edge[1] - (v1[1] - center[1]) * edge[0]) / denom
                s = ((v1[0] - center[0]) * direction[1] - (v1[1] - center[1]) * direction[0]) / denom
                if t > 0 and 0 <= s <= 1:
                    r_max = min(r_max, t)

        return r_max


class Particle:
    """
    Faces and vertices of discretized particle.

    The particle faces can be either triangles or quadrilaterals, or both.
    Matches MATLAB MNPBEM @particle class exactly.

    Parameters
    ----------
    verts : ndarray, shape (nverts, 3)
        Vertex coordinates [x, y, z]
    faces : ndarray, shape (nfaces, 3) or (nfaces, 4)
        Face connectivity (0-indexed vertex indices)
    interp : str
        'flat' or 'curv' for particle boundaries

    Attributes
    ----------
    verts : ndarray
        Vertices
    faces : ndarray, shape (nfaces, 4)
        Triangle or quadrilateral faces (NaN for 4th vertex if triangle)
    pos : ndarray, shape (nfaces, 3)
        Centroid positions of faces
    vec : list of ndarray
        Basis vectors [vec1, vec2, nvec] (matches MATLAB obj.vec cell array)
        vec[0] : First tangent vector (shape: nfaces, 3)
        vec[1] : Second tangent vector (shape: nfaces, 3)
        vec[2] : Normal vector (shape: nfaces, 3)
    area : ndarray, shape (nfaces,)
        Area of each face
    nvec : property -> vec[2]
        Outward normal vectors (matches MATLAB obj.nvec)
    tvec1 : property -> vec[0]
        First tangent vector (matches MATLAB obj.tvec1)
    tvec2 : property -> vec[1]
        Second tangent vector (matches MATLAB obj.tvec2)
    nverts : property
        Number of vertices
    nfaces : property
        Number of faces

    Examples
    --------
    >>> verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    >>> faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
    >>> p = Particle(verts, faces)
    >>> print(f"Particle: {p.nverts} vertices, {p.nfaces} faces")
        Centroids of faces
    vec : list
        [tvec1, tvec2, nvec] tangential and normal vectors at centroids
    area : ndarray
        Area of faces
    quad : QuadFace
        Quadrature rules for boundary element integration
    verts2 : ndarray or None
        Additional vertices for curved particle boundary
    faces2 : ndarray or None
        Additional faces for curved particle boundary
    """

    def __init__(self, verts, faces=None, interp='flat', norm='on', **kwargs):
        """
        Initialize particle from vertices and faces.

        MATLAB: obj = particle(verts, faces, op, PropertyPair)
        """
        # Return empty particle if no verts
        if verts is None or len(verts) == 0:
            self.verts = np.array([]).reshape(0, 3)
            self.faces = np.array([]).reshape(0, 4)
            self.pos = np.array([]).reshape(0, 3)
            self.vec = [np.array([]).reshape(0, 3)] * 3
            self.area = np.array([])
            self.verts2 = None
            self.faces2 = None
            self.interp = interp
            self.quad = QuadFace()
            return

        verts = np.asarray(verts, dtype=float)

        # Handle face format
        if faces is None:
            self.verts = verts
            self.faces = np.array([]).reshape(0, 4)
        else:
            faces = np.asarray(faces, dtype=float)

            if faces.shape[1] == 3:
                # Only triangular elements - add NaN column
                self.verts = verts
                nan_col = np.full((faces.shape[0], 1), np.nan)
                self.faces = np.hstack([faces, nan_col])
            elif faces.shape[1] == 4:
                # Triangular and/or quadrilateral elements
                self.verts = verts
                self.faces = faces
            else:
                # Intermediate points for curved particle boundary
                self.verts2 = verts
                self.faces2 = faces
                # Extract corner vertices
                corner_faces = faces[:, :4].reshape(-1)
                valid_idx = ~np.isnan(corner_faces)
                unique_verts, inv_idx = np.unique(corner_faces[valid_idx].astype(int), return_inverse=True)
                self.verts = verts[unique_verts]
                # Remap face indices
                new_faces = np.full_like(faces[:, :4], np.nan)
                new_faces.flat[valid_idx] = inv_idx
                self.faces = new_faces

        # Curved boundary data (None for flat)
        if not hasattr(self, 'verts2'):
            self.verts2 = None
            self.faces2 = None

        # Quadrature rules
        rule = kwargs.get('rule', 3)
        npol = kwargs.get('npol', 5)
        self.quad = QuadFace(rule=rule, npol=npol)

        # Interpolation type
        self.interp = interp

        # Compute geometric properties
        if norm != 'off' and len(self.faces) > 0:
            self._norm()
        else:
            self.pos = np.array([]).reshape(0, 3)
            self.vec = [np.array([]).reshape(0, 3)] * 3
            self.area = np.array([])

    # ==================== Properties (MATLAB subsref) ====================

    @property
    def nvec(self):
        """Normal vectors of surface elements (MATLAB: obj.nvec)."""
        return self.vec[2]

    @nvec.setter
    def nvec(self, value):
        """Set normal vectors."""
        self.vec[2] = value

    @property
    def tvec(self):
        """Tangential vectors (MATLAB: obj.tvec)."""
        return [self.vec[0], self.vec[1]]

    @property
    def tvec1(self):
        """First tangential vector (MATLAB: obj.tvec1)."""
        return self.vec[0]

    @tvec1.setter
    def tvec1(self, value):
        """Set first tangential vector."""
        self.vec[0] = value

    @property
    def tvec2(self):
        """Second tangential vector (MATLAB: obj.tvec2)."""
        return self.vec[1]

    @tvec2.setter
    def tvec2(self, value):
        """Set second tangential vector."""
        self.vec[1] = value

    @property
    def nfaces(self):
        """Number of surface elements (MATLAB: obj.nfaces, obj.n, obj.size)."""
        return self.faces.shape[0]

    @property
    def n(self):
        """Number of surface elements (alias)."""
        return self.nfaces

    @property
    def nverts(self):
        """Number of vertices (MATLAB: obj.nverts)."""
        return self.verts.shape[0]

    # ==================== Geometry computation ====================

    def _norm(self):
        """
        Compute auxiliary information for discretized particle surface.

        MATLAB: obj = norm(obj)
        """
        if self.interp == 'flat':
            self._norm_flat()
        else:
            self._norm_curv()

    def _norm_flat(self):
        """
        Compute centroids, areas, and basis vectors for flat elements.

        MATLAB: norm_flat.m
        """
        n = self.faces.shape[0]
        ind3, ind4 = self.index34()

        # Compute centroids
        self.pos = np.zeros((n, 3))

        if len(ind3) > 0:
            f3 = self.faces[ind3, :3].astype(int)
            self.pos[ind3] = (self.verts[f3[:, 0]] +
                              self.verts[f3[:, 1]] +
                              self.verts[f3[:, 2]]) / 3.0

        if len(ind4) > 0:
            f4 = self.faces[ind4].astype(int)
            self.pos[ind4] = (self.verts[f4[:, 0]] +
                              self.verts[f4[:, 1]] +
                              self.verts[f4[:, 2]] +
                              self.verts[f4[:, 3]]) / 4.0

        # Split into triangles
        tri_faces, ind4_split = self.totriangles()

        # Get triangle vertices
        v1 = self.verts[tri_faces[:, 0].astype(int)]
        v2 = self.verts[tri_faces[:, 1].astype(int)]
        v3 = self.verts[tri_faces[:, 2].astype(int)]

        # Triangle vectors
        vec1 = v1 - v2
        vec2 = v3 - v2

        # Normal vector
        nvec = np.cross(vec1, vec2)

        # Area of triangles
        area = 0.5 * np.linalg.norm(nvec, axis=1)

        # Normalize vectors
        vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
        nvec_norm = np.linalg.norm(nvec, axis=1, keepdims=True)
        vec1 = vec1 / np.maximum(vec1_norm, 1e-14)
        nvec = nvec / np.maximum(nvec_norm, 1e-14)

        # Orthogonal basis
        vec2 = np.cross(nvec, vec1)

        if len(ind4_split) == 0:
            # Only triangles
            self.area = area
            self.vec = [vec1, vec2, nvec]
        else:
            # Accumulate area for quads (two triangles per quad)
            self.area = np.zeros(n)
            for i in range(len(area)):
                if i < n:
                    self.area[i] = area[i]
                else:
                    # Second triangle of quad
                    orig_idx = ind4_split[i - n, 0]
                    self.area[orig_idx] += area[i]

            # Select vectors from larger triangle
            vec1_out = vec1[:n].copy()
            vec2_out = vec2[:n].copy()
            nvec_out = nvec[:n].copy()

            if len(ind4_split) > 0:
                area1 = area[ind4_split[:, 0]]
                area2 = area[ind4_split[:, 1]]
                larger = area2 > area1

                for i, (idx1, idx2) in enumerate(ind4_split):
                    if larger[i]:
                        vec1_out[idx1] = vec1[idx2]
                        vec2_out[idx1] = vec2[idx2]
                        nvec_out[idx1] = nvec[idx2]

            self.vec = [vec1_out, vec2_out, nvec_out]

    def _norm_curv(self):
        """
        Compute centroids, areas, and basis vectors for curved elements.

        MATLAB: norm_curv.m
        """
        n = self.faces.shape[0]

        # Get area from integration weights
        _, w = self.quad_integration()
        self.area = np.array(w.sum(axis=1)).flatten()

        ind3, ind4 = self.index34()
        faces = self.totriangles()[0]

        # Allocate arrays
        pos = np.zeros((n, 3))
        vec1 = np.zeros((n, 3))
        vec2 = np.zeros((n, 3))

        # Triangular elements
        if len(ind3) > 0:
            # Shape functions at centroid
            tri = np.array([-1, -1, -1, 4, 4, 4]) / 9
            trix = np.array([1, 0, -1, 4, -4, 0]) / 3
            triy = np.array([0, 1, -1, 4, 0, -4]) / 3

            for i in ind3:
                face_idx = self.faces2[i, [0, 1, 2, 4, 5, 6]].astype(int)
                for j in range(6):
                    pos[i] += tri[j] * self.verts2[face_idx[j]]
                    vec1[i] += triy[j] * self.verts2[face_idx[j]]
                    vec2[i] += trix[j] * self.verts2[face_idx[j]]

        # Quadrilateral elements
        if len(ind4) > 0:
            for i in ind4:
                # Centroid is last midpoint
                face_idx = self.faces2[i, 5].astype(int)
                pos[i] = self.verts2[face_idx]

                # Derivatives
                trix = np.array([1, 0, -1, 0, 0, 0])
                triy = np.array([0, -1, -1, 2, 2, -2])

                face_idx = self.faces2[i, :6].astype(int)
                for j in range(6):
                    vec1[i] += triy[j] * self.verts2[face_idx[j]]
                    vec2[i] += trix[j] * self.verts2[face_idx[j]]

        # Normalize
        nvec = np.cross(vec1, vec2)
        nvec_norm = np.linalg.norm(nvec, axis=1, keepdims=True)
        nvec = nvec / np.maximum(nvec_norm, 1e-14)

        vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
        vec1 = vec1 / np.maximum(vec1_norm, 1e-14)

        vec2 = np.cross(nvec, vec1)

        self.pos = pos
        self.vec = [vec1, vec2, nvec]

    # ==================== Index methods ====================

    def index34(self, ind=None):
        """
        Index to triangular and quadrilateral boundary elements.

        MATLAB: [ind3, ind4] = index34(obj, ind)

        Parameters
        ----------
        ind : array_like, optional
            Index to specific boundary elements

        Returns
        -------
        ind3 : ndarray
            Indices of triangular faces
        ind4 : ndarray
            Indices of quadrilateral faces
        """
        if ind is None:
            is_tri = np.isnan(self.faces[:, 3])
            ind3 = np.where(is_tri)[0]
            ind4 = np.where(~is_tri)[0]
        else:
            ind = np.asarray(ind)
            is_tri = np.isnan(self.faces[ind, 3])
            ind3 = np.where(is_tri)[0]
            ind4 = np.where(~is_tri)[0]

        return ind3, ind4

    def totriangles(self, ind=None):
        """
        Split quadrilateral face elements to triangles.

        MATLAB: [faces, ind4] = totriangles(obj, ind)

        Returns
        -------
        faces : ndarray
            Triangle faces
        ind4 : ndarray
            Pointer to split quadrilaterals [original_idx, new_idx]
        """
        if self.interp == 'flat':
            return self._totriangles_flat(ind)
        else:
            return self._totriangles_curv(ind)

    def _totriangles_flat(self, ind=None):
        """Split quads to triangles (flat)."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        _, ind4 = self.index34(ind)

        # Start with first 3 vertices of each face
        faces = self.faces[ind, :3].copy()

        if len(ind4) > 0:
            # Add second triangles for quads: v3, v4, v1
            quad_faces = np.column_stack([
                self.faces[ind[ind4], 2],
                self.faces[ind[ind4], 3],
                self.faces[ind[ind4], 0]
            ])
            faces = np.vstack([faces, quad_faces])

            # Index mapping: [original_quad_idx, new_triangle_idx]
            ind4_out = np.column_stack([
                ind4,
                len(ind) + np.arange(len(ind4))
            ])
        else:
            ind4_out = np.array([]).reshape(0, 2).astype(int)

        return faces, ind4_out

    def _totriangles_curv(self, ind=None):
        """Split quads to triangles (curved)."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        ind3, ind4 = self.index34(ind)

        # Allocate output
        faces = np.zeros((len(ind), 6))

        # Triangular elements
        if len(ind3) > 0:
            faces[ind3] = self.faces2[ind[ind3]][:, [0, 1, 2, 4, 5, 6]]

        # Quadrilateral elements
        if len(ind4) > 0:
            # First triangle
            faces[ind4] = self.faces2[ind[ind4]][:, [0, 1, 2, 4, 5, 8]]
            # Second triangle
            second_tri = self.faces2[ind[ind4]][:, [2, 3, 0, 6, 7, 8]]
            faces = np.vstack([faces, second_tri])

            ind4_out = np.column_stack([
                ind4,
                len(ind) + np.arange(len(ind4))
            ])
        else:
            ind4_out = np.array([]).reshape(0, 2).astype(int)

        return faces, ind4_out

    def vertices(self, ind, close=False):
        """
        Vertices of indexed face.

        MATLAB: v = vertices(obj, ind, 'close')

        Parameters
        ----------
        ind : int
            Face index
        close : bool
            If True, close the face indices

        Returns
        -------
        v : ndarray
            Vertices of the face
        """
        face = self.faces[ind]
        face = face[~np.isnan(face)].astype(int)

        if close:
            face = np.append(face, face[0])

        return self.verts[face]

    # ==================== Geometry transformations ====================

    def shift(self, vec):
        """
        Shift (translate) particle.

        MATLAB: obj = shift(obj, vec)

        Parameters
        ----------
        vec : array_like, shape (3,)
            Translation vector

        Returns
        -------
        self : Particle
            Shifted particle
        """
        vec = np.asarray(vec)
        self.verts = self.verts + vec
        if self.verts2 is not None:
            self.verts2 = self.verts2 + vec
        self._norm()
        return self

    def rot(self, angle, dir=None):
        """
        Rotate particle.

        MATLAB: obj = rot(obj, angle, dir)

        Parameters
        ----------
        angle : float
            Rotation angle in degrees
        dir : array_like, shape (3,), optional
            Rotation axis (default: z-axis [0,0,1])

        Returns
        -------
        self : Particle
            Rotated particle
        """
        if dir is None:
            dir = np.array([0, 0, 1])
        dir = np.asarray(dir, dtype=float)
        dir = dir / np.linalg.norm(dir)

        # Convert to radians
        angle_rad = angle * np.pi / 180

        # Rotation generators (skew-symmetric matrices)
        j1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        j2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
        j3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)

        # Rotation matrix via matrix exponential
        R = expm(-angle_rad * (dir[0]*j1 + dir[1]*j2 + dir[2]*j3))

        # Rotate vertices
        self.verts = self.verts @ R
        if self.verts2 is not None:
            self.verts2 = self.verts2 @ R

        self._norm()
        return self

    def scale(self, scale_factor):
        """
        Scale particle coordinates.

        MATLAB: obj = scale(obj, scale)

        Parameters
        ----------
        scale_factor : float or array_like
            Scaling factor (scalar or vector for each axis)

        Returns
        -------
        self : Particle
            Scaled particle
        """
        scale_factor = np.asarray(scale_factor)
        self.verts = self.verts * scale_factor
        if self.verts2 is not None:
            self.verts2 = self.verts2 * scale_factor
        self._norm()
        return self

    def flip(self, dir=0):
        """
        Flip particle along given direction.

        MATLAB: obj = flip(obj, dir)

        Parameters
        ----------
        dir : int
            Direction to flip (0=x, 1=y, 2=z). Default: 0

        Returns
        -------
        self : Particle
            Flipped particle
        """
        self.verts[:, dir] = -self.verts[:, dir]
        if self.verts2 is not None:
            self.verts2[:, dir] = -self.verts2[:, dir]

        return self.flipfaces()

    def flipfaces(self):
        """
        Flip orientation of surface elements (reverse face winding).

        MATLAB: obj = flipfaces(obj)

        Returns
        -------
        self : Particle
            Particle with flipped faces
        """
        ind3, ind4 = self.index34()

        # Flip triangular faces
        if len(ind3) > 0:
            self.faces[ind3, :3] = self.faces[ind3, :3][:, ::-1]

        # Flip quadrilateral faces
        if len(ind4) > 0:
            self.faces[ind4, :4] = self.faces[ind4, :4][:, ::-1]

        # Also flip faces2 if present
        if self.faces2 is not None:
            if len(ind3) > 0:
                # Flip: [1,2,3,5,6,7] -> [3,2,1,6,5,7]
                self.faces2[ind3][:, [0, 1, 2, 4, 5, 6]] = \
                    self.faces2[ind3][:, [2, 1, 0, 5, 4, 6]]
            if len(ind4) > 0:
                # Flip: [1,2,3,4,5,6,7,8] -> [4,3,2,1,7,6,5,8]
                self.faces2[ind4][:, [0, 1, 2, 3, 4, 5, 6, 7]] = \
                    self.faces2[ind4][:, [3, 2, 1, 0, 6, 5, 4, 7]]

        self._norm()
        return self

    # ==================== Selection and merging ====================

    def select(self, index=None, carfun=None, polfun=None, sphfun=None):
        """
        Select parts of discretized particle surface.

        MATLAB: [obj1, obj2] = select(obj, 'PropertyName', PropertyValue)

        Parameters
        ----------
        index : array_like, optional
            Index to selected elements
        carfun : callable, optional
            Function f(x, y, z) returning True for selected elements
        polfun : callable, optional
            Function f(phi, r, z) in cylindrical coordinates
        sphfun : callable, optional
            Function f(phi, theta, r) in spherical coordinates

        Returns
        -------
        obj1 : Particle
            Particle with selected faces
        obj2 : Particle or None
            Complement (if requested)
        """
        x = self.pos[:, 0]
        y = self.pos[:, 1]
        z = self.pos[:, 2]

        if index is not None:
            idx = np.asarray(index)
        elif carfun is not None:
            idx = np.where(carfun(x, y, z))[0]
        elif polfun is not None:
            phi = np.arctan2(y, x)
            r = np.sqrt(x**2 + y**2)
            idx = np.where(polfun(phi, r, z))[0]
        elif sphfun is not None:
            phi = np.arctan2(y, x)
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(np.sqrt(x**2 + y**2), z)
            idx = np.where(sphfun(phi, theta, r))[0]
        else:
            raise ValueError("Must specify index, carfun, polfun, or sphfun")

        obj1 = self._compress(idx)
        obj2 = self._compress(np.setdiff1d(np.arange(self.nfaces), idx))

        return obj1, obj2

    def _compress(self, index):
        """Compress particle and remove unused vertices."""
        if len(index) == 0:
            return Particle(np.array([]).reshape(0, 3), np.array([]).reshape(0, 4))

        faces = self.faces[index].copy()

        # Find unique vertices
        flat_faces = faces[~np.isnan(faces)].astype(int)
        unique_verts = np.unique(flat_faces)

        # Create remapping table
        remap = np.zeros(self.nverts, dtype=int)
        remap[unique_verts] = np.arange(len(unique_verts))

        # Remap faces
        valid_mask = ~np.isnan(faces)
        faces[valid_mask] = remap[faces[valid_mask].astype(int)]

        new_verts = self.verts[unique_verts]

        # Handle curved data
        new_verts2 = None
        new_faces2 = None
        if self.verts2 is not None and self.faces2 is not None:
            faces2 = self.faces2[index].copy()
            flat_faces2 = faces2[~np.isnan(faces2)].astype(int)
            unique_verts2 = np.unique(flat_faces2)
            remap2 = np.zeros(len(self.verts2), dtype=int)
            remap2[unique_verts2] = np.arange(len(unique_verts2))
            valid_mask2 = ~np.isnan(faces2)
            faces2[valid_mask2] = remap2[faces2[valid_mask2].astype(int)]
            new_verts2 = self.verts2[unique_verts2]
            new_faces2 = faces2

        new_particle = Particle(new_verts, faces, interp=self.interp, norm='off')
        new_particle.verts2 = new_verts2
        new_particle.faces2 = new_faces2
        new_particle._norm()

        return new_particle

    def __add__(self, other):
        """Concatenate particles using + operator."""
        return self.vertcat(other)

    def vertcat(self, *others):
        """
        Concatenate particles vertically.

        MATLAB: obj = vertcat(obj1, obj2, obj3, ...)
        MATLAB: obj = [obj1; obj2; obj3; ...]

        Returns
        -------
        obj : Particle
            Combined particle surface
        """
        for other in others:
            # Extend face-vertex list
            offset = self.verts.shape[0]
            self.faces = np.vstack([self.faces, other.faces + offset])
            self.verts = np.vstack([self.verts, other.verts])

            # Extend curved data if present
            if self.verts2 is not None and other.verts2 is not None:
                offset2 = self.verts2.shape[0]
                self.faces2 = np.vstack([self.faces2, other.faces2 + offset2])
                self.verts2 = np.vstack([self.verts2, other.verts2])

        self._norm()
        return self

    # ==================== Edge and boundary methods ====================

    def edges(self):
        """
        Find unique edges of particle.

        MATLAB: [net, faces] = edges(obj)

        Returns
        -------
        net : ndarray, shape (nedges, 2)
            List of unique edges (vertex indices)
        edge_faces : ndarray
            Edge indices for each face
        """
        ind3, ind4 = self.index34()
        faces = self.faces

        # Build edge list
        edge_list = []

        # Triangular edges
        if len(ind3) > 0:
            f3 = faces[ind3, :3].astype(int)
            edge_list.extend([
                np.column_stack([f3[:, 0], f3[:, 1]]),
                np.column_stack([f3[:, 1], f3[:, 2]]),
                np.column_stack([f3[:, 2], f3[:, 0]])
            ])

        # Quadrilateral edges
        if len(ind4) > 0:
            f4 = faces[ind4, :4].astype(int)
            edge_list.extend([
                np.column_stack([f4[:, 0], f4[:, 1]]),
                np.column_stack([f4[:, 1], f4[:, 2]]),
                np.column_stack([f4[:, 2], f4[:, 3]]),
                np.column_stack([f4[:, 3], f4[:, 0]])
            ])

        if not edge_list:
            return np.array([]).reshape(0, 2), np.array([])

        all_edges = np.vstack(edge_list)

        # Sort edge vertices and find unique
        sorted_edges = np.sort(all_edges, axis=1)
        net, inv_idx = np.unique(sorted_edges, axis=0, return_inverse=True)

        # Build face edge index
        edge_faces = np.full_like(faces, np.nan)
        offset = 0

        if len(ind3) > 0:
            n3 = len(ind3)
            edge_faces[ind3, 0] = inv_idx[offset:offset + n3]
            edge_faces[ind3, 1] = inv_idx[offset + n3:offset + 2*n3]
            edge_faces[ind3, 2] = inv_idx[offset + 2*n3:offset + 3*n3]
            offset += 3 * n3

        if len(ind4) > 0:
            n4 = len(ind4)
            edge_faces[ind4, 0] = inv_idx[offset:offset + n4]
            edge_faces[ind4, 1] = inv_idx[offset + n4:offset + 2*n4]
            edge_faces[ind4, 2] = inv_idx[offset + 2*n4:offset + 3*n4]
            edge_faces[ind4, 3] = inv_idx[offset + 3*n4:offset + 4*n4]

        return net, edge_faces

    def border(self):
        """
        Find border (single edges) of particle.

        MATLAB: net = border(obj)

        Returns
        -------
        net : ndarray, shape (n_border, 2)
            Border edge list
        """
        ind3, ind4 = self.index34()
        faces = self.faces

        # Build edge list
        edge_list = []

        if len(ind3) > 0:
            f3 = faces[ind3, :3].astype(int)
            edge_list.extend([
                np.column_stack([f3[:, 0], f3[:, 1]]),
                np.column_stack([f3[:, 1], f3[:, 2]]),
                np.column_stack([f3[:, 2], f3[:, 0]])
            ])

        if len(ind4) > 0:
            f4 = faces[ind4, :4].astype(int)
            edge_list.extend([
                np.column_stack([f4[:, 0], f4[:, 1]]),
                np.column_stack([f4[:, 1], f4[:, 2]]),
                np.column_stack([f4[:, 2], f4[:, 3]]),
                np.column_stack([f4[:, 3], f4[:, 0]])
            ])

        if not edge_list:
            return np.array([]).reshape(0, 2)

        all_edges = np.vstack(edge_list)
        sorted_edges = np.sort(all_edges, axis=1)

        # Find edges that appear only once
        _, idx, counts = np.unique(sorted_edges, axis=0,
                                   return_index=True, return_counts=True)
        single_edges = all_edges[idx[counts == 1]]

        return single_edges

    # ==================== Integration methods ====================

    def quad_integration(self, ind=None):
        """
        Quadrature points and weights for boundary element integration.

        MATLAB: [pos, w, iface] = quad(obj, ind)

        Parameters
        ----------
        ind : array_like, optional
            Face indices (default: all faces)

        Returns
        -------
        pos : ndarray
            Integration points
        w : sparse matrix
            Integration weights
        """
        if self.interp == 'flat':
            return self._quad_flat(ind)
        else:
            return self._quad_curv(ind)

    def _quad_flat(self, ind=None):
        """Quadrature for flat elements."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        # Decompose into triangles
        tri_faces, ind4_split = self._totriangles_flat(ind)

        # Get triangle indices
        ind3 = list(range(len(ind)))
        if len(ind4_split) > 0:
            ind3.extend(ind4_split[:, 0].tolist())

        # Normal vectors and areas
        v1 = self.verts[tri_faces[:, 0].astype(int)]
        v2 = self.verts[tri_faces[:, 1].astype(int)]
        v3 = self.verts[tri_faces[:, 2].astype(int)]

        nvec = np.cross(v2 - v1, v3 - v1)
        area = 0.5 * np.linalg.norm(nvec, axis=1)

        # Integration points and weights
        x, y, w = self.quad.x, self.quad.y, self.quad.w
        m = len(w)
        n_total = m * len(ind3)

        pos = np.zeros((n_total, 3))
        weights = np.zeros(n_total)
        rows = np.zeros(n_total, dtype=int)

        # Shape functions: [x, y, 1-x-y]
        tri_shape = np.column_stack([x, y, 1 - x - y])

        offset = 0
        for i, face_idx in enumerate(ind3):
            it = slice(offset, offset + m)
            face = tri_faces[i, :3].astype(int)
            pos[it] = tri_shape @ self.verts[face]
            weights[it] = w * area[i]
            rows[it] = face_idx
            offset += m

        # Create sparse weight matrix
        cols = np.arange(n_total)
        w_sparse = csr_matrix((weights, (rows, cols)), shape=(len(ind), n_total))

        return pos, w_sparse

    def _quad_curv(self, ind=None):
        """Quadrature for curved elements."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        tri_faces, ind4_split = self._totriangles_curv(ind)

        ind3 = list(range(len(ind)))
        if len(ind4_split) > 0:
            ind3.extend(ind4_split[:, 0].tolist())

        x, y, w = self.quad.x, self.quad.y, self.quad.w
        m = len(w)
        n_total = m * len(ind3)

        pos = np.zeros((n_total, 3))
        weights = np.zeros(n_total)
        rows = np.zeros(n_total, dtype=int)

        # 6-node triangle shape functions
        tri_shape = self._tri6_shape(x, y)
        tri_dx, tri_dy = self._tri6_deriv(x, y)

        offset = 0
        for i, face_idx in enumerate(ind3):
            it = slice(offset, offset + m)
            face = tri_faces[i].astype(int)

            # Interpolate positions
            pos[it] = tri_shape @ self.verts2[face]

            # Derivatives for Jacobian
            posx = tri_dx @ self.verts2[face]
            posy = tri_dy @ self.verts2[face]

            nvec = np.cross(posx, posy)
            jac = 0.5 * np.linalg.norm(nvec, axis=1)

            weights[it] = w * jac
            rows[it] = face_idx
            offset += m

        cols = np.arange(n_total)
        w_sparse = csr_matrix((weights, (rows, cols)), shape=(len(ind), n_total))

        return pos, w_sparse

    def quadpol(self, ind=None):
        """
        Quadrature points and weights for polar integration.

        MATLAB: [pos, weight, row] = quadpol(obj, ind)

        Parameters
        ----------
        ind : array_like, optional
            Face indices

        Returns
        -------
        pos : ndarray
            Integration points
        weight : ndarray
            Integration weights
        row : ndarray
            Face index for each integration point
        """
        if self.interp == 'flat':
            return self._quadpol_flat(ind)
        else:
            return self._quadpol_curv(ind)

    def _quadpol_flat(self, ind=None):
        """Polar quadrature for flat elements."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        ind3, ind4 = self.index34(ind)
        q = self.quad

        m3, m4 = len(q.x3), len(q.x4)
        n_total = len(ind3) * m3 + len(ind4) * m4

        pos = np.zeros((n_total, 3))
        weight = np.zeros(n_total)
        row = np.zeros(n_total, dtype=int)

        offset = 0

        # Triangular elements
        if len(ind3) > 0:
            tri_shape = np.column_stack([q.x3, q.y3, 1 - q.x3 - q.y3])

            for i in ind3:
                it = slice(offset, offset + m3)
                face = self.faces[ind[i], :3].astype(int)
                pos[it] = tri_shape @ self.verts[face]
                weight[it] = q.w3 * self.area[ind[i]]
                row[it] = i
                offset += m3

        # Quadrilateral elements
        if len(ind4) > 0:
            quad_shape = self._quad4_shape(q.x4, q.y4)
            quad_dx, quad_dy = self._quad4_deriv(q.x4, q.y4)

            for i in ind4:
                it = slice(offset, offset + m4)
                face = self.faces[ind[i], :4].astype(int)
                pos[it] = quad_shape @ self.verts[face]

                posx = quad_dx @ self.verts[face]
                posy = quad_dy @ self.verts[face]
                nvec = np.cross(posx, posy)
                jac = np.linalg.norm(nvec, axis=1)

                weight[it] = q.w4 * jac
                row[it] = i
                offset += m4

        return pos, weight, row

    def _quadpol_curv(self, ind=None):
        """Polar quadrature for curved elements."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        ind3, ind4 = self.index34(ind)
        q = self.quad
        faces = self.faces2[ind]

        m3, m4 = len(q.x3), len(q.x4)
        n_total = len(ind3) * m3 + len(ind4) * m4

        pos = np.zeros((n_total, 3))
        weight = np.zeros(n_total)
        row = np.zeros(n_total, dtype=int)

        offset = 0

        # Triangular elements
        if len(ind3) > 0:
            tri_shape = self._tri6_shape(q.x3, q.y3)
            tri_dx, tri_dy = self._tri6_deriv(q.x3, q.y3)

            for i in ind3:
                it = slice(offset, offset + m3)
                face_idx = faces[i, [0, 1, 2, 4, 5, 6]].astype(int)

                pos[it] = tri_shape @ self.verts2[face_idx]
                posx = tri_dx @ self.verts2[face_idx]
                posy = tri_dy @ self.verts2[face_idx]

                nvec = np.cross(posx, posy)
                jac = 0.5 * np.linalg.norm(nvec, axis=1)

                weight[it] = q.w3 * jac
                row[it] = i
                offset += m3

        # Quadrilateral elements
        if len(ind4) > 0:
            quad_shape = self._quad9_shape(q.x4, q.y4)
            quad_dx, quad_dy = self._quad9_deriv(q.x4, q.y4)

            for i in ind4:
                it = slice(offset, offset + m4)
                face_idx = faces[i, :9].astype(int)

                pos[it] = quad_shape @ self.verts2[face_idx]
                posx = quad_dx @ self.verts2[face_idx]
                posy = quad_dy @ self.verts2[face_idx]

                nvec = np.cross(posx, posy)
                jac = np.linalg.norm(nvec, axis=1)

                weight[it] = q.w4 * jac
                row[it] = i
                offset += m4

        return pos, weight, row

    # ==================== Shape functions ====================

    @staticmethod
    def _tri6_shape(x, y):
        """6-node triangle shape functions."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        L1, L2, L3 = x, y, 1 - x - y

        # Shape functions for 6-node triangle (corners + midpoints)
        N = np.zeros((len(x), 6))
        N[:, 0] = L1 * (2*L1 - 1)  # Corner 1
        N[:, 1] = L2 * (2*L2 - 1)  # Corner 2
        N[:, 2] = L3 * (2*L3 - 1)  # Corner 3
        N[:, 3] = 4 * L1 * L2      # Midpoint 12
        N[:, 4] = 4 * L2 * L3      # Midpoint 23
        N[:, 5] = 4 * L3 * L1      # Midpoint 31

        return N

    @staticmethod
    def _tri6_deriv(x, y):
        """Derivatives of 6-node triangle shape functions."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        L1, L2, L3 = x, y, 1 - x - y

        # dN/dL1, dN/dL2 derivatives, then chain rule for dN/dx, dN/dy
        dNdx = np.zeros((len(x), 6))
        dNdy = np.zeros((len(x), 6))

        # dN/dx = dN/dL1 (since L1=x)
        dNdx[:, 0] = 4*L1 - 1
        dNdx[:, 1] = 0
        dNdx[:, 2] = -4*L3 + 1
        dNdx[:, 3] = 4*L2
        dNdx[:, 4] = -4*L2
        dNdx[:, 5] = 4*(L3 - L1)

        # dN/dy = dN/dL2 (since L2=y)
        dNdy[:, 0] = 0
        dNdy[:, 1] = 4*L2 - 1
        dNdy[:, 2] = -4*L3 + 1
        dNdy[:, 3] = 4*L1
        dNdy[:, 4] = 4*(L3 - L2)
        dNdy[:, 5] = -4*L1

        return dNdx, dNdy

    @staticmethod
    def _quad4_shape(x, y):
        """4-node quadrilateral shape functions (bilinear)."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)

        N = np.zeros((len(x), 4))
        N[:, 0] = 0.25 * (1 - x) * (1 - y)
        N[:, 1] = 0.25 * (1 + x) * (1 - y)
        N[:, 2] = 0.25 * (1 + x) * (1 + y)
        N[:, 3] = 0.25 * (1 - x) * (1 + y)

        return N

    @staticmethod
    def _quad4_deriv(x, y):
        """Derivatives of 4-node quad shape functions."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)

        dNdx = np.zeros((len(x), 4))
        dNdx[:, 0] = -0.25 * (1 - y)
        dNdx[:, 1] = 0.25 * (1 - y)
        dNdx[:, 2] = 0.25 * (1 + y)
        dNdx[:, 3] = -0.25 * (1 + y)

        dNdy = np.zeros((len(x), 4))
        dNdy[:, 0] = -0.25 * (1 - x)
        dNdy[:, 1] = -0.25 * (1 + x)
        dNdy[:, 2] = 0.25 * (1 + x)
        dNdy[:, 3] = 0.25 * (1 - x)

        return dNdx, dNdy

    @staticmethod
    def _quad9_shape(x, y):
        """9-node quadrilateral shape functions (biquadratic)."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)

        N = np.zeros((len(x), 9))

        # Corner nodes
        N[:, 0] = 0.25 * x * (x - 1) * y * (y - 1)
        N[:, 1] = 0.25 * x * (x + 1) * y * (y - 1)
        N[:, 2] = 0.25 * x * (x + 1) * y * (y + 1)
        N[:, 3] = 0.25 * x * (x - 1) * y * (y + 1)

        # Edge midpoints
        N[:, 4] = 0.5 * (1 - x**2) * y * (y - 1)
        N[:, 5] = 0.5 * x * (x + 1) * (1 - y**2)
        N[:, 6] = 0.5 * (1 - x**2) * y * (y + 1)
        N[:, 7] = 0.5 * x * (x - 1) * (1 - y**2)

        # Center node
        N[:, 8] = (1 - x**2) * (1 - y**2)

        return N

    @staticmethod
    def _quad9_deriv(x, y):
        """Derivatives of 9-node quad shape functions."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)

        dNdx = np.zeros((len(x), 9))
        dNdy = np.zeros((len(x), 9))

        # Corner nodes
        dNdx[:, 0] = 0.25 * (2*x - 1) * y * (y - 1)
        dNdx[:, 1] = 0.25 * (2*x + 1) * y * (y - 1)
        dNdx[:, 2] = 0.25 * (2*x + 1) * y * (y + 1)
        dNdx[:, 3] = 0.25 * (2*x - 1) * y * (y + 1)

        dNdy[:, 0] = 0.25 * x * (x - 1) * (2*y - 1)
        dNdy[:, 1] = 0.25 * x * (x + 1) * (2*y - 1)
        dNdy[:, 2] = 0.25 * x * (x + 1) * (2*y + 1)
        dNdy[:, 3] = 0.25 * x * (x - 1) * (2*y + 1)

        # Edge midpoints
        dNdx[:, 4] = -x * y * (y - 1)
        dNdx[:, 5] = 0.5 * (2*x + 1) * (1 - y**2)
        dNdx[:, 6] = -x * y * (y + 1)
        dNdx[:, 7] = 0.5 * (2*x - 1) * (1 - y**2)

        dNdy[:, 4] = 0.5 * (1 - x**2) * (2*y - 1)
        dNdy[:, 5] = -x * (x + 1) * y
        dNdy[:, 6] = 0.5 * (1 - x**2) * (2*y + 1)
        dNdy[:, 7] = -x * (x - 1) * y

        # Center node
        dNdx[:, 8] = -2 * x * (1 - y**2)
        dNdy[:, 8] = -2 * (1 - x**2) * y

        return dNdx, dNdy

    # ==================== Interpolation mode ====================

    def flat(self):
        """
        Set to flat interpolation mode.

        MATLAB: obj = flat(obj)
        """
        self.interp = 'flat'
        self._norm()
        return self

    def curved(self, key='flat'):
        """
        Set to curved interpolation mode.

        MATLAB: obj = curved(obj)

        Parameters
        ----------
        key : str
            'flat' or 'curv' for midpoint computation
        """
        if self.verts2 is None:
            self._midpoints(key)

        self.interp = 'curv'
        self._norm()
        return self

    def _midpoints(self, key='flat'):
        """
        Add midpoints for curved particle boundaries.

        MATLAB: obj = midpoints(obj, key)
        """
        self.interp = key

        if key == 'flat':
            # Add midpoints for flat boundary elements
            net, edge_faces = self.edges()
            n = self.nverts

            # Midpoint vertices
            midpts = 0.5 * (self.verts[net[:, 0]] + self.verts[net[:, 1]])
            self.verts2 = np.vstack([self.verts, midpts])

            ind3, ind4 = self.index34()

            # Allocate faces2
            self.faces2 = np.full((self.nfaces, 9), np.nan)
            self.faces2[:, :4] = self.faces

            # Extend face list for triangles
            if len(ind3) > 0:
                self.faces2[ind3, 4:7] = n + edge_faces[ind3, :3]

            # Extend face list for quadrilaterals
            if len(ind4) > 0:
                self.faces2[ind4, 4:8] = n + edge_faces[ind4, :4]
                # Add centroids
                centroid_idx = self.verts2.shape[0] + np.arange(len(ind4))
                self.faces2[ind4, 8] = centroid_idx

                # Compute centroids
                f4 = self.faces[ind4].astype(int)
                centroids = 0.25 * (self.verts[f4[:, 0]] + self.verts[f4[:, 1]] +
                                    self.verts[f4[:, 2]] + self.verts[f4[:, 3]])
                self.verts2 = np.vstack([self.verts2, centroids])
        else:
            # Use curvature-based refinement
            self._refine()

        if self.interp == 'curv':
            self._norm()

    def _refine(self):
        """
        Refine particle boundary using curvature (B-spline interpolation).

        MATLAB: refine.m (simplified version)
        """
        # Simplified: just use linear midpoints
        self._midpoints('flat')

    # ==================== Mesh cleaning ====================

    def clean(self, cutoff=1e-10):
        """
        Remove multiple vertices and elements with too small areas.

        MATLAB: obj = clean(obj, cutoff)

        Parameters
        ----------
        cutoff : float
            Keep only elements with area > cutoff * mean(area)

        Returns
        -------
        self : Particle
            Cleaned particle
        """
        # Round vertices to avoid floating point issues
        verts_rounded = np.round(self.verts, 8)

        # Find unique vertices
        unique_verts, inv_idx = np.unique(verts_rounded, axis=0, return_inverse=True)

        if len(unique_verts) != len(self.verts):
            # Remap faces
            ind3, ind4 = self.index34()
            faces = self.faces.copy()

            if len(ind3) > 0:
                for j in range(3):
                    faces[ind3, j] = inv_idx[faces[ind3, j].astype(int)]
            if len(ind4) > 0:
                for j in range(4):
                    faces[ind4, j] = inv_idx[faces[ind4, j].astype(int)]

            self.verts = unique_verts
            self.faces = faces

        # Remove quads with duplicate vertices
        ind4 = np.where(~np.isnan(self.faces[:, 3]))[0]
        for i in ind4:
            face = self.faces[i]
            unique_vals = np.unique(face[~np.isnan(face)])
            if len(unique_vals) == 3:
                # Degenerate quad -> triangle
                self.faces[i] = np.array([unique_vals[0], unique_vals[1],
                                          unique_vals[2], np.nan])

        self._norm()

        # Keep only elements with sufficient area
        mean_area = np.mean(self.area)
        valid_idx = np.where(self.area > cutoff * mean_area)[0]

        if len(valid_idx) < self.nfaces:
            result, _ = self.select(index=valid_idx)
            self.verts = result.verts
            self.faces = result.faces
            self.verts2 = result.verts2
            self.faces2 = result.faces2
            self._norm()

        return self

    # ==================== Interpolation ====================

    def interp_values(self, v, method='area'):
        """
        Interpolate values from faces to vertices or vice versa.

        MATLAB: [vi, mat] = interp(obj, v, key)

        Parameters
        ----------
        v : ndarray
            Values at faces (nfaces,) or vertices (nverts,)
        method : str
            'area' for area-weighted, 'pinv' for pseudo-inverse

        Returns
        -------
        vi : ndarray
            Interpolated values
        mat : sparse matrix
            Interpolation matrix
        """
        ind3, ind4 = self.index34()
        nfaces, nverts = self.nfaces, self.nverts

        # Build connectivity
        faces3 = self.faces[ind3, :3].astype(int) if len(ind3) > 0 else np.array([]).reshape(0, 3).astype(int)
        faces4 = self.faces[ind4, :4].astype(int) if len(ind4) > 0 else np.array([]).reshape(0, 4).astype(int)

        n = len(v)

        if n == nfaces:
            # Interpolate from faces to vertices
            if method == 'area':
                # Area-weighted average
                data, rows, cols = [], [], []

                if len(ind3) > 0:
                    for j in range(3):
                        rows.extend(faces3[:, j].tolist())
                        cols.extend(ind3.tolist())
                        data.extend(self.area[ind3].tolist())

                if len(ind4) > 0:
                    for j in range(4):
                        rows.extend(faces4[:, j].tolist())
                        cols.extend(ind4.tolist())
                        data.extend(self.area[ind4].tolist())

                mat = csr_matrix((data, (rows, cols)), shape=(nverts, nfaces))
                # Normalize
                row_sums = np.array(mat.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1
                mat = diags(1.0 / row_sums) @ mat
            else:
                # Pseudo-inverse method
                data, rows, cols = [], [], []

                if len(ind3) > 0:
                    for j in range(3):
                        rows.extend(ind3.tolist())
                        cols.extend(faces3[:, j].tolist())
                        data.extend([1/3] * len(ind3))

        # Store as list (matching MATLAB cell array structure: obj.vec = {vec1, vec2, nvec})
        # MATLAB uses 1-based indexing: vec{1}, vec{2}, vec{3}
        # Python uses 0-based indexing: vec[0], vec[1], vec[2]
        vec1_array = np.array(all_vec1)
        vec2_array = np.array(all_vec2)
        nvec_array = np.array(all_nvec)

        self.vec = [vec1_array, vec2_array, nvec_array]
        self.area = np.array(all_area)

    # Properties matching MATLAB subsref interface
    @property
    def nvec(self):
        """Normal vectors (matches MATLAB obj.nvec -> obj.vec{3})."""
        return self.vec[2]

    @property
    def tvec1(self):
        """First tangent vector (matches MATLAB obj.tvec1 -> obj.vec{1})."""
        return self.vec[0]

    @property
    def tvec2(self):
        """Second tangent vector (matches MATLAB obj.tvec2 -> obj.vec{2})."""
        return self.vec[1]

    @property
    def nverts(self):
        """Number of vertices."""
        return len(self.verts)
                if len(ind4) > 0:
                    for j in range(4):
                        rows.extend(ind4.tolist())
                        cols.extend(faces4[:, j].tolist())
                        data.extend([1/4] * len(ind4))

                con = csr_matrix((data, (rows, cols)), shape=(nfaces, nverts))
                mat = np.linalg.pinv(con.toarray())
        else:
            # Interpolate from vertices to faces
            data, rows, cols = [], [], []

            if len(ind3) > 0:
                for j in range(3):
                    rows.extend(ind3.tolist())
                    cols.extend(faces3[:, j].tolist())
                    data.extend([1/3] * len(ind3))

            if len(ind4) > 0:
                for j in range(4):
                    rows.extend(ind4.tolist())
                    cols.extend(faces4[:, j].tolist())
                    data.extend([1/4] * len(ind4))

            mat = csr_matrix((data, (rows, cols)), shape=(nfaces, nverts))

        vi = mat @ v
        return vi, mat

    # ==================== Visualization ====================

    def plot(self, val=None, **kwargs):
        """
        Plot particle surface.

        MATLAB: plot(obj, val, 'PropertyName', PropertyValue)

        Parameters
        ----------
        val : ndarray, optional
            Values to display on surface
        **kwargs : dict
            Plotting options (EdgeColor, FaceAlpha, etc.)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("matplotlib not available for plotting")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Get face vertices
        face_verts = []
        for i in range(self.nfaces):
            face = self.faces[i]
            idx = face[~np.isnan(face)].astype(int)
            face_verts.append(self.verts[idx])

        # Create 3D collection
        if val is not None:
            # Color by value
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable, viridis

            norm = Normalize(vmin=np.min(val), vmax=np.max(val))
            mapper = ScalarMappable(norm=norm, cmap=viridis)

            facecolors = [mapper.to_rgba(val[i]) for i in range(self.nfaces)]
            collection = Poly3DCollection(face_verts, facecolors=facecolors,
                                          edgecolor=kwargs.get('EdgeColor', 'k'),
                                          alpha=kwargs.get('FaceAlpha', 1.0))
        else:
            collection = Poly3DCollection(face_verts,
                                          facecolor=kwargs.get('FaceColor', [0.8, 0.8, 0.9]),
                                          edgecolor=kwargs.get('EdgeColor', 'k'),
                                          alpha=kwargs.get('FaceAlpha', 1.0))

        ax.add_collection3d(collection)

        # Set axis limits
        all_verts = np.vstack([self.verts[f[~np.isnan(f)].astype(int)]
                               for f in self.faces])
        max_range = np.max(all_verts.max(axis=0) - all_verts.min(axis=0)) / 2
        mid = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        return fig, ax

    # ==================== String representations ====================

    def __repr__(self):
        return f"Particle(nverts={self.nverts}, nfaces={self.nfaces}, interp='{self.interp}')"

    def __str__(self):
        return (
            f"Particle:\n"
            f"  Vertices: {self.nverts}\n"
            f"  Faces: {self.nfaces}\n"
            f"  Total area: {self.area.sum():.2f}\n"
            f"  Interpolation: {self.interp}"
        )
