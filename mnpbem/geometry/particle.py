"""
Particle class for discretized surfaces.
"""

import numpy as np


class Particle:
    """
    Discretized particle surface with triangular mesh.

    Represents a particle surface using triangular boundary elements.
    Computes geometric properties: centroids, normals, tangent vectors, areas.

    Parameters
    ----------
    verts : ndarray, shape (nverts, 3)
        Vertex coordinates [x, y, z] in nm
    faces : ndarray, shape (nfaces, 3) or (nfaces, 4)
        Face connectivity (0-indexed vertex indices)
        For triangles: shape (nfaces, 3)
        For mixed tri/quad: shape (nfaces, 4) with NaN for triangles
    interp : str, optional
        Interpolation type: 'flat' or 'curv'. Default: 'flat'

    Attributes
    ----------
    verts : ndarray, shape (nverts, 3)
        Vertex positions
    faces : ndarray, shape (nfaces, 4)
        Face indices (triangles have NaN in 4th column)
    pos : ndarray, shape (nfaces, 3)
        Centroid positions of faces
    nvec : ndarray, shape (nfaces, 3)
        Outward normal vectors (unit vectors)
    vec1 : ndarray, shape (nfaces, 3)
        First tangent vector (unit vector)
    vec2 : ndarray, shape (nfaces, 3)
        Second tangent vector (unit vector, vec2 = nvec × vec1)
    area : ndarray, shape (nfaces,)
        Area of each face
    nverts : int
        Number of vertices
    nfaces : int
        Number of faces

    Examples
    --------
    >>> verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    >>> faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
    >>> p = Particle(verts, faces)
    >>> print(f"Particle: {p.nverts} vertices, {p.nfaces} faces")
    """

    def __init__(self, verts, faces, interp='flat'):
        """
        Initialize particle from vertices and faces.

        Parameters
        ----------
        verts : ndarray
            Vertex coordinates
        faces : ndarray
            Face connectivity
        interp : str
            Interpolation: 'flat' or 'curv'
        """
        self.verts = np.asarray(verts, dtype=float)
        self.interp = interp

        # Handle face format (3 or 4 columns)
        faces = np.asarray(faces, dtype=float)
        if faces.shape[1] == 3:
            # Pure triangular mesh - add NaN column
            nan_col = np.full((faces.shape[0], 1), np.nan)
            self.faces = np.hstack([faces, nan_col])
        elif faces.shape[1] == 4:
            self.faces = faces
        else:
            raise ValueError(f"Faces must have 3 or 4 columns, got {faces.shape[1]}")

        # Compute geometric properties
        self._compute_geometry()

    def _compute_geometry(self):
        """
        Compute centroids, normals, tangent vectors, and areas.
        """
        nfaces = self.faces.shape[0]

        # Identify triangular vs quadrilateral faces
        is_triangle = np.isnan(self.faces[:, 3])

        # Compute centroids
        self.pos = np.zeros((nfaces, 3))

        # Triangular elements: centroid = (v1 + v2 + v3) / 3
        tri_idx = np.where(is_triangle)[0]
        if len(tri_idx) > 0:
            v1 = self.verts[self.faces[tri_idx, 0].astype(int)]
            v2 = self.verts[self.faces[tri_idx, 1].astype(int)]
            v3 = self.verts[self.faces[tri_idx, 2].astype(int)]
            self.pos[tri_idx] = (v1 + v2 + v3) / 3.0

        # Quadrilateral elements: centroid = (v1 + v2 + v3 + v4) / 4
        quad_idx = np.where(~is_triangle)[0]
        if len(quad_idx) > 0:
            v1 = self.verts[self.faces[quad_idx, 0].astype(int)]
            v2 = self.verts[self.faces[quad_idx, 1].astype(int)]
            v3 = self.verts[self.faces[quad_idx, 2].astype(int)]
            v4 = self.verts[self.faces[quad_idx, 3].astype(int)]
            self.pos[quad_idx] = (v1 + v2 + v3 + v4) / 4.0

        # Compute normals and areas by splitting all faces into triangles
        all_vec1 = []
        all_vec2 = []
        all_nvec = []
        all_area = []

        for i in range(nfaces):
            if is_triangle[i]:
                # Single triangle
                idx = self.faces[i, :3].astype(int)
                v1, v2, v3 = self.verts[idx]

                # Triangle vectors
                vec_a = v1 - v2
                vec_b = v3 - v2

                # Normal (cross product)
                normal = np.cross(vec_a, vec_b)
                area = 0.5 * np.linalg.norm(normal)

                # Normalize
                vec1 = vec_a / np.linalg.norm(vec_a)
                nvec = normal / np.linalg.norm(normal)
                vec2 = np.cross(nvec, vec1)

                all_vec1.append(vec1)
                all_vec2.append(vec2)
                all_nvec.append(nvec)
                all_area.append(area)

            else:
                # Quadrilateral - split into two triangles
                idx = self.faces[i].astype(int)
                v1, v2, v3, v4 = self.verts[idx]

                # Triangle 1: v1, v2, v3
                vec_a1 = v1 - v2
                vec_b1 = v3 - v2
                normal1 = np.cross(vec_a1, vec_b1)
                area1 = 0.5 * np.linalg.norm(normal1)

                # Triangle 2: v1, v3, v4
                vec_a2 = v1 - v3
                vec_b2 = v4 - v3
                normal2 = np.cross(vec_a2, vec_b2)
                area2 = 0.5 * np.linalg.norm(normal2)

                # Use larger triangle for vectors
                if area1 >= area2:
                    vec_a, vec_b, normal = vec_a1, vec_b1, normal1
                else:
                    vec_a, vec_b, normal = vec_a2, vec_b2, normal2

                # Total area
                area = area1 + area2

                # Normalize
                vec1 = vec_a / np.linalg.norm(vec_a)
                nvec = normal / np.linalg.norm(normal)
                vec2 = np.cross(nvec, vec1)

                all_vec1.append(vec1)
                all_vec2.append(vec2)
                all_nvec.append(nvec)
                all_area.append(area)

        # Store as arrays
        self.vec1 = np.array(all_vec1)
        self.vec2 = np.array(all_vec2)
        self.nvec = np.array(all_nvec)
        self.area = np.array(all_area)

    @property
    def nverts(self):
        """Number of vertices."""
        return len(self.verts)

    @property
    def nfaces(self):
        """Number of faces."""
        return len(self.faces)

    def __repr__(self):
        return f"Particle(nverts={self.nverts}, nfaces={self.nfaces})"

    def __str__(self):
        return (
            f"Particle:\n"
            f"  Vertices: {self.nverts}\n"
            f"  Faces: {self.nfaces}\n"
            f"  Total area: {self.area.sum():.2f} nm²\n"
            f"  Interpolation: {self.interp}"
        )
