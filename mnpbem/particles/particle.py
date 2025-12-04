"""
Discretized particle surface class.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field

from ..misc.helpers import vecnorm, vecnormalize


@dataclass
class QuadFace:
    """Quadrature rules for boundary element integration."""
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))


class Particle:
    """
    Faces and vertices of a discretized particle surface.

    The particle faces can be either triangles or quadrilaterals, or both.

    Parameters
    ----------
    verts : ndarray
        Vertices of boundary elements, shape (n_verts, 3).
    faces : ndarray
        Faces of boundary elements, shape (n_faces, 3) or (n_faces, 4).
        For triangles, use 3 columns. For quads, use 4 columns.
        Use NaN to indicate missing vertices for mixed meshes.
    interp : str, optional
        Interpolation type: 'flat' or 'curv'. Default is 'flat'.

    Attributes
    ----------
    verts : ndarray
        Vertices of the particle, shape (n_verts, 3).
    faces : ndarray
        Face indices, shape (n_faces, 4). Triangles have NaN in 4th column.
    pos : ndarray
        Centroids of faces, shape (n_faces, 3).
    nvec : ndarray
        Normal vectors at centroids, shape (n_faces, 3).
    tvec1, tvec2 : ndarray
        Tangent vectors at centroids, shape (n_faces, 3).
    area : ndarray
        Area of faces, shape (n_faces,).
    """

    def __init__(
        self,
        verts: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        interp: str = 'flat',
        compute_normals: bool = True,
        **kwargs
    ):
        """
        Initialize particle surface.

        Parameters
        ----------
        verts : ndarray, optional
            Vertices of boundary elements.
        faces : ndarray, optional
            Faces of boundary elements.
        interp : str, optional
            'flat' or 'curv' particle boundaries.
        compute_normals : bool, optional
            Whether to compute normal vectors. Default True.
        """
        self.interp = interp
        self.verts2 = None
        self.faces2 = None

        if verts is None or faces is None:
            self.verts = np.array([]).reshape(0, 3)
            self.faces = np.array([]).reshape(0, 4)
            self.pos = np.array([]).reshape(0, 3)
            self.nvec = np.array([]).reshape(0, 3)
            self.tvec1 = np.array([]).reshape(0, 3)
            self.tvec2 = np.array([]).reshape(0, 3)
            self.area = np.array([])
            return

        self.verts = np.asarray(verts, dtype=float)
        faces = np.asarray(faces)

        # Ensure faces has 4 columns (pad with NaN for triangles)
        if faces.ndim == 1:
            faces = faces.reshape(1, -1)
        if faces.shape[1] == 3:
            self.faces = np.column_stack([faces, np.full(len(faces), np.nan)])
        else:
            self.faces = faces.astype(float)

        # Compute auxiliary information
        if compute_normals:
            self._compute_normals()
        else:
            n = len(self.faces)
            self.pos = np.zeros((n, 3))
            self.nvec = np.zeros((n, 3))
            self.tvec1 = np.zeros((n, 3))
            self.tvec2 = np.zeros((n, 3))
            self.area = np.zeros(n)

    def _compute_normals(self) -> None:
        """Compute face centroids, normals, tangents, and areas."""
        n_faces = len(self.faces)
        self.pos = np.zeros((n_faces, 3))
        self.nvec = np.zeros((n_faces, 3))
        self.tvec1 = np.zeros((n_faces, 3))
        self.tvec2 = np.zeros((n_faces, 3))
        self.area = np.zeros(n_faces)

        for i, face in enumerate(self.faces):
            # Get valid vertex indices
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            vertices = self.verts[indices]

            # Centroid
            self.pos[i] = vertices.mean(axis=0)

            if len(indices) >= 3:
                # Compute two edge vectors
                v0, v1, v2 = vertices[0], vertices[1], vertices[2]
                e1 = v1 - v0
                e2 = v2 - v0

                # Normal vector (cross product)
                normal = np.cross(e1, e2)
                norm_mag = np.linalg.norm(normal)

                if norm_mag > 1e-12:
                    self.nvec[i] = normal / norm_mag

                    # Tangent vectors
                    self.tvec1[i] = vecnormalize(e1)
                    self.tvec2[i] = np.cross(self.nvec[i], self.tvec1[i])

                # Area computation
                if len(indices) == 3:
                    # Triangle area = 0.5 * |e1 x e2|
                    self.area[i] = 0.5 * norm_mag
                else:
                    # Quadrilateral: sum of two triangles
                    v3 = vertices[3]
                    e3 = v3 - v0
                    area1 = 0.5 * np.linalg.norm(np.cross(e1, e2))
                    area2 = 0.5 * np.linalg.norm(np.cross(e2, e3))
                    self.area[i] = area1 + area2

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self.faces)

    @property
    def n_verts(self) -> int:
        """Number of vertices."""
        return len(self.verts)

    def is_triangle(self) -> np.ndarray:
        """Return boolean array indicating which faces are triangles."""
        return np.isnan(self.faces[:, 3])

    def is_quad(self) -> np.ndarray:
        """Return boolean array indicating which faces are quadrilaterals."""
        return ~np.isnan(self.faces[:, 3])

    def shift(self, displacement: np.ndarray) -> 'Particle':
        """
        Shift particle by given displacement.

        Parameters
        ----------
        displacement : array_like
            Displacement vector [dx, dy, dz].

        Returns
        -------
        Particle
            Shifted particle (new object).
        """
        displacement = np.asarray(displacement)
        new_verts = self.verts + displacement
        p = Particle(new_verts, self.faces.copy(), self.interp)
        if self.verts2 is not None:
            p.verts2 = self.verts2 + displacement
            p.faces2 = self.faces2.copy()
        return p

    def scale(self, factor: Union[float, np.ndarray]) -> 'Particle':
        """
        Scale particle by given factor.

        Parameters
        ----------
        factor : float or array_like
            Scale factor. Can be scalar or [sx, sy, sz].

        Returns
        -------
        Particle
            Scaled particle (new object).
        """
        factor = np.asarray(factor)
        new_verts = self.verts * factor
        p = Particle(new_verts, self.faces.copy(), self.interp)
        if self.verts2 is not None:
            p.verts2 = self.verts2 * factor
            p.faces2 = self.faces2.copy()
        return p

    def rot(self, angle: float, axis: int = 2) -> 'Particle':
        """
        Rotate particle around specified axis.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : int
            Axis of rotation: 0=x, 1=y, 2=z.

        Returns
        -------
        Particle
            Rotated particle (new object).
        """
        c, s = np.cos(angle), np.sin(angle)

        if axis == 0:  # x-axis
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1:  # y-axis
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:  # z-axis
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        new_verts = self.verts @ R.T
        p = Particle(new_verts, self.faces.copy(), self.interp)
        if self.verts2 is not None:
            p.verts2 = self.verts2 @ R.T
            p.faces2 = self.faces2.copy()
        return p

    def flip(self, axis: int) -> 'Particle':
        """
        Flip particle along specified axis.

        Parameters
        ----------
        axis : int
            Axis to flip: 0=x, 1=y, 2=z.

        Returns
        -------
        Particle
            Flipped particle (new object).
        """
        new_verts = self.verts.copy()
        new_verts[:, axis] = -new_verts[:, axis]

        # Reverse face winding to maintain outward normals
        new_faces = self.faces.copy()
        new_faces[:, :3] = new_faces[:, [0, 2, 1]]

        p = Particle(new_verts, new_faces, self.interp)
        if self.verts2 is not None:
            p.verts2 = self.verts2.copy()
            p.verts2[:, axis] = -p.verts2[:, axis]
            p.faces2 = self.faces2.copy()
        return p

    def flipfaces(self) -> 'Particle':
        """
        Flip face orientations (reverse normals).

        Returns
        -------
        Particle
            Particle with flipped faces.
        """
        new_faces = self.faces.copy()
        # Reverse vertex order for first 3 vertices
        new_faces[:, :3] = new_faces[:, [0, 2, 1]]
        return Particle(self.verts.copy(), new_faces, self.interp)

    def clean(self, tol: float = 1e-10) -> 'Particle':
        """
        Remove duplicate vertices and update face indices.

        Parameters
        ----------
        tol : float
            Tolerance for considering vertices as duplicates.

        Returns
        -------
        Particle
            Cleaned particle.
        """
        from scipy.spatial import cKDTree

        # Find unique vertices
        tree = cKDTree(self.verts)
        groups = tree.query_ball_tree(tree, tol)

        # Map each vertex to its representative (first in group)
        vertex_map = np.arange(len(self.verts))
        for group in groups:
            if len(group) > 1:
                rep = min(group)
                for idx in group:
                    vertex_map[idx] = rep

        # Get unique vertices
        unique_mask = np.array([vertex_map[i] == i for i in range(len(self.verts))])
        unique_verts = self.verts[unique_mask]

        # Create mapping from old to new indices
        old_to_new = np.zeros(len(self.verts), dtype=int)
        new_idx = 0
        for i in range(len(self.verts)):
            if unique_mask[i]:
                old_to_new[i] = new_idx
                new_idx += 1
            else:
                old_to_new[i] = old_to_new[vertex_map[i]]

        # Update face indices
        new_faces = self.faces.copy()
        valid = ~np.isnan(new_faces)
        new_faces[valid] = old_to_new[new_faces[valid].astype(int)]

        return Particle(unique_verts, new_faces, self.interp)

    def midpoints(self, mode: str = 'flat') -> 'Particle':
        """
        Add midpoints to edges for curved interpolation.

        Parameters
        ----------
        mode : str
            'flat' or 'curv'.

        Returns
        -------
        Particle
            Particle with midpoint vertices added.
        """
        # For triangular faces, add midpoints on each edge
        edges = []
        edge_midpoints = {}

        # Collect all edges
        for i, face in enumerate(self.faces):
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            n = len(indices)
            for j in range(n):
                v1, v2 = indices[j], indices[(j + 1) % n]
                edge = tuple(sorted([v1, v2]))
                if edge not in edge_midpoints:
                    midpoint = 0.5 * (self.verts[v1] + self.verts[v2])
                    edge_midpoints[edge] = len(self.verts) + len(edge_midpoints)
                    edges.append(midpoint)

        if len(edges) == 0:
            return self

        # Create extended vertex list
        new_verts = np.vstack([self.verts, np.array(edges)])

        # Create extended faces with midpoint indices
        # faces2 format: [v0, v01, v1, v12, v2, v20, v3, v30, ...]
        new_faces = []
        for face in self.faces:
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            n = len(indices)
            new_face = []
            for j in range(n):
                v1, v2 = indices[j], indices[(j + 1) % n]
                edge = tuple(sorted([v1, v2]))
                mid_idx = edge_midpoints[edge]
                new_face.extend([indices[j], mid_idx])
            # Pad to uniform length
            while len(new_face) < 8:
                new_face.append(np.nan)
            new_faces.append(new_face[:8])

        p = Particle(self.verts.copy(), self.faces.copy(), self.interp, compute_normals=False)
        p.verts2 = new_verts
        p.faces2 = np.array(new_faces)

        return p

    def curvature(self) -> np.ndarray:
        """
        Compute mean curvature at face centroids.

        Returns
        -------
        ndarray
            Mean curvature at each face.
        """
        # Simplified curvature estimation using discrete differential geometry
        curv = np.zeros(self.n_faces)

        # Build vertex-face adjacency
        vertex_faces = [[] for _ in range(self.n_verts)]
        for i, face in enumerate(self.faces):
            valid = ~np.isnan(face)
            for idx in face[valid].astype(int):
                vertex_faces[idx].append(i)

        # Estimate curvature from normal variation
        for i in range(self.n_faces):
            valid = ~np.isnan(self.faces[i])
            indices = self.faces[i][valid].astype(int)

            # Get neighboring faces
            neighbor_faces = set()
            for idx in indices:
                neighbor_faces.update(vertex_faces[idx])
            neighbor_faces.discard(i)

            if len(neighbor_faces) > 0:
                neighbor_normals = self.nvec[list(neighbor_faces)]
                # Curvature from normal deviation
                dot_products = np.dot(neighbor_normals, self.nvec[i])
                curv[i] = np.arccos(np.clip(dot_products, -1, 1)).mean()

        return curv

    def __add__(self, other: 'Particle') -> 'Particle':
        """Concatenate two particles."""
        if self.n_faces == 0:
            return other
        if other.n_faces == 0:
            return self

        # Offset face indices for second particle
        n_verts1 = len(self.verts)
        new_faces2 = other.faces.copy()
        valid = ~np.isnan(new_faces2)
        new_faces2[valid] = new_faces2[valid] + n_verts1

        new_verts = np.vstack([self.verts, other.verts])
        new_faces = np.vstack([self.faces, new_faces2])

        return Particle(new_verts, new_faces, self.interp)

    def __repr__(self) -> str:
        return f"Particle(n_verts={self.n_verts}, n_faces={self.n_faces}, interp='{self.interp}')"
