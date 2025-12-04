"""
Triangulated sphere generation.
"""

import numpy as np
from typing import Optional
from scipy.spatial import ConvexHull

from ..particle import Particle


def sphtriangulate(verts: np.ndarray) -> np.ndarray:
    """
    Triangulate points on a sphere surface using convex hull.

    Parameters
    ----------
    verts : ndarray
        Vertices on sphere surface, shape (n, 3).

    Returns
    -------
    faces : ndarray
        Triangle face indices, shape (n_faces, 3).
    """
    # Use convex hull for triangulation
    hull = ConvexHull(verts)
    return hull.simplices


def _generate_fibonacci_sphere(n: int) -> np.ndarray:
    """
    Generate approximately uniform points on a sphere using Fibonacci spiral.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    points : ndarray
        Points on unit sphere, shape (n, 3).
    """
    indices = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    y = 1 - (indices / (n - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)

    theta = phi * indices

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.column_stack([x, y, z])


def _generate_icosphere(subdivisions: int = 3) -> np.ndarray:
    """
    Generate icosphere vertices.

    Parameters
    ----------
    subdivisions : int
        Number of subdivisions of icosahedron.

    Returns
    -------
    verts : ndarray
        Vertices on unit sphere.
    """
    # Start with icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0

    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])

    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        edge_cache = {}

        for face in faces:
            v = [vertices[face[i]] for i in range(3)]

            # Get midpoints
            mid = []
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge in edge_cache:
                    mid.append(edge_cache[edge])
                else:
                    midpoint = (v[i] + v[(i + 1) % 3]) / 2
                    midpoint = midpoint / np.linalg.norm(midpoint)
                    mid_idx = len(vertices)
                    vertices = np.vstack([vertices, midpoint])
                    edge_cache[edge] = mid_idx
                    mid.append(mid_idx)

            # Create 4 new faces
            new_faces.append([face[0], mid[0], mid[2]])
            new_faces.append([face[1], mid[1], mid[0]])
            new_faces.append([face[2], mid[2], mid[1]])
            new_faces.append([mid[0], mid[1], mid[2]])

        faces = np.array(new_faces)

    return vertices


# Map of available sphere point counts
SPHERE_POINTS = {
    32: 2, 60: 2, 144: 3, 169: 3, 225: 3, 256: 3, 289: 3, 324: 3,
    361: 3, 400: 4, 441: 4, 484: 4, 529: 4, 576: 4, 625: 4,
    676: 4, 729: 4, 784: 4, 841: 4, 900: 4, 961: 5, 1024: 5,
    1225: 5, 1444: 5
}


def trisphere(
    n: int = 144,
    diameter: float = 1.0,
    interp: str = 'curv',
    **kwargs
) -> Particle:
    """
    Create triangulated sphere.

    Parameters
    ----------
    n : int
        Approximate number of vertices. Available: 32, 60, 144, 169, 225, 256,
        289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900,
        961, 1024, 1225, 1444. Other values will use closest available.
    diameter : float
        Diameter of sphere.
    interp : str
        Interpolation type: 'flat' or 'curv'.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Triangulated sphere.

    Examples
    --------
    >>> sphere = trisphere(144, 10)  # 144 vertices, 10 nm diameter
    >>> sphere.n_verts
    144
    """
    # Find closest available n
    available = list(SPHERE_POINTS.keys())
    idx = np.argmin(np.abs(np.array(available) - n))
    n_actual = available[idx]

    if n != n_actual:
        print(f"trisphere: using {n_actual} vertices (closest to {n})")

    # Determine subdivision level
    if n_actual <= 60:
        subdivisions = 2
    elif n_actual <= 400:
        subdivisions = 3
    elif n_actual <= 960:
        subdivisions = 4
    else:
        subdivisions = 5

    # Generate icosphere or fibonacci sphere
    if n_actual in [32, 60]:
        verts = _generate_icosphere(subdivisions)
    else:
        # Use Fibonacci sphere for more control over point count
        verts = _generate_fibonacci_sphere(n_actual)

    # Triangulate
    faces = sphtriangulate(verts)

    # Scale to desired diameter
    verts = verts * (diameter / 2)

    # Create basic particle
    p = Particle(verts, faces, interp='flat', compute_normals=False)

    # Add midpoints for curved interpolation
    p = p.midpoints('flat')

    # Rescale midpoint vertices to sphere surface
    if p.verts2 is not None:
        norms = np.linalg.norm(p.verts2, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        p.verts2 = p.verts2 / norms * (diameter / 2)

    # Create final particle with curved boundary data
    if p.verts2 is not None:
        p_final = Particle(p.verts2, p.faces2, interp=interp, **kwargs)
    else:
        p_final = Particle(verts, faces, interp=interp, **kwargs)

    return p_final
