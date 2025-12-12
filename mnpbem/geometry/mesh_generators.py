"""
Mesh generation functions for various particle shapes.
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from .particle import Particle


def trisphere(n, diameter=1.0):
    """
    Generate a triangulated sphere using quasi-uniform point distribution.

    Uses Fibonacci sphere algorithm for uniform point distribution and
    Delaunay triangulation for face generation.

    Parameters
    ----------
    n : int
        Number of vertices (approximate). Will use closest available value.
    diameter : float, optional
        Diameter of sphere in nm. Default: 1.0

    Returns
    -------
    particle : Particle
        Triangulated sphere particle

    Examples
    --------
    >>> # Create 10nm sphere with ~144 vertices
    >>> sphere = trisphere(144, 10.0)
    >>> print(f"Vertices: {sphere.nverts}, Faces: {sphere.nfaces}")
    """
    # Generate Fibonacci sphere points for quasi-uniform distribution
    verts = _fibonacci_sphere(n)

    # Perform spherical Delaunay triangulation
    faces = _sphere_triangulate(verts)

    # Scale to desired diameter
    verts = verts * (diameter / 2.0)

    # Create particle
    p = Particle(verts, faces)

    return p


def _fibonacci_sphere(n):
    """
    Generate approximately n points uniformly distributed on unit sphere
    using Fibonacci spiral.

    Parameters
    ----------
    n : int
        Approximate number of points

    Returns
    -------
    points : ndarray, shape (m, 3)
        Points on unit sphere (m ≈ n)
    """
    indices = np.arange(0, n, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / n)  # Latitude
    theta = np.pi * (1 + 5**0.5) * indices  # Golden angle spiral

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    points = np.column_stack([x, y, z])

    # Normalize to ensure points are exactly on unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    return points


def _sphere_triangulate(verts):
    """
    Triangulate points on sphere using stereographic projection.

    Based on MATLAB MNPBEM sphtriangulate.m

    Parameters
    ----------
    verts : ndarray, shape (n, 3)
        Vertices on sphere surface

    Returns
    -------
    faces : ndarray, shape (m, 3)
        Triangle face indices (0-indexed)
    """
    n = len(verts)

    # Step 1: Use first vertex as projection center
    center = verts[0]

    # Build rotation matrix that rotates first point to [0, 0, -1]
    r3 = -center
    if center[2] != 0:
        r2 = np.array([0, -r3[2], r3[1]])
    else:
        r2 = np.array([-r3[1], r3[0], 0])
    r2 = r2 / np.linalg.norm(r2)
    r1 = np.cross(r3, r2)

    rot = np.array([r1, r2, r3])

    # Rotate all vertices except first
    vertr = verts[1:] @ rot.T

    # Project to z=0 plane from center [0, 0, -1]
    tp = -1.0 / (vertr[:, 2] + 1)
    xp = vertr[:, 0] * tp
    yp = vertr[:, 1] * tp

    # Step 2: Delaunay triangulation of projected points
    points_2d = np.column_stack([xp, yp])
    tri = Delaunay(points_2d)
    faces = tri.simplices

    # Ensure outward-pointing normals
    vertp = np.column_stack([xp, yp, np.zeros(n - 1)])
    u = vertp[faces[:, 0]] - vertp[faces[:, 1]]
    v = vertp[faces[:, 2]] - vertp[faces[:, 1]]
    w = np.cross(u, v)

    # Flip faces with inward normals
    flip_idx = w[:, 2] > 0
    faces[flip_idx, :] = faces[flip_idx, :][:, [0, 2, 1]]

    # Step 3: Connect projection center to convex hull
    hull = ConvexHull(points_2d)
    hull_verts = hull.vertices

    # Create triangles connecting to center (vertex 0)
    n_hull = len(hull_verts)
    hull_faces = np.zeros((n_hull, 3), dtype=int)
    for i in range(n_hull):
        hull_faces[i] = [hull_verts[(i+1) % n_hull] + 1,
                         hull_verts[i] + 1,
                         0]

    # Combine all faces (shift indices by +1 for Delaunay part)
    all_faces = np.vstack([faces + 1, hull_faces])

    return all_faces


def triellipsoid(n, axes):
    """
    Generate triangulated ellipsoid.

    Parameters
    ----------
    n : int
        Number of vertices
    axes : array_like, shape (3,)
        Semi-axes lengths [a, b, c] in nm

    Returns
    -------
    particle : Particle
        Triangulated ellipsoid
    """
    # Start with unit sphere
    verts = _fibonacci_sphere(n)
    faces = _sphere_triangulate(verts)

    # Scale by axes
    verts = verts * np.array(axes)

    return Particle(verts, faces)
