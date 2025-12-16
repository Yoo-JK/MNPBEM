"""
Mesh generation functions for various particle shapes.
"""

import numpy as np
import os
from scipy.spatial import Delaunay, ConvexHull
from scipy.io import loadmat
from .particle import Particle


def trisphere(n, diameter=1.0, **kwargs):
    """
    Generate a triangulated sphere with curved boundaries.

    Loads pre-computed sphere vertices from MATLAB trisphere.mat file and
    creates curved triangular elements by adding midpoints on sphere surface.

    MATLAB: Particles/particleshapes/trisphere.m

    Parameters
    ----------
    n : int
        Number of vertices. Will use closest available value from:
        [32, 60, 144, 169, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576,
         625, 676, 729, 784, 841, 900, 961, 1024, 1225, 1444]
    diameter : float, optional
        Diameter of sphere in nm. Default: 1.0
    **kwargs : dict
        Additional arguments passed to Particle constructor

    Returns
    -------
    particle : Particle
        Triangulated sphere with curved boundaries (verts2, faces2)

    Examples
    --------
    >>> # Create 80nm sphere with ~144 vertices
    >>> sphere = trisphere(144, 80.0)
    >>> print(f"Vertices: {sphere.nverts}, Faces: {sphere.nfaces}")
    """
    # Saved vertex counts in MATLAB trisphere.mat
    # MATLAB: trisphere.m line 20-21
    nsav = np.array([32, 60, 144, 169, 225, 256, 289, 324, 361, 400, 441, 484,
                     529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1225, 1444])

    # Find closest available number
    # MATLAB: trisphere.m line 24
    ind = np.argmin(np.abs(nsav - n))
    n_actual = nsav[ind]

    if n != n_actual:
        print(f'trisphere: using {n_actual} vertices (closest to requested {n})')

    # Load data from MATLAB .mat file
    # MATLAB: trisphere.m line 26-34
    mat_file = os.path.join(os.path.dirname(__file__), '..', '..',
                            'Particles', 'particleshapes', 'trisphere.mat')

    if not os.path.exists(mat_file):
        # Fallback to Fibonacci sphere if .mat file not found
        print(f'Warning: trisphere.mat not found, using Fibonacci sphere instead')
        return _trisphere_fibonacci(n, diameter, **kwargs)

    # Try to load MATLAB-triangulated version first (for sphere144 only, as test)
    mat_file_tri = os.path.join(os.path.dirname(__file__), '..', '..',
                                'Particles', 'particleshapes', 'trisphere_triangulated.mat')

    faces = None
    if n_actual == 144 and os.path.exists(mat_file_tri):
        try:
            data_tri = loadmat(mat_file_tri, simplify_cells=True)
            sphere_tri = data_tri['sphere144_tri']
            verts = np.column_stack([sphere_tri['x'], sphere_tri['y'], sphere_tri['z']]).astype(float)
            faces = sphere_tri['faces'].astype(int) - 1  # MATLAB 1-indexed to Python 0-indexed
            print(f'trisphere: loaded MATLAB triangulation for sphere{n_actual}')
        except Exception as e:
            print(f'Warning: Could not load MATLAB triangulation: {e}')

    # Load sphere data from original file if not loaded yet
    if faces is None:
        try:
            data = loadmat(mat_file, simplify_cells=True)
            sphere_key = f'sphere{n_actual}'
            sphere = data[sphere_key]

            # Extract vertices
            # MATLAB: trisphere.m line 38
            verts = np.column_stack([sphere['x'], sphere['y'], sphere['z']]).astype(float)
        except (KeyError, ValueError) as e:
            print(f'Warning: Could not load sphere{n_actual} from trisphere.mat: {e}')
            print('Falling back to Fibonacci sphere')
            return _trisphere_fibonacci(n, diameter, **kwargs)

        # Triangulate sphere using Python
        # MATLAB: trisphere.m line 40
        faces = _sphere_triangulate(verts)

    # Rescale to diameter
    # MATLAB: trisphere.m line 49
    verts = 0.5 * verts * diameter

    # Create particle without computing normals
    # MATLAB: trisphere.m line 52
    p = Particle(verts, faces, norm=False)

    # Add midpoints for curved particle boundary
    # MATLAB: trisphere.m line 56
    p = _add_midpoints_flat(p)

    # Project midpoints onto sphere surface
    # MATLAB: trisphere.m line 58-59
    norms = np.linalg.norm(p.verts2, axis=1, keepdims=True)
    verts2 = 0.5 * diameter * (p.verts2 / norms)

    # Create final particle with curved boundaries
    # MATLAB: trisphere.m line 61
    p = Particle(verts2, p.faces2, **kwargs)

    # Set curved interpolation mode
    # MATLAB: trisphere.m uses 'curv' option in particle init
    p.interp = 'curv'
    p._norm()  # Recompute normals for curved boundaries

    return p


def _add_midpoints_flat(p):
    """
    Add midpoints for curved particle boundaries (flat interpolation).

    MATLAB: Particles/@particle/midpoints.m (case 'flat')

    Parameters
    ----------
    p : Particle
        Particle with verts and faces

    Returns
    -------
    p : Particle
        Particle with verts2 and faces2 added
    """
    # Get edges of particle
    # MATLAB: midpoints.m line 20
    edges, edge_indices = _get_edges(p.verts, p.faces)

    # Number of vertices
    # MATLAB: midpoints.m line 22
    n = len(p.verts)

    # Add midpoints to vertex list
    # MATLAB: midpoints.m line 24-25
    edge_midpoints = 0.5 * (p.verts[edges[:, 0]] + p.verts[edges[:, 1]])
    p.verts2 = np.vstack([p.verts, edge_midpoints])

    # Extend face list
    # MATLAB: midpoints.m line 29
    nfaces = len(p.faces)
    p.faces2 = np.column_stack([p.faces, np.full((nfaces, 5), np.nan)])

    # For triangular faces, add edge midpoint indices
    # MATLAB: midpoints.m line 32
    # faces2 columns: [v0, v1, v2, nan, e01, e12, e20, nan, nan]
    # where e01 is midpoint between v0 and v1, etc.
    p.faces2[:, 4] = n + edge_indices[:, 0]  # edge 0-1
    p.faces2[:, 5] = n + edge_indices[:, 1]  # edge 1-2
    p.faces2[:, 6] = n + edge_indices[:, 2]  # edge 2-0

    return p


def _get_edges(verts, faces):
    """
    Get unique edges and their indices for each face.

    MATLAB: Particles/@particle/edges.m

    Parameters
    ----------
    verts : ndarray
        Vertices
    faces : ndarray
        Face indices (triangles)

    Returns
    -------
    edges : ndarray, shape (n_edges, 2)
        Unique edges as pairs of vertex indices
    edge_indices : ndarray, shape (n_faces, 3)
        Index of each edge in the edges array for each face
    """
    nfaces = len(faces)

    # All edges in all faces (each triangle has 3 edges)
    # MATLAB stores edges as (v1, v2), (v2, v3), (v3, v1)
    all_edges = np.zeros((nfaces * 3, 2), dtype=int)
    all_edges[0::3] = faces[:, [0, 1]]  # edge 0: v0-v1
    all_edges[1::3] = faces[:, [1, 2]]  # edge 1: v1-v2
    all_edges[2::3] = faces[:, [2, 0]]  # edge 2: v2-v0

    # Sort edge vertices so (v1, v2) and (v2, v1) are treated as same edge
    all_edges_sorted = np.sort(all_edges, axis=1)

    # Find unique edges
    unique_edges, inverse_indices = np.unique(
        all_edges_sorted, axis=0, return_inverse=True
    )

    # Reshape inverse indices to (nfaces, 3)
    edge_indices = inverse_indices.reshape(nfaces, 3)

    return unique_edges, edge_indices


def _trisphere_fibonacci(n, diameter=1.0, **kwargs):
    """
    Fallback: Generate sphere using Fibonacci algorithm.

    This is used when the MATLAB .mat file is not available.
    """
    # Generate Fibonacci sphere points for quasi-uniform distribution
    verts = _fibonacci_sphere(n)

    # Perform spherical Delaunay triangulation
    faces = _sphere_triangulate(verts)

    # Scale to desired diameter
    verts = verts * (diameter / 2.0)

    # Create particle
    p = Particle(verts, faces, **kwargs)

    # Add midpoints and project to sphere
    p = _add_midpoints_flat(p)
    norms = np.linalg.norm(p.verts2, axis=1, keepdims=True)
    verts2 = (diameter / 2.0) * (p.verts2 / norms)
    p = Particle(verts2, p.faces2, **kwargs)

    # Set curved interpolation mode
    p.interp = 'curv'
    p._norm()  # Recompute normals for curved boundaries

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
