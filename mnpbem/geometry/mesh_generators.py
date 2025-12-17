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


def fvgrid(x, y, triangles=False, **kwargs):
    """
    Convert 2D grid to face-vertex structure.

    MATLAB: Particles/particleshapes/misc/fvgrid.m

    Parameters
    ----------
    x : array_like
        x-coordinates of grid
    y : array_like
        y-coordinates of grid
    triangles : bool
        If True, use triangles rather than quadrilaterals

    Returns
    -------
    verts : ndarray
        Vertices of triangulated grid (with midpoints)
    faces : ndarray
        Faces of triangulated grid
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Create meshgrid if 1D arrays
    if x.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y

    # Create faces and vertices using surf2patch equivalent
    verts, faces = _surf2patch(X, Y, np.zeros_like(X), triangles)

    # Create particle and add midpoints
    p = Particle(verts, faces[:, ::-1], norm=False)  # fliplr(faces) for correct orientation
    p = _add_midpoints_flat(p)

    return p.verts2, p.faces2


def _surf2patch(X, Y, Z, triangles=False):
    """
    Convert surface mesh to face-vertex representation.

    Python equivalent of MATLAB surf2patch.

    Parameters
    ----------
    X, Y, Z : ndarray
        Meshgrid coordinates
    triangles : bool
        If True, split quads into triangles

    Returns
    -------
    verts : ndarray, shape (n_verts, 3)
        Vertex coordinates
    faces : ndarray, shape (n_faces, 3 or 4)
        Face indices (0-indexed)
    """
    ny, nx = X.shape

    # Flatten vertices
    verts = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

    if triangles:
        # Create triangular faces
        faces = []
        for i in range(ny - 1):
            for j in range(nx - 1):
                # Vertex indices of current quad
                v0 = i * nx + j
                v1 = i * nx + (j + 1)
                v2 = (i + 1) * nx + (j + 1)
                v3 = (i + 1) * nx + j

                # Split quad into two triangles
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        faces = np.array(faces)
    else:
        # Create quadrilateral faces
        faces = []
        for i in range(ny - 1):
            for j in range(nx - 1):
                v0 = i * nx + j
                v1 = i * nx + (j + 1)
                v2 = (i + 1) * nx + (j + 1)
                v3 = (i + 1) * nx + j

                faces.append([v0, v1, v2, v3])

        faces = np.array(faces)

    return verts, faces


def trispheresegment(phi, theta, diameter=1.0, triangles=False, **kwargs):
    """
    Discretized surface of sphere segment.

    MATLAB: Particles/particleshapes/trispheresegment.m

    Parameters
    ----------
    phi : array_like
        Azimuthal angles
    theta : array_like
        Polar angles
    diameter : float
        Diameter of sphere
    triangles : bool
        Use triangles rather than quadrilaterals
    **kwargs : dict
        Additional arguments passed to Particle

    Returns
    -------
    p : Particle
        Discretized particle surface
    """
    phi = np.atleast_1d(phi)
    theta = np.atleast_1d(theta)

    # Grid of phi and theta values
    PHI, THETA = np.meshgrid(phi, theta)

    # Sphere coordinates
    x = diameter / 2 * np.sin(THETA) * np.cos(PHI)
    y = diameter / 2 * np.sin(THETA) * np.sin(PHI)
    z = diameter / 2 * np.cos(THETA)

    # Create faces and vertices
    verts, faces = _surf2patch(x, y, z, triangles)

    # Create particle and clean
    p = Particle(verts, faces, **kwargs)
    p = p.clean()

    # Add midpoints for curved boundary
    p = _add_midpoints_flat(p)

    # Rescale vertices to sphere surface
    norms = np.linalg.norm(p.verts2, axis=1, keepdims=True)
    verts2 = 0.5 * diameter * (p.verts2 / norms)

    # Create final particle
    return Particle(verts2, p.faces2, **kwargs)


def trirod(diameter, height, n=None, triangles=False, **kwargs):
    """
    Generate rod-shaped particle (cylinder with hemispherical caps).

    MATLAB: Particles/particleshapes/trirod.m

    Parameters
    ----------
    diameter : float
        Diameter of rod
    height : float
        Total height (length) of rod
    n : array_like, optional
        Number of discretization points [nphi, ntheta, nz]
        Default: [15, 20, 20]
    triangles : bool
        Use triangles rather than quadrilaterals
    **kwargs : dict
        Additional arguments passed to Particle

    Returns
    -------
    p : Particle
        Triangulated rod

    Examples
    --------
    >>> rod = trirod(20, 60)  # 20nm diameter, 60nm height
    """
    if n is None:
        n = [15, 20, 20]
    n = np.atleast_1d(n)
    assert len(n) == 3

    # Angles
    phi = np.linspace(0, 2 * np.pi, n[0])
    theta = np.linspace(0, 0.5 * np.pi, n[1])

    # z-values of cylinder
    z = 0.5 * np.linspace(-1, 1, n[2]) * (height - diameter)

    # Upper cap
    cap1 = trispheresegment(phi, theta, diameter, triangles, **kwargs)
    cap1 = cap1.shift([0, 0, 0.5 * (height - diameter)])

    # Lower cap
    cap2 = cap1.flip(2)  # Flip along z-axis

    # Grid for cylinder discretization
    verts, faces = fvgrid(phi, z, triangles, **kwargs)

    # Cylinder coordinates
    phi_cyl = verts[:, 0]
    z_cyl = verts[:, 1]

    # Make cylinder
    x = 0.5 * diameter * np.cos(phi_cyl)
    y = 0.5 * diameter * np.sin(phi_cyl)

    # Create cylinder particle
    cyl_verts = np.column_stack([x, y, z_cyl])
    cyl = Particle(cyl_verts, faces, **kwargs)

    # Compose particle
    p = cap1 + cap2 + cyl
    return p.clean()


def tricube(n, length=1.0, e=0.25, **kwargs):
    """
    Generate cube particle with rounded edges.

    MATLAB: Particles/particleshapes/tricube.m

    Parameters
    ----------
    n : int
        Grid size
    length : float or array_like
        Length of cube edges. If array, [lx, ly, lz]
    e : float
        Round-off parameter for edges, default 0.25
    **kwargs : dict
        Additional arguments passed to Particle

    Returns
    -------
    p : Particle
        Triangulated cube with rounded edges

    Examples
    --------
    >>> cube = tricube(10, 50)  # 50nm cube
    >>> cube = tricube(10, [40, 40, 60])  # Rectangular box
    """
    # Make length an array
    length = np.atleast_1d(length)
    if len(length) == 1:
        length = np.repeat(length, 3)

    def square(n_pts, e_param):
        """Triangulate square."""
        u = np.linspace(-0.5 ** e_param, 0.5 ** e_param, n_pts)
        verts, faces = fvgrid(u, u)
        # Spacing for grid
        x = np.sign(verts[:, 0]) * np.abs(verts[:, 0]) ** (1 / e_param)
        y = np.sign(verts[:, 1]) * np.abs(verts[:, 1]) ** (1 / e_param)
        return x, y, faces

    # Discretize single side of cube
    x, y, faces = square(n, e)
    z = 0.5 * np.ones_like(x)

    # Put together cube sides
    p1 = Particle(np.column_stack([x, y, z]), faces)
    p2 = Particle(np.column_stack([y, x, -z]), faces)
    p3 = Particle(np.column_stack([y, z, x]), faces)
    p4 = Particle(np.column_stack([x, -z, y]), faces)
    p5 = Particle(np.column_stack([z, x, y]), faces)
    p6 = Particle(np.column_stack([-z, y, x]), faces)

    p = p1 + p2 + p3 + p4 + p5 + p6
    p = p.clean()

    # Get vertex positions in spherical coordinates
    verts = p.verts2 if hasattr(p, 'verts2') else p.verts
    r = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2 + verts[:, 2]**2)
    phi = np.arctan2(verts[:, 1], verts[:, 0])
    theta = np.arcsin(verts[:, 2] / np.maximum(r, 1e-10))

    # Signed sin and cos
    def isin(x):
        return np.sign(np.sin(x)) * np.abs(np.sin(x)) ** e

    def icos(x):
        return np.sign(np.cos(x)) * np.abs(np.cos(x)) ** e

    # Use super-sphere for rounding-off edges
    x_new = 0.5 * icos(theta) * icos(phi)
    y_new = 0.5 * icos(theta) * isin(phi)
    z_new = 0.5 * isin(theta)

    # Make particle object and scale
    faces2 = p.faces2 if hasattr(p, 'faces2') else p.faces
    p = Particle(np.column_stack([x_new, y_new, z_new]), faces2, **kwargs)
    return p.scale(length)


def tritorus(diameter, rad, n=None, triangles=False, **kwargs):
    """
    Generate triangulated torus.

    MATLAB: Particles/particleshapes/tritorus.m

    Parameters
    ----------
    diameter : float
        Diameter of folded cylinder (major diameter)
    rad : float
        Radius of torus tube (minor radius)
    n : array_like, optional
        Number of discretization points [n_major, n_minor]
        Default: [21, 21]
    triangles : bool
        Use triangles rather than quadrilaterals
    **kwargs : dict
        Additional arguments passed to Particle

    Returns
    -------
    p : Particle
        Triangulated torus

    Examples
    --------
    >>> torus = tritorus(50, 10)  # 50nm diameter, 10nm tube radius
    """
    if n is None:
        n = [21, 21]
    n = np.atleast_1d(n)
    if len(n) == 1:
        n = np.array([n[0], n[0]])

    # Grid triangulation
    verts, faces = fvgrid(
        np.linspace(0, 2 * np.pi, n[0]),
        np.linspace(0, 2 * np.pi, n[1]),
        triangles
    )

    # Angles
    phi = verts[:, 0]
    theta = verts[:, 1]

    # Coordinates of torus
    x = (0.5 * diameter + rad * np.cos(theta)) * np.cos(phi)
    y = (0.5 * diameter + rad * np.cos(theta)) * np.sin(phi)
    z = rad * np.sin(theta)

    # Make torus
    p = Particle(np.column_stack([x, y, z]), faces, **kwargs)
    return p.clean()


def trispherescale(p, scale, unit=False):
    """
    Deform surface of sphere by scaling radially.

    MATLAB: Particles/particleshapes/trispherescale.m

    Parameters
    ----------
    p : Particle or ComParticle
        Particle to deform
    scale : ndarray
        Scale factors for each vertex or face
    unit : bool
        If True, normalize scale to maximum of 1

    Returns
    -------
    p : Particle or ComParticle
        Deformed particle
    """
    if unit:
        scale = scale / np.max(scale)

    # Check if p is a ComParticle
    if hasattr(p, 'p') and hasattr(p, 'index'):
        # p is a comparticle
        for i in range(len(p.p)):
            idx = p.index == i
            p.p[i] = trispherescale(p.p[i], scale[idx])
        p._norm()
    else:
        # p is a particle
        scale = np.atleast_1d(scale)
        if len(scale) == p.nfaces:
            # Interpolate from faces to vertices
            scale = p.interp_values(scale)
        p.verts = p.verts * scale[:, np.newaxis]

    return p
