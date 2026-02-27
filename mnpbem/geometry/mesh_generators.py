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


# ===========================================================================
# Helper: surf2patch equivalent
# ===========================================================================

def _surf2patch(x, y, z, triangles = False):
    # Python equivalent of MATLAB surf2patch(x, y, z) / surf2patch(x, y, z, 'triangles')
    # x, y, z are 2D arrays of shape (m, n). Returns (faces, verts).
    m, n = x.shape
    verts = np.column_stack([x.ravel(order = 'F'), y.ravel(order = 'F'), z.ravel(order = 'F')])

    faces_list = []
    for j in range(n - 1):
        for i in range(m - 1):
            # Vertex indices (column-major order like MATLAB)
            v00 = j * m + i
            v10 = j * m + (i + 1)
            v01 = (j + 1) * m + i
            v11 = (j + 1) * m + (i + 1)

            if triangles:
                # Two triangles per quad
                faces_list.append([v00, v10, v11])
                faces_list.append([v00, v11, v01])
            else:
                # Quadrilateral
                faces_list.append([v00, v10, v11, v01])

    faces = np.array(faces_list, dtype = int)
    return faces, verts


# ===========================================================================
# fvgrid: convert parametric surface to face-vertex structure
# ===========================================================================

def fvgrid(x: np.ndarray,
        y: np.ndarray,
        triangles: bool = False) -> tuple:
    # MATLAB: Particles/particleshapes/misc/fvgrid.m
    # Convert 2D grid to face-vertex structure
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)

    # If 1D, meshgrid them
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)

    z = np.zeros_like(x)

    # Use surf2patch equivalent
    faces, verts = _surf2patch(x, y, z, triangles = triangles)

    # MATLAB: fliplr(faces) -- reverse column order for correct normals
    faces = faces[:, ::-1]

    # Create particle with norm='off'
    p = Particle(verts, faces, norm = 'off')

    # Add midpoints (flat)
    p = _add_midpoints_flat(p)

    return p.verts2, p.faces2


# ===========================================================================
# trispheresegment: discretized surface of sphere segment
# ===========================================================================

def trispheresegment(phi: np.ndarray,
        theta: np.ndarray,
        diameter: float = 1.0,
        **kwargs) -> 'Particle':
    # MATLAB: Particles/particleshapes/trispheresegment.m
    phi = np.asarray(phi, dtype = float)
    theta = np.asarray(theta, dtype = float)

    # Meshgrid phi and theta
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Spherical to cartesian
    x = diameter / 2.0 * np.sin(theta_grid) * np.cos(phi_grid)
    y = diameter / 2.0 * np.sin(theta_grid) * np.sin(phi_grid)
    z = diameter / 2.0 * np.cos(theta_grid)

    # Use surf2patch for quadrilateral faces
    faces, verts = _surf2patch(x, y, z, triangles = False)
    faces = faces[:, ::-1]  # flip for correct normals

    p = Particle(verts, faces)
    p = p.clean()

    # Add midpoints for curved particle boundary
    p = _add_midpoints_flat(p)

    # Rescale vertices to sphere surface
    norms = np.sqrt(np.sum(p.verts2 ** 2, axis = 1, keepdims = True))
    # Avoid division by zero for points at origin
    norms = np.maximum(norms, 1e-30)
    verts2 = 0.5 * diameter * (p.verts2 / norms)

    # Create particle with midpoints
    p = Particle(verts2, p.faces2, **kwargs)

    return p


# ===========================================================================
# trirod: cylinder with hemispherical caps
# ===========================================================================

def trirod(diameter: float,
        height: float,
        n: list = None,
        **kwargs) -> 'Particle':
    # MATLAB: Particles/particleshapes/trirod.m
    if n is None:
        n = [15, 20, 20]
    assert len(n) == 3, '[error] n must have 3 elements [nphi, ntheta, nz]'

    nphi, ntheta, nz_cyl = n

    # Angles
    phi = np.linspace(0, 2 * np.pi, nphi)
    theta = np.linspace(0, 0.5 * np.pi, ntheta)

    # Upper cap: sphere segment shifted up
    cap1 = trispheresegment(phi, theta, diameter, **kwargs)
    cap1.shift([0, 0, 0.5 * (height - diameter)])

    # Lower cap: flip cap1 along z-axis
    cap2 = cap1.flip(2)

    # z-values for cylinder
    z_vals = 0.5 * np.linspace(-1, 1, nz_cyl) * (height - diameter)

    # Grid for cylinder
    verts_grid, faces_grid = fvgrid(phi, z_vals)
    # Extract phi and z from grid vertices
    phi_cyl = verts_grid[:, 0]
    z_cyl = verts_grid[:, 1]

    # Cylinder coordinates
    x_cyl = 0.5 * diameter * np.cos(phi_cyl)
    y_cyl = 0.5 * diameter * np.sin(phi_cyl)

    # Create cylinder particle
    cyl_verts = np.column_stack([x_cyl, y_cyl, z_cyl])
    cyl = Particle(cyl_verts, faces_grid, **kwargs)

    # Compose particle: cap1 + cap2 + cylinder, then clean
    p = (cap1 + cap2 + cyl).clean()

    return p


# ===========================================================================
# tricube: cube with rounded edges
# ===========================================================================

def _square_grid(n: int, e: float) -> tuple:
    # MATLAB: square() subfunction in tricube.m
    u = np.linspace(-0.5 ** e, 0.5 ** e, n)

    verts, faces = fvgrid(u, u)

    x = np.sign(verts[:, 0]) * np.abs(verts[:, 0]) ** (1.0 / e)
    y = np.sign(verts[:, 1]) * np.abs(verts[:, 1]) ** (1.0 / e)

    return x, y, faces


def tricube(n: int,
        length: float = 1.0,
        e: float = 0.25,
        **kwargs) -> 'Particle':
    # MATLAB: Particles/particleshapes/tricube.m
    # Make length an array of 3
    if np.isscalar(length):
        length = np.array([length, length, length], dtype = float)
    else:
        length = np.asarray(length, dtype = float)
        if length.size != 3:
            length = np.full(3, length.flat[0])

    # Discretize single side of cube
    x, y, faces = _square_grid(n, e)
    z = 0.5 * np.ones_like(x)

    # Put together 6 cube sides
    # MATLAB:  [x, y, z], [y, x, -z], [y, z, x], [x, -z, y], [z, x, y], [-z, y, x]
    p1 = Particle(np.column_stack([x, y, z]), faces)
    p2 = Particle(np.column_stack([y, x, -z]), faces)
    p3 = Particle(np.column_stack([y, z, x]), faces)
    p4 = Particle(np.column_stack([x, -z, y]), faces)
    p5 = Particle(np.column_stack([z, x, y]), faces)
    p6 = Particle(np.column_stack([-z, y, x]), faces)

    p = (p1 + p2 + p3 + p4 + p5 + p6).clean()

    # Convert to spherical coordinates for super-sphere rounding
    phi_sph, theta_sph = _cart2sph(p.verts2[:, 0], p.verts2[:, 1], p.verts2[:, 2])

    # Signed power functions
    def isin(x):
        return np.sign(np.sin(x)) * np.abs(np.sin(x)) ** e

    def icos(x):
        return np.sign(np.cos(x)) * np.abs(np.cos(x)) ** e

    # Super-sphere vertices
    x_new = 0.5 * icos(theta_sph) * icos(phi_sph)
    y_new = 0.5 * icos(theta_sph) * isin(phi_sph)
    z_new = 0.5 * isin(theta_sph)

    # Create final particle and scale
    p = Particle(np.column_stack([x_new, y_new, z_new]), p.faces2, **kwargs)
    p.scale(length)

    return p


def _cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    # MATLAB cart2sph equivalent
    # Returns (azimuth, elevation) -- note MATLAB convention
    hxy = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)  # azimuth
    theta = np.arctan2(z, hxy)  # elevation
    return phi, theta


# ===========================================================================
# tritorus: triangulated torus
# ===========================================================================

def tritorus(diameter: float,
        rad: float,
        n: list = None,
        **kwargs) -> 'Particle':
    # MATLAB: Particles/particleshapes/tritorus.m
    if n is None:
        n = [21, 21]
    if np.isscalar(n):
        n = [n, n]

    # Grid triangulation
    verts_grid, faces_grid = fvgrid(
        np.linspace(0, 2 * np.pi, n[0]),
        np.linspace(0, 2 * np.pi, n[1]))

    # Angles
    phi = verts_grid[:, 0]
    theta = verts_grid[:, 1]

    # Coordinates of torus
    x = (0.5 * diameter + rad * np.cos(theta)) * np.cos(phi)
    y = (0.5 * diameter + rad * np.cos(theta)) * np.sin(phi)
    z = rad * np.sin(theta)

    # Make torus
    p = Particle(np.column_stack([x, y, z]), faces_grid, **kwargs).clean()

    return p


# ===========================================================================
# trispherescale: deform surface of sphere
# ===========================================================================

def trispherescale(p: 'Particle',
        scale: np.ndarray,
        unit: bool = False) -> 'Particle':
    # MATLAB: Particles/particleshapes/trispherescale.m
    scale = np.asarray(scale, dtype = float)

    if unit:
        scale = scale / np.max(scale)

    # If scale has same length as nfaces, interpolate to vertices
    if scale.size == p.nfaces:
        scale = p.interp_to_verts(scale)

    # Scale vertex positions
    p.verts = np.reshape(scale, (-1, 1)) * p.verts
    if p.verts2 is not None:
        # For verts2, we need to handle the extra midpoints
        # The first nverts entries match verts, rest are midpoints
        scale2 = np.empty(len(p.verts2), dtype = float)
        scale2[:len(scale)] = scale
        # Interpolate scale for midpoints
        if len(p.verts2) > len(scale):
            edges, _ = _get_edges(p.verts, p.faces)
            for i in range(len(edges)):
                scale2[len(scale) + i] = 0.5 * (scale[edges[i, 0]] + scale[edges[i, 1]])
        p.verts2 = np.reshape(scale2, (-1, 1)) * p.verts2

    p._norm()
    return p


# ===========================================================================
# tripolygon: 3D particle from 2D polygon + edge profile
# ===========================================================================

def tripolygon(poly, edge, **kwargs):
    # MATLAB: Particles/particleshapes/tripolygon.m
    # Creates 3D nanostructure from 2D polygon cross-section + edge profile
    from .polygon3 import Polygon3
    from .edgeprofile import EdgeProfile

    # handle single polygon or list of polygons
    if not isinstance(poly, (list, tuple)):
        polys = [poly]
    else:
        polys = list(poly)

    # check edge profile type: rounded or sharp edges
    has_nan = np.any(np.isnan(edge.pos[:, 0]))
    nan_count_at_zero = np.sum(edge.pos[:, 0] == 0)
    all_not_nan = not has_nan

    if all_not_nan or (has_nan and nan_count_at_zero != 1):
        # both edges rounded (or both sharp -- mode '11')
        p = _tripolygon_both_rounded(polys, edge, **kwargs)
    elif np.isnan(edge.pos[0, 0]):
        # sharp lower edge
        p = _tripolygon_sharp_lower(polys, edge, **kwargs)
    else:
        # sharp upper edge
        p = _tripolygon_sharp_upper(polys, edge, **kwargs)

    return p


def _tripolygon_both_rounded(polys, edge, **kwargs):
    # MATLAB tripolygon.m -- case: both edges rounded
    from .polygon3 import Polygon3

    # create polygon3 objects at zmin and zmax
    polys1 = [Polygon3(p, edge.zmin) for p in polys]
    polys2 = [Polygon3(p, edge.zmax) for p in polys]

    # lower plate (dir = -1)
    plates1 = []
    for p3 in polys1:
        plate, _ = p3.plate(dir = -1, edge = edge, **kwargs)
        plates1.append(plate)

    # upper plate (dir = +1)
    polys_out = []
    plates2 = []
    for p3 in polys2:
        plate, p3_out = p3.plate(dir = 1, edge = edge, **kwargs)
        plates2.append(plate)
        polys_out.append(p3_out)

    # vertical ribbon (side walls)
    ribbons = []
    for p3_out in polys_out:
        ribbon, _, _ = p3_out.vribbon(edge = edge)
        ribbons.append(ribbon)

    # combine all particles
    all_parts = plates1 + plates2 + ribbons
    p = all_parts[0]
    for part in all_parts[1:]:
        p = p + part

    p = p.clean()
    return p


def _tripolygon_sharp_lower(polys, edge, **kwargs):
    # MATLAB tripolygon.m -- case: sharp lower edge (NaN at start)
    from .polygon3 import Polygon3

    # polygon3 objects at zmax
    polys3 = [Polygon3(p, edge.zmax) for p in polys]

    # upper plate
    polys_out = []
    plates1 = []
    for p3 in polys3:
        plate, p3_out = p3.plate(dir = 1, edge = edge, **kwargs)
        plates1.append(plate)
        polys_out.append(p3_out)

    # vertical ribbon
    ribbons = []
    lo_polys = []
    for p3_out in polys_out:
        ribbon, _, lo = p3_out.vribbon(edge = edge)
        ribbons.append(ribbon)
        lo_polys.append(lo)

    # lower plate (at zmin, using the lower boundary polygon)
    plates2 = []
    for lo_p3 in lo_polys:
        lo_p3.z = edge.zmin
        plate, _ = lo_p3.plate(dir = -1, edge = edge, **kwargs)
        plates2.append(plate)

    # combine
    all_parts = plates1 + ribbons + plates2
    p = all_parts[0]
    for part in all_parts[1:]:
        p = p + part

    p = p.clean()
    return p


def _tripolygon_sharp_upper(polys, edge, **kwargs):
    # MATLAB tripolygon.m -- case: sharp upper edge (NaN at end)
    from .polygon3 import Polygon3

    # polygon3 objects at zmin
    polys3 = [Polygon3(p, edge.zmin) for p in polys]

    # lower plate
    polys_out = []
    plates1 = []
    for p3 in polys3:
        plate, p3_out = p3.plate(dir = -1, edge = edge, **kwargs)
        plates1.append(plate)
        polys_out.append(p3_out)

    # vertical ribbon
    ribbons = []
    up_polys = []
    for p3_out in polys_out:
        ribbon, up, _ = p3_out.vribbon(edge = edge)
        ribbons.append(ribbon)
        up_polys.append(up)

    # upper plate (at zmax, using the upper boundary polygon)
    plates2 = []
    for up_p3 in up_polys:
        up_p3.z = edge.zmax
        plate, _ = up_p3.plate(dir = 1, edge = edge, **kwargs)
        plates2.append(plate)

    # combine
    all_parts = plates1 + ribbons + plates2
    p = all_parts[0]
    for part in all_parts[1:]:
        p = p + part

    p = p.clean()
    return p
