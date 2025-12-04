"""
Triangulated cube with rounded edges.
"""

import numpy as np
from typing import Union, Tuple

from ..particle import Particle
from .utils import fvgrid, cart2sph


def tricube(
    n: int = 10,
    length: Union[float, np.ndarray] = 1.0,
    edge_rounding: float = 0.25,
    interp: str = 'curv',
    **kwargs
) -> Particle:
    """
    Create cube particle with rounded edges.

    Parameters
    ----------
    n : int
        Grid size per face.
    length : float or array_like
        Length of cube edges. Can be scalar or [lx, ly, lz].
    edge_rounding : float
        Round-off parameter for edges (0 = sharp, 1 = very round).
    interp : str
        Interpolation type: 'flat' or 'curv'.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Triangulated cube.

    Examples
    --------
    >>> cube = tricube(10, 20)  # 10x10 grid, 20 nm edge length
    >>> cube = tricube(10, [20, 20, 40])  # Rectangular cuboid
    """
    length = np.atleast_1d(length)
    if len(length) == 1:
        length = np.array([length[0], length[0], length[0]])

    e = edge_rounding

    # Create single face
    x, y, faces = _square_face(n, e)
    z = 0.5 * np.ones_like(x)

    # Create 6 faces of cube
    particles = []

    # +z face
    p1 = Particle(np.column_stack([x, y, z]), faces, interp='flat', compute_normals=False)
    particles.append(p1)

    # -z face (flip z and reverse winding)
    p2 = Particle(np.column_stack([y, x, -z]), faces, interp='flat', compute_normals=False)
    particles.append(p2)

    # +x face
    p3 = Particle(np.column_stack([z, y, x]), faces, interp='flat', compute_normals=False)
    particles.append(p3)

    # -x face
    p4 = Particle(np.column_stack([-z, x, y]), faces, interp='flat', compute_normals=False)
    particles.append(p4)

    # +y face
    p5 = Particle(np.column_stack([x, z, y]), faces, interp='flat', compute_normals=False)
    particles.append(p5)

    # -y face
    p6 = Particle(np.column_stack([y, -z, x]), faces, interp='flat', compute_normals=False)
    particles.append(p6)

    # Combine and clean
    p = particles[0]
    for pi in particles[1:]:
        p = p + pi
    p = p.clean()

    # Apply super-ellipsoid rounding
    if p.verts2 is None:
        p = p.midpoints('flat')

    verts2 = p.verts2 if p.verts2 is not None else p.verts

    # Convert to spherical-like coordinates
    r, theta, phi = cart2sph(verts2[:, 0], verts2[:, 1], verts2[:, 2])

    # Super-ellipsoid formula for rounded cube
    def signed_pow(x, p):
        return np.sign(x) * np.abs(x) ** p

    x_new = 0.5 * signed_pow(np.cos(theta), e) * signed_pow(np.cos(phi), e)
    y_new = 0.5 * signed_pow(np.cos(theta), e) * signed_pow(np.sin(phi), e)
    z_new = 0.5 * signed_pow(np.sin(theta), e)

    verts_rounded = np.column_stack([x_new, y_new, z_new])

    # Scale
    verts_rounded = verts_rounded * length

    # Create final particle
    if p.verts2 is not None:
        p_final = Particle(verts_rounded, p.faces2, interp=interp, **kwargs)
    else:
        p_final = Particle(verts_rounded, p.faces, interp=interp, **kwargs)

    return p_final


def _square_face(n: int, e: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create triangulated square face.

    Parameters
    ----------
    n : int
        Grid size.
    e : float
        Edge rounding parameter.

    Returns
    -------
    x, y : ndarray
        Vertex coordinates.
    faces : ndarray
        Face indices.
    """
    # Use rounding-aware spacing
    u = np.linspace(-0.5 ** e, 0.5 ** e, n)

    # Create grid
    verts, faces = fvgrid(u, u)

    # Transform coordinates
    x = np.sign(verts[:, 0]) * np.abs(verts[:, 0]) ** (1 / e) if e != 0 else verts[:, 0]
    y = np.sign(verts[:, 1]) * np.abs(verts[:, 1]) ** (1 / e) if e != 0 else verts[:, 1]

    # Convert quads to triangles
    tri_faces = []
    for face in faces:
        tri_faces.append([face[0], face[1], face[2]])
        tri_faces.append([face[0], face[2], face[3]])

    return x, y, np.array(tri_faces)
