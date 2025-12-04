"""
Triangulated torus.
"""

import numpy as np
from typing import Tuple

from ..particle import Particle
from .utils import fvgrid


def tritorus(
    major_radius: float,
    minor_radius: float,
    n: Tuple[int, int] = (30, 20),
    interp: str = 'curv',
    **kwargs
) -> Particle:
    """
    Create triangulated torus (donut shape).

    Parameters
    ----------
    major_radius : float
        Distance from center of torus to center of tube.
    minor_radius : float
        Radius of the tube.
    n : tuple of int
        Number of discretization points [n_toroidal, n_poloidal].
    interp : str
        Interpolation type: 'flat' or 'curv'.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Triangulated torus.

    Examples
    --------
    >>> torus = tritorus(20, 5)  # R=20 nm, r=5 nm
    """
    n_toroidal, n_poloidal = n

    # Angles
    phi = np.linspace(0, 2 * np.pi, n_toroidal, endpoint=False)  # Toroidal angle
    theta = np.linspace(0, 2 * np.pi, n_poloidal, endpoint=False)  # Poloidal angle

    # Create grid (both directions are periodic)
    verts, faces = fvgrid(phi, theta, periodic_u=True, periodic_v=True)

    # Convert to 3D coordinates
    phi_v = verts[:, 0]
    theta_v = verts[:, 1]

    # Torus parametric equations
    x = (major_radius + minor_radius * np.cos(theta_v)) * np.cos(phi_v)
    y = (major_radius + minor_radius * np.cos(theta_v)) * np.sin(phi_v)
    z = minor_radius * np.sin(theta_v)

    verts_3d = np.column_stack([x, y, z])

    # Convert quads to triangles
    tri_faces = []
    for face in faces:
        tri_faces.append([face[0], face[1], face[2]])
        tri_faces.append([face[0], face[2], face[3]])

    return Particle(verts_3d, np.array(tri_faces), interp=interp, **kwargs)
