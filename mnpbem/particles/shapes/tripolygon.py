"""
Particle from polygon extrusion or revolution.
"""

import numpy as np
from typing import Optional, Tuple

from ..particle import Particle
from .utils import fvgrid


def tripolygon(
    vertices: np.ndarray,
    height: float,
    n_z: int = 10,
    closed: bool = True,
    interp: str = 'curv',
    **kwargs
) -> Particle:
    """
    Create particle by extruding a 2D polygon along z-axis.

    Parameters
    ----------
    vertices : ndarray
        2D polygon vertices, shape (n, 2).
    height : float
        Extrusion height.
    n_z : int
        Number of z-divisions.
    closed : bool
        Whether to close the top and bottom.
    interp : str
        Interpolation type: 'flat' or 'curv'.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Extruded polygon particle.

    Examples
    --------
    >>> # Hexagonal prism
    >>> angles = np.linspace(0, 2*np.pi, 7)[:-1]
    >>> vertices = np.column_stack([np.cos(angles), np.sin(angles)]) * 10
    >>> prism = tripolygon(vertices, 20)
    """
    vertices = np.asarray(vertices)
    n_poly = len(vertices)

    # Z coordinates
    z_vals = np.linspace(-height / 2, height / 2, n_z)

    # Create side walls
    all_verts = []
    all_faces = []

    for i_z, z in enumerate(z_vals):
        for v in vertices:
            all_verts.append([v[0], v[1], z])

    all_verts = np.array(all_verts)

    # Create side faces
    for i_z in range(n_z - 1):
        for i_poly in range(n_poly):
            i_next = (i_poly + 1) % n_poly

            # Bottom-left, bottom-right, top-right, top-left
            v0 = i_z * n_poly + i_poly
            v1 = i_z * n_poly + i_next
            v2 = (i_z + 1) * n_poly + i_next
            v3 = (i_z + 1) * n_poly + i_poly

            # Two triangles per quad
            all_faces.append([v0, v1, v2])
            all_faces.append([v0, v2, v3])

    # Create top and bottom caps if closed
    if closed:
        # Bottom cap (z = -height/2)
        bottom_center = len(all_verts)
        all_verts = np.vstack([all_verts, [0, 0, -height / 2]])

        for i in range(n_poly):
            i_next = (i + 1) % n_poly
            all_faces.append([bottom_center, i_next, i])  # Reversed for outward normal

        # Top cap (z = height/2)
        top_center = len(all_verts)
        all_verts = np.vstack([all_verts, [0, 0, height / 2]])

        base_top = (n_z - 1) * n_poly
        for i in range(n_poly):
            i_next = (i + 1) % n_poly
            all_faces.append([top_center, base_top + i, base_top + i_next])

    return Particle(all_verts, np.array(all_faces), interp=interp, **kwargs)


def trirevolution(
    profile: np.ndarray,
    n_phi: int = 30,
    interp: str = 'curv',
    **kwargs
) -> Particle:
    """
    Create particle by revolving a 2D profile around z-axis.

    Parameters
    ----------
    profile : ndarray
        Profile curve as (r, z) coordinates, shape (n, 2).
    n_phi : int
        Number of azimuthal divisions.
    interp : str
        Interpolation type: 'flat' or 'curv'.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Revolved profile particle.

    Examples
    --------
    >>> # Wine glass shape
    >>> z = np.linspace(0, 10, 20)
    >>> r = 2 + np.sin(z * 0.5)
    >>> profile = np.column_stack([r, z])
    >>> glass = trirevolution(profile)
    """
    profile = np.asarray(profile)
    n_profile = len(profile)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Create vertices
    all_verts = []
    for p in profile:
        r, z = p[0], p[1]
        for ph in phi:
            x = r * np.cos(ph)
            y = r * np.sin(ph)
            all_verts.append([x, y, z])

    all_verts = np.array(all_verts)

    # Create faces
    all_faces = []
    for i_p in range(n_profile - 1):
        for i_phi in range(n_phi):
            i_phi_next = (i_phi + 1) % n_phi

            v0 = i_p * n_phi + i_phi
            v1 = i_p * n_phi + i_phi_next
            v2 = (i_p + 1) * n_phi + i_phi_next
            v3 = (i_p + 1) * n_phi + i_phi

            all_faces.append([v0, v1, v2])
            all_faces.append([v0, v2, v3])

    return Particle(np.array(all_verts), np.array(all_faces), interp=interp, **kwargs)
