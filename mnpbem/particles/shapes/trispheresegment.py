"""
Triangulated sphere segment.
"""

import numpy as np
from typing import Union

from ..particle import Particle
from .utils import fvgrid


def trispheresegment(
    phi: np.ndarray,
    theta: np.ndarray,
    diameter: float = 1.0,
    interp: str = 'curv',
    **kwargs
) -> Particle:
    """
    Create spherical segment (partial sphere surface).

    Parameters
    ----------
    phi : ndarray
        Azimuthal angles (0 to 2*pi for full circle).
    theta : ndarray
        Polar angles (0 = north pole, pi/2 = equator, pi = south pole).
    diameter : float
        Diameter of sphere.
    interp : str
        Interpolation type: 'flat' or 'curv'.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Spherical segment.

    Examples
    --------
    >>> # Upper hemisphere
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> theta = np.linspace(0, np.pi/2, 10)
    >>> hemisphere = trispheresegment(phi, theta, 10)
    """
    phi = np.asarray(phi)
    theta = np.asarray(theta)

    # Check if phi is periodic (full circle)
    periodic_phi = np.abs(phi[-1] - phi[0] - 2 * np.pi) < 1e-10

    # Create grid
    verts, faces = fvgrid(phi, theta, periodic_u=periodic_phi)

    # Convert to Cartesian coordinates
    phi_v = verts[:, 0]
    theta_v = verts[:, 1]

    r = diameter / 2
    x = r * np.sin(theta_v) * np.cos(phi_v)
    y = r * np.sin(theta_v) * np.sin(phi_v)
    z = r * np.cos(theta_v)

    verts_3d = np.column_stack([x, y, z])

    # Convert quads to triangles for better quality
    tri_faces = []
    for face in faces:
        if len(face) == 4 and not np.isnan(face[3]):
            tri_faces.append([face[0], face[1], face[2]])
            tri_faces.append([face[0], face[2], face[3]])
        else:
            tri_faces.append(face[:3])

    return Particle(verts_3d, np.array(tri_faces), interp=interp, **kwargs)
