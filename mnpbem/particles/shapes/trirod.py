"""
Triangulated rod (cylinder with hemispherical caps).
"""

import numpy as np
from typing import Tuple, List

from ..particle import Particle
from .trispheresegment import trispheresegment
from .utils import fvgrid


def trirod(
    diameter: float,
    height: float,
    n: Tuple[int, int, int] = (15, 20, 20),
    interp: str = 'curv',
    use_triangles: bool = False,
    **kwargs
) -> Particle:
    """
    Create rod-shaped particle (cylinder with hemispherical caps).

    Parameters
    ----------
    diameter : float
        Diameter of the rod.
    height : float
        Total height (length) of the rod including caps.
    n : tuple of int
        Number of discretization points [nphi, ntheta, nz].
    interp : str
        Interpolation type: 'flat' or 'curv'.
    use_triangles : bool
        Use triangles instead of quadrilaterals for cylinder.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Triangulated rod.

    Examples
    --------
    >>> rod = trirod(10, 50)  # 10 nm diameter, 50 nm total length
    """
    nphi, ntheta, nz = n

    # Cylinder height (without caps)
    cyl_height = height - diameter

    # Angles
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    theta = np.linspace(0, np.pi / 2, ntheta)

    # Upper hemispherical cap
    cap1 = trispheresegment(phi, theta, diameter, interp='flat')
    cap1 = cap1.shift([0, 0, cyl_height / 2])

    # Lower hemispherical cap (flipped)
    cap2 = cap1.flip(2)

    # Cylinder
    z = np.linspace(-cyl_height / 2, cyl_height / 2, nz)
    verts_cyl, faces_cyl = fvgrid(phi, z, periodic_u=True)

    # Convert to 3D coordinates
    phi_cyl = verts_cyl[:, 0]
    z_cyl = verts_cyl[:, 1]
    x_cyl = 0.5 * diameter * np.cos(phi_cyl)
    y_cyl = 0.5 * diameter * np.sin(phi_cyl)

    verts_3d = np.column_stack([x_cyl, y_cyl, z_cyl])

    # Convert quads to triangles if requested
    if use_triangles:
        tri_faces = []
        for face in faces_cyl:
            tri_faces.append([face[0], face[1], face[2]])
            tri_faces.append([face[0], face[2], face[3]])
        faces_cyl = np.array(tri_faces)

    cyl = Particle(verts_3d, faces_cyl, interp='flat')

    # Combine all parts
    p = cap1 + cap2 + cyl
    p = p.clean()

    # Set interpolation
    p.interp = interp

    return p
