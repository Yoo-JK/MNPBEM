"""
Particle shape generators for MNPBEM.

This module provides functions to create common particle geometries:
- trisphere: Triangulated sphere
- tricube: Cube with rounded edges
- trirod: Rod-shaped particle (cylinder with hemispherical caps)
- tritorus: Torus (donut shape)
- trispheresegment: Spherical segment
- tripolygon: Particle from polygon extrusion
"""

from .trisphere import trisphere, sphtriangulate
from .tricube import tricube
from .trirod import trirod
from .tritorus import tritorus
from .trispheresegment import trispheresegment
from .tripolygon import tripolygon
from .utils import fvgrid

__all__ = [
    "trisphere",
    "sphtriangulate",
    "tricube",
    "trirod",
    "tritorus",
    "trispheresegment",
    "tripolygon",
    "fvgrid",
]
