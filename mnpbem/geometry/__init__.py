"""
Geometry and mesh generation module.

Classes:
- Particle: Basic particle with triangular mesh
- ComParticle: Compound particle with multiple materials
- ComParticleMirror: Compound particle with mirror symmetry
- Point: Collection of points
- ComPoint: Compound of points in a dielectric environment
- Polygon: 2D polygons for mesh generation
- LayerStructure: Dielectric layer structures

Functions:
- trisphere: Generate triangulated sphere
- trirod: Generate rod-shaped particle (cylinder with caps)
- tricube: Generate cube with rounded edges
- tritorus: Generate triangulated torus
- trispheresegment: Generate sphere segment
- trispherescale: Deform sphere surface
- triellipsoid: Generate triangulated ellipsoid
- fvgrid: Convert 2D grid to face-vertex structure
- connect: Compute connectivity between particles
- polygon_union: Combine multiple polygons
"""

from .particle import Particle
from .comparticle import ComParticle
from .comparticlemirror import ComParticleMirror
from .point import Point
from .compoint import ComPoint, Compound
from .polygon import Polygon, polygon_union
from .layerstructure import LayerStructure, LayerOptions, PositionStruct
from .mesh_generators import (
    trisphere, trirod, tricube, tritorus, trispheresegment,
    trispherescale, triellipsoid, fvgrid
)
from .connect import connect

__all__ = [
    "Particle",
    "ComParticle",
    "ComParticleMirror",
    "Point",
    "ComPoint",
    "Compound",
    "Polygon",
    "polygon_union",
    "LayerStructure",
    "LayerOptions",
    "PositionStruct",
    "trisphere",
    "trirod",
    "tricube",
    "tritorus",
    "trispheresegment",
    "trispherescale",
    "triellipsoid",
    "fvgrid",
    "connect",
]
