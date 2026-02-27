"""
Geometry and mesh generation module.

Classes:
- Particle: Basic particle with triangular mesh
- ComParticle: Compound particle with multiple materials
- ComParticleMirror: Compound particle with mirror symmetry
- CompStructMirror: Structure for compound with mirror symmetry
- EdgeProfile: Edge rounding profile for nanostructures

Functions:
- trisphere: Generate triangulated sphere
- trirod: Generate triangulated nanorod
- tricube: Generate triangulated nanocube with rounded edges
- tritorus: Generate triangulated torus
- trispheresegment: Generate triangulated sphere segment
- trispherescale: Scale a sphere to create ellipsoid
- fvgrid: Convert parametric surface to face-vertex structure
- connect: Compute connectivity between particles
"""

from .particle import Particle
from .comparticle import ComParticle
from .comparticle_mirror import ComParticleMirror, CompStructMirror
from .mesh_generators import (
    trisphere,
    trirod,
    tricube,
    tritorus,
    trispheresegment,
    trispherescale,
    fvgrid,
)
from .edgeprofile import EdgeProfile
from .connect import connect

__all__ = [
    "Particle",
    "ComParticle",
    "ComParticleMirror",
    "CompStructMirror",
    "EdgeProfile",
    "trisphere",
    "trirod",
    "tricube",
    "tritorus",
    "trispheresegment",
    "trispherescale",
    "fvgrid",
    "connect",
]
