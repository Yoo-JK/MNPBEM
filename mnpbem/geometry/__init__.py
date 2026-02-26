"""
Geometry and mesh generation module.

Classes:
- Particle: Basic particle with triangular mesh
- ComParticle: Compound particle with multiple materials
- ComParticleMirror: Compound particle with mirror symmetry
- CompStructMirror: Structure for compound with mirror symmetry

Functions:
- trisphere: Generate triangulated sphere
- connect: Compute connectivity between particles
"""

from .particle import Particle
from .comparticle import ComParticle
from .comparticle_mirror import ComParticleMirror, CompStructMirror
from .mesh_generators import trisphere
from .connect import connect

__all__ = [
    "Particle",
    "ComParticle",
    "ComParticleMirror",
    "CompStructMirror",
    "trisphere",
    "connect",
]
